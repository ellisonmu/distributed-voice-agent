from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

import httpx

from voiceagent.audio.snr import estimate_snr_from_wav
from voiceagent.clients import ASRClient, LLMClient, Pool, TTSClient
from voiceagent.config import Config
from voiceagent.filler import FillerBank
from voiceagent.player import STALL_THRESHOLD_S, Player
from voiceagent.prompts import HEDGE, ROUTER, ROUTER_GENERATED_FILLER, build_messages
from voiceagent.rag import Passage
from voiceagent.records import StageRecord, TurnRecord
from voiceagent.scheduler import LatencyModel, Scheduler
from voiceagent.text import ESCALATIONS, RouterHead, SentenceSplitter


async def _prepend(head: str, rest: AsyncIterator[str]) -> AsyncIterator[str]:
    if head:
        yield head
    async for delta in rest:
        yield delta


class Orchestrator:
    def __init__(self, cfg: Config, http: httpx.AsyncClient, retriever, clock: Callable[[], float]):
        self.cfg, self.retriever, self.clock = cfg, retriever, clock
        self.pools = {
            "asr_small": Pool("asr_small", cfg.asr_small),
            "asr_large": Pool("asr_large", cfg.asr_large),
            "llm_small": Pool("llm_small", cfg.llm_small.urls),
            "llm_large": Pool("llm_large", cfg.llm_large.urls),
            "tts": Pool("tts", cfg.tts),
        }
        self.asr = {size: ASRClient(self.pools[f"asr_{size}"], http) for size in ("small", "large")}
        self.llm = {
            "small": LLMClient(self.pools["llm_small"], cfg.llm_small.model, http),
            "large": LLMClient(self.pools["llm_large"], cfg.llm_large.model, http),
        }
        if cfg.llm_turbo:
            self.pools["llm_turbo"] = Pool("llm_turbo", cfg.llm_turbo.urls)
            self.llm["turbo"] = LLMClient(self.pools["llm_turbo"], cfg.llm_turbo.model, http)
        self.tts = TTSClient(self.pools["tts"], http)
        self.scheduler = Scheduler(cfg, LatencyModel(cfg.latency), self.pools)
        self.filler = FillerBank()

    async def start(self) -> None:
        await self.filler.prerender(self.tts)

    async def handle_turn(self, rec: TurnRecord, wav: bytes) -> None:
        now = self.clock
        rec.t_start = now()
        rec.est_snr_db = await asyncio.to_thread(estimate_snr_from_wav, wav)

        rec.asr_model = self.scheduler.choose_asr(rec.policy, rec.est_snr_db)
        pool = self.pools[f"asr_{rec.asr_model}"]
        inflight, t0 = pool.inflight, now()
        rec.transcript = await self.asr[rec.asr_model].transcribe(wav)
        rec.t_asr_done = now()
        rec.stages.append(StageRecord(f"asr_{rec.asr_model}", inflight, t0, rec.t_asr_done))

        passages = await asyncio.to_thread(self.retriever.search, rec.transcript, int(self.cfg.rag.get("k", 3)))
        rec.t_retrieval_done = now()

        player = Player(self.tts, now, rec.stages)
        try:
            if rec.policy in ("always_small", "always_large", "always_turbo"):
                size = rec.policy.removeprefix("always_")
                rec.route, rec.llm_model = "plain", size
                rec.answer = await self._speak_new(size, build_messages(rec.transcript, passages), player, rec)
            else:
                await self._routed_turn(rec, passages, player)
            rec.t_playback_end = await player.finish()
        finally:
            player.cancel()

        stalls = [g for g in player.gaps if g > STALL_THRESHOLD_S]
        rec.stall_count, rec.stall_total_s = len(stalls), sum(stalls)
        rec.max_gap_s = max(player.gaps, default=0.0)

        if player.clips:
            rec.t_first_audio = player.clips[0].ready
            rec.t_audio_done = player.clips[-1].ready
            rec.ttfa = rec.t_first_audio - rec.t_start
            answer_clips = [c for c in player.clips if c.kind == "answer"]
            if answer_clips:
                rec.t_first_answer_audio = answer_clips[0].ready
                rec.ttsa = rec.t_first_answer_audio - rec.t_start

    async def _routed_turn(self, rec: TurnRecord, passages: list[Passage], player: Player) -> None:
        now = self.clock
        extra = ROUTER + (ROUTER_GENERATED_FILLER if self.cfg.filler_mode == "generated" else "")
        inflight, t0 = self.pools["llm_small"].inflight, now()
        # Only the routing call uses the fine-tuned router; baselines and hedged answers use the base 8B.
        small = self.llm["small"].stream(build_messages(rec.transcript, passages, extra), self.cfg.max_tokens,
                                         model=self.cfg.llm_small.router_model)
        try:
            head, after_token = RouterHead(), ""
            async for delta in small:
                after_token = head.feed(delta)
                if head.decision:
                    break
            rec.router_raw = head.buf[:40]
            rec.router_decision = head.decision or "UNKNOWN"
            rec.stages.append(StageRecord("llm_small_router", inflight, t0, now()))

            if head.decision not in ESCALATIONS:
                rec.route, rec.llm_model = "answer", "small"
                rec.answer = await self._stream_sentences(
                    _prepend(after_token, small), "llm_small_first_sentence", inflight, t0, rec,
                    lambda s: player.say(s, "answer"),
                )
                return

            tier = self.scheduler.choose_escalation(head.decision, now() - rec.t_start)
            if tier is None:
                await small.aclose()
                rec.route, rec.llm_model = "escalate_hedge", "small"
                hedged = build_messages(rec.transcript, passages, HEDGE)
                rec.answer = await self._speak_new("small", hedged, player, rec)
                return

            rec.route, rec.llm_model = f"escalate_{tier}", tier
            sentences: asyncio.Queue = asyncio.Queue()
            escalated = asyncio.create_task(
                self._tier_to_queue(tier, build_messages(rec.transcript, passages), sentences, rec))
            try:
                if rec.policy == "adaptive":
                    await self._play_filler(after_token, small, player, rec)
                await small.aclose()
                spoken = []
                while (sentence := await sentences.get()) is not None:
                    player.say(sentence, "answer")
                    spoken.append(sentence)
                await escalated
                rec.answer = " ".join(spoken)
            finally:
                if not escalated.done():
                    escalated.cancel()
        finally:
            await small.aclose()

    async def _play_filler(self, after_token: str, small: AsyncIterator[str], player: Player, rec: TurnRecord) -> None:
        if self.cfg.filler_mode == "generated":
            splitter = SentenceSplitter(min_chars=1)
            found = splitter.feed(after_token)
            if not found:
                async for delta in small:
                    if found := splitter.feed(delta):
                        break
                else:
                    found = splitter.flush()
            if found:
                rec.filler_text = found[0]
                player.say(found[0], "filler")
                return
        line, duration = self.filler.pick()
        rec.filler_text = line
        player.play_prerendered(line, duration)

    async def _tier_to_queue(self, tier: str, messages: list[dict], out: asyncio.Queue, rec: TurnRecord) -> None:
        inflight, t0 = self.pools[f"llm_{tier}"].inflight, self.clock()
        try:
            await self._stream_sentences(
                self.llm[tier].stream(messages, self.cfg.max_tokens),
                f"llm_{tier}_first_sentence", inflight, t0, rec, out.put_nowait,
            )
        finally:
            out.put_nowait(None)

    async def _speak_new(self, size: str, messages: list[dict], player: Player, rec: TurnRecord) -> str:
        inflight, t0 = self.pools[f"llm_{size}"].inflight, self.clock()
        return await self._stream_sentences(
            self.llm[size].stream(messages, self.cfg.max_tokens),
            f"llm_{size}_first_sentence", inflight, t0, rec, lambda s: player.say(s, "answer"),
        )

    async def _stream_sentences(self, deltas: AsyncIterator[str], key: str, inflight: int, t0: float,
                                rec: TurnRecord, on_sentence: Callable[[str], None]) -> str:
        splitter, parts, t_first = SentenceSplitter(), [], None
        async for delta in deltas:
            parts.append(delta)
            for sentence in splitter.feed(delta):
                t_first = t_first if t_first is not None else self.clock()
                on_sentence(sentence)
        for sentence in splitter.flush():
            t_first = t_first if t_first is not None else self.clock()
            on_sentence(sentence)
        rec.stages.append(StageRecord(key, inflight, t0, t_first if t_first is not None else self.clock()))
        return "".join(parts).strip()
