from __future__ import annotations

from voiceagent.rag import Passage

SAFETY = (
    "You are a voice assistant that gives general consumer health information. "
    "Answer only from the reference passages below; if they do not cover the question, say so. "
    "Do not diagnose or prescribe. If the caller describes an emergency, tell them to call emergency services now. "
    "Your reply is spoken aloud: use plain sentences, no lists or markdown, at most four sentences."
)

ROUTER = (
    "Begin your reply with exactly one word: ANSWER, ESCALATE or EXPERT. "
    "Write ANSWER followed by your answer if the passages let you answer accurately and completely. "
    "Write ESCALATE if the question needs more care than you can give, so a more capable model should answer it. "
    "Write EXPERT if it needs specialist depth, such as drug interactions, rare conditions or treatment details, "
    "so the most capable model should answer it."
)

ROUTER_GENERATED_FILLER = (
    " After ESCALATE or EXPERT, write one short, natural sentence telling the caller you are looking into their "
    "question. That sentence must not contain any medical information."
)

HEDGE = (
    "You may not have enough information to answer fully. Give a brief, cautious answer "
    "and tell the caller to confirm with a clinician."
)


def build_messages(question: str, passages: list[Passage], extra: str = "") -> list[dict]:
    context = "\n\n".join(f"[{i + 1}] {p.title}: {p.text}" for i, p in enumerate(passages))
    system = f"{SAFETY} {extra}".strip() + f"\n\nReference passages:\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]
