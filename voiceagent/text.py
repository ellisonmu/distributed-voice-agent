from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+")
_ABBREVIATIONS = {"e.g.", "i.e.", "dr.", "mr.", "mrs.", "ms.", "vs.", "etc.", "approx.", "no.", "st."}
ROUTER_TOKENS = ("ANSWER", "ESCALATE", "EXPERT")
ESCALATIONS = ("ESCALATE", "EXPERT")


class SentenceSplitter:
    """Turns a token stream into complete sentences, so TTS can start on sentence one
    while the LLM is still writing sentence two."""

    def __init__(self, min_chars: int = 20):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out, pos = [], 0
        while m := _SENTENCE_END.search(self.buf, pos):
            candidate = self.buf[: m.end()].strip()
            last_word = candidate.split()[-1].lower() if candidate else ""
            if len(candidate) >= self.min_chars and last_word not in _ABBREVIATIONS:
                out.append(candidate)
                self.buf, pos = self.buf[m.end() :], 0
            else:
                pos = m.end()
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


class RouterHead:
    """Reads the small LLM's leading ANSWER/ESCALATE/EXPERT token off a stream. feed() returns the
    text that follows the token (to be spoken or used as filler); until the token is
    decided it returns ''."""

    def __init__(self):
        self.buf = ""
        self.decision: str | None = None
        self._leading = True

    def feed(self, delta: str) -> str:
        if self.decision is None:
            self.buf += delta
            s = self.buf.lstrip()
            upper = s.upper()
            for token in ROUTER_TOKENS:
                if upper.startswith(token):
                    self.decision, delta = token, s[len(token) :]
                    break
            else:
                if any(token.startswith(upper) for token in ROUTER_TOKENS):
                    return ""
                self.decision, delta = "UNKNOWN", s
        if self._leading:
            delta = delta.lstrip(" :-\n\t")
            self._leading = not delta
        return delta
