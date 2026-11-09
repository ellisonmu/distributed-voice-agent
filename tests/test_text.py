from voiceagent.text import RouterHead, SentenceSplitter


def feed_all(splitter, chunks):
    out = []
    for c in chunks:
        out += splitter.feed(c)
    return out + splitter.flush()


def test_splitter_emits_sentences_as_they_complete():
    s = SentenceSplitter()
    assert s.feed("Asthma is a long-term lung condition. It ") == ["Asthma is a long-term lung condition."]
    assert s.feed("causes wheezing") == []
    assert s.flush() == ["It causes wheezing"]


def test_splitter_does_not_split_on_abbreviations_or_short_fragments():
    text = "Talk to your Dr. Smith about inhalers, e.g. steroid inhalers. Ok. Then follow the plan."
    tokens = [w + " " for w in text.split()]
    assert feed_all(SentenceSplitter(), tokens) == [
        "Talk to your Dr. Smith about inhalers, e.g. steroid inhalers.",
        "Ok. Then follow the plan.",
    ]


def test_router_head_split_across_tokens():
    h = RouterHead()
    assert h.feed("ESC") == "" and h.decision is None
    assert h.feed("ALATE") == "" and h.decision == "ESCALATE"
    assert h.feed(": Let me") == "Let me"
    assert h.feed(" check.") == " check."


def test_router_head_waits_while_escalate_and_expert_are_ambiguous():
    h = RouterHead()
    assert h.feed("E") == "" and h.decision is None
    assert h.feed("XPERT: One") == "One" and h.decision == "EXPERT"


def test_router_head_answer_and_unknown():
    h = RouterHead()
    assert h.feed("ANSWER Asthma is") == "Asthma is" and h.decision == "ANSWER"
    u = RouterHead()
    assert u.feed("Asthma is") == "Asthma is" and u.decision == "UNKNOWN"
