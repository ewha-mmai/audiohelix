from __future__ import annotations

SYSTEM_PROMPT = "You are a helpful assistant."

# AF3 requires the option letter and exact option text for text-based scoring.
_ANSWER_FORMAT_RULE = (
    "Answer with the option letter followed by that option's exact text, copied "
    "verbatim from the list above — for example, if the correct option is 'B. 3', "
    "respond with exactly 'B. 3'. Do not describe what you heard or add explanation. "
    "If the option text is a number, respond with just the letter and that number."
)

_LABEL_MEANING = (
    "Each retrieved audio segment is labeled with its position within its "
    "source chunk, how many segments were considered from that chunk, and "
    "the time range it covers — use this to judge how complete the "
    "retrieved evidence is and to follow the order of events."
)
_UNIFIED_EVIDENCE_BASE = (
    "You are given audio segments retrieved from a long audio as relevant to the "
    "question, along with textual information associated with the retrieved "
    "segments. Carefully analyze the provided audio and text, integrating "
    "information across the retrieved segments to answer the question."
)
UNIFIED_EVIDENCE_TEXT = f"{_UNIFIED_EVIDENCE_BASE} {_LABEL_MEANING}"

_AF3_EVIDENCE_BASE = (
    "You are given concatenated audio segments retrieved from a long audio as "
    "relevant to the question, along with textual information associated with the "
    "retrieved segments. Carefully analyze the provided audio and text, integrating "
    "information across the retrieved segments to answer the question."
)
_AF3_EVIDENCE_TEXT = f"{_AF3_EVIDENCE_BASE} {_LABEL_MEANING}"

INSTRUCTION = (
    "Question: {question}\n\nOptions:\n{options}\n\n"
    + UNIFIED_EVIDENCE_TEXT + " Respond with the single correct option letter and "
    "its text, e.g. 'A. Boba tea'."
)
INSTRUCTION_NO_OPTIONS = f"Question: {{question}}\n\n{UNIFIED_EVIDENCE_TEXT}"

AF3_INSTRUCTION = (f"{_AF3_EVIDENCE_TEXT}\n\n"
                   "Question: {question}\n\n"
                   "Options:\n{options}\n\n"
                   f"{_ANSWER_FORMAT_RULE}")
AF3_INSTRUCTION_NO_OPTIONS = f"{_AF3_EVIDENCE_TEXT}\n\nQuestion: {{question}}"

EVIDENCE_FRAMING = (
    "You are given audio extracted from a video. The audio is a concatenation of the "
    "important segments of the video selected in advance as the key evidence needed to "
    "answer the question."
)


def instruction(record: dict, model: str, *, concat: bool = False,
                evidence_framing: bool = False) -> str:
    """Build the question and options instruction for one record."""
    if model == "audio_flamingo3":
        tmpl, tmpl_no_opt = AF3_INSTRUCTION, AF3_INSTRUCTION_NO_OPTIONS
    else:
        tmpl, tmpl_no_opt = INSTRUCTION, INSTRUCTION_NO_OPTIONS
    options = record.get("options", "")
    if options:
        base = tmpl.format(question=record["question"], options=options)
    else:
        base = tmpl_no_opt.format(question=record["question"])

    if evidence_framing and concat and model != "audio_flamingo3":
        return f"{EVIDENCE_FRAMING}\n\n{base}"
    return base


def system(model: str) -> str | None:
    """Return the system message, or None for Voxtral."""
    if model == "voxtral":
        return None
    return SYSTEM_PROMPT
