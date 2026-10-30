from __future__ import annotations

from audiohelix.prompts import options_text

_PROMPT_MULTI = """\
You are given audio segments retrieved from a long audio as relevant to the question, along with textual information associated with the retrieved segments. Carefully analyze the provided audio and text, integrating information across the retrieved segments to answer the question. Provide only the letter(s) representing your choice (A/B/C/D). Do not output the full text of any option. Do not provide any explanation, reasoning, commentary, or additional text. Your response must contain ONLY the final answer letter(s) and nothing else. The problem could be a single-choice question or a multiple-choice question. If multiple options are correct, return letters joined by commas (example: A,C). Output each correct letter exactly once.

Question:
{question}

Options:
{options}"""

_PROMPT_CONCAT = """\
You are given concatenated audio segments retrieved from a long audio as relevant to the question, along with textual information associated with the retrieved segments. Carefully analyze the provided audio and text, integrating information across the retrieved segments to answer the question. Provide only the letter(s) representing your choice (A/B/C/D). Do not output the full text of any option. Do not provide any explanation, reasoning, commentary, or additional text. Your response must contain ONLY the final answer letter(s) and nothing else. The problem could be a single-choice question or a multiple-choice question. If multiple options are correct, return letters joined by commas (example: A,C). Output each correct letter exactly once.

Question:
{question}

Options:
{options}"""

_SYSTEM = "You are a helpful assistant."


def instruction(record: dict, model: str, *, concat: bool, evidence_framing: bool = False) -> str:
    """Build the TraceAV instruction for one question."""
    del concat, evidence_framing
    tpl = _PROMPT_CONCAT if model == "audio_flamingo3" else _PROMPT_MULTI
    try:
        return tpl.format(question=record["question"],
                          options=options_text(record.get("options", "")))
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(
            "instruction substitution failed -- only {question} and {options} "
            f"placeholders are allowed, any other brace must be doubled ({e})"
        ) from e


def prompt_source_of(model: str) -> str:
    """Return the instruction template name used by the model."""
    return "_PROMPT_CONCAT" if model == "audio_flamingo3" else "_PROMPT_MULTI"


def system(model: str) -> str | None:
    """Return the system message, or None for Voxtral and AF3."""
    if model in {"voxtral", "audio_flamingo3"}:
        return None
    return _SYSTEM
