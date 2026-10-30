"""AudioMarathon instruction templates and prompt assembly."""

from __future__ import annotations

import re

SYSTEM_PROMPT = "You are a helpful assistant."

_TASK_DESCRIPTION = {
    "race": ("You are a helpful assistant that analyzes reading comprehension "
             "passages with audio narration. Please listen to the passage and "
             "answer the multiple-choice question based on what you heard."),
    "gtzan": "Listen to this audio segment and identify the music genre based on what you hear.",
    "tau": ("You are a helpful assistant that analyzes urban soundscape audio to "
            "identify acoustic scenes. Please listen to the audio carefully and "
            "classify the scene type."),
    "slue": ("You are a helpful assistant that analyzes speech audio for named "
             "entity recognition. Please listen carefully and extract the "
             "requested named entities from the speech."),
    "vesus": ("You are a helpful assistant that analyzes speech audio to recognize "
              "emotions. Please listen to the voice carefully and identify the "
              "emotional state of the speaker."),
    "desed": ("You are a helpful assistant that analyzes audio to detect and "
              "classify sound events. Please listen carefully and select the most "
              "appropriate answer from the given choices."),
    "vox_age": ("You are a helpful assistant that analyzes speech audio to estimate "
                "speaker age. Please listen to the voice carefully and classify the "
                "speaker's age group."),
    "had": "You are a helpful assistant that analyzes audio to determine authenticity.",
    "vox": ("You are a helpful assistant that analyzes speech audio to classify "
            "speaker gender. Please listen to the voice carefully and determine "
            "the speaker's gender."),
}

# had/vox: a fixed question that never changes per item -- scoring expects fixed a/b.
_HAD_QUESTION = (
    "Listen to this audio clip carefully. Is this audio completely authentic "
    "(real) or does it contain any artificially synthesized segments (fake)? "
    "If it is completely real, answer 'a'. If it contains any fake segments, "
    "answer 'b'. Answer with only 'a' or 'b'."
)
_VOX_QUESTION = (
    "Listen to this audio and identify the speaker's gender. Is this a male or "
    "female voice? If it is a male, answer 'a'. If it is a female, answer 'b'. "
    "Answer with only 'a' or 'b'."
)

# Accept the option formats emitted by the adapters.
_CHOICE_LINE = re.compile(r"^([A-Za-z])[.)\:]\s*(.+)$")


def _extract_choices(options_text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in options_text.splitlines():
        m = _CHOICE_LINE.match(line.strip())
        if m:
            out.append((m.group(1).upper(), m.group(2).strip()))
    return out


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

_EVIDENCE_FRAMING_BASE = (
    "You are given concatenated audio segments retrieved from a long audio as "
    "relevant to the question, along with textual information associated with the "
    "retrieved segments. Carefully analyze the provided audio and text, integrating "
    "information across the retrieved segments to answer the question."
)
EVIDENCE_FRAMING = f"{_EVIDENCE_FRAMING_BASE} {_LABEL_MEANING}"


def official_instruction(suite: str, question: str, options: str, model: str = "") -> str:
    """Build the suite instruction, including the AF3 variant."""
    is_af3 = model == "audio_flamingo3"
    if suite == "had":
        evidence = EVIDENCE_FRAMING if is_af3 else UNIFIED_EVIDENCE_TEXT
        task = _TASK_DESCRIPTION["had"]
        return f"{evidence}\n\n{task}\n\n{_HAD_QUESTION}"
    if suite == "vox":
        evidence = EVIDENCE_FRAMING if is_af3 else UNIFIED_EVIDENCE_TEXT
        task = _TASK_DESCRIPTION["vox"]
        return f"{evidence}\n\n{task}\n\n{_VOX_QUESTION}"

    choices = _extract_choices(options)
    if not choices:
        raise ValueError(f"{suite}: could not extract option lines from options -- check the format\n{options!r}")
    letters = [c for c, _ in choices]
    choice_block = "\n".join(f"{letter}. {text}" for letter, text in choices)
    letter_desc = (", ".join(letters[:-1]) + f", or {letters[-1]}") if len(letters) > 1 else letters[0]
    task_desc = _TASK_DESCRIPTION.get(suite, "")
    task_part = f"{task_desc}\n\n" if task_desc else ""
    format_rule = (
        f"Respond with only the letter of the correct option ({letter_desc}). "
        f"Do not output the full text of any option. Do not provide any "
        f"explanation, reasoning, commentary, or additional text. Your response "
        f"must contain ONLY the final answer letter and nothing else."
    )

    if is_af3:
        return f"{EVIDENCE_FRAMING} {format_rule}\n\n{task_part}Question: {question}\n\nOptions:\n{choice_block}"

    return f"{task_part}Question: {question}\n\nOptions:\n{choice_block}\n\n{UNIFIED_EVIDENCE_TEXT} {format_rule}"


def instruction(record: dict, model: str, *, concat: bool, evidence_framing: bool = False) -> str:
    """Build the instruction text for one question."""
    return official_instruction(record["suite"], record["question"],
                                record.get("options", ""), model)


def system(model: str) -> str | None:
    """Return the system text, or None for Voxtral."""
    return None if model == "voxtral" else SYSTEM_PROMPT


def prompt_txt(suites: list[str], model: str) -> str:
    """Build the prompt template saved with a run."""
    is_af3 = model == "audio_flamingo3"
    lines = [f"[SYSTEM] {SYSTEM_PROMPT}", ""]
    for s in suites:
        if s in ("had", "vox"):
            lines.append(f"[{s}] {official_instruction(s, '', '', model)}")
        elif is_af3:
            task = _TASK_DESCRIPTION.get(s, "")
            task_part = f"{task}\\n\\n" if task else ""
            lines.append(
                f"[{s}] {EVIDENCE_FRAMING} Respond with only the letter of "
                f"the correct option (...). Do not output the full text of any option. "
                f"Do not provide any explanation, reasoning, commentary, or additional "
                f"text. Your response must contain ONLY the final answer letter and "
                f"nothing else.\\n\\n{task_part}Question: {{question}}\\n\\nOptions:\\n{{extracted choices}}")
        else:
            task = _TASK_DESCRIPTION.get(s, "")
            task_part = f"{task}\\n\\n" if task else ""
            lines.append(
                f"[{s}] {task_part}Question: {{question}}\\n\\nOptions:\\n{{extracted choices}}"
                f"\\n\\n{UNIFIED_EVIDENCE_TEXT} Respond with only the letter of "
                f"the correct option (...). Do not output the full text of any option. "
                f"Do not provide any explanation, reasoning, commentary, or additional "
                f"text. Your response must contain ONLY the final answer letter and "
                f"nothing else.")
    return "\n".join(lines)
