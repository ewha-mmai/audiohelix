from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Caption/transcription/concept extraction
# ---------------------------------------------------------------------------

CAPTION_PROMPT = (
    'Listen to the audio clip and produce exactly three labeled lines.\n\n'
    'Follow these rules:\n\n'
    '- Line 1 starts with "Transcription:" followed by everything spoken, word-for-word, in whatever language(s) it was actually spoken in. Do not translate, paraphrase, or summarize. First check whether the clip contains any intelligible speech, commentary, dialogue, narration, or sung words. If any of these are audible, transcribe them. Use "No speech" only when none of them is audible anywhere in the clip. For singing, transcribe only words actually audible in this clip. Do not continue a lyric from prior knowledge. Do not repeat a completed lyric line unless it is audibly repeated again. Transcribe the audio only once, in the order it is heard, and stop when the audible speech ends.\n\n'
    '- Line 2 starts with "Caption:" followed by declarative sentences describing the audible characteristics of the audio. Caption only audible evidence. Do not repeat the transcribed words; describe music, sound events, ambience, and vocal delivery instead. The Caption should contain at least two distinct audible details whenever the clip provides them. Prefer specific audible details over generic statements. Never write "No speech" on the Caption line.\n\n'
    '- Line 3 starts with "Entities:" followed by a comma-separated list of semantic objects or semantic concepts mentioned in the Transcription or Caption that could serve as knowledge-graph nodes. Include meaningful people, animals, objects, places, organizations, works, events, or distinctive concepts that have a specific referent or identifiable meaning in the audio. Exclude overly generic audio terms such as music, speech, singing, ambience, or sound effects. Do not repeat any entity.\n\n'
    'Output exactly these three lines and nothing else.'
)

CAPTION_ASSISTANT_PREFIX = "Transcription:"


def parse_raw_output(raw_text: str) -> dict:
    """Parses "Transcription: ...\\nCaption: ...\\nEntities: a, b, c" into
    {"transcription", "caption", "concepts"}."""
    caption_idx = raw_text.find("Caption:")
    concepts_idx = raw_text.find("Entities:")

    if caption_idx == -1:
        transcription = raw_text.strip()
        if transcription.startswith("Transcription:"):
            transcription = transcription[len("Transcription:"):].strip()
        return {"transcription": transcription, "caption": "", "concepts": []}

    transcription = raw_text[:caption_idx].strip()
    if transcription.startswith("Transcription:"):
        transcription = transcription[len("Transcription:"):].strip()

    if concepts_idx == -1:
        caption = raw_text[caption_idx + len("Caption:"):].strip()
        concepts: list[str] = []
    else:
        caption = raw_text[caption_idx + len("Caption:"):concepts_idx].strip()
        concepts_str = raw_text[concepts_idx + len("Entities:"):].strip()
        concepts = [e.strip().strip('"').strip("'") for e in concepts_str.split(",") if e.strip()]

    return {"transcription": transcription, "caption": caption, "concepts": concepts}


def options_text(options) -> str:
    """`options` may be a JSON string or a dict (TraceAV questions file)."""
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            options = {}
    return "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def normalize_option_letters(raw) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        c = str(x).strip().upper()
        if c in {"A", "B", "C", "D"} and c not in seen:
            seen.add(c)
            out.append(c)
    return sorted(out)


def parse_response_to_prediction(text: str) -> list[str]:
    """Extracts A-D letters from generated text anywhere -- handles multi-answer
    (comma-separated), code blocks, and JSON ("answer" key)."""
    s = (text or "").strip()
    if s.startswith("```json"):
        s = s[7:].strip()
    if s.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    try:
        obj = json.loads(s)
        ans = obj.get("answer")
        if isinstance(ans, list):
            return normalize_option_letters(ans)
        if isinstance(ans, str):
            return normalize_option_letters(re.findall(r"[A-D]", ans.upper()))
    except Exception:
        pass
    m = re.search(r"Answer:\s*([A-D](?:\s*[,;/]\s*[A-D])*)", text or "", re.IGNORECASE | re.DOTALL)
    if m:
        return normalize_option_letters(re.findall(r"[A-D]", m.group(1).upper()))
    return normalize_option_letters(re.findall(r"\b([A-D])\b", (text or "").upper()))


def normalize_correct_options(correct_options) -> list[str]:
    """Normalizes a questions-file correct_options value (e.g. a '["C"]' JSON string)."""
    if isinstance(correct_options, str):
        try:
            correct_options = json.loads(correct_options)
        except Exception:
            correct_options = [correct_options]
    return normalize_option_letters(correct_options)
