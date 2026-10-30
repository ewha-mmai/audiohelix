from __future__ import annotations

import functools
import os
from typing import Any, Dict, List, Protocol

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from audiohelix.config import LLMConfig


class OmniLLMBackend(Protocol):
    def caption_audio(self, audio_path: str, instruction: str, assistant_prefix: str = "Transcription:") -> str:
        ...

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512) -> str:
        """Text-only generate (used for question NER, etc.)."""
        ...

    def _hf_generate(self, conversation: List[Dict[str, Any]], max_new_tokens: int,
                     assistant_prefix: str | None = None,
                     do_sample: bool = False, no_repeat_ngram_size: int | None = None,
                     min_new_tokens: int = 0) -> str:
        """Multimodal conversation -> generated text."""
        ...


def _install_lm_head_patch(model) -> None:
    """Avoids a prefill OOM by applying lm_head to only the last token
    (hidden (B, L, d) -> (B, 1, d) at prefill time), which is lossless since
    generation only ever needs the last position's logits."""

    def _last_token_lm_head_hook(module, args):
        hidden = args[0]
        if isinstance(hidden, torch.Tensor) and hidden.dim() == 3 and hidden.shape[1] > 1:
            return (hidden[:, -1:, :],) + args[1:]
        return None

    model.thinker.lm_head.register_forward_pre_hook(_last_token_lm_head_hook)


class Qwen25OmniBackend:
    def __init__(self, model_id: str | None = None):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        model_id = model_id or LLMConfig().qwen25_model_id
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
        ).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        _install_lm_head_patch(self.model)

        # Drops input_features when no audio placeholder token is present in
        # input_ids, since Qwen2.5 otherwise rejects audio inputs on text-only calls.
        audio_token_id = getattr(self.model.config.thinker_config, "audio_token_index", None)
        if audio_token_id is None:
            return
        thinker = self.model.thinker
        _orig_forward = thinker.forward

        @functools.wraps(_orig_forward)
        def _patched_forward(self_, input_ids=None, input_features=None, **kwargs):
            if (
                input_features is not None
                and input_ids is not None
                and not bool((input_ids == audio_token_id).any())
            ):
                input_features = None
            return _orig_forward(input_ids=input_ids, input_features=input_features, **kwargs)

        thinker.forward = _patched_forward.__get__(thinker, type(thinker))

    def _hf_generate_batch(self, conversations: list, max_new_tokens: int,
                            do_sample: bool = False, min_new_tokens: int = 0) -> list[str]:
        """Batches several conversations into one generate() call (QA only, no
        assistant_prefix support). Requires left-padding: _install_lm_head_patch
        takes only the last prefill position, which must be the true last
        token for every sample in the batch -- right-padding would put a pad
        token there for shorter samples."""
        from qwen_omni_utils import process_mm_info

        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
        kwargs = dict(
            text=texts, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False, truncation=False,
        )
        orig_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        try:
            try:
                inputs = self.processor(audio=audios, **kwargs)
            except TypeError:
                inputs = self.processor(audios=audios, **kwargs)
        finally:
            self.processor.tokenizer.padding_side = orig_padding_side
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                use_audio_in_video=False, return_audio=False,
            )
        input_len = inputs["input_ids"].shape[1]  # left-padding -- same length for all in batch
        raws = self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        del inputs, output_ids
        torch.cuda.empty_cache()
        return [r.strip() for r in raws]

    def _hf_generate(self, conversation, max_new_tokens, assistant_prefix=None,
                     do_sample=False, no_repeat_ngram_size=None,
                     min_new_tokens=0) -> str:
        # min_new_tokens is ignored -- Qwen2.5-Omni has no such concept (kept for signature parity).
        from qwen_omni_utils import process_mm_info

        if assistant_prefix:
            conversation = conversation + [
                {"role": "assistant", "content": [{"type": "text", "text": assistant_prefix}]}
            ]
            text = self.processor.apply_chat_template(
                conversation, continue_final_message=True, tokenize=False,
            )
        else:
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False,
            )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        kwargs = dict(
            text=text, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False, truncation=False,
        )
        try:
            inputs = self.processor(audio=audios, **kwargs)
        except TypeError:
            inputs = self.processor(audios=audios, **kwargs)
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                use_audio_in_video=False, return_audio=False,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        del inputs, output_ids
        torch.cuda.empty_cache()
        if assistant_prefix:
            raw = assistant_prefix + raw
        return raw

    def caption_audio(self, audio_path, instruction, assistant_prefix="Transcription:") -> str:
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": instruction},
            ]},
        ]
        return self._hf_generate(conversation, max_new_tokens=1024, assistant_prefix=assistant_prefix)

    def generate(self, messages, max_new_tokens=512) -> str:
        conversation = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        return self._hf_generate(conversation, max_new_tokens=max_new_tokens)


class Qwen3OmniBackend:
    def __init__(self, model_id: str | None = None):
        from transformers import Qwen3OmniMoeForConditionalGeneration, AutoProcessor

        model_id = model_id or LLMConfig().qwen3_model_id
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        _install_lm_head_patch(self.model)

    def _hf_generate(self, conversation, max_new_tokens, assistant_prefix=None,
                     do_sample=False, no_repeat_ngram_size=None,
                     min_new_tokens=0) -> str:
        from qwen_omni_utils import process_mm_info

        if assistant_prefix:
            conversation = conversation + [
                {"role": "assistant", "content": [{"type": "text", "text": assistant_prefix}]}
            ]
            text = self.processor.apply_chat_template(
                conversation, continue_final_message=True, tokenize=False,
            )
        else:
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False,
            )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        # truncation=False set explicitly: the processor default is already False,
        # but WhisperFeatureExtractor's own default is True + max_length=30s,
        # so this is set explicitly as a guard against future transformers versions.
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False, truncation=False,
        )
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        # thinker_* prefixed kwargs get stripped and forwarded to self.thinker.generate.
        gen_kwargs: dict = dict(
            thinker_max_new_tokens=max_new_tokens,
            thinker_min_new_tokens=min_new_tokens,
            thinker_do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(thinker_temperature=0.6, thinker_top_p=0.95, thinker_top_k=20)
        if no_repeat_ngram_size is not None:
            gen_kwargs["thinker_no_repeat_ngram_size"] = no_repeat_ngram_size
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, use_audio_in_video=False, return_audio=False, **gen_kwargs,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        del inputs, output_ids
        torch.cuda.empty_cache()
        if assistant_prefix:
            raw = assistant_prefix + raw
        return raw

    def caption_audio(self, audio_path, instruction, assistant_prefix="Transcription:") -> str:
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": instruction},
            ]},
        ]
        # min_new_tokens=32 avoids an empty <think></think> block on captioning.
        return self._hf_generate(
            conversation, max_new_tokens=1024, assistant_prefix=assistant_prefix,
            do_sample=False, min_new_tokens=LLMConfig().caption_min_new_tokens,
        )

    def generate(self, messages, max_new_tokens=512) -> str:
        conversation = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        return self._hf_generate(conversation, max_new_tokens=max_new_tokens)


class VoxtralBackend:
    def __init__(self, model_id: str | None = None):
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        model_id = model_id or LLMConfig().voxtral_model_id

        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=True,
        )

        # Avoids an OOM from the audio tower's eager attention on long audio.
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "audio_tower"):
                self.model.model.audio_tower.config._attn_implementation = "sdpa"
            if hasattr(self.model.model, "language_model"):
                self.model.model.language_model.config._attn_implementation = "sdpa"

    @staticmethod
    def _normalize_conversation(conversation):
        """Converts a Qwen-style conversation into Voxtral's input format."""
        normalized = []

        for message in conversation:
            # Dropped: this Voxtral tokenizer version rejects a system role before audio.
            if message.get("role") == "system":
                continue

            content = []
            for item in message.get("content", []):
                item_type = item.get("type")

                if item_type == "audio":
                    audio_value = item.get("path", item.get("audio"))

                    if not isinstance(audio_value, str):
                        raise TypeError(
                            "Voxtral audio input must be a file path string. "
                            "Use separate audio paths for Voxtral."
                        )

                    content.append({
                        "type": "audio",
                        "path": audio_value,
                    })
                else:
                    content.append(item)

            normalized.append({
                "role": message.get("role", "user"),
                "content": content,
            })

        return normalized

    def _hf_generate(
        self,
        conversation,
        max_new_tokens,
        assistant_prefix=None,
        do_sample=False,
        no_repeat_ngram_size=None,
        min_new_tokens=0,
    ) -> str:
        conversation = self._normalize_conversation(conversation)

        if assistant_prefix:
            conversation = conversation + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": assistant_prefix,
                        }
                    ],
                }
            ]

        # mistral-common requires continue_final_message=True for a conversation
        # ending in an assistant message (prefix continuation). The Voxtral
        # processor drops that named kwarg in _merge_kwargs, so it's passed
        # via the text_kwargs group instead.
        pk = {}
        if assistant_prefix:
            pk["text_kwargs"] = {"continue_final_message": True}
        inputs = self.processor.apply_chat_template(
            conversation,
            processor_kwargs=pk,
        )

        device = next(self.model.parameters()).device
        inputs = inputs.to(device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if no_repeat_ngram_size is not None:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        if min_new_tokens:
            generate_kwargs["min_new_tokens"] = min_new_tokens

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        raw = self.processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        del inputs, output_ids
        torch.cuda.empty_cache()

        if assistant_prefix:
            raw = assistant_prefix + raw

        return raw

    def caption_audio(
        self,
        audio_path,
        instruction,
        assistant_prefix="Transcription:",
    ) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": audio_path,
                    },
                    {
                        "type": "text",
                        "text": instruction,
                    },
                ],
            }
        ]

        return self._hf_generate(
            conversation,
            max_new_tokens=LLMConfig().caption_max_new_tokens,
            assistant_prefix=assistant_prefix,
            do_sample=False,
            min_new_tokens=LLMConfig().caption_min_new_tokens,
        )

    def generate(self, messages, max_new_tokens=512) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message["content"],
                    }
                    for message in messages
                    if message.get("role") != "system"
                ],
            }
        ]

        return self._hf_generate(
            conversation,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )


class AudioFlamingo3Backend:
    SAMPLE_RATE = 16000

    def __init__(self, model_id: str | None = None):
        from transformers import (
            AudioFlamingo3ForConditionalGeneration,
            AutoProcessor,
        )

        model_id = model_id or LLMConfig().audio_flamingo3_model_id

        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=True,
        )

        # Removes the processor's default 600s audio length cap.
        self.processor.max_audio_len = 1_000_000

    @classmethod
    def _load_audio(cls, audio_path: str):
        import librosa

        waveform, _ = librosa.load(
            audio_path,
            sr=cls.SAMPLE_RATE,
            mono=True,
        )

        return waveform.astype("float32")

    @classmethod
    def _normalize_conversation(cls, conversation):
        """Converts to AF3's input format. AF3 accepts only one audio per
        conversation, so the caller is expected to already pass a concatenated
        waveform."""

        normalized = []

        for message in conversation:
            content = []

            for item in message.get("content", []):
                item_type = item.get("type")

                if item_type == "audio":
                    audio_value = item.get("audio")

                    if isinstance(audio_value, str):
                        audio_value = cls._load_audio(audio_value)

                    content.append({
                        "type": "audio",
                        "audio": audio_value,
                    })
                else:
                    content.append(item)

            normalized.append({
                "role": message.get("role", "user"),
                "content": content,
            })

        return normalized

    def _hf_generate(
        self,
        conversation,
        max_new_tokens,
        assistant_prefix=None,
        do_sample=False,
        no_repeat_ngram_size=None,
        min_new_tokens=0,
    ) -> str:
        conversation = self._normalize_conversation(conversation)

        if assistant_prefix:
            conversation = conversation + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": assistant_prefix,
                        }
                    ],
                }
            ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=not bool(assistant_prefix),
            continue_final_message=bool(assistant_prefix),
            return_dict=True,
        )

        device = next(self.model.parameters()).device
        moved_inputs = {}

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.float32:
                    value = value.to(torch.bfloat16)
                moved_inputs[key] = value.to(device)
            else:
                moved_inputs[key] = value

        input_len = moved_inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if no_repeat_ngram_size is not None:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        if min_new_tokens:
            generate_kwargs["min_new_tokens"] = min_new_tokens

        with torch.inference_mode():
            output_ids = self.model.generate(
                **moved_inputs,
                **generate_kwargs,
            )

        raw = self.processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        del moved_inputs, output_ids
        torch.cuda.empty_cache()

        if assistant_prefix:
            raw = assistant_prefix + raw

        return raw

    def caption_audio(
        self,
        audio_path,
        instruction,
        assistant_prefix="Transcription:",
    ) -> str:
        waveform = self._load_audio(audio_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    },
                    {
                        "type": "audio",
                        "audio": waveform,
                    },
                ],
            }
        ]

        return self._hf_generate(
            conversation,
            max_new_tokens=LLMConfig().caption_max_new_tokens,
            assistant_prefix=assistant_prefix,
            do_sample=False,
            min_new_tokens=LLMConfig().caption_min_new_tokens,
        )

    def generate(self, messages, max_new_tokens=512) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message["content"],
                    }
                    for message in messages
                    if message.get("role") != "system"
                ],
            }
        ]

        return self._hf_generate(
            conversation,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )


OMNI_LLM_REGISTRY = {
    "qwen25_omni": Qwen25OmniBackend,
    "qwen3_omni": Qwen3OmniBackend,
    "voxtral": VoxtralBackend,
    "audio_flamingo3": AudioFlamingo3Backend,
}


def build_omni_llm(name: str = "qwen3_omni", **kwargs) -> OmniLLMBackend:
    if name not in OMNI_LLM_REGISTRY:
        raise ValueError(f"Unknown Omni LLM backend: {name!r}. Choose from {list(OMNI_LLM_REGISTRY)}")
    return OMNI_LLM_REGISTRY[name](**kwargs)
