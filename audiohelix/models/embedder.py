from __future__ import annotations

from typing import List, Protocol

import numpy as np
import torch


class ChunkEmbedder(Protocol):
    def encode_audio(self, audio_paths: List[str]) -> np.ndarray:
        """Audio chunk wav paths -> (N, dim) L2-normalized embeddings."""
        ...

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Text -> (N, dim) L2-normalized embeddings, same space as encode_audio."""
        ...


class PEAVChunkEmbedder:
    def __init__(self, model_id: str = "facebook/pe-av-large", device: str = "cuda"):
        from transformers import PeAudioVideoModel, PeAudioVideoProcessor

        self.device = device
        self.model = PeAudioVideoModel.from_pretrained(model_id).to(device).eval()
        self.processor = PeAudioVideoProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def encode_audio(self, audio_paths: List[str], batch_size: int = 8) -> np.ndarray:
        """Batched to avoid OOM -- PE-AV's conv stack uses GB-scale activations per clip."""
        audio_paths = list(audio_paths)
        if not audio_paths:
            return np.zeros((0, 0))
        chunks = []
        for i in range(0, len(audio_paths), batch_size):
            inputs = self.processor(
                audio=audio_paths[i:i + batch_size], return_tensors="pt", padding=True,
            ).to(self.device)
            embeds = self.model.get_audio_embeds(**inputs)
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
            chunks.append(embeds.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    @torch.inference_mode()
    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batched -- long/many units otherwise pad to a huge single forward pass."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0))
        chunks = []
        for i in range(0, len(texts), batch_size):
            inputs = self.processor(
                text=texts[i:i + batch_size], return_tensors="pt", padding=True,
            ).to(self.device)
            # get_text_audio_embeds() has a library bug; call text_model + text_audio_head directly.
            text_outputs = self.model.audio_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
                return_dict=True,
            )
            text_audio_embeds = text_outputs.hidden_states[-1][:, 0]
            embeds = self.model.audio_model.text_audio_head(text_audio_embeds)
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
            chunks.append(embeds.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)


CHUNK_EMBEDDER_REGISTRY = {
    "pe_av": PEAVChunkEmbedder,
}


def build_chunk_embedder(name: str = "pe_av", **kwargs) -> ChunkEmbedder:
    if name not in CHUNK_EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown chunk embedder: {name!r}. Choose from {list(CHUNK_EMBEDDER_REGISTRY)}")
    return CHUNK_EMBEDDER_REGISTRY[name](**kwargs)


class _EmbeddingStoreEncodeView:
    """Thin adapter giving ChunkEmbedder the
    .encode(items, normalize_embeddings=True, show_progress_bar=False, batch_size=...)
    signature EmbeddingStore expects."""

    def __init__(self, embedder: ChunkEmbedder, modality: str):
        assert modality in ("audio", "text")
        self.embedder = embedder
        self.modality = modality

    def encode(self, items, normalize_embeddings: bool = True, show_progress_bar: bool = False, batch_size: int = 32):
        items = list(items)
        if not items:
            return np.zeros((0, 0))
        if self.modality == "audio":
            return self.embedder.encode_audio(items, batch_size=batch_size)
        return self.embedder.encode_text(items, batch_size=batch_size)


def as_audio_encoder(embedder: ChunkEmbedder) -> "_EmbeddingStoreEncodeView":
    return _EmbeddingStoreEncodeView(embedder, "audio")


def as_text_encoder(embedder: ChunkEmbedder) -> "_EmbeddingStoreEncodeView":
    return _EmbeddingStoreEncodeView(embedder, "text")
