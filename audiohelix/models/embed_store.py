from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from audiohelix.utils import compute_mdhash_id


class EmbeddingStore:
    def __init__(self, embedding_model, db_filename: str, batch_size: int, namespace: str):
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace

        self.hash_ids: List[str] = []
        self.texts: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.hash_id_to_text: dict[str, str] = {}
        self.hash_id_to_idx: dict[str, int] = {}
        self.text_to_hash_id: dict[str, str] = {}

        self._load_data()

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def insert(self, ids: List[str], contents: List[str], embed_inputs: Optional[List] = None,
               embeddings: Optional[np.ndarray] = None) -> None:
        """Stores ids/contents, encoding embed_inputs (default: contents).

        If `embeddings` is given, uses it directly instead of encoding. Ids
        already present are skipped (resume support)."""
        ids = list(ids)
        contents = list(contents)
        assert len(ids) == len(contents)
        if embed_inputs is None:
            embed_inputs = contents
        if embeddings is not None:
            assert embeddings.shape[0] == len(ids)

        nodes = {}
        for node_id, content, embed_input in zip(ids, contents, embed_inputs):
            h = compute_mdhash_id(str(node_id), prefix=self.namespace + "-")
            nodes[h] = {"content": content, "embed_input": embed_input, "id": str(node_id)}

        all_hash_ids = list(nodes.keys())
        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]
        if not missing_ids:
            return

        texts_to_encode = [nodes[h]["embed_input"] for h in missing_ids]

        if embeddings is not None:
            emb_by_id = {str(i): e for i, e in zip(ids, embeddings)}
            new_embeddings = np.stack([emb_by_id[nodes[h]["id"]] for h in missing_ids])
        else:
            new_embeddings = self.embedding_model.encode(
                texts_to_encode, normalize_embeddings=True,
                show_progress_bar=False, batch_size=self.batch_size,
            )
        new_texts = [nodes[h]["content"] for h in missing_ids]
        self._upsert(missing_ids, new_texts, new_embeddings)

    def insert_text(self, text_list: List[str]) -> None:
        """For nodes where id == text (text units / text concepts)."""
        self.insert(ids=text_list, contents=text_list)

    def _upsert(self, hash_ids, texts, embeddings) -> None:
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)

        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}

        self._save_data()

    def _save_data(self) -> None:
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "text": self.texts,
            "embedding": self.embeddings,
        })
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        data_to_save.to_parquet(self.db_filename, index=False)

    def _load_data(self) -> None:
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["text"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()

            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
            self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
            print(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def hash_id_of(self, node_id: str) -> str:
        """Deterministic node id -> hash_id, matching insert()'s rule."""
        return compute_mdhash_id(str(node_id), prefix=self.namespace + "-")

    def get_hash_id_to_text(self):
        from copy import deepcopy
        return deepcopy(self.hash_id_to_text)

    def encode_texts(self, texts):
        return self.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size
        )

    def get_embeddings(self, hash_ids):
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings)[indices]
        return embeddings
