from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Cluster:
    concept_id: str            # f"cpt{seq:04d}"
    centroid: np.ndarray      # (dim,) float32, L2-normalized
    member_unit_ids: List[str]
    seq: int = 0


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def cluster_spectral(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Spectral clustering with a precomputed cosine-similarity affinity matrix.
    Negative similarities are clipped to 0 (treated as "no edge")."""
    from sklearn.cluster import SpectralClustering

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    affinity = np.clip(unit @ unit.T, 0.0, None)
    np.fill_diagonal(affinity, 1.0)

    model = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed",
        assign_labels="kmeans", random_state=random_state,
    )
    return model.fit_predict(affinity).astype(np.int64)


def cluster_unit_features(
    embeddings: np.ndarray,
    unit_ids: List[str],
    cfg=None,
) -> List[Cluster]:
    """cfg is a audiohelix.config.ClusterConfig (defaults to one if omitted).
    A cluster of one member becomes its own singleton Cluster."""
    from audiohelix.config import ClusterConfig

    if cfg is None:
        cfg = ClusterConfig()
    if not cfg.spectral_n_clusters:
        raise ValueError(
            "ClusterConfig.spectral_n_clusters is unset -- the index path tag "
            "would not name the k actually used.")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(unit_ids)
    assert embeddings.shape[0] == n, f"feature count ({embeddings.shape[0]}) != unit count ({n})"
    if n == 0:
        return []

    labels = cluster_spectral(embeddings, cfg.spectral_n_clusters)

    clusters: List[Cluster] = []
    for label in sorted(np.unique(labels)):  # sorted for deterministic output
        member_idx = np.where(labels == label)[0]
        member_ids = [unit_ids[i] for i in member_idx.tolist()]
        centroid = _normalize(embeddings[member_idx].mean(axis=0, keepdims=True))[0]
        clusters.append(
            Cluster(
                concept_id=f"cpt{len(clusters):04d}",
                centroid=centroid.astype(np.float32),
                member_unit_ids=member_ids,
                seq=len(clusters),
            )
        )
    return clusters


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    base = _normalize(rng.normal(size=(3, 8)))
    feats = np.vstack([
        base[0] + rng.normal(scale=0.02, size=(10, 8)),
        base[1] + rng.normal(scale=0.02, size=(10, 8)),
        base[2] + rng.normal(scale=0.02, size=(10, 8)),
        rng.normal(size=(4, 8)),  # outliers
    ])
    feats = _normalize(feats)
    ids = [f"s{i:03d}" for i in range(len(feats))]

    from audiohelix.config import ClusterConfig
    cfg = ClusterConfig(spectral_n_clusters=3)
    clusters = cluster_unit_features(feats, ids, cfg=cfg)
    covered = sum(len(c.member_unit_ids) for c in clusters)
    print(f"clusters: {len(clusters)}, singleton: {sum(1 for c in clusters if len(c.member_unit_ids)==1)}")
    print(f"total members (full coverage): {covered} / {len(feats)}")
    for c in clusters[:5]:
        print(c.concept_id, len(c.member_unit_ids), c.centroid.dtype, c.centroid.shape)
