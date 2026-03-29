from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset, RandomSampler

from nepa3d.data.mixed_pretrain import MixedPretrainDataset, MixtureSampler, load_mix_config
from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import CQA_VOCAB_VERSION
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import (
    PackedCQATaskSpec,
    PackedPrimitiveCQADataset,
    V2PrimitiveCQADataset,
    _normalize_query_src_filter,
)


def build_mixed_pretrain_cqa(
    mix_config_path: str,
    *,
    n_ctx: int,
    n_qry: int,
    mode: str = "train",
    eval_seed: int = 0,
    query_order: str | None = None,
    codec_version: str | None = None,
) -> Tuple[MixedPretrainDataset, MixtureSampler, Dict[str, Any]]:
    specs, cfg = load_mix_config(mix_config_path)
    default_codec = str(codec_version or cfg.get("codec_version", CQA_VOCAB_VERSION))

    datasets: List[Dataset] = []
    names: List[str] = []
    weights: List[float] = []
    codec_versions: List[str] = []
    for s in specs:
        paths = list_npz(s.cache_root, s.split)
        if len(paths) == 0:
            raise FileNotFoundError(f"no npz found: cache_root={s.cache_root} split={s.split}")
        task_name = str(s.extra.get("task_name", s.name))
        context_source = str(s.extra.get("context_source", "surf"))
        ds_codec = str(s.extra.get("codec_version", default_codec))
        ds = V2PrimitiveCQADataset(
            paths,
            task_name=task_name,
            context_source=context_source,
            n_ctx=int(s.extra.get("n_ctx", n_ctx)),
            n_qry=int(s.extra.get("n_qry", n_qry)),
            seed=int(eval_seed),
            mode=str(mode),
            query_src_filter=s.extra.get("query_src_filter", None),
            query_dist_min=s.extra.get("query_dist_min", None),
            query_dist_max=s.extra.get("query_dist_max", None),
            query_order=query_order,
            codec_version=ds_codec,
        )
        datasets.append(ds)
        names.append(s.name)
        weights.append(float(s.weight))
        codec_versions.append(ds_codec)

    unique_codecs = sorted(set(codec_versions))
    if len(unique_codecs) != 1:
        raise ValueError(f"mixed codec versions in one run are not supported: {unique_codecs}")

    mixed = MixedPretrainDataset(datasets, names)
    num_samples = int(cfg.get("mix_num_samples", len(mixed)))
    replacement = bool(cfg.get("replacement", True))
    seed = int(cfg.get("mix_seed", 0))
    sampler = MixtureSampler(
        dataset_sizes=mixed.sizes,
        dataset_weights=weights,
        num_samples=num_samples,
        replacement=replacement,
        seed=seed,
    )
    info = {
        "names": names,
        "weights": weights,
        "sizes": mixed.sizes,
        "num_samples": num_samples,
        "replacement": replacement,
        "seed": seed,
        "codec_versions": codec_versions,
        "codec_version": unique_codecs[0],
    }
    return mixed, sampler, info


def build_packed_pretrain_cqa(
    mix_config_path: str,
    *,
    n_ctx: int,
    n_qry: int,
    mode: str = "train",
    eval_seed: int = 0,
    query_order: str | None = None,
    codec_version: str | None = None,
) -> Tuple[PackedPrimitiveCQADataset, RandomSampler, Dict[str, Any]]:
    specs, cfg = load_mix_config(mix_config_path)
    default_codec = str(codec_version or cfg.get("codec_version", CQA_VOCAB_VERSION))
    if len(specs) <= 0:
        raise ValueError("packed CQA requires at least one dataset spec")

    key_to_path_per_spec: list[dict[str, str]] = []
    for s in specs:
        paths = list_npz(s.cache_root, s.split)
        if len(paths) == 0:
            raise FileNotFoundError(f"no npz found: cache_root={s.cache_root} split={s.split}")
        split_root = Path(s.cache_root) / s.split
        key_to_path = {str(Path(p).relative_to(split_root)): str(p) for p in paths}
        key_to_path_per_spec.append(key_to_path)

    common_keys = set(key_to_path_per_spec[0].keys())
    for key_to_path in key_to_path_per_spec[1:]:
        common_keys &= set(key_to_path.keys())
    common_keys = set(sorted(common_keys))
    if len(common_keys) <= 0:
        raise ValueError(
            "packed CQA requires a common shape support across task specs; "
            "got empty intersection. Use a common split/root for all packed tasks."
        )

    canonical_paths = [key_to_path_per_spec[0][k] for k in sorted(common_keys)]
    packed_task_specs: list[PackedCQATaskSpec] = []
    context_sources = {str(s.extra.get("context_source", "surf")) for s in specs}
    if len(context_sources) != 1:
        raise ValueError(f"packed CQA currently requires one shared context_source, got {sorted(context_sources)}")
    for s in specs:
        packed_task_specs.append(
            PackedCQATaskSpec(
                task_name=str(s.extra.get("task_name", s.name)),
                context_source=str(s.extra.get("context_source", "surf")),
                query_src_filter=_normalize_query_src_filter(s.extra.get("query_src_filter", None)),
                query_dist_min=None if s.extra.get("query_dist_min", None) is None else float(s.extra.get("query_dist_min")),
                query_dist_max=None if s.extra.get("query_dist_max", None) is None else float(s.extra.get("query_dist_max")),
            )
        )

    dataset = PackedPrimitiveCQADataset(
        canonical_paths,
        task_specs=packed_task_specs,
        n_ctx=int(n_ctx),
        n_qry=int(n_qry),
        seed=int(eval_seed),
        mode=str(mode),
        query_order=query_order,
        codec_version=default_codec,
    )
    mix_num_samples = int(cfg.get("mix_num_samples", len(canonical_paths) * max(1, len(packed_task_specs))))
    num_shapes = int(math.ceil(float(mix_num_samples) / float(max(1, len(packed_task_specs)))))
    replacement = bool(cfg.get("replacement", True))
    sampler = RandomSampler(
        dataset,
        replacement=replacement,
        num_samples=num_shapes if replacement else min(num_shapes, len(dataset)),
    )
    info = {
        "names": [str(s.name) for s in specs],
        "task_names": [str(spec.task_name) for spec in packed_task_specs],
        "shape_count": int(len(dataset)),
        "num_tasks_per_shape": int(len(packed_task_specs)),
        "num_shape_samples": int(num_shapes if replacement else min(num_shapes, len(dataset))),
        "effective_task_samples": int((num_shapes if replacement else min(num_shapes, len(dataset))) * max(1, len(packed_task_specs))),
        "replacement": replacement,
        "seed": int(cfg.get("mix_seed", 0)),
        "codec_version": default_codec,
        "common_shape_support": int(len(common_keys)),
        "sampling_protocol": "packed",
        "paths_source": "common_intersection",
    }
    return dataset, sampler, info
