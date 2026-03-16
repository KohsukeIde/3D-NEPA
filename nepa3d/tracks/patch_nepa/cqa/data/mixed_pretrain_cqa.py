from __future__ import annotations

from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset

from nepa3d.data.mixed_pretrain import MixedPretrainDataset, MixtureSampler, load_mix_config
from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import V2PrimitiveCQADataset


def build_mixed_pretrain_cqa(
    mix_config_path: str,
    *,
    n_ctx: int,
    n_qry: int,
    mode: str = "train",
    eval_seed: int = 0,
) -> Tuple[MixedPretrainDataset, MixtureSampler, Dict[str, Any]]:
    specs, cfg = load_mix_config(mix_config_path)

    datasets: List[Dataset] = []
    names: List[str] = []
    weights: List[float] = []
    for s in specs:
        paths = list_npz(s.cache_root, s.split)
        if len(paths) == 0:
            raise FileNotFoundError(f"no npz found: cache_root={s.cache_root} split={s.split}")
        task_name = str(s.extra.get("task_name", s.name))
        context_source = str(s.extra.get("context_source", "surf"))
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
        )
        datasets.append(ds)
        names.append(s.name)
        weights.append(float(s.weight))

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
    }
    return mixed, sampler, info
