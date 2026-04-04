import torch


class KNN:
    """Fallback KNN wrapper that mimics the knn_cuda interface used by PointGPT."""

    def __init__(self, k, transpose_mode=False):
        self.k = int(k)
        self.transpose_mode = bool(transpose_mode)

    def __call__(self, ref, query):
        if not self.transpose_mode:
            raise NotImplementedError("Fallback KNN only supports transpose_mode=True")
        if ref.dim() != 3 or query.dim() != 3:
            raise ValueError("ref/query must be rank-3 tensors")

        distances = torch.cdist(query, ref)
        dist_k, idx = torch.topk(distances, k=self.k, dim=-1, largest=False, sorted=False)
        return dist_k, idx
