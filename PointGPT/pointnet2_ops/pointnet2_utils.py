import torch


def furthest_point_sample(xyz, npoint):
    """Pure PyTorch FPS fallback matching the small subset of the PointNet2 API used here."""
    if xyz.dim() != 3 or xyz.size(-1) != 3:
        raise ValueError(f"xyz must be [B, N, 3], got {tuple(xyz.shape)}")

    batch_size, num_points, _ = xyz.shape
    npoint = int(npoint)
    if npoint <= 0:
        raise ValueError(f"npoint must be positive, got {npoint}")
    npoint = min(npoint, num_points)

    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch_size, num_points), float("inf"), device=xyz.device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def gather_operation(features, idx):
    """Gather fallback for features shaped [B, C, N] and indices [B, S]."""
    if features.dim() != 3:
        raise ValueError(f"features must be [B, C, N], got {tuple(features.shape)}")
    if idx.dim() != 2:
        raise ValueError(f"idx must be [B, S], got {tuple(idx.shape)}")

    if features.size(0) != idx.size(0):
        raise ValueError("features and idx batch dimensions must match")

    expanded_idx = idx.unsqueeze(1).expand(-1, features.size(1), -1)
    return torch.gather(features, 2, expanded_idx)
