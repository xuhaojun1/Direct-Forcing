"""Memory-bounded construction of the same Nyström estimator on token vectors.

The complete pool may be a CPU numpy mmap. Only landmarks and small row
blocks reside on the GPU; no N x D FP64 copies or covariance matrices.
"""
import torch


@torch.no_grad()
def build_empirical_reference(pool, sigma, *, k_rr):
    """All real rows with uniform weights: exact biased empirical RBF MMD.

    The existing kernel assembly accepts a weighted bank, so no Gram solve or
    approximation is needed. M is a legacy storage field, not a landmark budget.
    """
    if pool.ndim != 2 or len(pool) < 2 or not float(sigma) > 0:
        raise ValueError('Empirical reference needs at least two rows and positive bandwidth')
    if not torch.isfinite(torch.tensor([float(sigma), float(k_rr)])).all():
        raise ValueError('Invalid empirical kernel statistics')
    n, d = pool.shape
    return dict(Z=pool.detach().to(device='cpu', dtype=torch.float32),
                alpha=torch.full((n,), 1./n, dtype=torch.float32),
                sigma=float(sigma), k_rr=float(k_rr), M=n, d_in=d)


def rows(pool, start, stop, device):
    return torch.as_tensor(pool[start:stop]).to(device=device, dtype=torch.float32)


def costs(a, b):
    return (a.square().sum(1)[:, None]+b.square().sum(1)[None, :]-2*a@b.T).clamp_min(0)


@torch.no_grad()
def pool_cost_matrix(pool, device='cuda', block=128):
    """Store only the N x N distances on CPU; mirror upper blocks exactly."""
    n = len(pool)
    result = torch.empty(n, n, dtype=torch.float32)
    for i in range(0, n, block):
        a = rows(pool, i, i+block, device)
        for j in range(i, n, block):
            b = rows(pool, j, j+block, device)
            c = costs(a, b).cpu()
            result[i:i+len(a), j:j+len(b)] = c
            result[j:j+len(b), i:i+len(a)] = c.T
    result.fill_diagonal_(0)
    return result


@torch.no_grad()
def build_token_reference(pool, sigma, *, n_landmarks=4096, iterations=20,
                          seed=0, jitter=1e-6, device='cuda', block=64,
                          k_rr=None, progress=lambda **kw: None):
    n, d = pool.shape
    if not 1 <= n_landmarks <= n:
        raise ValueError('Landmarks exceed reference population')
    rng = torch.Generator().manual_seed(seed)
    # Match existing builder's fit shuffle, then k-means initial shuffle.
    fit_ids = torch.randperm(n, generator=rng)
    init_ids = fit_ids[torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:n_landmarks]]
    centers = torch.stack([torch.as_tensor(pool[int(i)]).float() for i in init_ids]).to(device)
    # A small pilot can retain every empirical reference sample. No clustering
    # is needed when the reference population itself fits within the budget.
    if n_landmarks == n:
        iterations = 0
        progress(phase='all_samples_as_landmarks', landmarks=n)
    sums = x = None
    for iteration in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(n_landmarks, device=device)
        for i in range(0, n, block):
            # Keep row order stable and avoid a second full-size permuted pool.
            x = rows(pool, i, i+block, device)
            labels = costs(x, centers).argmin(1)
            sums.index_add_(0, labels, x)
            counts.index_add_(0, labels, torch.ones(len(x), device=device))
        empty = counts == 0
        sums.div_(counts.clamp_min(1)[:, None])
        if empty.any():
            ids = torch.randperm(n, generator=rng)[:int(empty.sum())]
            sums[empty] = torch.stack([torch.as_tensor(pool[int(i)]).float() for i in ids]).to(device)
        centers = sums
        progress(phase='kmeans', iteration=iteration+1, iterations=iterations,
                 empty_clusters=int(empty.sum()))
    gamma = 1/(2*float(sigma)**2)
    # Preserve the original ViRDM FP64 landmark Gram/solve arithmetic. Only
    # landmarks, never the full pool, need the larger dtype.
    saved_centers = centers.cpu()
    del sums, centers, x
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    centers = saved_centers.to(device=device, dtype=torch.float64)
    # A full squared landmark matrix is another 40 GiB for K-token features.
    # Reduce complete rows in bounded blocks, preserving the FP64 row sums.
    norm = torch.empty(n_landmarks, device=device, dtype=torch.float64)
    for i in range(0, n_landmarks, block):
        norm[i:i+block] = centers[i:i+block].square().sum(1)
    mean = torch.zeros(n_landmarks, device=device, dtype=torch.float64)
    for i in range(0, n, block):
        x = rows(pool, i, i+block, device).double()
        distance = (x.square().sum(1)[:, None]+norm[None, :]-2*x@centers.T).clamp_min(0)
        mean += torch.exp(-gamma*distance).double().sum(0)
    mean /= n
    kernel = torch.empty(n_landmarks, n_landmarks, device=device, dtype=torch.float64)
    for i in range(0, n_landmarks, block):
        distance = (norm[i:i+block,None]+norm[None,:]
                    -2*centers[i:i+block]@centers.T).clamp_min(0)
        kernel[i:i+block] = torch.exp(-gamma*distance)
    kernel = (kernel+kernel.T)*.5
    kernel.diagonal().fill_(1+jitter)
    cholesky = torch.linalg.cholesky(kernel)
    alpha = torch.cholesky_solve(mean[:, None], cholesky).squeeze(1)
    if k_rr is None:
        distances = pool_cost_matrix(pool, device, block)
        k_rr = float(torch.exp(-gamma*distances.double()).mean())
    if not torch.isfinite(alpha).all():
        raise FloatingPointError('Invalid Nyström solve')
    return dict(Z=saved_centers, alpha=alpha.float().cpu(), sigma=float(sigma),
                k_rr=float(k_rr), M=n_landmarks, d_in=d)
