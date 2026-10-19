"""Synthesize a Hi-C contact matrix from a TAD hierarchy (Lun/Smyth recipe).

For each node we build a dense N x N matrix whose expected counts follow a
distance-decay background modulated by domain enrichment, then add Negative
Binomial biological noise and a uniform random-ligation background:

    lambda_ij = A * (|i-j| + 1)^(-alpha)
    for every domain containing both i and j:  lambda_ij *= enrichment   (compounds
                                                for nested domains)
    for adjacent sibling domains: contacts crossing their shared boundary are
        multiplied by boundary_depletion to create insulation
    around every domain boundary: narrow stripe contacts are depleted to create
        local insulation minima for OnTAD
    M_ij ~ NegBinom(mean = lambda_ij, BCV = b)
    add fraction rho of total counts spread uniformly (random ligation)
    symmetrize; optionally rescale to a target depth D.

The matrix is written in the dense N x N text format OnTAD consumes.
"""

from __future__ import annotations

import random

import numpy as np

from hierarchy import iter_domains


def _deplete_sibling_cross_contacts(lam: np.ndarray, domains, depletion: float) -> None:
    """Reduce expected contacts crossing adjacent sibling-domain boundaries."""
    if depletion >= 1.0:
        return
    for left, right in zip(domains, domains[1:]):
        if left.end == right.start:
            lam[left.start : left.end, right.start : right.end] *= depletion
            lam[right.start : right.end, left.start : left.end] *= depletion
    for d in domains:
        _deplete_sibling_cross_contacts(lam, d.children, depletion)


def _boundary_positions(hier, n_bins: int) -> set[int]:
    bounds: set[int] = set()
    for d in iter_domains(hier):
        if 0 < d.start < n_bins:
            bounds.add(d.start)
        if 0 < d.end < n_bins:
            bounds.add(d.end)
    return bounds


def _deplete_boundary_stripes(lam: np.ndarray, boundaries: set[int], width: int, depletion: float) -> None:
    """Create narrow low-contact stripes crossing each boundary."""
    if width <= 0 or depletion >= 1.0:
        return
    n_bins = lam.shape[0]
    for b in boundaries:
        left = slice(max(0, b - width), b)
        right = slice(b, min(n_bins, b + width))

        # Immediate cross-boundary square.
        lam[left, right] *= depletion * 0.05
        lam[right, left] *= depletion * 0.05

        # Vertical/horizontal stripe contacts spanning the opposite side.
        lam[:b, right] *= depletion
        lam[right, :b] *= depletion
        lam[left, b:] *= depletion
        lam[b:, left] *= depletion


def _sample_negbinom(mu: np.ndarray, bcv: float, rng: np.random.Generator) -> np.ndarray:
    """Sample counts with mean ``mu`` and biological CV ``bcv``.

    NegBinom variance = mu + bcv^2 * mu^2, i.e. size r = 1/bcv^2 and success
    probability p = r / (r + mu). ``bcv <= 0`` falls back to Poisson.
    """
    mu = np.maximum(mu, 1e-9)
    if bcv <= 0:
        return rng.poisson(mu)
    r = 1.0 / (bcv * bcv)
    p = r / (r + mu)
    return rng.negative_binomial(r, p)


def simulate_matrix(hier, n_bins: int, params, rng: np.random.Generator) -> np.ndarray:
    """Return a symmetric integer N x N contact matrix for ``hier``."""
    idx = np.arange(n_bins)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    lam = params.A * (dist + 1.0) ** (-params.alpha)

    # Domain enrichment, compounding for nested domains. Per-domain jitter
    # prevents a perfectly periodic synthetic matrix, which OnTAD tends to
    # collapse into only a few calls.
    for d in iter_domains(hier):
        jitter = rng.lognormal(mean=0.0, sigma=params.domain_strength_jitter)
        lam[d.start : d.end, d.start : d.end] *= params.enrichment * jitter

    _deplete_sibling_cross_contacts(lam, hier, params.boundary_depletion)
    _deplete_boundary_stripes(
        lam,
        _boundary_positions(hier, n_bins),
        params.boundary_stripe_width,
        params.boundary_stripe_depletion,
    )

    # Sample on the upper triangle (incl. diagonal), then mirror for symmetry.
    upper = np.triu(np.ones((n_bins, n_bins), dtype=bool))
    counts = np.zeros((n_bins, n_bins), dtype=np.int64)
    counts[upper] = _sample_negbinom(lam[upper], params.bcv, rng)

    # Random-ligation background: rho of total counts, uniform over the triangle.
    if params.noise > 0:
        n_upper = int(upper.sum())
        noise_mass = params.noise * counts.sum()
        per_entry = noise_mass / max(n_upper, 1)
        counts[upper] += rng.poisson(per_entry, size=n_upper)

    # Mirror to a full symmetric matrix.
    M = np.triu(counts, 0)
    M = M + M.T - np.diag(np.diag(M))

    # Optional depth rescaling to a target total.
    if params.depth is not None and M.sum() > 0:
        M = np.rint(M * (params.depth / M.sum())).astype(np.int64)

    return M


def write_matrix(M: np.ndarray, path) -> None:
    """Write a dense integer matrix in OnTAD's tab-separated N x N format."""
    np.savetxt(path, M, fmt="%d", delimiter="\t")
