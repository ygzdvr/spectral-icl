"""Block commutant utilities for theorem-C.

Plan correspondence: EXPERIMENT_PLAN_FINAL.MD §4.6 (partitions + commutants).
Step-1 Generator / Utility Specification v4 §6.

The **coarse block commutant** of a partition ``B`` is

    C(B) = { Q in R^{D x D} : Q = sum_b q_b * Pi_b,  q in R^{n_blocks} },
    Pi_b  = sum_{k in b} e_k e_k^T.

This is the space of matrices that are constant-diagonal within each block and
zero everywhere else. It is **tighter** than block-diagonal matrices (which
would allow any m_b x m_b matrix per block); the coarse form is the class that
survives the Haar average over each block's rotation, and it is the relevant
class for theorem-C's spectral-only obstruction and the C6 oracle hybrid.

Binding (v4 §6.3). The theorem-C **oracle hybrid** is the minimizer of the
population loss over the refined commutant class; it is NOT a learned
projector, NOT a trained network, NOT an estimated basis. That minimizer is
produced by ``metrics.oracle_commutant_loss`` (v4 §7.5). Learned projectors
belong to the architecture-aligned tier and use a separate API.

Block ordering. Every output indexed by "block index" follows exactly the
ordering of ``partition.blocks`` (no re-sorting, no alphabetisation).
"""

from __future__ import annotations

import torch

from scripts.thesis.utils.partitions import BlockPartition


# ---------------------------------------------------------------------------
# Internal validation helper
# ---------------------------------------------------------------------------


def _check_square_matching(
    Q: torch.Tensor, partition: BlockPartition, name: str = "Q"
) -> None:
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"{name} must be square 2D; got shape {tuple(Q.shape)}")
    if Q.shape[0] != partition.D:
        raise ValueError(
            f"{name}.shape[0] = {Q.shape[0]} does not match partition.D = {partition.D}"
        )
    if not Q.is_floating_point():
        raise TypeError(f"{name} must be real floating-point; got dtype {Q.dtype}")


# ---------------------------------------------------------------------------
# Block-scalar extraction / reconstruction
# ---------------------------------------------------------------------------


def extract_block_scalars(
    Q: torch.Tensor, partition: BlockPartition
) -> torch.Tensor:
    """Return the per-block diagonal averages

        q[b] = (1/m_b) * sum_{k in block b} Q[k, k]

    as a length-``n_blocks`` tensor. Block ordering matches ``partition.blocks``
    exactly. Output dtype and device match Q's.
    """
    _check_square_matching(Q, partition, name="Q")
    diag = torch.diagonal(Q)
    q = torch.zeros(partition.n_blocks, dtype=Q.dtype, device=Q.device)
    for b_idx, block in enumerate(partition.blocks):
        q[b_idx] = diag[list(block)].mean()
    return q


def reconstruct_from_block_scalars(
    q: torch.Tensor, partition: BlockPartition
) -> torch.Tensor:
    """Build the matrix

        sum_b q[b] * Pi_b   in   R^{D x D}

    from a block-scalar vector ``q`` of shape ``(n_blocks,)``. The result is a
    dense diagonal tensor (zeros off-diagonal). Output dtype/device match q's.
    """
    if q.ndim != 1 or q.shape[0] != partition.n_blocks:
        raise ValueError(
            f"q must have shape ({partition.n_blocks},); got {tuple(q.shape)}"
        )
    if not q.is_floating_point():
        raise TypeError(f"q must be real floating-point; got dtype {q.dtype}")
    D = partition.D
    diag_values = torch.zeros(D, dtype=q.dtype, device=q.device)
    for b_idx, block in enumerate(partition.blocks):
        diag_values[list(block)] = q[b_idx]
    return torch.diag(diag_values)


# ---------------------------------------------------------------------------
# Commutant projection and violation
# ---------------------------------------------------------------------------


def commutant_projection(
    Q: torch.Tensor, partition: BlockPartition
) -> torch.Tensor:
    """Project Q onto the block commutant ``C(B)``:

        Pi_C(Q) = sum_b q_b * Pi_b,
        q_b     = (1/m_b) * sum_{k in block b} Q[k, k].

    Returns a D x D diagonal tensor with dtype/device matching Q's.
    Off-diagonal entries and within-block diagonal variance are discarded.
    """
    _check_square_matching(Q, partition, name="Q")
    q = extract_block_scalars(Q, partition)
    return reconstruct_from_block_scalars(q, partition)


def commutant_violation(
    Q: torch.Tensor, partition: BlockPartition, *, normalize: bool = True
) -> float:
    """Squared Frobenius distance from Q to the commutant,

        ||Q - Pi_C(Q)||_F^2,

    optionally divided by ``||Q||_F^2`` (the default). Returns a Python float.

    Equals 0 up to float eps iff ``Q in C(B)``. For ``Q == 0`` with
    ``normalize=True`` the function returns ``0.0`` (vacuously in the
    commutant).
    """
    _check_square_matching(Q, partition, name="Q")
    proj = commutant_projection(Q, partition)
    resid_sq = (Q - proj).pow(2).sum().item()
    if not normalize:
        return float(resid_sq)
    q_sq = Q.pow(2).sum().item()
    if q_sq == 0.0:
        return 0.0
    return float(resid_sq / q_sq)


# ---------------------------------------------------------------------------
# Refinement predicate
# ---------------------------------------------------------------------------


def refines(fine: BlockPartition, coarse: BlockPartition) -> bool:
    """Return True iff ``fine`` refines ``coarse``: every block of ``fine`` is
    contained in exactly one block of ``coarse``.

    Both partitions must share the same ambient ``D``; if not, returns False.
    Any partition refines itself; the predicate is reflexive on equal
    partitions.
    """
    if fine.D != coarse.D:
        return False
    for fine_block in fine.blocks:
        coarse_ids = {coarse.block_of(k) for k in fine_block}
        if len(coarse_ids) != 1:
            return False
    return True


__all__ = [
    "extract_block_scalars",
    "reconstruct_from_block_scalars",
    "commutant_projection",
    "commutant_violation",
    "refines",
]
