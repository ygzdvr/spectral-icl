"""Block partitions and mass-preserving heterogeneity.

Plan correspondence: EXPERIMENT_PLAN_FINAL.MD Section 4.6 (partitions + commutants
utilities), Step-1 Generator / Utility Specification v4 §5.

This module provides:

- :class:`BlockPartition`, a frozen dataclass describing a partition of
  ``[0, D)`` into disjoint index blocks.
- Ladder constructors :func:`equal_blocks`, :func:`dyadic_ladder`,
  :func:`custom_ladder` (the last validates a refinement chain).
- Mass-preserving heterogeneity constructors
  :func:`mass_preserving_block_spectrum` (for λ) and
  :func:`mass_preserving_block_task` (for ω).

Mass-preserving formula (v4 §5.3):

    lambda_{b,j} = lambda_bar_b * kappa_b^{xi_j}
                   / ( (1/m_b) * sum_{u=1..m_b} kappa_b^{xi_u} )

which enforces the block-mean invariant

    (1/m_b) * sum_{j=1..m_b} lambda_{b,j}  =  lambda_bar_b

for every value of kappa_b. The default xi_shape "linear" uses

    xi_j = (j - (m_b + 1)/2) / (m_b - 1),   j = 1, ..., m_b   (m_b >= 2)
    lambda_{b,1} = lambda_bar_b                                 (m_b == 1)

so that lambda_max / lambda_min = kappa_b exactly within each block of size >= 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# BlockPartition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockPartition:
    """Partition of ``[0, D)`` into ``blocks``: a tuple of disjoint index tuples
    whose union is ``{0, 1, ..., D - 1}``.

    Validated at construction. All blocks must be nonempty, pairwise disjoint,
    and cover every index. A precomputed per-index block lookup table is
    attached via :func:`object.__setattr__` so that :meth:`block_of` is O(1).

    The dataclass is frozen and hashable on its ``(D, blocks)`` fields.
    """

    D: int
    blocks: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.D < 1:
            raise ValueError(f"D must be positive, got {self.D}")
        if not isinstance(self.blocks, tuple):
            raise TypeError(
                f"blocks must be a tuple of int-tuples, got {type(self.blocks).__name__}"
            )
        seen = [False] * self.D
        block_of_arr = [-1] * self.D
        for b_idx, block in enumerate(self.blocks):
            if not isinstance(block, tuple):
                raise TypeError(
                    f"blocks[{b_idx}] must be a tuple of ints, got {type(block).__name__}"
                )
            if len(block) == 0:
                raise ValueError(
                    f"blocks[{b_idx}] is empty; every block must be nonempty"
                )
            for k in block:
                if not isinstance(k, int) or isinstance(k, bool):
                    raise TypeError(f"blocks[{b_idx}] contains non-int {k!r}")
                if k < 0 or k >= self.D:
                    raise ValueError(
                        f"index {k} in blocks[{b_idx}] out of range [0, {self.D})"
                    )
                if seen[k]:
                    raise ValueError(f"index {k} appears in multiple blocks")
                seen[k] = True
                block_of_arr[k] = b_idx
        missing = [k for k, was_seen in enumerate(seen) if not was_seen]
        if missing:
            raise ValueError(f"indices {missing} missing from blocks")
        object.__setattr__(self, "_block_of_arr", tuple(block_of_arr))

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the partition."""
        return len(self.blocks)

    @property
    def sizes(self) -> tuple[int, ...]:
        """Tuple of block sizes ``(m_0, m_1, ..., m_{n_blocks - 1})``."""
        return tuple(len(b) for b in self.blocks)

    def block_of(self, k: int) -> int:
        """Return the block index ``b`` such that ``k in self.blocks[b]``."""
        if not (0 <= k < self.D):
            raise ValueError(f"k = {k} out of range [0, {self.D})")
        return self._block_of_arr[k]  # type: ignore[attr-defined]

    def indicator_matrix(self) -> torch.Tensor:
        """Real ``(D, n_blocks)`` float64 tensor with
        ``I[k, b] == 1`` iff ``k`` is in block ``b``, else 0.
        """
        I = torch.zeros((self.D, self.n_blocks), dtype=torch.float64)
        for b_idx, block in enumerate(self.blocks):
            for k in block:
                I[k, b_idx] = 1.0
        return I

    def block_projector(self, b: int) -> torch.Tensor:
        """Return the ``(D, D)`` diagonal 0/1 projector

            Pi_b = sum_{k in block b} e_k e_k^T,

        as a real float64 tensor.
        """
        if not (0 <= b < self.n_blocks):
            raise ValueError(f"b = {b} out of range [0, {self.n_blocks})")
        diag = torch.zeros(self.D, dtype=torch.float64)
        for k in self.blocks[b]:
            diag[k] = 1.0
        return torch.diag(diag)


# ---------------------------------------------------------------------------
# Ladder constructors
# ---------------------------------------------------------------------------


def equal_blocks(D: int, m: int) -> BlockPartition:
    """Partition ``[0, D)`` into ``D // m`` consecutive blocks of size ``m``.

    Requires ``D % m == 0``.
    """
    if D < 1:
        raise ValueError(f"D must be positive, got {D}")
    if m < 1:
        raise ValueError(f"m must be positive, got {m}")
    if D % m != 0:
        raise ValueError(f"equal_blocks: D = {D} must be divisible by m = {m}")
    n = D // m
    blocks = tuple(tuple(range(b * m, (b + 1) * m)) for b in range(n))
    return BlockPartition(D=D, blocks=blocks)


def dyadic_ladder(D: int, J: int | None = None) -> list[BlockPartition]:
    """Dyadic refinement ladder on ``[0, D)``, coarsest to finest.

    Level ``0`` has one block of size ``D``; level ``j`` has ``2**j`` blocks of
    size ``D / 2**j``. Requires ``D`` to be a power of 2. Returns a list of
    length ``J + 1``; default ``J = log2(D)``, which produces block counts
    ``1, 2, 4, ..., D``.
    """
    if D < 1:
        raise ValueError(f"D must be positive, got {D}")
    if D & (D - 1) != 0:
        raise ValueError(f"dyadic_ladder: D = {D} must be a power of 2")
    max_J = int(math.log2(D))
    if J is None:
        J = max_J
    if J < 0 or J > max_J:
        raise ValueError(
            f"dyadic_ladder: J = {J} out of range [0, {max_J}] for D = {D}"
        )
    return [equal_blocks(D, D >> j) for j in range(J + 1)]


def custom_ladder(levels: list[BlockPartition]) -> list[BlockPartition]:
    """Validate that ``levels`` is a refinement chain, coarsest to finest, and
    return a fresh list.

    A valid chain requires every block of ``levels[j + 1]`` to be contained in
    some block of ``levels[j]`` (for all ``j``), and every level to share the
    same ambient ``D``. Raises ``ValueError`` on any violation.
    """
    if len(levels) == 0:
        raise ValueError("custom_ladder: at least one level is required")
    for j, level in enumerate(levels):
        if not isinstance(level, BlockPartition):
            raise TypeError(
                f"custom_ladder: levels[{j}] is not a BlockPartition "
                f"(got {type(level).__name__})"
            )
    D = levels[0].D
    for j, level in enumerate(levels):
        if level.D != D:
            raise ValueError(
                f"custom_ladder: levels[{j}] has D = {level.D}, expected {D}"
            )
    for j in range(len(levels) - 1):
        coarse = levels[j]
        fine = levels[j + 1]
        for b_idx, fine_block in enumerate(fine.blocks):
            coarse_ids = {coarse.block_of(k) for k in fine_block}
            if len(coarse_ids) != 1:
                raise ValueError(
                    f"custom_ladder: levels[{j + 1}] block {b_idx} = {fine_block} "
                    f"spans multiple blocks of levels[{j}]"
                )
    return list(levels)


# ---------------------------------------------------------------------------
# Mass-preserving heterogeneity
# ---------------------------------------------------------------------------


def _build_xi(
    partition: BlockPartition,
    xi_shape: str,
    xi_custom: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the length-D exponent vector ``xi`` for mass-preserving heterogeneity.

    For ``xi_shape == "linear"``: within each block of size ``m``,
        xi_j = (j - (m + 1)/2) / (m - 1),  j = 1, ..., m    (m >= 2)
    and ``xi[k] = 0`` for singleton blocks. For ``xi_shape == "custom"``,
    ``xi_custom`` is used verbatim after shape/dtype validation.
    """
    D = partition.D
    xi = torch.zeros(D, dtype=dtype)
    if xi_shape == "linear":
        for block in partition.blocks:
            m = len(block)
            if m == 1:
                xi[block[0]] = 0.0
                continue
            half = (m + 1) / 2.0
            denom = float(m - 1)
            for k_local, k in enumerate(block):
                j = k_local + 1  # 1-indexed within the block
                xi[k] = (j - half) / denom
        return xi
    if xi_shape == "custom":
        if xi_custom is None:
            raise ValueError("xi_shape='custom' requires xi_custom to be provided")
        if xi_custom.ndim != 1 or xi_custom.shape[0] != D:
            raise ValueError(
                f"xi_custom must have shape ({D},); got {tuple(xi_custom.shape)}"
            )
        if not xi_custom.is_floating_point():
            raise TypeError(
                f"xi_custom must be real floating-point; got dtype {xi_custom.dtype}"
            )
        return xi_custom.to(dtype).clone()
    raise ValueError(
        f"unknown xi_shape: {xi_shape!r}; expected 'linear' or 'custom'"
    )


def _mass_preserving_core(
    partition: BlockPartition,
    block_means: torch.Tensor,
    block_kappas: torch.Tensor,
    xi_shape: str,
    xi_custom: torch.Tensor | None,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Shared implementation for mass-preserving block spectrum / task constructors."""
    D = partition.D
    n = partition.n_blocks
    if block_means.ndim != 1 or block_means.shape[0] != n:
        raise ValueError(
            f"{name}: block_means must have shape ({n},); got {tuple(block_means.shape)}"
        )
    if not block_means.is_floating_point():
        raise TypeError(
            f"{name}: block_means must be real floating-point; got dtype {block_means.dtype}"
        )
    if (block_means <= 0).any():
        raise ValueError(f"{name}: block_means must all be positive")
    if block_kappas.ndim != 1 or block_kappas.shape[0] != n:
        raise ValueError(
            f"{name}: block_kappas must have shape ({n},); got {tuple(block_kappas.shape)}"
        )
    if not block_kappas.is_floating_point():
        raise TypeError(
            f"{name}: block_kappas must be real floating-point; got dtype {block_kappas.dtype}"
        )
    if (block_kappas < 1).any():
        raise ValueError(f"{name}: block_kappas must all be >= 1")
    bm = block_means.to(dtype)
    bk = block_kappas.to(dtype)
    xi = _build_xi(partition, xi_shape, xi_custom, dtype)
    lam = torch.zeros(D, dtype=dtype)
    for b_idx, block in enumerate(partition.blocks):
        idx = torch.tensor(block, dtype=torch.long)
        xi_block = xi[idx]
        num = bk[b_idx] ** xi_block              # (m,)
        denom = num.mean()                        # (1/m) sum kappa^{xi_u}
        lam[idx] = bm[b_idx] * num / denom
    return lam


def mass_preserving_block_spectrum(
    partition: BlockPartition,
    block_means: torch.Tensor,
    block_kappas: torch.Tensor,
    *,
    xi_shape: str = "linear",
    xi_custom: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return an eigenvalue vector ``lambda ∈ R^D`` with mass-preserving
    within-block heterogeneity.

    For each block ``b`` of size ``m_b`` with mean ``lambda_bar_b`` and condition
    number ``kappa_b``,

        lambda_{b,j} = lambda_bar_b * kappa_b^{xi_j}
                       / ((1/m_b) * sum_{u=1..m_b} kappa_b^{xi_u})

    which enforces ``(1/m_b) * sum_j lambda_{b,j} == lambda_bar_b`` for every
    ``kappa_b``. Under the default ``xi_shape="linear"``, the within-block
    ``lambda_max / lambda_min`` equals ``kappa_b`` exactly (for blocks of size
    >= 2); singleton blocks return ``lambda_bar_b`` unconditionally.

    Parameters
    ----------
    partition
        :class:`BlockPartition` of ``[0, D)``.
    block_means
        Real tensor of shape ``(n_blocks,)`` with positive entries.
    block_kappas
        Real tensor of shape ``(n_blocks,)`` with entries ``>= 1``.
    xi_shape
        ``"linear"`` (default) or ``"custom"``.
    xi_custom
        Required when ``xi_shape == "custom"``; real tensor of shape ``(D,)``.
    dtype
        Output dtype. Defaults to ``torch.float64``.

    Returns
    -------
    Tensor of shape ``(D,)``, ordered per the partition's index convention.
    """
    return _mass_preserving_core(
        partition,
        block_means,
        block_kappas,
        xi_shape,
        xi_custom,
        dtype,
        name="mass_preserving_block_spectrum",
    )


def mass_preserving_block_task(
    partition: BlockPartition,
    block_means: torch.Tensor,
    block_kappas: torch.Tensor,
    *,
    xi_shape: str = "linear",
    xi_custom: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a task-variance vector ``omega ∈ R^D`` with mass-preserving
    within-block heterogeneity.

    Contract and formula are identical to
    :func:`mass_preserving_block_spectrum`; ``omega`` plays the role of
    task-covariance rather than input-covariance.
    """
    return _mass_preserving_core(
        partition,
        block_means,
        block_kappas,
        xi_shape,
        xi_custom,
        dtype,
        name="mass_preserving_block_task",
    )


__all__ = [
    "BlockPartition",
    "equal_blocks",
    "dyadic_ladder",
    "custom_ladder",
    "mass_preserving_block_spectrum",
    "mass_preserving_block_task",
]
