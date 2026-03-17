"""
Negacyclic Number Theoretic Transform (NTT) implementation.

The negacyclic NTT computes polynomial evaluation at odd powers of a primitive
root. Given coefficients x[0], x[1], ..., x[N-1], the output is:

    y[k] = Σ_{n=0}^{N-1} x[n] · ψ^{(2k+1)·n}  (mod q)

where ψ is a primitive 2N-th root of unity (ψ^N ≡ -1 mod q).

This is equivalent to a cyclic NTT on "twisted" input, where each coefficient
x[n] is first multiplied by ψ^n.
"""

import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------

def mod_add(a, b, q):
    """Return (a + b) mod q, elementwise."""
    q64 = jnp.uint64(q)
    s = jnp.asarray(a, dtype=jnp.uint64) + jnp.asarray(b, dtype=jnp.uint64)
    s = jnp.where(s >= q64, s - q64, s)
    return s.astype(jnp.uint32)


def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise."""
    q64 = jnp.uint64(q)
    d = jnp.asarray(a, dtype=jnp.uint64) + q64 - jnp.asarray(b, dtype=jnp.uint64)
    d = jnp.where(d >= q64, d - q64, d)
    return d.astype(jnp.uint32)


def mod_mul(a, b, q):
    """Return (a * b) mod q, elementwise."""
    q64 = jnp.uint64(q)
    p = (jnp.asarray(a, dtype=jnp.uint64) * jnp.asarray(b, dtype=jnp.uint64)) % q64
    return p.astype(jnp.uint32)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _bit_reverse_indices(N):
    """Return bit-reversal permutation for power-of-two size N."""
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("NTT size N must be a positive power of two")

    bits = N.bit_length() - 1
    perm = []
    for i in range(N):
        r = 0
        x = i
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        perm.append(r)
    return jnp.asarray(perm, dtype=jnp.uint32)


# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------

def ntt(x, *, q, psi_powers, twiddles):
    """
    Compute the forward negacyclic NTT using:
      1) negacyclic twist by ψ^n
      2) bit-reversal of the twisted input
      3) iterative radix-2 Cooley-Tukey cyclic NTT with ω = ψ^2

    Args:
        x: Input coefficients, shape (batch, N), values in [0, q)
        q: Prime modulus satisfying (q - 1) % (2N) == 0
        psi_powers: Precomputed ψ^n table for n = 0..N-1
        twiddles: Packed table containing:
            - first (N - 1) entries: stage-packed twiddles
            - last N entries: bit-reversal permutation

    Returns:
        jnp.ndarray: NTT output in normal order, same shape as input
    """
    x = jnp.asarray(x, dtype=jnp.uint32)
    psi_powers = jnp.asarray(psi_powers, dtype=jnp.uint32)
    twiddles = jnp.asarray(twiddles, dtype=jnp.uint32)

    B, N = x.shape

    if psi_powers.shape[0] != N:
        raise ValueError(f"psi_powers must have length N={N}, got {psi_powers.shape[0]}")

    expected_tw_len = (N - 1) + N
    if twiddles.shape[0] != expected_tw_len:
        raise ValueError(
            f"twiddles must have length {(N - 1)} + {N} = {expected_tw_len}, "
            f"got {twiddles.shape[0]}"
        )

    # Unpack tables
    stage_twiddles = twiddles[: N - 1]
    bitrev_perm = twiddles[N - 1 :].astype(jnp.int32)

    # -------------------------------------------------------------------------
    # Step 1: twist input by ψ^n
    #
    # y[k] = sum_n x[n] * ψ^{(2k+1)n}
    #      = sum_n (x[n] * ψ^n) * (ψ^2)^{kn}
    #
    # So this is a cyclic NTT of the twisted input with root ω = ψ^2.
    # -------------------------------------------------------------------------
    a = mod_mul(x, psi_powers[None, :], q)

    # -------------------------------------------------------------------------
    # Step 2: bit-reverse input
    #
    # This iterative radix-2 DIT Cooley-Tukey schedule expects bit-reversed
    # input and produces normal-order output.
    # -------------------------------------------------------------------------
    a = a[:, bitrev_perm]

    # -------------------------------------------------------------------------
    # Step 3: iterative radix-2 Cooley-Tukey cyclic NTT
    #
    # Implement each stage with reshape/split/concat instead of many .at[] sets.
    # This is much friendlier to JAX/XLA than nested scatter-style updates.
    # -------------------------------------------------------------------------
    length = 1
    tw_offset = 0

    while length < N:
        stage_tw = stage_twiddles[tw_offset : tw_offset + length]
        tw_offset += length

        block = 2 * length

        # Reshape into blocks of size `block`
        a3 = a.reshape(B, N // block, block)

        left = a3[:, :, :length]      # (B, nblocks, length)
        right = a3[:, :, length:]     # (B, nblocks, length)

        # Broadcast stage twiddles across batch and blocks
        t = mod_mul(right, stage_tw[None, None, :], q)

        out_left = mod_add(left, t, q)
        out_right = mod_sub(left, t, q)

        a = jnp.concatenate([out_left, out_right], axis=2).reshape(B, N)

        length *= 2

    return a.astype(jnp.uint32)


def prepare_tables(*, q, psi_powers, twiddles):
    """
    One-time table preparation.

    Builds:
      1) stage-packed cyclic NTT twiddles for ω = ψ^2
      2) bit-reversal permutation

    Packs both into a single returned `twiddles` array because the required API
    only allows returning (psi_powers, twiddles).

    Layout of returned `twiddles`:
      - first N-1 entries: stage-packed twiddles
      - last N entries: bit-reversal permutation

    Stage twiddle layout:
        stage 0, length=1:   [ω^0]
        stage 1, length=2:   [ω^0, ω^(N/4)]
        stage 2, length=4:   [ω^0, ω^(N/8), ω^(2N/8), ω^(3N/8)]
        ...
        final stage:         length = N/2

    Total stage twiddle count = N - 1
    Total packed table count  = (N - 1) + N = 2N - 1
    """
    del twiddles  # ignored; rebuilt here

    psi_powers = jnp.asarray(psi_powers, dtype=jnp.uint32)
    N = int(psi_powers.shape[0])

    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("NTT size N must be a positive power of two")

    # ω = ψ^2
    omega = mod_mul(psi_powers[1], psi_powers[1], q)

    # Build ω^k table for k = 0..N-1
    omega_powers = [jnp.uint32(1)]
    cur = jnp.uint32(1)
    for _ in range(1, N):
        cur = mod_mul(cur, omega, q)
        omega_powers.append(cur)
    omega_powers = jnp.asarray(omega_powers, dtype=jnp.uint32)

    # Pack stage twiddles
    stage_tables = []
    length = 1
    while length < N:
        stride = N // (2 * length)
        exponents = jnp.arange(length, dtype=jnp.int32) * stride
        stage_tables.append(omega_powers[exponents])
        length *= 2

    stage_twiddles = jnp.concatenate(stage_tables, axis=0)   # length N - 1

    # Precompute bit-reversal permutation once
    bitrev_perm = _bit_reverse_indices(N)                    # length N

    # Pack both into returned twiddles
    packed_twiddles = jnp.concatenate([stage_twiddles, bitrev_perm], axis=0)

    return psi_powers, packed_twiddles