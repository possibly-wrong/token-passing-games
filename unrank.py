import math
import numpy as np

def rank_subset(S, n=None):
    """Return colex rank of k-subset S of {0..n-1}."""
    return sum(math.comb(c, j + 1) for j, c in enumerate(S))

# Different implementations of unrank_subset may be fastest depending
# on application. This is the direct combinadic approach.
def unrank_subset(r, n, k):
    """Return k-subset S of {0..n-1} with colex rank r."""
    S = [0] * k
    while k > 0:
        n = n - 1
        offset = math.comb(n, k)
        if r >= offset:
            r = r - offset
            k = k - 1
            S[k] = n
    return S

# Speed up calculation of "adjacent" binomial coefficients.
def unrank_subset(r, n, k):
    """Return k-subset S of {0..n-1} with colex rank r."""
    S = [0] * k
    offset = math.comb(n, k)
    while k > 0:
        # Decrease n and update offset to comb(n, k).
        offset = offset * (n - k) // n
        n = n - 1
        if r >= offset:
            r = r - offset
            k = k - 1
            S[k] = n
            if k < n:
                offset = offset * (k + 1) // (n - k)
    return S

# Use binary search.
def unrank_subset(r, n, k):
    """Return k-subset S of {0..n-1} with colex rank r."""
    S = [0] * k
    while k > 0:
        # Use binary search to decrease n until r >= comb(n, k).
        lower = k - 1
        while lower < n:
            mid = (lower + n + 1) // 2
            if r < math.comb(mid, k):
                n = mid - 1
            else:
                lower = mid
        r = r - math.comb(n, k)
        k = k - 1
        S[k] = n
    return S

def rank_subset_lex(S, n):
    """Return lex rank of k-subset S of {0..n-1}."""
    return math.comb(n, len(S)) - 1 - rank_subset_colex([n - 1 - c
                                                         for c in reversed(S)])

def unrank_subset_lex(r, n, k):
    """Return k-subset S of {0..n-1} with lex rank r."""
    return [n - 1 - c for c in
            reversed(unrank_subset_colex(math.comb(n, k) - 1 - r, n, k))]

def rank_weak_composition(x):
    """Return colex rank of weak composition x."""
    return rank_subset(np.cumsum(np.array(x[:-1]) + 1) - 1)
    
def unrank_weak_composition(r, n, s):
    """Return weak composition with colex rank r, length n, sum s."""
    return list(np.diff(np.array(unrank_subset(r, s + n - 1, n - 1), dtype=int),
                        prepend=-1, append=s + n - 1) - 1)
