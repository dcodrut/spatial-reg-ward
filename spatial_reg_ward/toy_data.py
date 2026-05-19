import numpy as np


def make_half_moon_toy_data(
    n: int = 300,
    *,
    seed: int = 1234,
    n1: int = None,
    n2: int = None,
    n3: int = None,
    radial_noise: float = 0.1,
    betas=None,
    sigmas=None,
):
    """Create the synthetic half-moon toy dataset used in examples/tests.

    Returns a dict with keys: x, y, coords, regime_true, theta, v.
    """
    if n1 is None or n2 is None or n3 is None:
        # Default proportions based on (60, 140, 100) for n=300
        weights = np.array([60, 140, 100], dtype=float)
        weights /= weights.sum()
        n1, n2 = (weights[:2] * n).round().astype(int)
        n3 = n - n1 - n2

    if n1 + n2 + n3 != n:
        raise ValueError("n1 + n2 + n3 must equal n.")

    rng = np.random.default_rng(seed=seed)

    # Half-moon geometry
    theta = rng.uniform(0.0, np.pi, size=n)
    noise_radial = rng.normal(0.0, radial_noise, size=n)
    radius = 1.0 + noise_radial
    x_coord = radius * np.cos(theta)
    y_coord = radius * np.sin(theta)

    # Regimes: strictly ordered along the arc
    idx_sorted = np.argsort(theta)
    regime_true = np.empty(n, dtype=int)
    regime_true[idx_sorted[:n1]] = 3
    regime_true[idx_sorted[n1:n1 + n2]] = 2
    regime_true[idx_sorted[n1 + n2:]] = 1

    # Single covariate
    v = rng.normal(0.0, 1.0, size=n)

    # Regime-specific parameters
    if betas is None:
        betas = {
            1: (0.5, 1.0),
            2: (0.5, -0.3),
            3: (-1.0, 1.0),
        }
    if sigmas is None:
        sigmas = {
            1: 0.3,
            2: 0.5,
            3: 0.4,
        }

    beta0 = np.array([betas[c][0] for c in regime_true])
    beta1 = np.array([betas[c][1] for c in regime_true])
    sigma_eps = np.array([sigmas[c] for c in regime_true])

    eps = rng.normal(0.0, sigma_eps, size=n)
    r = beta0 + beta1 * v + eps

    x = v.reshape(-1, 1)
    y = r
    coords = np.column_stack([x_coord, y_coord])

    return {
        "x": x,
        "y": y,
        "coords": coords,
        "regime_true": regime_true,
        "theta": theta,
        "v": v,
    }
