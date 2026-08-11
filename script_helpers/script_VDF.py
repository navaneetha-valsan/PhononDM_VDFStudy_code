"""
Velocity-distribution utilities for the paper plotting notebooks.

This module provides normalized speed distributions and band-building helpers
for the three halo models used in the analysis:

* SHM: truncated Maxwell-Boltzmann distribution.
* TSA: Tsallis distribution with ``q = 1 - V0**2 / VESC**2``.
* EMP: empirical Mao et al.-style distribution with an exponential core and
  power-law cutoff.

The notebook scripts use the first group of functions for plotting
``4 pi v^2 f(v)`` directly and the second group for rms-matching studies,
where the model-dependent velocity parameter is chosen to match the SHM rms
speed at fixed ``VESC``.
"""

import numpy as np
from scipy.special import erf

from scipy.integrate import quad
# from scipy.integrate import simps
from scipy.integrate import simpson as simps

from scipy.optimize import root_scalar



### Normalized speed distributions, used directly in VDF comparison plots. ###

# 1. Truncated Maxwell-Boltzmann
def f_MB_norm(v, v0, v_esc):
    """    Truncated Maxwell-Boltzmann distribution with normalization.
        Cutoff set to the corresponding v_esc value
    Args:
        v (array): Speed values.
        v0 (float): Most probable speed.
        v_esc (float): Escape velocity.
    Returns:
        array: Normalized distribution values.
    """
    sigma_v = v0 * np.sqrt(3 / 2)
    z = v_esc / v0
    N_esc = erf(z) - (2 / np.sqrt(np.pi)) * z * np.exp(-z**2)
    prefactor = (1 / N_esc) * (3 / (2 * np.pi * sigma_v**2))**(3/2) * 4 * np.pi
    return np.where(v < v_esc, prefactor * v**2 * np.exp(-3 * v**2 / (2 * sigma_v**2)), 0)

# 2. Tsallis distribution with q parameter set by escape velocity
def f_Tsallis_norm(v, v0, v_esc):
    """   Tsallis distribution with q parameter set by escape velocity and normalization.
        Cutoff set to the corresponding v_esc value
    Args:
        v (array): Speed values.
        v0 (float): Most probable speed.
        v_esc (float): Escape velocity.
    Returns:
        array: Normalized distribution values.
    """
    q = 1 - (v0**2 / v_esc**2)
    arg = 1 - (1 - q) * v**2 / v0**2
    f = np.where((v < v_esc) & (arg > 0), v**2 * arg**(1 / (1 - q)), 0)
    norm = simps(f, v)
    return f / norm if norm > 0 else f


# 3. Empirical with exponential falloff
def f_Empirical_norm(v, v0, v_esc, p):
    """    Empirical distribution with exponential falloff and normalization.
        Cutoff set to the corresponding v_esc value
    Args:
        v (array): Speed values.
        v0 (float): Most probable speed.
        v_esc (float): Escape velocity.
        p (float): Empirical distribution parameter.
    Returns:
        array: Normalized distribution values.
    """
    core = v_esc**2 - v**2
    f = np.where((v < v_esc) & (core > 0), v**2 * np.exp(-v / v0) * core**p, 0)
    norm = simps(f, v)
    return f / norm if norm > 0 else f

# Compute min/max bands when velocity parameters are varied independently.
def get_band(distribution_func, v, v0_vals, vesc_vals, **kwargs):
    """     Compute the min/max bands for the distribution function over a range of (v0, v_esc) pairs.
    Args:
        distribution_func (function): The distribution function to compute.
        v (array): Speed values.
        v0_vals (array): Array of most probable speed values.
        vesc_vals (array): Array of escape velocity values.
        **kwargs: Additional parameters for the distribution function.
    Returns:
        tuple: Minimum and maximum distribution values across all (v0, v_esc) pairs.
    """
    all_distributions = []
    for v0 in v0_vals:
        for vesc in vesc_vals:
            f = distribution_func(v, v0, vesc, **kwargs)
            all_distributions.append(f * 1e0)
    return np.min(all_distributions, axis=0), np.max(all_distributions, axis=0)

# Compute min/max bands for the empirical-shape parameter p.
def get_p_band(distribution_func, v, v0_val, vesc_val, p_vals):
    """    
        Computes the minimum and maximum of the distribution function over a range of p values.
    Args:
        distribution_func: Function to compute the distribution.
        v: Velocity array.
        v0_val: Most probable speed.
        vesc_val: Escape velocity.
        p_vals: Array of p values for the empirical distribution.
    Returns:
        Tuple of arrays (min_distribution, max_distribution) containing the minimum and maximum values of the distribution function.
    """
    all_distributions = []
    for p in p_vals:
        f = distribution_func(v, v0_val, vesc_val, p)
        all_distributions.append(f * 1e0)
    return np.min(all_distributions, axis=0), np.max(all_distributions, axis=0)

# Compute all distributions for explicit (v0, v_esc) pairs.
def get_all_distributions(distribution_func, v, v0_vals, vesc_vals, **kwargs):
    """     Compute all distributions for the given (v0, v_esc) pairs.
    Args:
        distribution_func (function): The distribution function to compute.
        v (array): Speed values.
        v0_vals (array): Array of most probable speed values.
        vesc_vals (array): Array of escape velocity values.
        **kwargs: Additional parameters for the distribution function.
    Returns:
        tuple: List of distribution values and corresponding (v0, vesc) parameter labels.
    """
    all_distributions = []
    param_labels = []
    
    for v0 in v0_vals:
        for vesc in vesc_vals:
            f = distribution_func(v, v0, vesc, **kwargs)
            all_distributions.append(f * 1e0)
            param_labels.append((v0, vesc))
    
    return all_distributions, param_labels



### Raw distributions plus explicit normalization for rms-matching studies. ###

# 1. Truncated Maxwell-Boltzmann
def f_MB(v, V0, VESC):
    """    Truncated Maxwell-Boltzmann distribution.
        Cutoff set to the corresponding VESC value
    Args:
        v (array): Speed values.
        V0 (float): Most probable speed.
        VESC (float): Escape velocity.
    Returns:
        array: Distribution values.
    """
    cut_off = VESC  # Set cutoff to the corresponding VESC value
    # return np.exp(-v**2 / V0**2) * (v <= cut_off)
    return np.where(v <= cut_off, np.exp(-v**2 / V0**2), 0)

# 2. Tsallis distribution with q parameter set by escape velocity
def f_Tsallis(v, V0, VESC):
    """    Tsallis distribution with q parameter set by escape velocity.
        Cutoff set to the corresponding VESC value
    Args:
        v (array): Speed values.
        V0 (float): Most probable speed.
        VESC (float): Escape velocity.
    Returns:
        array: Distribution values.
    """
    cut_off = VESC  # Set cutoff to the corresponding VESC value
    q = 1 - V0**2/VESC**2
    # return (1 - (1 - q) * v**2 / V0**2)**(1 / (1 - q)) * (v <= cut_off)
    return np.where(v <= cut_off, (1 - (1 - q) * v**2 / V0**2)**(1 / (1 - q)), 0)

# 3. Empirical with exponential falloff
def f_Empirical(v, V0, VESC, p):
    """    Empirical distribution with exponential falloff.
        Cutoff set to the corresponding VESC value
    Args:
        v (array): Speed values.
        V0 (float): Most probable speed.
        VESC (float): Escape velocity.
        p (float): Empirical distribution parameter.
    Returns:
        array: Distribution values.
    """
    # p = 3/2
    cut_off = VESC  # Set cutoff to the corresponding VESC value
    # return np.exp(-v / V0) * (VESC**2 - v**2)**p * (v <= cut_off)
    return np.where(v <= cut_off, np.exp(-v / V0) * (VESC**2 - v**2)**p, 0.0)

# 3b. Empirical without exponential falloff (v0 → ∞)
def f_Empirical_infinite(v, V0, VESC, **kwargs):
    """Empirical v0 -> infinity limit used when rms matching has no finite root."""
    p = kwargs["p"]
    return np.where(v <= VESC, (VESC**2 - v**2)**p, 0.0)



# Normalize the 3D distribution with the spherical measure 4*pi*v^2.
def compute_normalization_constant(dist_func, V0, VESC, v_min, cut_off, **kwargs):
    """     Compute the normalization constant k for the 3D velocity distribution.
    Args:
        dist_func (function): The distribution function to normalize.
        V0 (float): Most probable speed.
        VESC (float): Escape velocity.
        v_min (float): Minimum velocity for integration.
        cut_off (float): Cutoff velocity for integration.
        **kwargs: Additional parameters for the distribution function.
    Returns:
        float: Normalization constant k.
    """
    integrand = lambda v: dist_func(v, V0, VESC, **kwargs) * 4 * np.pi * v**2
    integral, _ = quad(integrand, v_min, cut_off)
    return 1 / integral

# Compute normalized 4*pi*v^2*f(v) values for SHM, TSA, and EMP models.
def compute_distributions(dist, v_min, V0, VESC, **kwargs):
    """     Compute the normalized distributions 4πv²f(v) for SHM, TSA, and EMP models.
    Args:
        dist (function): Distribution function to compute.
        v_min (float): Minimum velocity for integration.
        V0 (float): Most probable speed.
        VESC (float): Escape velocity.
        **kwargs: Additional parameters for the distribution function.
    Returns:
        array: Normalized distribution values for 4πv²f(v).
    """
    # cut_off = VESC  # Set cutoff to the corresponding VESC value
    cut_off = VESC
    v_values = np.linspace(v_min, 800, 1000)

    # Normalization constants
    k = compute_normalization_constant(dist, V0, VESC, v_min, cut_off, **kwargs)

    # Raw distributions
    norm = k  * dist(v_values, V0, VESC, **kwargs)

    # Multiply by 4πv²
    dist_val = 4 * np.pi * v_values**2 * norm

    return dist_val


### RMS matching helpers. ###

def _velocity_grid(v=None, v_min=0.0):
    """Return the notebook-compatible default velocity grid when none is supplied."""
    if v is None:
        return np.linspace(v_min, 800, 1000)
    return np.asarray(v)


def _compute_distributions_on_grid(dist, v, v_min, V0, VESC, **kwargs):
    """Compute normalized 4*pi*v^2*f(v) values on the supplied velocity grid."""
    k = compute_normalization_constant(dist, V0, VESC, v_min, VESC, **kwargs)
    norm = k * dist(v, V0, VESC, **kwargs)
    return 4 * np.pi * v**2 * norm


### Function to compute the rms of an normalized distribution f(v)
def find_rms(f, v):
    """     Compute the standard deviation of the distribution.
    Args:
        f (array): Normalized distribution values.
        v (array): Speed values.
    Returns:
        float: Root mean square (rms) value of the distribution.
    """
    v = np.asarray(v)
    mean_sq = simps(v**2 * f, x=v)               # <v^2>
    vrms = np.sqrt(mean_sq)                # rms = sqrt(<v^2>)
    return vrms


def objective_MB(v0, v, v_min, v_esc):
    v = _velocity_grid(v, v_min)
    dist = _compute_distributions_on_grid(f_MB, v, v_min, v0, v_esc)
    rms = find_rms(dist, v)
    return rms


def objective_tsallis(x, v, v_min, v_esc, trgt_rms):
    v = _velocity_grid(v, v_min)
    dist = _compute_distributions_on_grid(f_Tsallis, v, v_min, x, v_esc)
    rms = find_rms(dist, v)
    return rms - trgt_rms


def objective_empirical(x, v, v_min, v_esc, trgt_rms, p):
    v = _velocity_grid(v, v_min)
    dist = _compute_distributions_on_grid(f_Empirical, v, v_min, x, v_esc, p=p)
    rms = find_rms(dist, v)
    return rms - trgt_rms


def rms_empirical_infinite(v, v_min, v_esc, p):
    v = _velocity_grid(v, v_min)
    dist = _compute_distributions_on_grid(
        f_Empirical_infinite,  # function
        v,
        v_min,                 # v_min (positional!)
        np.inf,                # v0 -> infinity
        v_esc,                 # escape speed
        p=p
    )
    return find_rms(dist, v)


def match_v0_Tsallis(trgt_rms, v_esc, v=None, v_min=0.0, bracket=(100, 450), n_scan=500):
    v = _velocity_grid(v, v_min)
    a, b = bracket  # bracket

    f_a = objective_tsallis(a, v, v_min, v_esc, trgt_rms)
    f_b = objective_tsallis(b, v, v_min, v_esc, trgt_rms)

    # Try brentq if sign change exists
    if f_a * f_b < 0:
        sol = root_scalar(
            objective_tsallis,
            args=(v, v_min, v_esc, trgt_rms),
            bracket=[a, b],
            method='brentq'
        )
        return sol.root, 0.0  # matched exactly

    # Fallback: scan bracket and find x minimizing |rms - trgt_rms|
    xs = np.linspace(a, b, n_scan)
    diffs = [
        (
            x,
            abs(objective_tsallis(x, v, v_min, v_esc, trgt_rms)),
            objective_tsallis(x, v, v_min, v_esc, trgt_rms)
        )
        for x in xs
    ]
    best_x, min_abs_diff, signed_diff = min(diffs, key=lambda tup: tup[1])
    print(
        f"[INFO] Fallback match for v0={trgt_rms}, v_esc={v_esc} "
        f"-> v0_tsallis={best_x:.2f}, delta_rms={signed_diff:.4f}"
    )
    return best_x, signed_diff


def match_v0_Empirical(trgt_rms, v_esc, v=None, v_min=0.0, p=1.5, bracket=(100.0, 15000.0)):
    v = _velocity_grid(v, v_min)

    # --- Step 1: infinite-v0 limit ---
    rms_inf = rms_empirical_infinite(v, v_min, v_esc, p)

    if trgt_rms > rms_inf:
        # no finite solution exists
        return np.inf, trgt_rms - rms_inf

    # --- Step 2: attempt finite solution ---
    a, b = bracket   # your chosen physical bracket

    fa = objective_empirical(a, v, v_min, v_esc, trgt_rms, p)
    fb = objective_empirical(b, v, v_min, v_esc, trgt_rms, p)

    # If no sign change, treat as infinite-v0 case
    if fa * fb > 0:
        return np.inf, trgt_rms - rms_inf

    sol = root_scalar(
        objective_empirical,
        args=(v, v_min, v_esc, trgt_rms, p),
        bracket=[a, b],
        method="brentq"
    )

    return sol.root, 0.0
