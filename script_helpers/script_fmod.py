"""
Daily-modulation helpers for the paper plotting notebooks.

The functions here read the modified PhonoDark HDF5 layout and compute either
the modulation amplitude

    f_mod = max(|R - <R>|) / <R>

or the normalized daily rate ``R / <R>``.  The same routines are used for the
SHM, Tsallis, and empirical halo-model comparisons in the paper.

Functions ending in ``_from_diff`` recompute an effective total rate by
integrating ``data/diff_rate`` above a new threshold.  The threshold arguments
use the same unit as the HDF5 file, which is eV in the paper notebooks
(``0.001`` for 1 meV, ``0.02`` for 20 meV).
"""

import os
import h5py
import numpy as np


# Amount of daily modulation of a material:
# f_mod = max(|R - <R>|) / <R>.

def generate_file_prefixes(target, mediators, numerics, mod, v_0, v_e, v_esc):
    """
    Generate file prefixes based on the target, mediators, numerics, mod, and velocity parameters.
    Args:
        target (str): The target material.
        mediators (list): List of mediators.
        numerics (str): Numeric suffix.
        mod (str): Modulation type.
        v_0 (list): List of v_0 values.
        v_e (list): List of v_e values.
        v_esc (list): List of v_esc values.
    Returns:
        list: List of generated file prefixes.
    """
    prefixes = []

    for mediator in mediators:
        base_prefix = f"{target}_{mediator}_{numerics}"

        # for mod in mods:
        for i in v_0:
            for j in v_e:
                for k in v_esc:
                    # Keep this in sync with calculator.py output filenames.
                    full_prefix = f"{base_prefix}_{mod}_{i}_{j}_{k}"
                    prefixes.append(full_prefix)

    return prefixes



def calculate_fmod(h5file, time, dm_mass, th):
    """
    Calculate f_mod for a single HDF5 file.
    Args:
        h5file (h5py.File): The HDF5 file to read data from.
        time (list): List of time values.
        dm_mass (list): List of dark matter masses.
        th (int or str): Energy-threshold index in the HDF5 group path.
    Returns:
        list: List of modulation factors for each dark matter mass.
    """
    fmod = []
    for m in range(len(dm_mass)):
        rate = []
        for i in range(len(time)):
            rate.append(h5file[f'data/rate/{i}/{m}/{th}'][()])
        rate = np.array(rate)
        avg_rate = np.average(rate)
        modulation = max(abs(rate - avg_rate)) / avg_rate
        fmod.append(modulation)
    return fmod



def calculate_fmod_maxmin(files, time, dm_mass, th):
    """
    Calculate the envelope of f_mod values across a list of HDF5 files.
    Args:
        files (list): List of HDF5 files to read data from.
        time (list): List of time values.
        dm_mass (list): List of dark matter masses.
        th (str): The energy threshold index.
    Returns:
        tuple: Two lists containing the maximum and minimum modulation factors for each dark matter mass.
    """
    fmod_max = []
    fmod_min = []
    for m in range(len(dm_mass)):
        fmod_mass = []
        for file in files:
            rate = []
            for i in range(len(time)):
                rate.append(file[f'data/rate/{i}/{m}/{th}'][()])
            rate = np.array(rate)
            avg_rate = np.average(rate)
            modulation = max(abs(rate - avg_rate)) / avg_rate  
            fmod_mass.append(modulation)
        fmod_max.append(max(fmod_mass))
        fmod_min.append(min(fmod_mass))
    return fmod_max, fmod_min


"""
Daily modulation of rate (R/<R>) for different dark matter masses (m) and energy thresholds (th).
"""

def calculate_dmod(h5file, time, m, th):
    """
    Calculate R/<R> for one mass and threshold in a single HDF5 file.
    Args:
        h5file (h5py.File): The HDF5 file to read data from.
        time (list): List of time values.
        m (int): Dark-matter mass index in the HDF5 group path.
        th (int or str): Energy-threshold index in the HDF5 group path.
    Returns:
        list: List of modulation  for each dark matter mass.
    """
    rate = []
    for i in range(len(time)):
        rate.append(h5file[f'data/rate/{i}/{m}/{th}'][()])
    rates = np.array(rate)
    dmod = rates / np.average(rates)

    return dmod


def calculate_dailymod_maxmin(files, time, m, th):
    """
    Calculate the envelope of R/<R> across a list of HDF5 files.
    Args:
        files (list): List of HDF5 files to read data from.
        time (list): List of time values.
        m (int): Dark-matter mass index in the HDF5 group path.
        th (int or str): Energy-threshold index in the HDF5 group path.
    Returns:
        tuple: Two lists containing the maximum and minimum modulation for each dark matter mass.
    """
    dmod_max = None
    dmod_min = None
    
    for file in files:
        rate = []
        for i in range(len(time)):
            rate.append(file[f'data/rate/{i}/{m}/{th}'][()])
        rates = np.array(rate)
        dmod = rates / np.mean(rates)

        if dmod_max is None:
            dmod_max = dmod.copy()
            dmod_min = dmod.copy()
        else:
            dmod_max = np.maximum(dmod_max, dmod)
            dmod_min = np.minimum(dmod_min, dmod)

    return dmod_max, dmod_min

# Helpers for imposing a different analysis threshold during post-processing.


def total_rate_above_threshold(diff_rate, new_threshold, threshold_run, energy_bin_width):
    """
    Integrate a stored differential-rate spectrum above a new threshold.

    ``new_threshold``, ``threshold_run``, and ``energy_bin_width`` must use the
    same energy unit as the HDF5 file.  The paper notebooks use eV.
    """
    # If new_threshold is below or equal to the threshold used in the run,
    # just integrate from the first bin.
    if new_threshold <= threshold_run:
        i_cut = 0
    else:
        # First bin whose LOWER edge is >= new_threshold
        i_cut = int(np.ceil((new_threshold - threshold_run) / energy_bin_width))

        # i_cut = int(np.ceil((new_threshold - threshold_run) / energy_bin_width - 1))


    if i_cut >= len(diff_rate):
        return 0.0

    # diff_rate can be complex; usually you want the real part
    return np.sum(diff_rate[i_cut:]).real

def calculate_fmod_from_diff(h5file, time, dm_mass, th, new_threshold_meV, threshold_run_meV, energy_bin_width_meV):
    """
    Calculate the daily modulation factor using diff_rate, imposing a higher
    effective threshold (new_threshold_meV) than the one used in the run
    (threshold_run_meV).

    Args:
        h5file (h5py.File): The HDF5 file to read data from.
        time (list): List of time values (or just indices; only the length matters).
        dm_mass (list): List of dark matter masses (same ordering as in the HDF5).
        th (int or str): The energy-threshold index key used in the file.
        new_threshold_meV (float): Historical name; desired threshold in eV.
        threshold_run_meV (float): Historical name; run threshold in eV.
        energy_bin_width_meV (float): Historical name; bin width in eV.

    Returns:
        np.ndarray: Array of modulation factors for each dark matter mass.
    """

    fmod = []

    for m in range(len(dm_mass)):
        # Collect effective total rate at each time after re-thresholding
        rates = []

        for i in range(len(time)):
            # Read diff_rate array instead of a single total_rate
            # Adjust path if your HDF5 uses a different name
            diff_rate = h5file[f'data/diff_rate/{i}/{m}/{th}'][()]

            # Compute total rate above new_threshold_meV
            total_rate_t = total_rate_above_threshold(
                diff_rate=diff_rate,
                new_threshold=new_threshold_meV,
                threshold_run=threshold_run_meV,
                energy_bin_width=energy_bin_width_meV
            )

            rates.append(total_rate_t)

        rates = np.array(rates)
        avg_rate = np.average(rates)
        modulation = np.max(np.abs(rates - avg_rate)) / avg_rate
        # if avg_rate == 0:
        #     modulation = 0.0
        # else:
        #     modulation = np.max(np.abs(rates - avg_rate)) / avg_rate

        fmod.append(modulation)

    return np.array(fmod)

def calculate_fmod_maxmin_from_diff(files, time, dm_mass, th, new_threshold_meV, threshold_run_meV, energy_bin_width_meV):
    """
    Calculate the maximum and minimum daily modulation factors using diff_rate,
    imposing a higher effective threshold (new_threshold_meV) than the one used
    in the run (threshold_run_meV).

    Args:
        files (list): List of HDF5 files to read data from.
        time (list): List of time values (or just indices; only the length matters).
        dm_mass (list): List of dark matter masses (same ordering as in the HDF5).
        th (str): The energy-threshold index key used in the file, e.g. "1.0".
        new_threshold_meV (float): Historical name; desired threshold in eV.
        threshold_run_meV (float): Historical name; run threshold in eV.
        energy_bin_width_meV (float): Historical name; bin width in eV.

    Returns:
        tuple: Two lists containing the maximum and minimum modulation factors for each dark matter mass.
    """

    fmod_max = []
    fmod_min = []

    for m in range(len(dm_mass)):
        fmod_mass = []

        for file in files:
            # Collect effective total rate at each time after re-thresholding
            rates = []

            for i in range(len(time)):
                diff_rate = file[f'data/diff_rate/{i}/{m}/{th}'][()]

                total_rate_t = total_rate_above_threshold(
                    diff_rate=diff_rate,
                    new_threshold=new_threshold_meV,
                    threshold_run=threshold_run_meV,
                    energy_bin_width=energy_bin_width_meV
                )

                rates.append(total_rate_t)

            rates = np.array(rates)
            avg_rate = np.average(rates)
            modulation = np.max(np.abs(rates - avg_rate)) / avg_rate
            fmod_mass.append(modulation)

        fmod_max.append(max(fmod_mass))
        fmod_min.append(min(fmod_mass))

    return fmod_max, fmod_min

# Daily modulation R/<R> for thresholds imposed from diff_rate.

def calculate_dmod_from_diff(h5file, time, m, th, new_threshold_meV, threshold_run_meV, energy_bin_width_meV):
    """Calculate R/<R> from diff_rate after imposing a new threshold."""
    
    rate = []
    for i in range(len(time)):
        diff_rate = h5file[f'data/diff_rate/{i}/{m}/{th}'][()]

        # Compute total rate above new_threshold_meV
        total_rate_t = total_rate_above_threshold(
            diff_rate=diff_rate,
            new_threshold=new_threshold_meV,
            threshold_run=threshold_run_meV,
            energy_bin_width=energy_bin_width_meV
        )
        rate.append(total_rate_t)
    rates = np.array(rate)
    dmod = rates / np.average(rates)

    return dmod


def calculate_dailymod_maxmin_from_diff(files, time, m, th, new_threshold_meV, threshold_run_meV, energy_bin_width_meV):
    """Calculate the R/<R> envelope from diff_rate for a file list."""
    
    dmod_max = None
    dmod_min = None
    
    for file in files:
        rate = []
        for i in range(len(time)):
            diff_rate = file[f'data/diff_rate/{i}/{m}/{th}'][()]

            total_rate_t = total_rate_above_threshold(
                diff_rate=diff_rate,
                new_threshold=new_threshold_meV,
                threshold_run=threshold_run_meV,
                energy_bin_width=energy_bin_width_meV
            )

            rate.append(total_rate_t)
        rates = np.array(rate)
        dmod = rates / np.mean(rates)

        if dmod_max is None:
            dmod_max = dmod.copy()
            dmod_min = dmod.copy()
        else:
            dmod_max = np.maximum(dmod_max, dmod)
            dmod_min = np.minimum(dmod_min, dmod)

    return dmod_max, dmod_min
