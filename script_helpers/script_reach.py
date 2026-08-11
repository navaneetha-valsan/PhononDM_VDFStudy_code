"""
Projected-reach helpers for the paper plotting notebooks.

The functions in this module read the HDF5 files produced by the modified
PhonoDark driver and convert rates into the cross-section normalizations used
in the projected-reach plots.  They assume the filename convention

    <target>_<mediator>_<numerics>_<VDF>_<v0>_<ve>_<vesc>.hdf5

and the modified HDF5 rate layout

    data/rate/<time_index>/<mass_index>/<threshold_index>
    data/diff_rate/<time_index>/<mass_index>/<threshold_index>

The post-processing functions with the historical suffix ``_meV`` expect
threshold values in eV, matching the HDF5 files.  For example, pass ``0.001``
for 1 meV and ``0.02`` for 20 meV.
"""

import os
import h5py
import numpy as np
from collections import defaultdict
# The read_data_* functions return arrays with shape
# (n_thresholds, n_masses).  DM masses are converted from eV to MeV in the
# returned arrays because the plotting notebooks label the x axes in MeV.

### Light hadrophilic mediator ###
def read_data_light_hadrophilic(file_prefix, data_dir):
    """
    Read rate from hdf5 files in the format rate = fw[f'data/rate/{time=0}/{mass}/{threshold}'][()] 
    and calculate sigma from rate for Light hadrophilic mediator physics model.

    function take arguments:
        file_prefix: prefix of the file to read
        data_dir: directory where the data files are located

    function gives output as:
        DM_mass: DM mass in MeV
        threshold: threshold in meV
        rates_array: 2D array of shape (n_thresholds, n_masses) with rate values
        sigmas_array: 2D array of shape (n_thresholds, n_masses) with sigma values
    """
    filename = os.path.join(data_dir, file_prefix + '.hdf5')
    with h5py.File(filename, 'r') as fw:
        DM_mass = fw['particle_physics/dm_properties/mass_list'][()] * 1e-6  # [MeV]
        threshold = fw['particle_physics/threshold'][()] * 1e3  # (n_thresholds,) in [meV]
        n_masses = len(DM_mass)
        n_thresholds = len(threshold)

        # Initialize 2D arrays: shape (n_thresholds, n_masses)
        rates_array = np.zeros((n_thresholds, n_masses))
        sigmas_array = np.zeros((n_thresholds, n_masses))

        # for t_idx, th in enumerate(threshold):
        for t_idx in range(len(threshold)):
            for m_idx, mass in enumerate(DM_mass):
                rate = fw[f'data/rate/0/{m_idx}/{t_idx}'][()]
                rates_array[t_idx, m_idx] = rate

                sigma = (
                    3 / (2.69e58) / rate
                    * (1 / np.pi)
                    * (938 * mass)**2 / (938 + mass)**2
                    * 1e12
                    * (1 / (mass * 1e6 * 220 * 3.34e-6)**4)
                    * 3.88e-10
                )
                sigmas_array[t_idx, m_idx] = sigma

    return DM_mass, threshold, rates_array, sigmas_array

### Heavy hadrophilic mediator ###
def read_data_heavy_hadrophilic(file_prefix, data_dir):
    """
    Read rate from hdf5 files in the format rate = fw[f'data/rate/{time=0}/{mass}/{threshold}'][()] 
    and calculate sigma from rate for Heavy hadrophilic mediator physics model.

    function take arguments:
        file_prefix: prefix of the file to read
        data_dir: directory where the data files are located

    function gives output as:
        DM_mass: DM mass in MeV
        threshold: threshold in meV
        rates_array: 2D array of shape (n_thresholds, n_masses) with rate values
        sigmas_array: 2D array of shape (n_thresholds, n_masses) with sigma values
    """
    filename = os.path.join(data_dir, file_prefix + '.hdf5')
    with h5py.File(filename, 'r') as fw:
        DM_mass = fw['particle_physics/dm_properties/mass_list'][()] * 1e-6  # [GeV]
        threshold = fw['particle_physics/threshold'][()] * 1e3  # [meV]
        n_masses = len(DM_mass)
        n_thresholds = len(threshold)

        rates_array = np.zeros((n_thresholds, n_masses))
        sigmas_array = np.zeros((n_thresholds, n_masses))

        # for t_idx, th in enumerate(threshold):
        for t_idx in range(len(threshold)):
            for m_idx, mass in enumerate(DM_mass):
                rate = fw[f'data/rate/0/{m_idx}/{t_idx}'][()]
                rates_array[t_idx, m_idx] = rate

                sigma = (
                    3 / (2.69e58) / rate
                    * (1 / np.pi)
                    * (938 * mass)**2 / (938 + mass)**2
                    * 1e12
                    * 3.88e-10
                )
                sigmas_array[t_idx, m_idx] = sigma

    return DM_mass, threshold, rates_array, sigmas_array

### Light Dark Photon ###
def read_data_light_dark_photon(file_prefix, data_dir):
    """
    Read rate from hdf5 files in the format rate = fw[f'data/rate/{time=0}/{mass}/{threshold}'][()] 
    and calculate sigma from rate for Light dark photon mediator physics model.

    function take arguments:
        file_prefix: prefix of the file to read
        data_dir: directory where the data files are located

    function gives output as:
        DM_mass: DM mass in MeV
        threshold: threshold in meV
        rates_array: 2D array of shape (n_thresholds, n_masses) with rate values
        sigmas_array: 2D array of shape (n_thresholds, n_masses) with sigma values
    """
    filename = os.path.join(data_dir, file_prefix + '.hdf5')
    with h5py.File(filename, 'r') as fw:
        DM_mass = fw['particle_physics/dm_properties/mass_list'][()] * 1e-6  # [GeV]
        threshold = fw['particle_physics/threshold'][()] * 1e3  # [meV]
        n_masses = len(DM_mass)
        n_thresholds = len(threshold)

        rates_array = np.zeros((n_thresholds, n_masses))
        sigmas_array = np.zeros((n_thresholds, n_masses))

        # for t_idx, th in enumerate(threshold):
        for t_idx in range(len(threshold)):
            for m_idx, mass in enumerate(DM_mass):
                rate = fw[f'data/rate/0/{m_idx}/{t_idx}'][()]
                rates_array[t_idx, m_idx] = rate

                sigma = (
                    3 / (2.69e58) / rate
                    * (1 / np.pi)
                    * (0.511 * mass)**2 / (0.511 + mass)**2
                    * 1e12
                    * (137 / (0.511e6))**4
                    * 3.88e-10
                )
                sigmas_array[t_idx, m_idx] = sigma

    return DM_mass, threshold, rates_array, sigmas_array


### Generate file prefixes for DM reach studies based on input parameters ###
def generate_file_prefixes(Target, Mediators, Numerics, V_0, V_E, V_ESC, Mod):
    """
    Generate file prefixes for DM reach studies based on input parameters.

    Parameters:
        Target (str): The detector material, e.g., 'Al2O3'.
        Mediators (list of str): List of mediator types.
        Numerics (str): Numerical settings label, e.g., 'standard'.
        V_0 (list of int): List of V_0 values.
        V_E (list of int): List of V_E values.
        V_ESC (list of int): List of escape velocity values.

    Returns:
        grouped_prefixes (dict): Dictionary with keys (mediator, halo model)
                                    and values as lists of file name prefixes.
    """
    # Construct base prefixes
    base_prefixes = [f"{Target}_{mediator}_{Numerics}" for mediator in Mediators]

    # Grouped prefixes dict
    grouped_prefixes = defaultdict(list)

    for base_prefix in base_prefixes:
        # Extract just the mediator name from the base_prefix
        mediator = base_prefix.replace(f"{Target}_", "").replace(f"_{Numerics}", "")

        for mod in Mod:
            key = (mediator, mod)

            for v0 in V_0:
                for ve in V_E:
                    for vesc in V_ESC:
                        if isinstance(v0, str):
                            v0_str = v0
                        elif np.isinf(v0):
                            v0_str = "inf"
                        else:
                            v0_str = str(int(v0))

                        file_prefix = f"{base_prefix}_{mod}_{v0_str}_{ve}_{vesc}"
                        # file_prefix = f"{base_prefix}_{mod}_{v0}_{ve}_{vesc}"
                        grouped_prefixes[key].append(file_prefix)

    return grouped_prefixes


### Generate file prefixes for DM reach studies based on input parameters ###
def check_for_duplicates(grouped_prefixes):
    """
    Check for duplicate filenames in grouped_prefixes and print summary with group-wise counts.
    
    Parameters:
        grouped_prefixes (dict): Dictionary with (mediator, mod) as keys
                                 and list of generated file prefixes as values.

    Prints:
        - Total number of files
        - Number of unique files
        - List of duplicate filenames (if any)
    """
    all_filenames = []

    print("\n--- File Count per Group ---")
    for key, files in grouped_prefixes.items():
        print(f"{key}: {len(files)} files")
        all_filenames.extend(files)

    # Count duplicates
    unique_filenames = set(all_filenames)
    duplicates = [f for f in all_filenames if all_filenames.count(f) > 1]

    print("\n--- File Check ---")
    print(f"Total number of files: {len(all_filenames)}")
    print(f"Number of unique files: {len(unique_filenames)}")

    if duplicates:
        print(f"\nWarning: found {len(set(duplicates))} duplicate file(s):")
        for dup in set(duplicates):
            print(f"  {dup}")
    else:
        print("\nNo duplicate files found.")

### Extract v0 from file prefix ###
# def get_v0_from_prefix(prefix: str) -> int:
#     """
#     Extract v0 from a prefix of the form:
#         Target_mediator_Numerics_mod_v0_ve_vesc
#     """
#     parts = prefix.split("_")
#     try:
#         v0_str = parts[-3]   # ... _mod_ v0 _ ve _ vesc
#         return int(v0_str)
#     except (IndexError, ValueError) as e:
#         raise ValueError(f"Cannot extract v0 from prefix '{prefix}': {e}")

def get_v0_from_prefix(prefix: str):
    """
    Extract v0 from a prefix of the form:
        Target_mediator_Numerics_mod_v0_ve_vesc
    """
    parts = prefix.split("_")

    try:
        v0_str = parts[-3]   # ... _mod_ v0 _ ve _ vesc
    except IndexError:
        raise ValueError(f"Cannot extract v0 from prefix '{prefix}'")

    if v0_str == "inf":
        return np.inf

    try:
        return int(v0_str)
    except ValueError:
        raise ValueError(
            f"Cannot extract v0 from prefix '{prefix}': "
            f"invalid literal '{v0_str}'"
        )

### Read fiducial rates and sigmas for given mediators and fiducial parameters ###
def read_fiducial_data(data_dir, Target, Mediators, Numerics, fid_mod, fid_velocity):
    """
    Read fiducial rates and sigmas for given mediators and fiducial parameters.

    Parameters:
        data_dir (str): Path to the directory containing data files.
        target (str): Target material (e.g., "Al2O3").
        mediators (list): List of mediator types.
        numerics (str): Numerics tag (e.g., "standard").
        fid_mod (str): Fiducial halo model (e.g., "SHM").
        fid_velocity (tuple): Fiducial velocity parameters (v0, ve, vesc) as strings.

    Returns:
        dict: Nested dictionary structure of fiducial data:
              fiducial_data[mediator][threshold] = {
                  "dm_mass": [...],
                  "rates": [...],
                  "sigmas": [...]
              }
    """
    
    fiducial_data = {}

    # Ensure we consistently have strings for building the prefix
    v0_str, ve_str, vesc_str = map(str, fid_velocity)
    # And an int v0 for passing into the read functions
    v0_fid = int(v0_str)

    for med in Mediators:
        # # Construct file prefix
        # fid_file_prefix = f"{Target}_{med}_{Numerics}_{fid_mod}_{'_'.join(fid_velocity)}"
        # Construct file prefix: Target_mediator_Numerics_mod_v0_ve_vesc
        fid_file_prefix = f"{Target}_{med}_{Numerics}_{fid_mod}_{v0_str}_{ve_str}_{vesc_str}"

        # Dynamically get the appropriate read function
        read_func = globals().get(f"read_data_{med}")
        if read_func is None:
            print(f"Warning: read function for {med} not found. Skipping.")
            continue

        # Read data
        dm_mass, thresholds, rates_2D, sigmas_2D = read_func(fid_file_prefix, data_dir)

        # # Store in dictionary
        # for idx, th in enumerate(thresholds):
        #     if med not in fiducial_data:
        #         fiducial_data[med] = {}
        #     fiducial_data[med][th] = {
        #         "dm_mass": dm_mass,
        #         "rates": rates_2D[idx],
        #         "sigmas": sigmas_2D[idx]
        #     }
        for idx, th in enumerate(thresholds):
            # Initialize mediator entry if not already present
            if med not in fiducial_data:
                fiducial_data[med] = {
                    "thresholds": thresholds  # store thresholds once
                }

            # Store threshold-specific data
            fiducial_data[med][th] = {
                "dm_mass": dm_mass,
                "rates": rates_2D[idx],
                "sigmas": sigmas_2D[idx]
            }

    return fiducial_data


### Computes max/min rate and sigma values for each (mediator, model, threshold) group ###
def compute_group_extremes(grouped_prefixes, data_dir, fiducial_data):
    """
    Computes max/min rate and sigma values for each (mediator, model, threshold) group.

    Parameters:
        grouped_prefixes (dict): Mapping from (mediator, mod) → list of file prefixes.
        data_dir (str): Directory where HDF5 files are located.
        fid_dm_mass (list): List of fiducial DM masses to compare against.
        fid_threshold (list): List of threshold values (in same order as rate/sigma axes).

    Returns:
        Tuple of four dictionaries: max_rate, min_rate, max_sigma, min_sigma
    """
    # Initialize dictionaries to store results
    group_max_rate = {}
    group_min_rate = {}
    group_max_sigma = {}
    group_min_sigma = {}

    # Loop over each (mediator, mod) group
    for (mediator, mod), file_prefixes in grouped_prefixes.items():
        # Dynamically get the appropriate function for reading data
        read_data_function = globals().get(f"read_data_{mediator}")
        
        if read_data_function is None:
            print(f"Error: Function read_data_{mediator} not found. Skipping group ({mediator}, {mod})")
            continue

        # Loop over thresholds by index
        for th_idx, th in enumerate(fiducial_data[mediator]["thresholds"]):
            max_rate = []
            min_rate = []
            max_sigma = []
            min_sigma = []

            # Loop over mass points
            for i in range(len(fiducial_data[mediator][th]["dm_mass"])):
                rate_list = []
                sigma_list = []

                for prefix in file_prefixes:
                    try:
                        # Validate that the velocity token can be parsed,
                        # including the empirical v0 -> infinity case.
                        v0_from_prefix = get_v0_from_prefix(prefix)

                        dm_mass, threshold_vals, rates, sigmas = read_data_function(prefix, data_dir)
                    except Exception as e:
                        print(f"Warning: skipping file '{prefix}' due to read error: {e}")
                        continue

                    # Ensure the mass matches the expected one at index `i`
                    if i < len(dm_mass) and dm_mass[i] == fiducial_data[mediator][th]["dm_mass"][i]:
                        rate_list.append(rates[th_idx, i])
                        sigma_list.append(sigmas[th_idx, i])

                # Record min/max if values were collected
                if rate_list:
                    max_rate.append(max(rate_list))
                    min_rate.append(min(rate_list))
                    max_sigma.append(max(sigma_list))
                    min_sigma.append(min(sigma_list))
                else:
                    max_rate.append(None)
                    min_rate.append(None)
                    max_sigma.append(None)
                    min_sigma.append(None)

            # Store in dictionaries using (mediator, mod, threshold) as the key
            group_max_rate[(mediator, mod, th)] = max_rate
            group_min_rate[(mediator, mod, th)] = min_rate
            group_max_sigma[(mediator, mod, th)] = max_sigma
            group_min_sigma[(mediator, mod, th)] = min_sigma

    return group_max_rate, group_min_rate, group_max_sigma, group_min_sigma


### Computes rate uncertainties for a given mediator across different models and thresholds ###
def compute_group_uncertainties(mediator_key, fiducial_data, group_max_rate, group_min_rate, mod):
    """
    Computes rate uncertainties for a given mediator across different models and thresholds.

    Parameters:
    - mediator_key: str
    - fiducial_data: dict
    - group_max_rate: dict
    - group_min_rate: dict
    - mod: list of halo models (e.g. ["SHM", "TSA", "EMP"])

    Returns:
    - group_uncertainties: dict with (mediator, model, threshold) as keys
    """

    def compute_uncertainty(fid_dm_mass, fid_rates, max_rate, min_rate):
        return (
            [max_rate[i] / fid_rates[i] - 1 for i in range(len(fid_dm_mass))],
            [min_rate[i] / fid_rates[i] - 1 for i in range(len(fid_dm_mass))],
            [(max_rate[i] - min_rate[i]) / fid_rates[i] for i in range(len(fid_dm_mass))]
        )

    group_uncertainties = {}

    if mediator_key not in fiducial_data:
        print(f"Error: no data available for mediator '{mediator_key}'")
        return group_uncertainties

    for th in sorted(fiducial_data[mediator_key]["thresholds"]):
        fid_dm_mass = fiducial_data[mediator_key][th]["dm_mass"]
        fid_rates = fiducial_data[mediator_key][th]["rates"]

        for m in mod:
            key = (mediator_key, m, th)

            if key not in group_max_rate or key not in group_min_rate:
                print(f"Warning: skipping missing data for key: {key}")
                continue

            max_rate = group_max_rate[key]
            min_rate = group_min_rate[key]

            unc_max, unc_min, unc = compute_uncertainty(fid_dm_mass, fid_rates, max_rate, min_rate)

            group_uncertainties[key] = {
                "uncertainty_max": unc_max,
                "uncertainty_min": unc_min,
                "uncertainty": unc
            }

    return group_uncertainties


### Computes sigma uncertainties for a given mediator across different models and thresholds ###
def compute_group_sigma_uncertainties(mediator_key, fiducial_data, group_max_sigma, group_min_sigma, mod):
    """
    Computes sigma uncertainties for a given mediator across different models and thresholds.

    Parameters:
    - mediator_key: str
    - fiducial_data: dict
    - group_max_sigma: dict
    - group_min_sigma: dict
    - mod: list of halo models (e.g. ["SHM", "TSA", "EMP"])

    Returns:
    - group_uncertainties: dict with (mediator, model, threshold) as keys
    """

    def compute_uncertainty(fid_dm_mass, fid_sigmas, max_sigma, min_sigma):
        return (
            [max_sigma[i] / fid_sigmas[i] - 1 for i in range(len(fid_dm_mass))],
            [min_sigma[i] / fid_sigmas[i] - 1 for i in range(len(fid_dm_mass))],
            [(max_sigma[i] - min_sigma[i]) / fid_sigmas[i] for i in range(len(fid_dm_mass))]
        )

    group_uncertainties = {}

    if mediator_key not in fiducial_data:
        print(f"Error: no data available for mediator '{mediator_key}'")
        return group_uncertainties

    for th in sorted(fiducial_data[mediator_key]["thresholds"]):
        fid_dm_mass = fiducial_data[mediator_key][th]["dm_mass"]
        fid_sigmas = fiducial_data[mediator_key][th]["sigmas"]

        for m in mod:
            key = (mediator_key, m, th)

            if key not in group_max_sigma or key not in group_min_sigma:
                print(f"Warning: skipping missing data for key: {key}")
                continue

            max_sigma = group_max_sigma[key]
            min_sigma = group_min_sigma[key]

            unc_max, unc_min, unc = compute_uncertainty(fid_dm_mass, fid_sigmas, max_sigma, min_sigma)

            group_uncertainties[key] = {
                "uncertainty_max": unc_max,
                "uncertainty_min": unc_min,
                "uncertainty": unc
            }

    return group_uncertainties

### Computes model uncertainties for specific halo models relative to fiducial model ###
def compute_model_uncertainties(mediator_key, fiducial_data, model_data_dict):
    """
    Compute rate (and optionally sigma) uncertainties for specific halo models
    (e.g., TSA, EMP) relative to a fiducial model stored in `fiducial_data`.

    Parameters
    ----------
    mediator_key : str
        Mediator name key used in `fiducial_data`, e.g. "light_hadrophilic".
    fiducial_data : dict
        Output of `read_fiducial_data`. Expected structure:
            fiducial_data[mediator]["thresholds"] = [...]
            fiducial_data[mediator][threshold] = {
                "dm_mass": [...],
                "rates": [...],
                "sigmas": [...]
            }
    model_data_dict : dict
        Mapping of model name → (DM_mass, thresholds, rates_2D, sigmas_2D), e.g.:
            {
              "TSA": TSA_data,  # tuple from read_data_...
              "EMP": EMP_data
            }

    Returns
    -------
    model_uncertainties : dict
        Dictionary with keys (mediator_key, model_name, threshold) and values:
            {
                "dm_mass": [...],
                "rate_uncertainty": [...],   # (rate_model / rate_fid - 1)
                "sigma_uncertainty": [...]   # (sigma_model / sigma_fid - 1)
            }
    """

    model_uncertainties = {}

    if mediator_key not in fiducial_data:
        print(f"Error: no fiducial data available for mediator '{mediator_key}'")
        return model_uncertainties

    fid_thresholds = fiducial_data[mediator_key]["thresholds"]

    for model_name, model_data in model_data_dict.items():
        DM_mass_m, thresholds_m, rates_2D_m, sigmas_2D_m = model_data

        # Convert thresholds to numpy array for easy matching
        thresholds_m = np.array(thresholds_m)

        for th in fid_thresholds:
            fid_entry = fiducial_data[mediator_key][th]
            fid_dm_mass = np.array(fid_entry["dm_mass"])
            fid_rates = np.array(fid_entry["rates"])
            fid_sigmas = np.array(fid_entry["sigmas"])

            # Find matching threshold index in this model's data
            idx_arr = np.where(thresholds_m == th)[0]
            if len(idx_arr) == 0:
                print(f"Warning: threshold {th} not found for model '{model_name}'. Skipping.")
                continue
            th_idx = int(idx_arr[0])

            model_rates_1d = np.array(rates_2D_m[th_idx])
            model_sigmas_1d = np.array(sigmas_2D_m[th_idx])

            # Basic consistency check on DM mass
            if len(model_rates_1d) != len(fid_dm_mass):
                print(
                    f"Warning: mismatch in DM mass grid for (model={model_name}, threshold={th}). "
                    f"len(model)={len(model_rates_1d)}, len(fid)={len(fid_dm_mass)}. Skipping."
                )
                continue

            # Avoid divide-by-zero issues
            with np.errstate(divide="ignore", invalid="ignore"):
                rate_unc = model_rates_1d / fid_rates - 1.0
                sigma_unc = model_sigmas_1d / fid_sigmas - 1.0

            key = (mediator_key, model_name, th)
            model_uncertainties[key] = {
                "dm_mass": fid_dm_mass.tolist(),
                "rate_uncertainty": rate_unc.tolist(),
                "sigma_uncertainty": sigma_unc.tolist(),
            }

    return model_uncertainties


"-------------------------------------------------------"
"--------------For random threshold-------------------"

def total_rate_above_threshold(diff_rate, new_threshold, threshold_run, energy_bin_width):
    """
    Integrate a stored differential-rate spectrum above a new threshold.

    ``diff_rate`` is for a fixed time, mass, and run threshold.  The three
    threshold-like arguments must use the same energy unit as the HDF5 file.
    In the paper notebooks this unit is eV, so 20 meV is passed as 0.02.
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

### Light hadrophilic mediator ###
def read_data_light_hadrophilic_from_diff(file_prefix, data_dir, new_threshold_list_meV, time_index, use_run_threshold_index):
    """
    Read rates from ``data/diff_rate/<time>/<mass>/<threshold>`` and
    integrate them above one or more effective thresholds.
    and calculate sigma from rate for Light hadrophilic mediator physics model.

    function take arguments:
        file_prefix: prefix of the file to read
        data_dir: directory where the data files are located
        new_threshold_list_meV: historical name; values are in eV
        energy_bin_width_meV: historical name; values are in eV
        time_index: time index (default 0)
        use_run_threshold_index: threshold index used in the run (default 0)

    function gives output as:
        DM_mass: DM mass in MeV
        threshold_new: new thresholds in eV
        rates_array: 2D array of shape (n_new, n_masses) with rate values
        sigmas_array: 2D array of shape (n_new, n_masses) with sigma values
    """
    filename = os.path.join(data_dir, file_prefix + '.hdf5')

    threshold_new = np.atleast_1d(new_threshold_list_meV).astype(float)

    with h5py.File(filename, 'r') as fw:
        DM_mass = fw['particle_physics/dm_properties/mass_list'][()] * 1e-6  # [MeV]
        energy_bin_width_meV = fw['numerics/energy_bin_width'][()]
        # threshold = fw['particle_physics/threshold'][()] * 1e3  # (n_thresholds,) in [meV]
        n_masses = len(DM_mass)
        # n_thresholds = len(threshold)
        n_new = len(threshold_new)

        # Run thresholds are stored in eV; use the lowest index (0) by default.
        thr_eV = np.atleast_1d(fw["particle_physics/threshold"][()]).astype(float)
        if use_run_threshold_index >= len(thr_eV):
            raise IndexError(
                f"use_run_threshold_index={use_run_threshold_index} out of range; "
                f"file has {len(thr_eV)} threshold(s)."
            )
        threshold_run_meV = thr_eV[use_run_threshold_index]

        # Initialize 2D arrays: shape (n_new, n_masses)
        rates_array = np.zeros((n_new, n_masses))
        sigmas_array = np.zeros((n_new, n_masses))

        # for t_idx, th in enumerate(threshold):
        for t_idx, new_th in enumerate(threshold_new):
            for m_idx, mass in enumerate(DM_mass):
                diff_rate = fw[f"data/diff_rate/{time_index}/{m_idx}/{use_run_threshold_index}"][()]

                rate = total_rate_above_threshold(
                    diff_rate=diff_rate,
                    new_threshold=new_th,
                    threshold_run=threshold_run_meV,
                    energy_bin_width=energy_bin_width_meV,
                )
                rates_array[t_idx, m_idx] = rate

                # rate = fw[f'data/rate/0/{m_idx}/{t_idx}'][()]
                # rates_array[t_idx, m_idx] = rate

                sigma = (
                    3 / (2.69e58) / rate
                    * (1 / np.pi)
                    * (938 * mass)**2 / (938 + mass)**2
                    * 1e12
                    * (1 / (mass * 1e6 * 220 * 3.34e-6)**4)
                    * 3.88e-10
                )
                sigmas_array[t_idx, m_idx] = sigma

    return DM_mass, threshold_new, rates_array, sigmas_array

### Heavy hadrophilic mediator ###
def read_data_heavy_hadrophilic_from_diff(file_prefix, data_dir, new_threshold_list_meV, time_index, use_run_threshold_index):
    
    filename = os.path.join(data_dir, file_prefix + '.hdf5')
    threshold_new = np.atleast_1d(new_threshold_list_meV).astype(float)

    with h5py.File(filename, 'r') as fw:
        DM_mass = fw['particle_physics/dm_properties/mass_list'][()] * 1e-6  # [GeV]
        energy_bin_width_meV = fw['numerics/energy_bin_width'][()]
        # threshold = fw['particle_physics/threshold'][()] * 1e3  # [meV]
        n_masses = len(DM_mass)
        # n_thresholds = len(threshold)
        n_new = len(threshold_new)

        # Run thresholds are stored in eV; use the lowest index (0) by default.
        thr_eV = np.atleast_1d(fw["particle_physics/threshold"][()]).astype(float)
        if use_run_threshold_index >= len(thr_eV):
            raise IndexError(
                f"use_run_threshold_index={use_run_threshold_index} out of range; "
                f"file has {len(thr_eV)} threshold(s)."
            )
        threshold_run_meV = thr_eV[use_run_threshold_index]

        # Initialize 2D arrays: shape (n_new, n_masses)
        rates_array = np.zeros((n_new, n_masses))
        sigmas_array = np.zeros((n_new, n_masses))

        # for t_idx, th in enumerate(threshold):
        for t_idx, new_th in enumerate(threshold_new):
            for m_idx, mass in enumerate(DM_mass):
                diff_rate = fw[f"data/diff_rate/{time_index}/{m_idx}/{use_run_threshold_index}"][()]

                rate = total_rate_above_threshold(
                    diff_rate=diff_rate,
                    new_threshold=new_th,
                    threshold_run=threshold_run_meV,
                    energy_bin_width=energy_bin_width_meV,
                )
                rates_array[t_idx, m_idx] = rate

                # rate = fw[f'data/rate/0/{m_idx}/{t_idx}'][()]
                # rates_array[t_idx, m_idx] = rate

                sigma = (
                    3 / (2.69e58) / rate
                    * (1 / np.pi)
                    * (938 * mass)**2 / (938 + mass)**2
                    * 1e12
                    * 3.88e-10
                )
                sigmas_array[t_idx, m_idx] = sigma

    return DM_mass, threshold_new, rates_array, sigmas_array

### Light Dark Photon ###
def read_data_light_dark_photon_from_diff(file_prefix, data_dir, new_threshold_list_meV, time_index, use_run_threshold_index):
   
    filename = os.path.join(data_dir, file_prefix + '.hdf5')
    threshold_new = np.atleast_1d(new_threshold_list_meV).astype(float)

    with h5py.File(filename, 'r') as fw:
        DM_mass = fw['particle_physics/dm_properties/mass_list'][()] * 1e-6  # [GeV]
        energy_bin_width_meV = fw['numerics/energy_bin_width'][()]
        # threshold = fw['particle_physics/threshold'][()] * 1e3  # [meV]
        n_masses = len(DM_mass)
        # n_thresholds = len(threshold)
        n_new = len(threshold_new)
        
        # Run thresholds are stored in eV; use the lowest index (0) by default.
        thr_eV = np.atleast_1d(fw["particle_physics/threshold"][()]).astype(float)
        if use_run_threshold_index >= len(thr_eV):
            raise IndexError(
                f"use_run_threshold_index={use_run_threshold_index} out of range; "
                f"file has {len(thr_eV)} threshold(s)."
            )
        threshold_run_meV = thr_eV[use_run_threshold_index]

        # Initialize 2D arrays: shape (n_new, n_masses)
        rates_array = np.zeros((n_new, n_masses))
        sigmas_array = np.zeros((n_new, n_masses))

        # for t_idx, th in enumerate(threshold):
        for t_idx, new_th in enumerate(threshold_new):
            for m_idx, mass in enumerate(DM_mass):
                diff_rate = fw[f"data/diff_rate/{time_index}/{m_idx}/{use_run_threshold_index}"][()]

                rate = total_rate_above_threshold(
                    diff_rate=diff_rate,
                    new_threshold=new_th,
                    threshold_run=threshold_run_meV,
                    energy_bin_width=energy_bin_width_meV,
                )
                rates_array[t_idx, m_idx] = rate

                # rate = fw[f'data/rate/0/{m_idx}/{t_idx}'][()]
                # rates_array[t_idx, m_idx] = rate

                sigma = (
                    3 / (2.69e58) / rate
                    * (1 / np.pi)
                    * (0.511 * mass)**2 / (0.511 + mass)**2
                    * 1e12
                    * (137 / (0.511e6))**4
                    * 3.88e-10
                )
                sigmas_array[t_idx, m_idx] = sigma

    return DM_mass, threshold_new, rates_array, sigmas_array

### Read fiducial rates and sigmas for given mediators and fiducial parameters ###
def read_fiducial_data_from_diff(data_dir, Target, Mediators, Numerics, fid_mod, fid_velocity, new_threshold_list_meV, time_index=0, use_run_threshold_index=0):
    """
    Read fiducial rates and sigmas for given mediators and fiducial parameters.

    Parameters:
        data_dir (str): Path to the directory containing data files.
        target (str): Target material (e.g., "Al2O3").
        mediators (list): List of mediator types.
        numerics (str): Numerics tag (e.g., "standard").
        fid_mod (str): Fiducial halo model (e.g., "SHM").
        fid_velocity (tuple): Fiducial velocity parameters (v0, ve, vesc) as strings.
        new_threshold_list_meV (list): Historical name; threshold values in eV.
        time_index (int, optional): Time index for reading data. Default is 0.
        use_run_threshold_index (int, optional): Index of run threshold to use. Default is 0.

    Returns:
        dict: Nested dictionary structure of fiducial data:
              fiducial_data[mediator][threshold] = {
                  "dm_mass": [...],
                  "rates": [...],
                  "sigmas": [...]
              }
    """

    fiducial_data = {}

    # Ensure we consistently have strings for building the prefix
    v0_str, ve_str, vesc_str = map(str, fid_velocity)
    # And an int v0 for passing into the read functions
    v0_fid = int(v0_str)

    for med in Mediators:
        # # Construct file prefix
        # fid_file_prefix = f"{Target}_{med}_{Numerics}_{fid_mod}_{'_'.join(fid_velocity)}"
        # Construct file prefix: Target_mediator_Numerics_mod_v0_ve_vesc
        fid_file_prefix = f"{Target}_{med}_{Numerics}_{fid_mod}_{v0_str}_{ve_str}_{vesc_str}"

        # Dynamically get the appropriate read function
        read_func = globals().get(f"read_data_{med}_from_diff")
        if read_func is None:
            print(f"Warning: read function for {med} not found. Skipping.")
            continue

        # Read data
        dm_mass, thresholds, rates_2D, sigmas_2D = read_func(fid_file_prefix, data_dir, new_threshold_list_meV, time_index, use_run_threshold_index)

        # # Store in dictionary
        # for idx, th in enumerate(thresholds):
        #     if med not in fiducial_data:
        #         fiducial_data[med] = {}
        #     fiducial_data[med][th] = {
        #         "dm_mass": dm_mass,
        #         "rates": rates_2D[idx],
        #         "sigmas": sigmas_2D[idx]
        #     }
        for idx, th in enumerate(thresholds):
            # Initialize mediator entry if not already present
            if med not in fiducial_data:
                fiducial_data[med] = {
                    "thresholds": thresholds  # store thresholds once
                }

            # Store threshold-specific data
            fiducial_data[med][th] = {
                "dm_mass": dm_mass,
                "rates": rates_2D[idx],
                "sigmas": sigmas_2D[idx]
            }

    return fiducial_data


### Computes max/min rate and sigma values for each (mediator, model, threshold) group ###
def compute_group_extremes_from_diff(grouped_prefixes, data_dir, fiducial_data, new_threshold_list_meV, time_index=0, use_run_threshold_index=0):
    """
    Computes max/min rate and sigma values for each (mediator, model, threshold) group
    using the diff_rate method.

    Parameters:
        grouped_prefixes (dict): Mapping from (mediator, mod) → list of file prefixes.
        data_dir (str): Directory where HDF5 files are located.
        fiducial_data (dict): Fiducial data for comparison.
        new_threshold_list_meV (list): Historical name; threshold values in eV.

    Returns:
        Tuple of four dictionaries: max_rate, min_rate, max_sigma, min_sigma
    """
    # Initialize dictionaries to store results
    group_max_rate = {}
    group_min_rate = {}
    group_max_sigma = {}
    group_min_sigma = {}

    # Loop over each (mediator, mod) group
    for (mediator, mod), file_prefixes in grouped_prefixes.items():
        # Dynamically get the appropriate function for reading data
        read_data_function = globals().get(f"read_data_{mediator}_from_diff")
        
        if read_data_function is None:
            print(f"Error: Function read_data_{mediator} not found. Skipping group ({mediator}, {mod})")
            continue

        # Loop over thresholds by index
        for th_idx, th in enumerate(fiducial_data[mediator]["thresholds"]):
            max_rate = []
            min_rate = []
            max_sigma = []
            min_sigma = []

            # Loop over mass points
            for i in range(len(fiducial_data[mediator][th]["dm_mass"])):
                rate_list = []
                sigma_list = []

                for prefix in file_prefixes:
                    try:
                        # Validate the velocity token, including the empirical
                        # v0 -> infinity case used in rms matching.
                        v0_from_prefix = get_v0_from_prefix(prefix)

                        dm_mass, threshold_vals, rates, sigmas = read_data_function(prefix, data_dir, new_threshold_list_meV, time_index, use_run_threshold_index)
                    except Exception as e:
                        print(f"Warning: skipping file '{prefix}' due to read error: {e}")
                        continue

                    # Ensure the mass matches the expected one at index `i`
                    if i < len(dm_mass) and dm_mass[i] == fiducial_data[mediator][th]["dm_mass"][i]:
                        rate_list.append(rates[th_idx, i])
                        sigma_list.append(sigmas[th_idx, i])

                # Record min/max if values were collected
                if rate_list:
                    max_rate.append(max(rate_list))
                    min_rate.append(min(rate_list))
                    max_sigma.append(max(sigma_list))
                    min_sigma.append(min(sigma_list))
                else:
                    max_rate.append(None)
                    min_rate.append(None)
                    max_sigma.append(None)
                    min_sigma.append(None)

            # Store in dictionaries using (mediator, mod, threshold) as the key
            group_max_rate[(mediator, mod, th)] = max_rate
            group_min_rate[(mediator, mod, th)] = min_rate
            group_max_sigma[(mediator, mod, th)] = max_sigma
            group_min_sigma[(mediator, mod, th)] = min_sigma

    return group_max_rate, group_min_rate, group_max_sigma, group_min_sigma
