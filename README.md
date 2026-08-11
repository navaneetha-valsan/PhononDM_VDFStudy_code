# Reader Notes

The repository contains the code used to obtain the results presented in the paper (`arXiv:2606.04091`).

The release contains two distinct layers:

- the modified PhonoDark calculation code, which generates HDF5 files (see section `PhonoDark Modification Notes`);
- the post-processing notebooks in this repository, which read those HDF5 files and generate the figures and tables used in `arXiv:2606.04091` (see section `Paper Results Reproduction Notes`).


# PhonoDark Modification Notes

This note summarizes the relevant changes introduced in `PhonoDark_modified` relative to the original [`PhonoDark`](https://github.com/tanner-trickle/PhonoDark) package. 

Any comments on files that does not affect the paper reproduction workflow are intentionally omitted in this note.

## Purpose of the modifications

The modified code extends the original PhonoDark workflow from a single fixed Standard Halo Model velocity distribution to scan over range of velocity parameters for multiple velocity distribution functions. The main goal is to support:

- in addition to the Maxwell-Boltzmann velocity distribution corresponding to Standard Halo Model (`SHM`),
  - a Tsallis velocity distribution (`TSA`),
  - an empirical velocity distribution (`EMP`) from  Mao et al. 2013,
- scan over a range of each velocity-parameters namely `V0`, `VE`, and `VESC`,
- record the astrophysical parameters (`VDF`, `V0`, `VE`, and `VESC`) used in each run in corresponding output file.

## `calculator.py`

The original driver performed one calculation over dark matter mass and time using velocity constants fixed in `src/constants.py`. The modified driver wraps the original calculation workflow in an astrophysical parameter scan.

Modified:

- Added `scipy.special` and `datetime` imports.
- Removed top-level imports of `src.constants`, `src.parallel_util`, `src.mesh`, `src.phonopy_funcs`, `src.physics`, and `src.hdf5_output`; these are now imported after the velocity constants are set for each scan point.
- Creates one timestamped output directory per submitted run:
  `data/<material>_<physics-model>_<timestamp>/`.
- Broadcasts the timestamped output folder from the root MPI process to all MPI processes so every process writes into the same run directory.
- Adds common scan lists:
  - `VDF_models = ["SHM", "TSA", "EMP"]`.
    (Velocities in km/s conservative or aggressive range used in `arXiv:2606.04091`)
  - `V0_values = [280,220,200]` or `[239.5,238,236.5]`.
  - `VESC_values = [600, 544, 450]` or `[552, 528, 503]`.
  - `VE_values = [217, 232, 247]` or `[235.6, 250.6, 265.6]`.
- Adds `model_configs` instead of `V0_values`, a compact configuration dictionary for V0 scans in rms-matching prescription:
- Writes one HDF5 file per velocity combination. The filename now includes the velocity-distribution model and velocity parameters:
  `<material>_<physics-model>_<numerics>_<VDF>_<V0>_<VE>_<VESC>.hdf5`.

Why this was done:

- Adding scan lists avoids manually editing constants or velocity distribution funstion (VDF) for every run.
- The `VESC_to_V0_map` keeps the Tsallis and empirical V0 parameter combinations tied to the escape velocity they belong to.
- Timestamped folders and velocity-tagged filenames prevent output files from overwriting each other during large scans.

## `src/constants.py`

The original constants file hard-coded one value for velocity parameters:

- `V0 = 230 km/s`,
- `VE = 240 km/s`,
- `VESC = 600 km/s`,
- `N0`, `C1`, and `C2` computed immediately from those values.

Modified:

- `V0`, `VE`, `VESC`, `VDF`, `N0`, `C1`, and `C2` are initialized to `None`.
- `N0`, `C1`, and `C2` values are calculated dynamically inside `calculator.py` for every velocity scan point.

Why this was done:

- The velocity parameters cannot remain fixed global constants when scanning over multiple halo models.
- Recomputing the normalization constants per scan point ensures each distribution is evaluated with the correct velocity every scan point.

## `src/vel_g_function_integrals.py`

The original file only implemented the SHM velocity integral through closed-form `g0_func` and optimized `g0_func_opt`.

Modified:

- Added `from scipy.integrate import quad`.
- Added an explicit quadrature-based SHM implementation in addition to original SHM functions:
  - `integrandSHM(v, V0)`,
  - `KSHM(v, V0)`,
  - `g0_func_SHM(...)`.
- Added Tsallis velocity-distribution support:
  - `integrandTSA(v, VESC, V0) = v * (1 - v^2 / VESC^2)^(VESC^2 / V0^2)`,
  - `KTSA(v, VESC, V0)` for normalization,
  - `g0_func_TSA(...)` for the velocity integral above `v_minus`.
- Added empirical Mao et al.-style velocity-distribution support:
  - fixed `p = 3/2`,
  - `integrandEMP(v, p, V0, VESC) = v * exp(-v / V0) * (VESC^2 - v^2)^p`,
  - `KEMP(v, p, V0, VESC)` for normalization,
  - `g0_func_EMP(...)` for the velocity integral above `v_minus`.
- Both non-standard distributions compute their own normalization using `quad(..., 0, const.VESC, ...)`.

Why this was done:

- The original closed-form SHM expression cannot be reused directly for Tsallis or empirical velocity distributions.
- Numerical integration gives one common structure for evaluating all three distributions over the allowed speed range.
- Each `g0_func_*` returns zero when `v_minus > VESC`, preserving the physical velocity cutoff.

## `src/physics.py`

The modified physics code wires the new velocity distributions into the spin-independent rate calculation.

Modified:

- Inside `calc_diff_rates_SI`, the velocity integral is selected from `const.VDF`:
  - `SHM -> vel_g_function_integrals.g0_func_SHM`,
  - `TSA -> vel_g_function_integrals.g0_func_TSA`,
  - any other value -> `vel_g_function_integrals.g0_func_EMP`.
- The selected `g0` value is then used in the same rate expression as before.

Why this was done:

- The rate calculation must use the correct velocity integral for the selected distribution model.

Note: The VDF selection was added to `calc_diff_rates_SI`. Other rate calculation definitions `calc_diff_rates_general`, `calc_diff_rates_SI_q`, `calc_diff_rates_general_q` still call the original optimized SHM `g0_func_opt` path. 

## `src/hdf5_output.py`

The output writer now records the astrophysical parameters.

Modified:

- Imports `src.constants as const`.
- Adds an `astrophysical` group to every HDF5 output file.
- Stores:
  - `astrophysical/VDF`,
  - `astrophysical/V0`,
  - `astrophysical/VE`,
  - `astrophysical/VESC`,
  - `astrophysical/RHO_DM`.

Why this was done:

- Each output file must carry enough metadata to identify which velocity-distribution model and velocity parameters produced it.

## Numerics input choices

The default numerical input for the mesh used for all the calculations in `inputs/numerics/standard.py`:

- `n_a = 50`,
- `n_b = 25`,
- `n_c = 25`.

Other numerical inputs where chosen according to the mediator choice:
- Heavy hadrophilic scalar mediator
  - `power_a = 1`,
  - `special_mesh = False`.
- Light hadrophilic scalar and Light dark photon mediator
  - `power_a = 2`,
  - `special_mesh = True`.

## Physics model input choices

The following settings specify the physics inputs used to generate the data presented in the paper. These are run-configuration choices and do not modify the core rate calculation.

- Physics-model input files used: `inputs/physics_model/heavy_hadrophilic.py` or `light_hadrophilic.py` or `light_dark_photon.py`
-`log_start_mass = 3`, start mass 10^-3 MeV
-`log_end_mass = 10`, end mass 10^4 MeV (`log_end_mass = 6`, end mass 1 MeV for fmod plots)
-`n_masses = 100`, 100 mass points (16 mass points per decade for fmod plots)
-`'threshold' : [1*10**(-3)]`, 1 meV
-`'times'     : [0.]`, 0 for reach plots, `[0., 1., ..., 23.]` for daily modulation plots

## Threshold handling for the paper plots

The threshold-related code was added in two stages, and the paper figures use the second workflow.

Initial implementation:

- The driver was modified to treat `physics_parameters["threshold"]` as a possible scan axis.
- The MPI job indexing and HDF5 output structure were extended from `time/mass` to `time/mass/threshold`, allowing multiple thresholds to be included in a single run.

Workflow used for the paper:

- Most of the data used for the paper were generated with a 1 meV threshold (`threshold = 0.001` eV).
- The 20 meV results were obtained by post-processing the differential rates from the 1 meV run.
- The plotting scripts sum the differential-rate bins above the desired threshold using `total_rate_above_threshold(...)` in `script_helpers/script_reach.py` and `script_helpers/script_fmod.py`.

Thus, the threshold-indexed HDF5 structure is retained in the code, while the paper's 20 meV results are obtained by post-processing the differential-rate spectra from the 1 meV runs.


<!-- codex resume 019fde7a-566e-7ef1-9330-c32a539b5ea4 -->
# Paper Results Reproduction Notes

This note summarizes the scripts and notebooks used to reproduce the plots and tables presented in `arXiv:2606.04091` from the HDF5 files generated by the PhonoDark_modified code.

Run the notebooks from the `PhononDM_VDFStudy_code` directory so imports such as `script_helpers.script_reach` and relative files such as `output_contour.xlsx` resolve correctly. The HDF5 data used to reproduce the paper results are provided separately in the PhononDM_VDFStudy_DATA Zenodo repository `DOI:10.5281/zenodo.21892348`. For notebooks that read HDF5 data, set `DATA_ROOT` to the root of the downloaded PhononDM_VDFStudy_DATA repository:

```python
DATA_ROOT = "/path/to/PhononDM_VDFStudy_DATA"
```

The notebooks then append the required `Manuscript_data/...` paths automatically.

Alternatively, the HDF5 files can be generated using the modified PhonoDark code described above.

```python
DATA_ROOT = "/path/to/locally/generated/data"
```

## HDF5 Data Layout

The plotting notebooks expect HDF5 files below:

```text
<DATA_ROOT>/Manuscript_data/
```

Projected-reach data are organized as:

- `<Target>_projected_reach/files_QQQ/`: standard-prescription conservative velocity scan.
- `<Target>_projected_reach/files_vrms_QQQ/`: rms-matching prescription, conservative velocity range.
- `<Target>_projected_reach/files_vrms_aggr/`: rms-matching prescription, aggressive velocity range.

Daily-modulation data are organized as:

- `<Target>_daily_modulation/files/`: standard-prescription daily-modulation data for the main mass range.
- `<Target>_daily_modulation/files_QQQ_10-3/`: additional standard-prescription daily-modulation data used by the `f_mod` notebook to stitch in the low-mass range.
- `<Target>_daily_modulation/files_vrms_QQQ/`: rms-matching daily-modulation data used for Tsallis and empirical curves.

The filename convention is:

```text
<Target>_<mediator>_<numerics>_<VDF>_<V0>_<VE>_<VESC>.hdf5
```

where:

- `Target` is one of `Al2O3`, `CaWO4`, `SiO2`, or `GaAs` for projected-reach plots. Daily-modulation notebooks use `Al2O3`, `CaWO4`, and `SiO2`.
- `mediator` is one of `heavy_hadrophilic`, `light_hadrophilic`, or `light_dark_photon`.
- `numerics` is `standard` for the released paper data.
- `VDF` is one of `SHM`, `TSA`, or `EMP`.
- `V0`, `VE`, and `VESC` are velocity parameters in km/s. Empirical rms-matching files may use `inf` for `V0` when no finite empirical rms-matched solution exists.

## Helper Modules

The reusable Python helpers live in `script_helpers/`.

- `script_helpers/script_reach.py` reads HDF5 files, builds velocity-scan filename prefixes, integrates `diff_rate` above requested thresholds, converts rates to projected cross-section reaches, and computes min/max uncertainty envelopes and model-to-fiducial relative differences. It provides mediator-specific readers for `light_hadrophilic`,
`heavy_hadrophilic`, and `light_dark_photon`.
- `script_helpers/script_fmod.py` reads HDF5 files and computes either `f_mod = max(|R - <R>|) / <R>` or normalized daily rates `R / <R>`. The `_from_diff` functions impose a requested analysis threshold by integrating `diff_rate` above that threshold before computing modulation.
- `script_helpers/script_VDF.py` evaluates normalized SHM, Tsallis, and empirical speed distributions, builds velocity-parameter bands, computes rms speeds, solves for rms-matched `V0` values for Tsallis and empirical distributions, and handles the empirical `V0 -> infinity` case.

Note: Threshold values passed to the plotting helper functions are in eV, even where legacy argument names end in `_meV`.

## Jupiter Notebooks

- `script_rms_contour_plot.ipynb`: reproduces Fig. F01. It computes the rms-matching contour grid in Python using `script_helpers/script_VDF.py` and reads `output_contour.xlsx` to overlay the empirical no-finite-solution region flagged by the Mathematica check.
- `script_f(v)_plot.ipynb`: reproduces Fig. F02. It does not read HDF5 data; it uses `script_helpers/script_VDF.py` to compare SHM, Tsallis, and empirical speed distributions in the standard and rms-matching prescriptions.
- `script_reach_plot_rms_prescription.ipynb`: reproduces Figs. F03-F05 and F12-F14. Use projected-reach data from `files_vrms_QQQ` for the conservative rms-matching range or `files_vrms_aggr` for the aggressive rms-matching range.
- `script_reach_plot_std_prescription.ipynb`: reproduces Figs. F09-F11. Use projected-reach data from `files_QQQ` for the conservative range in standard prescription.
- `script_fmod_plot.ipynb`: reproduces the upper panels of Figs. F06-F08. It reads daily-modulation `files` and `files_QQQ_10-3` for the standard SHM prescription and `files_vrms_QQQ` for rms-matched Tsallis and empirical curves.
- `script_daily_modulation_plot.ipynb`: reproduces the lower panels of Figs. F06-F08. It reads daily-modulation `files` for the standard prescription and `files_vrms_QQQ` for rms-matched Tsallis and empirical curves.
- `script_rel_diff_table.ipynb`: reproduces Tables T2-T5. It reads projected reach folders, with the `data_dir` cell selecting `files_QQQ`, `files_vrms_QQQ`, or `files_vrms_aggr` depending on the prescription and velocity range being tabulated. The commented velocity blocks select whether the relative-difference scan varies `V0`, `VE`, or `VESC`.

## Mathematica Notebooks

Two Mathematica notebooks support the rms-matching prescription:

- `rms_matched_v0.nb` derives the rms speed for a truncated Maxwell-Boltzmann distribution and solves for the Tsallis and empirical parameters, `v0_TSA` and `v0_EMP`, that reproduce the same rms speed.
- `rms_matching_emp_check.nb` checks whether the empirical distribution has a finite `v0_EMP` solution when rms-matched to the Maxwell-Boltzmann target. It compares `vrms(MB)` with the empirical distribution's `V0 -> infinity` upper limit. Points with `MB > max? = True` are treated as empirical `V0 = inf` in the contour plot. The output workbook from this check is `output_contour.xlsx`, which is read by `script_rms_contour_plot.ipynb`.

### Code-Assistance Note

Some reade notes and code cleanup were assisted by GPT/Codex.
