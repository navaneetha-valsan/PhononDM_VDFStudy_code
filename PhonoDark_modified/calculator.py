"""
    Compute the DM - phonon scattering rate for a general effective operator.

    Necessary input files:
        - DFT input files:
            - POSCAR - equilibrium positions of atoms
            - FORCE_SETS - second order force constants
            (optional) BORN - Born effective charges

        - inputs/
            - material/
                - (material name)/ 
                    (material_info).py - set the material parameters such as nucleon numbers and spins
            - numerics/
                (numerics_info).py - list of numerics parameters
            - physics_model/
                (physics_model_info).py - parameters defining the scattering potential
                    

    Created by : Tanner Trickle, Zhengkang Zhang
"""

from mpi4py import MPI
import numpy as np
from scipy import special
#from time import time
import datetime

import phonopy
import os
import sys
import optparse

#####

version = "1.1.0"

# initializing MPI
comm = MPI.COMM_WORLD

# total number of processors
n_proc = comm.Get_size()

# processor if
proc_id = comm.Get_rank()

# ID of root process
root_process = 0

if proc_id == root_process:

    print('\n--- Dark Matter - Phonon Scattering Rate Calculator ---\n')
    print('  version: '+version+'\n')
    print('  Running on '+str(n_proc)+' processors\n')
    print('------\n')

# Parse the input parameters ###
cwd = os.getcwd()

parser = optparse.OptionParser()
parser.add_option('-m', action="store", default="", 
        help="Material info file. Contains the crystal lattice degrees of freedom for a given material.")
parser.add_option('-p', action="store", default="", 
        help="Physics model input. Contains the coupling coefficients and defines which operators enter the scattering potential.")
parser.add_option('-n', action="store", default="", 
        help="Numerics input. Sets the parameters used for the integration over momentum space and the input and output files.")

options_in, args = parser.parse_args()

options = vars(options_in)

def import_file(full_name, path):
    """
        Import a python module from a path. 3.4+ only.
        Does not call sys.modules[full_name] = path
    """
    from importlib import util

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

if options['m'] != '' and options['p'] != '' and options['n'] != '':

    # import modules
    material_input = options['m']
    physics_model_input = options['p']
    numerics_input = options['n']

    mat_input_mod_name = os.path.splitext(os.path.basename(material_input))[0] 
    phys_input_mod_name = os.path.splitext(os.path.basename(physics_model_input))[0] 
    num_input_mod_name = os.path.splitext(os.path.basename(numerics_input))[0] 

    mat_mod = import_file(mat_input_mod_name, os.path.join(cwd, material_input))
    phys_mod = import_file(phys_input_mod_name, os.path.join(cwd, physics_model_input))
    num_mod = import_file(num_input_mod_name, os.path.join(cwd, numerics_input))

    # Base output directory
    base_output_folder = num_mod.io_parameters['output_folder']

    # Only the root process generates the timestamp and creates the folder
    if proc_id == root_process:
        # Create a timestamp-based folder (e.g., output_2025-02-17_10-45-30)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = os.path.join(base_output_folder, f"{mat_input_mod_name}_{phys_input_mod_name}_{timestamp}")
        # Create the new folder
        os.makedirs(output_folder, exist_ok=True)
    else:
        timestamp = None
        output_folder = None
    
    # Broadcast the timestamp and output folder to all processes
    timestamp = comm.bcast(timestamp, root=root_process)
    output_folder = comm.bcast(output_folder, root=root_process)

    model_configs = {
        "SHM": {
            "V0_values": [239.5, 238, 236.5],
            "VESC_to_V0_map": None  # use all V0 for all VESC
        },
        "TSA": {
            "V0_values": [326.266, 323.248, 329.29, 318.872, 315.912, 321.84, 312.209, 309.313, 315.114],
            "VESC_to_V0_map": {
                503: [326.266, 323.248, 329.29],
                528: [318.872, 315.912, 321.84],
                552: [312.209, 309.313, 315.114],
            }
        },
        "EMP": {
            "V0_values": [298.812, 286.676, 311.876, 235.975, 228.475, 243.902, 200.052, 194.746, 205.602],
            "VESC_to_V0_map": {
                503: [298.812, 286.676, 311.876],
                528: [235.975, 228.475, 243.902],
                552: [200.052, 194.746, 205.602],
            }
        },
    }

    # VE_values  = [252, 250.6, 249.2]
    VE_values  = [217, 232, 247, 235.6, 250.6, 265.6]
    VESC_values = [552, 528, 503]
    VDF_models  = ["SHM", "TSA", "EMP"]

    # ### Looping of Astrophysical parameters
    # VDF_models = ["SHM","TSA","EMP"]
    # # # Velocity values in km/s 
    # V0_values = [239.5,238,236.5]
    # VE_values = [252,250.6,249.2]
    # VESC_values = [552,528,503]

    # VDF_models = ["TSA"]
    # V0_values = [326.266, 323.248, 329.29, 318.872, 315.912, 321.84, 312.209, 309.313, 315.114]

    # # Map specific VESC values to subsets of V0_values for TSA
    # VESC_to_V0_map = {
    #     503: [326.266, 323.248, 329.29],       # only use these V0 for VESC=503
    #     528: [318.872, 315.912, 321.84],       # only use this V0 for VESC=528
    #     552: [312.209, 309.313, 315.114],       # only use this V0 for VESC=552
    # }

    # VDF_models = ["EMP"]
    # V0_values = [298.812, 286.676, 311.876, 235.975, 228.475, 243.902, 200.052, 194.746, 205.602]

    # # Map specific VESC values to subsets of V0_values for EMP
    # VESC_to_V0_map = {
    #     503: [298.812, 286.676, 311.876],       # only use these V0 for VESC=450
    #     528: [235.975, 228.475, 243.902],       # only use this V0 for VESC=544
    #     552: [200.052, 194.746, 205.602],       # only use this V0 for VESC=600
    # }

    # ### if what to start loop from in between
    # start_VDF = "EMP"
    # start_V0 = 238
    # start_VE = 250.6
    # start_VESC = 528
    # start_found = False
    
    for VDF in VDF_models:
        config = model_configs[VDF]
        VESC_to_V0_map = config["VESC_to_V0_map"]
        for VESC in VESC_values:
            # Use mapped V0 list if available, else use full list
            if VESC_to_V0_map is not None:
                V0_list = VESC_to_V0_map.get(VESC, [])
            else:
                V0_list = config["V0_values"]
            # # choose which V0 list to use depending on VESC
            # V0_list = VESC_to_V0_map.get(VESC, V0_values)
            
            for V0 in V0_list:
            # for V0 in V0_values:
                for VE in VE_values:
                    ## Only start once the target combination is reached
                    # if not start_found:
                    #     if (VDF, V0, VE, VESC) == (start_VDF, start_V0, start_VE, start_VESC):
                    #         start_found = True
                    #     else:
                    #         continue  # skip until the starting point is found

                    if proc_id == root_process:
                        print(f"Calculating for VDF = {VDF}, V0 = {V0}, VE = {VE}, VESC = {VESC}.\n\n------\n")
                        # print(f"Calculating for VDF = {VDF}, VE = {VE}, VESC = {VESC}.\n\n------\n")
                    import src.constants as const
                    const.VDF = VDF #["SHM", "TSA", "EMP"]
                    const.V0 = V0*const.kmet_per_sec_to_none #[220, 200, 280]   # Example values in km/s
                    const.VE = VE*const.kmet_per_sec_to_none #[232, 217, 247]
                    const.VESC = VESC*const.kmet_per_sec_to_none #[544, 450, 600]
                    PI = np.pi
                    const.N0 = PI**(3/2)*const.V0**2*(const.V0*special.erf(const.VESC/const.V0) - (2/np.sqrt(PI))*const.VESC*np.exp(-const.VESC**2/const.V0**2))
                    const.C1 = PI*const.V0**2/const.N0
                    const.C2 = np.exp(-(const.VESC/const.V0)**2)

                    import src.parallel_util as put 
                    import src.mesh as mesh
                    import src.phonopy_funcs as phonopy_funcs
                    import src.physics as physics
                    import src.hdf5_output as hdf5_output

                    if proc_id == root_process:

                        print('Inputs :\n')
                        print('  Material :\n')
                        print('    Name : '+mat_mod.material+'\n')
                        print('    DFT supercell grid size : '+str(mat_mod.mat_properties_dict['supercell_dim'])+'\n')
                        print('    N : '+str(mat_mod.mat_properties_dict['N_list'])+'\n')
                        print('    L.S : '+str(mat_mod.mat_properties_dict['L_S_list'])+'\n')
                        print('    S : '+str(mat_mod.mat_properties_dict['S_list'])+'\n')
                        print('    L : '+str(mat_mod.mat_properties_dict['L_list'])+'\n')
                        print('    L.S_2 : '+str(mat_mod.mat_properties_dict['L_tens_S_list'])+'\n')
                        print('  Physics Model :\n')
                        print('    DM spin : '+str(phys_mod.dm_properties_dict['spin'])+'\n')
                        print('    DM mass : '+str(phys_mod.dm_properties_dict['mass_list'])+'\n')
                        print('    d log V / d log q : '+str(phys_mod.physics_parameters['power_V'])+'\n')
                        print('    - d log F_med / d log q : '+str(phys_mod.physics_parameters['Fmed_power'])+'\n')
                        print('    Time of day : '+str(phys_mod.physics_parameters['times'])+'\n')
                        print('    Threshold : '+str(phys_mod.physics_parameters['threshold'])+' eV\n')
                        print('    c coefficients : '+str(phys_mod.c_dict)+'\n')
                        print('  Numerics :\n')
                        print('    Energy bin width : '+str(num_mod.numerics_parameters['energy_bin_width'])+' eV\n')
                        print('    N_a : '+str(num_mod.numerics_parameters['n_a'])+'\n')
                        print('    N_b : '+str(num_mod.numerics_parameters['n_b'])+'\n')
                        print('    N_c : '+str(num_mod.numerics_parameters['n_c'])+'\n')
                        print('  Astrophysical parameters :\n')
                        print('    VDF : '+str(const.VDF)+' Model\n')
                        print('    V0 : '+str(const.V0/const.kmet_per_sec_to_none)+' Km/s\n')
                        print('    VE : '+str(const.VE/const.kmet_per_sec_to_none)+' Km/s\n')
                        print('    VESC : '+str(const.VESC/const.kmet_per_sec_to_none)+' Km/s\n')
                        print('    RHO_DM : '+str(const.RHO_DM/(const.GeV_To_eV*const.invcmet_To_eV**3))+' GeV/cm3\n')
                        if num_mod.numerics_parameters['special_mesh']:
                            print('    Special mesh : True\n')
                        else:
                            print('    power_a : '+str(num_mod.numerics_parameters['power_a'])+'\n')
                            print('    power_b : '+str(num_mod.numerics_parameters['power_b'])+'\n')
                            print('    power_c : '+str(num_mod.numerics_parameters['power_c'])+'\n')
                        print('    q cut : '+str(num_mod.numerics_parameters['q_cut'])+'\n')
                        print('    N_DW_x : '+str(num_mod.numerics_parameters['n_DW_x'])+'\n')
                        print('    N_DW_y : '+str(num_mod.numerics_parameters['n_DW_y'])+'\n')
                        print('    N_DW_z : '+str(num_mod.numerics_parameters['n_DW_z'])+'\n')
                        print('------\n')

                    job_list      = None
                    job_list_recv = None

                    if proc_id == root_process:

                        print('Configuring calculation ...\n')
                        
                        # number of jobs to do
                        num_masses    = len(phys_mod.dm_properties_dict['mass_list'])
                        num_times     = len(phys_mod.physics_parameters['times'])
                        num_threshold     = len(phys_mod.physics_parameters['threshold'])

                        num_jobs = num_masses*num_times*num_threshold
                        print('  Total number of jobs : '+str(num_jobs))
                        print()

                        total_job_list = []

                        for m in range(num_masses):
                            for w in range(num_threshold):
                                for t in range(num_times):

                                    total_job_list.append([m, w, t])

                        job_list = put.generate_job_list(n_proc, np.array(total_job_list))

                    # scatter the job list
                    job_list_recv = comm.scatter(job_list, root=root_process)
                    
                    diff_rate_list   = []
                    binned_rate_list = []
                    total_rate_list  = []

                    if proc_id == root_process:

                        print('Done configuring calculation\n\n------\n')
                        print('Loading DFT files ...\n')
                            
                    material = mat_mod.material

                    # load phonon file
                    poscar_path = os.path.join(
                            os.path.split(material_input)[0], 'POSCAR'
                            )
                    force_sets_path = os.path.join(
                            os.path.split(material_input)[0], 'FORCE_SETS'
                            )
                    born_path = os.path.join(
                            os.path.split(material_input)[0], 'BORN'
                            )

                    # check if the born file exists
                    if os.path.exists(born_path):
                        born_exists = True
                    else:
                        if proc_id == root_process: 
                            print('  There is no BORN file for '+material+'. PHONOPY calculations will process with .NAC. = FALSE\n')
                        born_exists = False

                    if born_exists: 
                        phonon_file = phonopy.load(
                                        supercell_matrix    = mat_mod.mat_properties_dict['supercell_dim'],
                                        primitive_matrix    = 'auto',
                                        unitcell_filename   = poscar_path,
                                        force_sets_filename = force_sets_path,
                                        is_nac              = True,
                                        born_filename       = born_path
                                    )
                    else:
                        phonon_file = phonopy.load(
                                            supercell_matrix    = mat_mod.mat_properties_dict['supercell_dim'],
                                            primitive_matrix    = 'auto',
                                            unitcell_filename   = poscar_path,
                                            force_sets_filename = force_sets_path
                                        )
                    if proc_id == root_process:
                        print('\nDone loading DFT files\n\n------\n')
                        print('Starting rate computation ...\n')

                    # run for the given jobs
                    first_job = True
                    for job in range(len(job_list_recv)):


                        if (job_list_recv[job, 0] != -1 and job_list_recv[job, 1] != -1 and job_list_recv[job, 2] != -1):

                            job_id = job_list_recv[job]

                            mass = phys_mod.dm_properties_dict['mass_list'][int(job_id[0])]
                            threshold = phys_mod.physics_parameters['threshold'][int(job_id[1])]
                            time = phys_mod.physics_parameters['times'][int(job_id[2])]


                            if first_job and proc_id == root_process:
                                print('  Loading data to PHONOPY ...\n')

                            phonopy_params = phonopy_funcs.get_phonon_file_data(phonon_file, born_exists)

                            if first_job and proc_id == root_process:

                                print('    Number of atoms : '+str(phonopy_params['num_atoms'])+'\n')
                                print('    Number of modes : '+str(phonopy_params['num_modes'])+'\n')
                                print('    Atom masses : '+str(phonopy_params['atom_masses'])+'\n')
                            
                            if born_exists and proc_id == root_process and first_job:
                                
                                print('    dielectric : \n')
                                print(phonopy_params['dielectric'])
                                print()
                            
                            if not phys_mod.include_screen:
                                if proc_id == root_process:
                                    print('  Include screen is FALSE. Setting the dielectric to the identity.\n')
                                    print()

                                phonopy_params['dielectric'] = np.identity(3)

                            if first_job and proc_id == root_process:
                                print('  Done loading data to PHONOPY\n')

                            # generate q mesh
                            vE_vec = physics.create_vE_vec(time)

                            delta = 2*phys_mod.physics_parameters['power_V'] - 2*phys_mod.physics_parameters['Fmed_power']

                            [q_XYZ_list, jacob_list] = mesh.create_q_mesh(mass, 
                                                        threshold, 
                                                        vE_vec, 
                                                        num_mod.numerics_parameters,
                                                        phonon_file,
                                                        phonopy_params['atom_masses'],
                                                        delta)

                            # Beta testing a uniform q mesh for different calculations...

                            # [q_XYZ_list, jacob_list] = mesh.create_q_mesh_uniform(mass, 
                            #                                threshold, 
                            #                                vE_vec, 
                            #                                num_mod.numerics_parameters,
                            #                                phonon_file,
                            #                                phonopy_params['atom_masses'],
                            #                                delta, 
                            #                                q_red_to_XYZ = phonopy_params['recip_red_to_XYZ'],
                            #                                mesh = [20, 20, 20]
                            #                                )

                            k_red_list = mesh.generate_k_red_mesh_from_q_XYZ_mesh(q_XYZ_list, phonopy_params['recip_red_to_XYZ'])
                            G_XYZ_list = mesh.get_G_XYZ_list_from_q_XYZ_list(q_XYZ_list, phonopy_params['recip_red_to_XYZ'])

                            # run phonopy
                            [ph_eigenvectors, ph_omega] = phonopy_funcs.run_phonopy(phonon_file, k_red_list)

                            # compute W tensor
                            W_tensor = physics.calculate_W_tensor(phonon_file,
                                                                    phonopy_params['num_atoms'],
                                                                    phonopy_params['atom_masses'],
                                                                    num_mod.numerics_parameters['n_DW_x'], 
                                                                    num_mod.numerics_parameters['n_DW_y'], 
                                                                    num_mod.numerics_parameters['n_DW_z'], 
                                                                    phonopy_params['recip_red_to_XYZ'])

                            # compute the differential, binned and total rates

                            if 'special_model' in phys_mod.physics_parameters.keys():

                                if first_job and proc_id == root_process:
                                    print('Computing the scattering rate for a specific model...')
                                    print()
                                    print('    Model : '+phys_mod.physics_parameters['special_model'])
                                    print()

                                if phys_mod.physics_parameters['special_model'] == 'SI':

                                    [diff_rate, binned_rate, total_rate] = physics.calc_diff_rates_SI(
                                                                    mass, threshold,
                                                                    q_XYZ_list, 
                                                                    G_XYZ_list, 
                                                                    jacob_list, 
                                                                    phys_mod.physics_parameters,
                                                                    vE_vec, 
                                                                    num_mod.numerics_parameters, 
                                                                    phonopy_params,
                                                                    ph_omega, 
                                                                    ph_eigenvectors,
                                                                    W_tensor, 
                                                                    mat_mod.mat_properties_dict, 
                                                                    phys_mod.dm_properties_dict, 
                                                                    phonon_file, phys_mod.c_dict)



                                else:

                                    if first_job and proc_id == root_process:
                                        print('    This specific model does not have a unique implementation.')
                                        print('    Try using the more general calculation by specifying the ')
                                        print('    coefficients of the operators you want to include.')
                                        print()

                            else:

                                [diff_rate, binned_rate, total_rate] = physics.calc_diff_rates_general(mass, 
                                                                    q_XYZ_list, 
                                                                    G_XYZ_list, 
                                                                    jacob_list, 
                                                                    phys_mod.physics_parameters,
                                                                    vE_vec, 
                                                                    num_mod.numerics_parameters, 
                                                                    phonopy_params,
                                                                    ph_omega, 
                                                                    ph_eigenvectors,
                                                                    W_tensor, 
                                                                    phys_mod.c_dict, 
                                                                    mat_mod.mat_properties_dict, 
                                                                    phys_mod.dm_properties_dict, 
                                                                    phys_mod.c_dict_form, phonon_file)

                            diff_rate_list.append([job_list_recv[job], np.real(diff_rate)])
                            binned_rate_list.append([job_list_recv[job], np.real(binned_rate)])
                            total_rate_list.append([job_list_recv[job], np.real(total_rate)])

                            first_job = False

                    if proc_id == root_process:
                        print('Done computing rate. Returning all data to root node to write.\n\n------\n')

                    # return data back to root
                    all_diff_rate_list   = comm.gather(diff_rate_list, root=root_process)
                    all_binned_rate_list = comm.gather(binned_rate_list, root=root_process)
                    all_total_rate_list  = comm.gather(total_rate_list, root=root_process)

                    # Converting threshold, V0,VE and VESC values to integer
                    # thr=int(threshold/(10**(-3)))
                    V0=int(const.V0/const.kmet_per_sec_to_none)
                    VE=int(const.VE/const.kmet_per_sec_to_none)
                    VESC=int(const.VESC/const.kmet_per_sec_to_none)

                    # write to output file
                    if proc_id == root_process:

                        out_filename = os.path.join(
                                output_folder, material+'_'+phys_input_mod_name+'_'+num_input_mod_name+'_'+str(const.VDF)+'_'+str(V0)+'_'+str(VE)+'_'+str(VESC)+'.hdf5')
                        # out_filename = os.path.join(
                        #         output_folder, material+'_'+phys_input_mod_name+'_'+num_input_mod_name+'_'+str(const.VDF)+'_'+str(VE)+'_'+str(VESC)+'.hdf5')
                        
                        hdf5_output.hdf5_write_output(out_filename,
                                                    num_mod.numerics_parameters,
                                                    phys_mod.physics_parameters,
                                                    phys_mod.dm_properties_dict,
                                                    phys_mod.c_dict,
                                                    all_total_rate_list,
                                                    n_proc, 
                                                    material,
                                                    all_diff_rate_list,
                                                    all_binned_rate_list)

                        print(f"Done writing rate for VDF = {const.VDF}, V0 = {const.V0/const.kmet_per_sec_to_none}, VE = {const.VE/const.kmet_per_sec_to_none}, VESC = {const.VESC/const.kmet_per_sec_to_none}.\n\n------\n")
                        # print(f"Done writing rate for VDF = {const.VDF}, VE = {const.VE/const.kmet_per_sec_to_none}, VESC = {const.VESC/const.kmet_per_sec_to_none}.\n\n------\n")
                
    if proc_id == root_process:
        print('Done writing rate for all loops.\n\n------\n')
else:

    if proc_id == root_process:
        print('ERROR:')
        print("\t- material info file\n"+
        "\t- physics model file\n"+
        "\t- numerics input file\n\n"+
        "must be present. These are added with the -m, -p, -n input flags. Add -h (--help) flags for more help.\n\n"+
        "See inputs/material/GaAs/GaAs_example.py, inputs/physics_model/dark_photon_example.py, inputs/numerics/standard.py for examples.")

