# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:30:49 2023

@author: user
"""

import numpy as np
import src.constants as const

material = 'SiO2'

# number of atoms in the primitive cell
num_atoms = 9
mat_properties_dict = {
        # dimension of supercell used in DFT calculation
        "supercell_dim": [3., 3., 3.], 
	"mass":{
		"e": const.M_ELEC,
		"p": const.M_NUCL,
		"n": const.M_NUCL
	},
	"N_list": {
		"e": np.array([
                    14.0 - 4.0, # Si⁴⁺ 
                    8.0+2.0  # O²⁻
                    ]),
		"p": np.array([
                    14.0, 
                    8.0
                    ]),
		"n": np.array([
            		28.085 - 14.0, 
                    15.999 - 8.0])                  
	},
	"L_S_list": {
		"e": np.zeros(num_atoms),
		"p": np.zeros(num_atoms),
		"n": np.zeros(num_atoms)
	},
	"S_list": {
		"e": np.zeros((num_atoms, 3)),
		"p": np.zeros((num_atoms, 3)),
		"n": np.zeros((num_atoms, 3))
	},
	"L_list": {
		"e": np.zeros((num_atoms, 3)),
		"p": np.zeros((num_atoms, 3)),
		"n": np.zeros((num_atoms, 3))
	},
	"L_tens_S_list": {
		"e": np.zeros((num_atoms, 3, 3)),
		"p": np.zeros((num_atoms, 3, 3)),
		"n": np.zeros((num_atoms, 3, 3))
	},
}