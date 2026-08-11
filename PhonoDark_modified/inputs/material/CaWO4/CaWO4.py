# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:30:49 2023

@author: user
"""

import numpy as np
import src.constants as const

material = 'CaWO4'

# number of atoms in the primitive cell
num_atoms = 24
mat_properties_dict = {
        # dimension of supercell used in DFT calculation
        "supercell_dim": [2., 2., 1.], 
	"mass":{
		"e": const.M_ELEC,
		"p": const.M_NUCL,
		"n": const.M_NUCL
	},
	"N_list": {
		"e": np.array([
                    20.0 - 2.0,   # Ca²⁺
        			74.0 - 6.0,   # W⁶⁺
        			8.0 + 2.0     # O²⁻
                    ]),
		"p": np.array([
                    20.0,  # Ca
        			74.0,  # W
        			8.0    # O
                    ]),
		"n": np.array([
            		40.078 - 20.0,
        			183.84 - 74.0,
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