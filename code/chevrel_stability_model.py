# -*- coding: utf-8 -*-
"""
Script to run DRIFT ML method on Chevrel dataset and regenerate H_delta.

More information on the machine learned descriptor and the ML method are available 
in the main text and SI of the corresponding manuscript. 

Inputs:
    data (dict): All individual entries and their corresponding features 
        and target property values.
    target_property (str): 
"""

import json
from SIFT import sift

data_file = '../data/CP_comp_features_dataset.json'
target_prop = 'dHd'
normalization_file = '../data/CP_feature_normalization_constants.json'

feat_units = {'Cavg__ICSDvolume': 'V', 'Ctotal__ICSDvolume': 'V', 'X__ICSDvolume': 'V',
              'Cavg__MiracleRadius': 'pm','Ctotal__MiracleRadius': 'pm', 'X__MiracleRadius': 'pm',
              'Cavg__Polarizability': 'cc', 'X__Polarizability': 'cc', 'Ctotal__Polarizability': 'cc', 
              'Ctotal__NValence': '', 'Cavg__NValence': '',
              'Cavg__allred_rochow': '', 'X__allred_rochow': '', 'X__X': '', 'Cavg__X': '', 'Ctotal__X': '',
              'Cavg__atomic_radius': 'pm', 'Ctotal__atomic_radius': 'pm', 'X__atomic_radius': 'pm',
              'Cavg__average_ionic_radius': 'pm','Ctotal__average_ionic_radius': 'pm','X__average_ionic_radius': 'pm',
              'Cavg__covalent_radius': 'pm', 'Ctotal__covalent_radius': 'pm', 'X__covalent_radius': 'pm',
              'Cavg__atomic_mass': 'amu', 'Ctotal__atomic_mass': 'amu', 'X__atomic_mass': 'amu',
              'Cavg__electron_affinity': 'eVat', 'Ctotal__electron_affinity': 'eVat', 'X__electron_affinity': 'eVat',
              'Cavg__phi':'eVat', 'Ctotal__phi':'eVat',
              'Cavg__cohesive_energies': 'eVat', 'Ctotal__cohesive_energies': 'eVat', 'X__cohesive_energies': 'eVat',
              'nels': '', 'C_occupation': '',
              'Cavg__mendeleev_no': '', 'Cavg__row': '', 'X__row': '', 'Cavg__group': '',
              'Cavg__first_IE': 'eVat', 'X__first_IE': 'eVat',
              'Cavg__second_IE': 'eVat', 'X__second_IE': 'eVat',
              'Cavg__min_oxidation_state': '', 'Cavg__max_oxidation_state': '',
              'Ctotal__min_oxidation_state': '', 'Ctotal__max_oxidation_state': '',
              'C__avg_oxidation_state': '', 'C__tot_oxidation_state': '', 'Mo__avg_oxidation_state': '',
              'els_donated': '', 'els_from_4': '', 
              'Cavg__fusion': 'eVat', 'Ctotal__fusion': 'eVat', 'X__fusion': 'eVat',
              'spacing_diff_r': 'pm', 'spacing_r': 'pm', 'spacing_d': 'pm', 'spacing': 'pm',
              }

with open(data_file,'r') as f:
    data = json.load(f)
with open(normalization_file,'r') as f:
    norm_constants = json.load(f)
    
dr = sift(data, target_prop, depth = 3, testing_split = 0.2, corr_function = 'rmse',
          use_units = True, unit_data = feat_units, keep_previous_set = True, 
          normalize = True, norm_constants = norm_constants, verbose = True,
          max_new_features = 500, max_complexity = 6, k_fold = (4,5))

results = dr.run()
