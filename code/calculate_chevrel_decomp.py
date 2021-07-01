# -*- coding: utf-8 -*-
'''

Simple script using Hd_predict.py functions to calculate 
decompositon enthalpy of Chevrel phase

'''

from Hd_predict import Hdelta

formula = 'Mg2Mo6S8'

# get dictionary representation
Hd = Hdelta()
element_dict = Hd.formula_to_dic(formula)

# get features for Hdelta
features = Hd.get_features(element_dict)

# get predicted decomposition enthalpy
predicted_decomp = Hd.predict(features)

print('\nChevrel phase '+formula+' has a predicted decomp.'+
      ' enthalpy of %.3f eV/atom.' % (predicted_decomp))
