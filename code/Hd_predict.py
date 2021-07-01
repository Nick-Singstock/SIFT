# -*- coding: utf-8 -*-

'''
Backend to use H_delta descriptor for CP stability prediction.
Also generates new Chevrel phase compositions and predicts stability
     and synthesizability

Author: Nick Singstock
'''

import json
from Equation import Expression # available at: https://pypi.org/project/Equation/
from pymatgen.core.composition import Composition
from pymatgen.core import Element
import os
import csv
import numpy as np
import re

class Hdelta():
    
    def __init__(self):
        self.descriptor = ('(((Cavg__Polarizability / X__Polarizability) - (Ctotal__fusion'+
              ' / X__cohesive_energies)) * (Cavg__first_IE * els_donated))')
        self.slope = -0.03804
        self.intercept = 0.02967
        
        self.features = ['Polarizability','fusion','cohesive_energies','first_IE']
        self.descriptor_features = ['Cavg__Polarizability','X__Polarizability','els_donated',
                                    'Ctotal__fusion','X__cohesive_energies','Cavg__first_IE']
        self.set_norms()
        self.set_feature_dics(self.features)
    
    def default_elements(self):
        single_site = ['Al', 'Ru', 'Hf', 'In', 'Pb', 'Os', 'Tb', 'Zn', 'Ge', 'Y', 'Ag', 'Cu',
                       'Ni', 'Au', 'Cr', 'Rh', 'Nb', 'Mn', 'Tc', 'Bi', 'Ta', 'Sn', 'Be', 'K', 
                       'Tl', 'Rb', 'Sb', 'Dy', 'Ti', 'Cd', 'Pd', 'Na', 'Pr', 'La', 'Ce', 'Ba', 
                       'Co', 'Sm', 'Si', 'Ga', 'Zr', 'Sc', 'Nd', 'Pt', 'Ir', 'V', 'Sr', 'Mg', 
                       'Hg', 'Re', 'Ca', 'Li', 'Fe', ]
        single_site += ['W','As','Pm','Eu','Gd','Ho','Er','Tm','Yb','Lu']
        
        exp_double_site = ['Cu','Fe','Ga','Mg','Ni',     'Na'] # TODO: Na  
        
        radii = self.read_feature('average_ionic_radius') #'average_ionic_radius', 'atomic_radius'
        max_r = np.max([radii[d] for d in exp_double_site])
        double_site = [el for el in single_site if radii[el] <= max_r]
        
        print(len(double_site))
#        double_site = single_site
        
        chalcogenides = [{'S': 8}, ]#{'S': 6, 'Se': 2}, {'S': 4, 'Se': 4}, {'S': 2, 'Se': 6},
#                         {'Se': 8}, {'Se': 6, 'Te': 2}, {'Se': 4, 'Te': 4}, {'Se': 2, 'Te': 6}, 
#                         {'Te': 8}, {'S': 6, 'Te': 2}, {'S': 4, 'Te': 4}, {'S': 2, 'Te': 6}, ]
        return single_site, double_site, chalcogenides
        
    
    def generate_all_comps(self):
        '''
        Generates set of all CP compositions studied in our work on CP stability
        '''
        single, double, chalc = self.default_elements()
        cation_combinations = []
        
        # combining rules:
        #   1) single cations up to stoich of 1
        #   2) double cations up to stoich of 2
        #   3) Allowed coeffs: 0.5, 1.0, 1.5, 2.0
        #   4) up to 3 cations per material
        #   5) double cation size limited by ionic radii of Mg
        for i1,c1 in enumerate(single):
            # Create combinations with single cation and coeffs. of 0.5 and 1
            cation_combinations.append({c1: 0.5})
            cation_combinations.append({c1: 1.0})
            if c1 in double:
                cation_combinations.append({c1: 1.5})
                cation_combinations.append({c1: 2.0})
            
            for i2,c2 in enumerate(single):
                if i1 >= i2: continue
                cation_combinations.append({c1: 0.5, c2: 0.5})

            if c1 in double:
                ii1 = double.index(c1)
                for i2,c2 in enumerate(double):
                    if c1 == c2: continue
                    cation_combinations.append({c1: 1.0, c2: 0.5})
                    cation_combinations.append({c1: 1.5, c2: 0.5})
                    if ii1 < i2: 
                        cation_combinations.append({c1: 1.0, c2: 1.0})
#                    for i3,c3 in enumerate(double):
#                        if c3 in [c1, c2]: 
#                            continue
#                        if i2 < i3:
#                            cation_combinations.append({c1: 1.0, c2: 0.5, c3: 0.5})
#                        if i2 < i3 and i2 < ii1:
#                            cation_combinations.append({c1: 0.5, c2: 0.5, c3: 0.5})
                    
        # generate all compositons with each anion set
        all_compositions = []
        for anion in chalc:
            for cat in cation_combinations:
                comp_dic = cat.copy()
                comp_dic['Mo'] = 6
                for an, coeff in anion.items():
                    comp_dic[an] = coeff
                all_compositions += [comp_dic.copy()]
        return all_compositions
    
    def get_features(self, comp_dic):
        if 'Mo' not in comp_dic or comp_dic['Mo'] != 6:
            print('WARNING: Script should be used for Chevrel-phases with Mo6 octahedra. '+
                  'Other octahedra not supported.')
        if sum([v for k,v in comp_dic.items() if k in ['S','Se','Te']]) != 8:
            n_chalc = sum([v for k,v in comp_dic.items() if k in ['S','Se','Te']])
            print('WARNING: Script should be used for Chevrel-phases '+
                  'with 8 chalcogenides, %i were given.'%n_chalc)
        
        feats = {}
        anions, cations = [], []
        for k,v in comp_dic.items():
            if k in ['Mo']: continue
            if k in ['S','Se','Te']:
                anions += [k]
            else:
                cations += [k]
        # add tabulated features
        n_cats = np.sum([v for k,v in comp_dic.items() if k in cations])
        for f in self.features:
            if f not in ['cohesive_energies']:
                feats['Ctotal__'+f] = np.sum([self.feature_dic[f][k]*v 
                                             for k,v in comp_dic.items() if k in cations ])
                feats['Cavg__'+f] = feats['Ctotal__'+f] / n_cats if n_cats > 0 else 0
            feats['X__'+f] = np.sum([self.feature_dic[f][k]*v 
                             for k,v in comp_dic.items() if k in anions ]) / 8
        # add calculated oxidation states and electrons donated from cations
        allowed_oxi_states = {a: [-2] for a in anions}
        allowed_oxi_states['Mo'] = [1,2,3]
        for cat in cations:
            oxis = list(Element(cat).icsd_oxidation_states) + list(Element(cat).common_oxidation_states)
            allowed_oxi_states[cat] = [o for o in oxis if o > 0]
        div = 1
        if any([v in [0.5,1.5,2.5] for v in comp_dic.values()]):
            div = 2
#            formula = self.get_formula({k: int(v*2) for k,v in comp_dic.items()})
            comp = Composition(self.get_formula({k: int(v*2) for k,v in comp_dic.items()}))
            oxi = comp.add_charges_from_oxi_state_guesses(oxi_states_override=allowed_oxi_states)
        else:
            comp = Composition(self.get_formula(comp_dic))
            oxi = comp.add_charges_from_oxi_state_guesses(oxi_states_override=allowed_oxi_states)
        total_charge = 0
        for ko, vo in oxi.as_dict().items():
            el = ko[:-2] if any([x in ko for x in self.numbers]) else ko[:-1]
            if el not in cations:
                continue
            oxi_state = (float(ko[-1]+ko[-2]) if any([x in ko for x in self.numbers]) 
                         else float(ko[-1]+'1'))
            total_charge += oxi_state * vo
        total_charge = total_charge / div
        feats['els_donated'] = total_charge if total_charge <= 4.0 else 4.0
        return feats

    @property
    def numbers(self):
        return ['1','2','3','4','5','6','7','8','9','0']
    
    def get_formula(self, comp_dic):
        assert 'Mo' in comp_dic, 'ERROR: Mo must be present in composition dictionary.'
        cations = [k for k in comp_dic if k not in ['Mo','S','Se','Te']]
        cations.sort()
        anions = [k for k in comp_dic if k  in ['S','Se','Te']]
        anions.sort()
        string = ''
        for c in cations:
            if comp_dic[c] == 1.0:
                string += c 
            elif comp_dic[c] == 2.0:
                string += c + str(int(comp_dic[c]))
            else:
                string += c + '%.1f'%(comp_dic[c])
        string += 'Mo' + str(int(comp_dic['Mo']))
        for a in anions:
            string += a + str(comp_dic[a])
        return string
    
    def formula_to_dic(self, formula):
        assert '.' not in formula, 'ERROR: Cannot include decimals in formula. Only integers allowed.'
        if '(' in formula:
            first = formula.split('(')[0]
            second = ''.join(formula.split('(')[1:]).split(')')[0]
            coeff = ''.join(formula.split('(')[1:]).split(')')[1]
            d = self.formula_to_dic(first)
            for k, v in self.formula_to_dic(second).items():
                if k in d:
                    d[k] += v*int(coeff)
                else:
                    d[k] = v*int(coeff)
            return d
        
        # split formula to list by number
        spl = [s for s in re.split('(\d+)',formula) if s != '']
        els_list = []; el_frac = [];
        for el in spl: 
            if sum(1 for c in el if c.isupper()) == 1 and len(el) < 3:
                # I am a single element, add me to the list please
                els_list.append(el)
                el_frac.append(1)
            elif el.isdigit() == True:
                # I am a number and should be calculated in to self.el_frac
                el_frac[-1] = el_frac[-1] * int(el)
            elif sum(1 for c in el if c.isupper()) > 1:
                # I am more than one element and am here to cause trouble
                els_split = re.sub( r"([A-Z])", r" \1", el).split()
                els_list += els_split
                for i in els_split:
                    el_frac.append(1)
        # combine duplicate elements
        for el in els_list:
            inds = [i for i, j in enumerate(els_list) if j == el]
            if len(inds) > 1:
                drop_ind = inds[1:]
                drop_ind = drop_ind[::-1]
                el_frac[inds[0]] = sum(el_frac[inds[i]] for i in range(len(inds)))
                for d in drop_ind: del els_list[d]; del el_frac[d]
        assert len(els_list) == len(el_frac), 'ERROR: elements and coefficients not balanced in "formula_to_dic".'
        return {els_list[i]: el_frac[i] for i,_ in enumerate(els_list)}
    
    def predict(self, features):
        normed = {k: v / self.norms[k] for k,v in features.items() if k in self.descriptor_features}
        return self.evaluate(self.descriptor, normed) * self.slope + self.intercept
    
    def analyze_predictions(self, prediction_dic):
        dHd = np.array([v[0] for k,v in prediction_dic.items()])
        total = len(dHd)
        stable = [x for x in dHd if x < 0]
        high_meta = [x for x in dHd if x < 0.047 and x >= 0]
        med_meta = [x for x in dHd if x < 0.065 and x >= 0.047]
        unstable = [x for x in dHd if x >= 0.065]
        print('\n\nTotal Materials: \t%i' % total)
        print('Stable: \t\t%i' % len(stable) + '\t%.1f%%' % (100*len(stable)/total))
        print('HighT Metastable: \t%i'%len(high_meta) + '\t%.1f%%' % (100*len(high_meta)/total))
        print('MedT Metastable: \t%i' % len(med_meta) + '\t%.1f%%' % (100*len(med_meta)/total))
        print('Unstable: \t\t%i' %len(unstable) + '\t%.1f%%' % (100*len(unstable)/total))
    
    def evaluate(self, s, dic):
        counter = 0
        var_map = {}
        arg_order = []
        for k in dic.keys():
            if k in s.replace('(','').replace(')','').split(): 
                counter += 1
                s = s.replace('('+k+' ', '(x'+str(counter)+' ')
                s = s.replace(' '+k+')', ' x'+str(counter)+')')
                var_map['x'+str(counter)] = k
                arg_order.append('x'+str(counter))
        ex = Expression(s, argorder = arg_order.copy())
        if counter == 1: return ex(dic[var_map[arg_order[0]]])
        if counter == 2: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]])
        if counter == 3: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]])
        if counter == 4: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]])
        if counter == 5: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]], dic[var_map[arg_order[4]]])
        if counter == 6: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]], dic[var_map[arg_order[4]]], dic[var_map[arg_order[5]]])

    def set_norms(self):
        with open(os.path.join('..','data','CP_feature_normalization_constants.json'),'r') as f:
            norms = json.load(f)
        self.norms = norms
    
    def set_feature_dics(self, features):
        self.feature_dic = {}
        for f in features:
            self.feature_dic[f] = self.read_feature(f)
    
    def read_feature(self, f, location = os.path.join('..','data','datasets')):
        with open(os.path.join(location, f+'.csv'), 'r') as csvfile:
            feat_data = csv.reader(csvfile)
            return {d[0]: float(d[1]) for d in feat_data}
    
if __name__ == '__main__':
    Hd = Hdelta()
    
    predictions = {}
    all_comps = Hd.generate_all_comps()
    print('Generated Compositions: %i' % len(all_comps))

    for i,comp_dic in enumerate(all_comps):
        frac_complete = i / len(all_comps)
        print("\rProgress {:2.1%}".format(frac_complete), end="\r")
        
        comp = Hd.get_formula(comp_dic)
        features = Hd.get_features(comp_dic)
        predictions[comp] = (Hd.predict(features), comp_dic)
    
    Hd.analyze_predictions(predictions)


#with open(os.path.join('..','data','predicted_decomp_all2Na.json'), 'w') as f:
#    json.dump(predictions, f)
    