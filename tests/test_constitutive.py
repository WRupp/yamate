import pytest
import numpy as np

from yamate.materials import mooney_rivlin as mrn
from yamate.materials import visco_hydrolysis as vvh
from yamate.procedures import uniaxial
from yamate.utils import conversor_medidas as cm

def test_mooney_rivlin():
    
    mr_mat = mrn.MooneyRivlin(alpha1=1.0, alpha2=2.0, k=2000)
    
    F = np.eye(3)
    trial_state = mr_mat.calculate_state(F=F)
    assert trial_state.cauchy_stress[0] == 0.0
 
    F[0,0] = 1.01
    trial_state = mr_mat.calculate_state(F=F)
    assert trial_state.cauchy_stress[0] > 0


def test_visco_hydrolysis():

    props = {
    "mu":948.45,
    "nu":0.0e0, 
    "Bulk":2459.47,
    "kfa":0.0e0, 
    "kc":36.31, 
    "keta":0.118, 
    "Sy0":60.0e0, 
    "kh":0.0e0, 
    "s0":113.34, 
    "scv":18.14, 
    "sg":140.78, 
    "sz":185.46, 
    "sb":164.82,
    "knh":10.37, 
    "kHiso":0.001, 
    "kcH":0.0e0,
    "knd":0.0e0, 
    "km":0.0e0, 
    "kR":120e12, 
    "kg":0.10, 
    "kS":1.0e0, 
    "kN":4.0e+12, 
    "threshold": 0.0e0,
    "FlagHardening":3, 
    "FlagPlasDam":1, 
    "FlagHidrDam":1, 
    "params":np.ones(3, dtype=np.int), 
    "alpha_guess":1.0e-12
    }

    meu_mat = vvh.VariationalViscoHydrolysisAxi(props=props)

    F = np.eye(3)
    time = 1.0
    trial_state =meu_mat.calculate_state(F, time=time)
    assert np.array_equal(trial_state.cauchy_stress, np.zeros((6)))

    F[0,0] = 1.2
    trial_state =meu_mat.calculate_state(F, time=time)
    cauchy_stress = trial_state.cauchy_stress
    assert cauchy_stress[0] > 0.0
    assert cauchy_stress[1] > 0.0
    assert cauchy_stress[0] > cauchy_stress[1]
    assert cauchy_stress[1] == cauchy_stress[2]


    F[0,0] = 0.98
    trial_state =meu_mat.calculate_state(F, time=time)
    cauchy_stress = trial_state.cauchy_stress
    assert cauchy_stress[0] < 0.0
    assert cauchy_stress[1] < 0.0
    assert cauchy_stress[0] < cauchy_stress[1]
    assert cauchy_stress[1] == cauchy_stress[2]