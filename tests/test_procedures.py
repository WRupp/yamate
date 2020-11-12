import numpy as np
from yamate.materials.mooney_rivlin import MooneyRivlin
from yamate.procedures.uniaxial import uniaxial_procedure

def test_uniaxial():
    times = np.linspace(0, 10, num=10)
    axial_stretches = np.linspace(1.0, 0.85, num=10)
    mr = MooneyRivlin(alpha1=0.293, alpha2=0.177, k=1410)
    results = uniaxial_procedure(times, axial_stretches, mr)
    
    assert results[1][-1] > 1.0
