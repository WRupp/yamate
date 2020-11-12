import copy

import numpy as np

from yamate.materials import material
from yamate.utils import mathroutines
# Computational Methods for Plasticity: Theory and Applications, Souza Neto.
# pg 526
    
class MooneyRivlin(material.Material):
    name = 'Mooney-Rivlin'

    def __init__(self, alpha1=np.inf, alpha2=np.inf, k=np.inf):
        super().__init__()
        self.ALPHA1 = alpha1
        self.ALPHA2 = alpha2
        self.K = k
        
    def calculate_state(self, F, **kwargs):
        """ Given a new set of kinematic state variables, calculates the material state under this new condition."""
        
        trial_state = copy.deepcopy(self.state)

        I = np.eye(3)
        
        J = np.linalg.det(F)
        b = np.matmul(F, F.transpose() )
        
        b_iso = (J**(-2.0e0/3.0e0)) *b
        trbi = np.trace(b_iso)

        dbi = mathroutines.deviatoric(b_iso)
        dbi2 = mathroutines.deviatoric(np.matmul(b_iso,b_iso))
    
        T_mr = (1.0/J)*(
        + 2.0 * (self.ALPHA1 + self.ALPHA2*trbi) * dbi
        - 2.0*self.ALPHA2 * dbi2 
        + self.K * np.log(J) * I
        )

        trial_state.F = copy.deepcopy(F)
        trial_state.cauchy_stress = mathroutines.to_voigt(T_mr)

        return trial_state