import copy

import numpy as np

from yamate.materials import material
from yamate.utils import mathroutines


class Neo_Hookean(material.Material):
    """Compressible form of the Neo-Hookean as described in 'Computational Methods for Plasticity: Theory and Applications - Souza Neto'
    pg 526, §13.47'"""

    def __init__(self, G=np.inf, K=np.inf):
        super().__init__()
        self.G = G  # shear modulus
        self.K = K  # logarithmic bulk modulus

    def calculate_state(self, F, **kwargs):
        trial_state = copy.deepcopy(self.state)
        trial_state.F = copy.deepcopy(F)
        trial_state.cauchy_stress = mathroutines.to_voigt(self.cauchy_stress(F))
        return trial_state

    def cauchy_stress(self, F):
        J = np.linalg.det(F)
        return self.kirchhoff_stress(F) / J

    def kirchhoff_stress(self, F):
        """ calculate Neo-Hookean stress tensor given a deformation tensor F"""
        I = np.eye(3)

        J = np.linalg.det(F)
        F_iso = J ** (-1.0e0 / 3.0e0) * F
        b_iso = np.dot(F_iso, F_iso.transpose())
        b_star_dev = mathroutines.deviatoric(b_iso)
        return self.G * b_star_dev + self.K * np.log(J) * I
