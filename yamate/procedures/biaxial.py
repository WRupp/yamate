import copy

import numpy as np
from scipy.optimize import minimize

from yamate.utils import deformation_gradients as grads

def biaxial_procedure(times, biaxial_stretches, material, tol=1.0e-6):
    """Performs an homogeneous biaxial test for isotropic materials given the axial stretches(axis 0 and 1) and times.
    returns the axial stress (axis 0 and 1 are equal) and the transversal stretch (axis 2).
    Note: the material state is modified.
    """
    
    """ For isotropic materials."""
    
    # Assert that all arrays have the same lenght
    try: 
        assert len(times) == len(biaxial_stretches)
    except AssertionError as err:
        err.args = ("Stretches and times arrays do not have the same lenght!")
        raise err

    biaxial_stresses = np.empty(len(biaxial_stretches))
    transversal_stretches = np.empty(len(biaxial_stretches))

    for index, (time, biaxial_stretch) in enumerate(zip(times, biaxial_stretches)):
        current_transversal_stretch = material.state.F[2,2]

        # find the transversal stretch, in z axis, so that the transversal stress is zero
        # i.e., minimize the error between the transversal stress and zero.
        result = minimize(
            transversal_stress_error,
            args=(biaxial_stretch, time, material),
            x0=current_transversal_stretch,
            tol=tol,
        )

        # save results and update material state
        transversal_stretches[index] = result.x[0]

        F = grads.F_biaxial(biaxial_stretch, transversal_stretches[index])
        print(F)
        trial_state = material.calculate_state(F, time=time)
        material.save_state(trial_state)
        biaxial_stresses[index] = material.state.cauchy_stress[0] 

    return biaxial_stresses, transversal_stretches


def transversal_stress_error(biaxial_stretch, transversal_stretch, time, material):
    """Calculates the stress tensor for the given stretches and returns the transversal(axis 2) component"""
    F_biaxial = grads.F_biaxial(biaxial_stretch, transversal_stretch)
    trial_state = material.calculate_state(F=F_biaxial, time=time)
    transversal_stress_trial = trial_state.cauchy_stress[2]
    return abs(transversal_stress_trial)  # equivalent to sqrt((valor-0)**2)