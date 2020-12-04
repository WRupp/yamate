import copy

import numpy as np
from scipy.optimize import minimize

from yamate.utils import deformation_gradients as grads


def uniaxial_procedure(times, axial_stretches, material, tol=1.0e-6):
    """Performs an homogeneous uniaxial test on the material given the axial stretches and times.
    returns the axial stress and the transversal stretch.
    Note: the material state is modified.
    """

    # Assert that all arrays have the same lenght
    try: 
        assert len(times) == len(axial_stretches)
    except AssertionError as err:
        err.args = ("Stretches and times arrays do not have the same lenght!")
        raise err
    
    axial_stresses = np.empty(len(axial_stretches))
    transversal_stretches = np.empty(len(axial_stretches))

    # find the stress-strain equilibrium state for a given axial stretch and elapsed time
    for index, (time, axial_stretch) in enumerate(zip(times, axial_stretches)):
        current_transversal_state = material.state.F[0, 0]

        # find the transversal stretch, x, so that the transversal stress is zero
        # i.e., minimize the error between the transversal stress and zero.
        result = minimize(
            transversal_stress_error,
            args=(axial_stretch, time, material),
            x0=current_transversal_state,
            tol=tol,
        )

        # save results and update material state
        transversal_stretches[index] = result.x[0]

        F = grads.F_uniaxial(axial_stretch, transversal_stretches[index])
        trial_state = material.calculate_state(F, time=time)
        material.save_state(trial_state)
        axial_stresses[index] = material.state.cauchy_stress[0]  
        # the axial component of the stress tensor

    return axial_stresses, transversal_stretches


def transversal_stress_error(transversal_stretch, axial_stretch, time, material):
    """Calculates the stress tensor for the given stretches and returns the transversal(axis 1) component"""
    F_uni = grads.F_uniaxial(axial_stretch, transversal_stretch)
    trial_state = material.calculate_state(F=F_uni, time=time)
    transversal_stress_trial = trial_state.cauchy_stress[1]
    return abs(transversal_stress_trial)  # equivalent to sqrt((valor-0)**2)
