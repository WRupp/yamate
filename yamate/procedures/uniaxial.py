import copy

import numpy as np

from scipy.optimize import minimize


def uniaxial_procedure(times, axial_stretches, material):
    """ Performs an homogeneous uniaxial test on the material given the axial stretches and times.
    returns the axial stress and the transversal strain.
    Note: the material state is modified.
    """

    # Assert that all arrays have the same lenght
    assert len(times) == len(axial_stretches)
    axial_stresses = np.empty(len(axial_stretches))
    transversal_stretches = np.empty(len(axial_stretches))

    # find the stress-strain equilibrium state for a given axial stretch and elapsed time
    for index, (time, axial_stretch) in enumerate(zip(times, axial_stretches)):     
        current_transversal_state = material.state.F[0,0]
                
        # find the transversal stretch, x, so that the transversal stress is zero
        # i.e., minimize the error between the transversal stress and zero.
        result = minimize(
            transversal_stress_error,
            args= (axial_stretch, time, material),
            x0= current_transversal_state,            
            tol= 1.0e-6
        )

        # save results and update material state
        transversal_stretches[index] = result.x[0]    
        
        F = F_uniaxial(transversal_stretches[index], axial_stretch)
        trial_state = material.calculate_state(F,time = time)
        material.save_state(trial_state)
        axial_stresses[index] = material.state.cauchy_stress[1] # the axial component of the stress tensor

    return axial_stresses, transversal_stretches

def F_uniaxial(XX_stretch,YY_stretch):
        """ Composes an uniaxial deformation tensor F from stretches.""" 
        F = np.eye(3)
        F[0,0] = XX_stretch
        F[1,1] = YY_stretch
        F[2,2] = XX_stretch
        return F

def transversal_stress_error(XX_stretch, YY_stretch, time, material):
    """Calculates the stress tensor for the given stretches and returns the transversal(XX or ZZ) component"""
    F_uni = F_uniaxial(XX_stretch, YY_stretch)
    trial_state = material.calculate_state(F=F_uni, time=time)
    transversal_stress_trial = trial_state.cauchy_stress[0]
    return abs(transversal_stress_trial) # equivalent to sqrt((valor-0)**2)