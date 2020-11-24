import numpy as np


def time_to_strain(strain_rate: float, time: np.ndarray) -> np.ndarray:
    """Calculates the strain given an elapsed time and a strain rate"""
    return strain_rate * time


def time_to_stretch(strain_rate_eng: float, time: np.ndarray) -> np.ndarray:
    """Calculates the stretch given an elapsed time and the engineering strain rate"""
    strain = time_to_strain(strain_rate_eng, time)
    return strain + 1.0


def time_to_stretch_true(strain_rate_true: float, time: np.ndarray ) -> np.ndarray:
    """Calculates the stretch given an elapsed time and the true strain rate"""
    strain = time_to_strain(strain_rate_true, time)
    return np.exp(strain)


def time_to_displacement(strain_rate: float, time: np.ndarray, initial_size=6.0) -> np.ndarray:
    """Calculates the displacement given the strain rate and the initial size of the specimen"""
    strain = time_to_strain(strain_rate, time)
    return initial_size * strain


def nominal_rate_with_compliance(total_time, initial_size, displacement) -> float:
    """Calculate the nominal strain rate, when machine compliance is present"""
    nominal_strain = displacement / initial_size
    return nominal_strain / total_time
