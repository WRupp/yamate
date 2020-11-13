from math import sqrt
import numpy as np


def rms_error(base_curve, fitted_curve):
    error = base_curve - fitted_curve
    return np.sqrt(np.mean(error ** 2))


def nrms_error(base_curve, fitted_curve, method="mean"):

    if method == "mean":
        Norm = np.mean(base_curve)
    elif method == "range":
        Norm = np.max(base_curve) - np.min(base_curve)
    elif method == "std":
        Norm = np.std(base_curve)
    elif method == "absmax":
        Norm = np.max((abs(np.max(base_curve)), abs(np.min(base_curve))))

    if Norm != 0.0:
        error = base_curve - fitted_curve
        return np.sqrt(np.mean((error / Norm) ** 2))

    else:
        raise NotImplementedError


def mae_error(base_curve, fitted_curve):
    abs_error = np.abs(base_curve - fitted_curve)
    return np.mean(abs_error)


def L2_norm(vector):
    return np.sqrt(np.sum(vector ** 2.0))


def p_norm(vector, p):

    if p < 1.0:
        print(" p must be greater or equal to 1. Stopping...")
        exit()

    return np.sum(np.abs(vector) ** p) ** (1.0 / p)


def mape_error(base_curve, fitted_curve):
    """	The mean absolute percentage error (MAPE)"""
    if 0 not in base_curve:
        Normed_error = (base_curve - fitted_curve) / base_curve
        return np.mean(Normed_error)
    else:
        raise ValueError
