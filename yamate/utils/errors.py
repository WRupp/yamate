from math import sqrt
import numpy as np


def RMS_error(BaseCurve, FittedCurve):
    error = BaseCurve - FittedCurve
    return np.sqrt(np.mean(error ** 2))


def NRMS_error(BaseCurve, FittedCurve, method="mean"):

    if method == "mean":
        Norm = np.mean(BaseCurve)
    elif method == "range":
        Norm = np.max(BaseCurve) - np.min(BaseCurve)
    elif method == "std":
        Norm = np.std(BaseCurve)
    elif method == "absmax":
        Norm = np.max((abs(np.max(BaseCurve)), abs(np.min(BaseCurve))))

    if Norm != 0.0:
        error = BaseCurve - FittedCurve
        return np.sqrt(np.mean((error / Norm) ** 2))

    else:
        raise NotImplementedError


def MAE_error(BaseCurve, FittedCurve):
    abs_error = np.abs(BaseCurve - FittedCurve)
    return np.mean(abs_error)


def L2_norm(vector):
    return np.sqrt(np.sum(vector ** 2.0))


def p_norm(vector, p):

    if p < 1.0:
        print(" p must be greater or equal to 1. Stopping...")
        exit()

    return np.sum(np.abs(vector) ** p) ** (1.0 / p)


def MAPE_error(BaseCurve, FittedCurve):
    """	The mean absolute percentage error (MAPE)"""
    if 0 not in BaseCurve:
        Normed_error = (BaseCurve - FittedCurve) / BaseCurve
        return np.mean(Normed_error)
    else:
        raise ValueError
