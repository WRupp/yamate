import numpy as np


def tensor_inner_product(a, b):
    return np.tensordot(a, b, axes=2)


# def Tensor_Product(a,b):
#     return np.tensordot(a, b, axes=0)


def tensor_product(a_vector, b_vector):
    "Pelo que entendi é o produto tensorial de dois vetores coluna"
    Tensor = np.empty((a_vector.shape[0], b_vector.shape[0]))

    for i in range(a_vector.shape[0]):
        for j in range(b_vector.shape[0]):
            Tensor[i, j] = a_vector[i] * b_vector[j]
    return Tensor


def norm(a):
    return np.sqrt(np.tensordot(a, a, axes=1))


def deviatoric(tensor):
    """Returns the deviatoric part of a tensor"""
    I = np.eye(tensor.shape[0])
    T_dev = tensor - (1.0e0 / 3.0e0) * np.trace(tensor) * I
    return T_dev


def to_voigt(tensor):
    """Converts a 3x3 tensor to voigt notation"""

    tensor_voigt = np.empty(6)
    tensor_voigt[0] = tensor[0, 0]
    tensor_voigt[1] = tensor[1, 1]
    tensor_voigt[2] = tensor[2, 2]
    tensor_voigt[3] = tensor[0, 1]
    tensor_voigt[4] = tensor[1, 2]
    tensor_voigt[5] = tensor[0, 2]
    return tensor_voigt
