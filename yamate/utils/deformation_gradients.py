# Note: all stretches are considered in the x direction. consider rotating the deformation gradient to fit your needs.
import numpy as np


def F_uniaxial(axial_stretch, transversal_stretch):
    """Composes an uniaxial, compressible, deformation tensor F from axial and transversal stretches."""
    F = np.eye(3)
    F[0,0] = axial_stretch
    F[1,1] = transversal_stretch
    F[2,2] = transversal_stretch
    return F

def F_uniaxial_incompressible(axial_stretch):
    """Composes an uniaxial, incompressible, deformation tensor F from axial stretch."""
    F = np.eye(3)
    F[0,0] = axial_stretch
    F[1,1] = 1/np.sqrt(axial_stretch)
    F[2,2] = 1/np.sqrt(axial_stretch)
    return F

def F_biaxial(biaxial_stretch, transversal_stretch):
    """Composes an biaxial, compressible, deformation tensor F from axial stretches in the xx and yy directions.
    The  transversal_stretch (zz direction - axis 2) is left free to be specified."""
    F = np.eye(3)
    F[0,0] = biaxial_stretch
    F[1,1] = biaxial_stretch
    F[2,2] = transversal_stretch
    return F

def F_biaxial_incompressible(biaxial_stretch):
    """Composes an biaxial, incompressible, deformation tensor F from equal stretches in the xx and yy directions."""
    F = np.eye(3)
    F[0,0] = biaxial_stretch
    F[1,1] = biaxial_stretch
    F[2,2] = 1/(biaxial_stretch**2.0)
    return F

def F_simple_shear(shear_strain):
    """Composes a pure shear deformation tensor F from shear strain in the xy plane."""
    F = np.eye(3)
    F[0,1] = shear_strain
    return F