import numpy as np
import pandas as pd

from yamate.materials import visco_hydrolysis as vvh
from yamate.procedures import uniaxial
from yamate.utils import measurement_converter as mc


def vvh_uniaxial():
    """ Performs an uniaxial test to the visco_hydrolysis material and save results of the experiment in a '.csv' file"""

    props = {
        "mu": 948.45,
        "nu": 0.0e0,
        "Bulk": 2459.47,
        "kfa": 0.0e0,
        "kc": 36.31,
        "keta": 0.118,
        "Sy0": 60.0e0,
        "kh": 0.0e0,
        "s0": 113.34,
        "scv": 18.14,
        "sg": 140.78,
        "sz": 185.46,
        "sb": 164.82,
        "knh": 10.37,
        "kHiso": 0.001,
        "kcH": 0.0e0,
        "knd": 0.0e0,
        "km": 0.0e0,
        "kR": 120e12,
        "kg": 0.10,
        "kS": 1.0e0,
        "kN": 4.0e12,
        "threshold": 0.0,
        "FlagHardening": 3.0,
        "FlagPlasDam": 1.0,
        "FlagHidrDam": 1.0,
        "params": np.ones(3, dtype=int),
        "alpha_guess": 1.0e-12,
    }

    meu_mat = vvh.VariationalViscoHydrolysisAxi(props=props)

    # defining the experiment
    initial_radius = 3.0
    initial_height = 6.0

    strain_rate = mc.nominal_rate_with_compliance(
        3000, initial_size=initial_height, displacement=-1.7
    )

    times = np.linspace(0.0, 3000.0, 301)
    axial_stretches = mc.time_to_stretch(
        strain_rate_eng=strain_rate, time=times
    )
    axial_displacements = mc.time_to_displacement(
        strain_rate=strain_rate, time=times, initial_size=initial_height
    )

    axial_stresses, transversal_stretches = uniaxial.uniaxial_procedure(
        times, axial_stretches, meu_mat
    )

    # operate and save results
    crosssectional_area = np.pi * (initial_radius * transversal_stretches) ** 2.0
    axial_forces = axial_stresses * crosssectional_area

    results = pd.DataFrame([], index=times)
    results["axial_stretch"] = axial_stretches
    results["transversal_stretch"] = transversal_stretches
    results["axial_displacement"] = axial_displacements
    results["axial_stress"] = axial_stresses
    results["axial_force"] = axial_forces
    results.to_csv("vvh_uniaxial_example.csv")


if __name__ == "__main__":
    vvh_uniaxial()
