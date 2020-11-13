from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from yamate.utils import errors
from yamate.utils import conversor_medidas as cm
from yamate.procedures import uniaxial
from yamate.materials import visco_hydrolysis as vvh

from yamate.fit import Model


def simulate_vvh_uniaxial_compression(material_properties):
    """Simulates the uniaxial procedure for the visco_hydrolysis material, where some of the constitutive parameters are unknown"""

    my_material = vvh.VarViscoHydrolysis_Axi(props=material_properties)

    # define and run the experiment
    initial_radius = 3.0
    initial_height = 6.0

    times = np.linspace(0.0, 3000.0, 301)
    strain_rate = cm.taxa_nominal_com_compliance(
        3000, tamanho_inicial=initial_height, deslocamento_real=-1.7
    )
    axial_stretches = cm.de_tempo_para_elongamento(
        taxa_def_eng=strain_rate, tempos=times
    )
    # axial_displacement = cm.de_tempo_para_deslocamento(taxa_def=strain_rate, tempos=times, tamanho_inicial=initial_height)

    axial_stresses, transversal_stretches = uniaxial.uniaxial_procedure(
        times, axial_stretches, my_material
    )

    # converts from stress to force
    cross_sectional_area = np.pi * (initial_radius * transversal_stretches) ** 2.0
    axial_force = axial_stresses * cross_sectional_area
    return axial_force


def uniaxial_force_l2(
    design_variables, true_curve, parameters_to_identify, additional_material_parameters
):

    # joins all material properties and run the simulation
    identifiable_parameters = {}
    for variable_name, variable_value in zip(parameters_to_identify, design_variables):
        identifiable_parameters[variable_name] = variable_value
    material_properties_trial = {
        **identifiable_parameters,
        **additional_material_parameters,
    }

    simulated_curve = simulate_vvh_uniaxial_compression(material_properties_trial)

    # calculates the error function
    error_curve = true_curve - simulated_curve
    return errors.L2_norm(error_curve)


def fit_vvh_to_uniaxial_data():

    # imports the data from real experiments
    plga_data_path = Path("yamate/data/PLGA/processed/")
    diss_result_forces = pd.read_csv(
        plga_data_path / "Forcas_resultadosDissertacao.csv"
    )
    true_curve = diss_result_forces["PHC1"].values

    # sets an initial guess
    parameter_guess = {
        "mu": 948.45,
        "Bulk": 2459.47,
        "kc": 36.31,
        "keta": 0.118,
        "s0": 113.34,
        "scv": 18.14,
        "sg": 140.78,
        "sz": 185.46,
        "sb": 164.82,
        "knh": 10.37,
        "kHiso": 0.001,
    }

    # sets the other necessary parameters
    additional_material_parameters = {
        "nu": 0.0e0,
        "kfa": 0.0e0,
        "Sy0": 60.0e0,
        "kh": 0.0e0,
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
        "params": np.ones(3, dtype=np.int),
        "alpha_guess": 1.0e-12,
    }

    # finally, we define the fitting job and get the results after it finishes (might take a while depending on the guess)
    model = Model(error_fn=uniaxial_force_l2)

    identified_parameters = model.fit(
        parameter_guess,
        goal_curve=true_curve,
        additional_material_parameters=additional_material_parameters,
    )

    print(identified_parameters)


if __name__ == "__main__":
    fit_vvh_to_uniaxial_data()
