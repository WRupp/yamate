import numpy as np
import copy

class State():
    """ State variables are variables that identify the state of a material. The standard State class contains only kinematic state variables. """
    F = np.eye(3)
    cauchy_stress = np.empty(6) #using voigt notation

    internal_variables = None

class Constitutive_Parameters():
    pass

class Internal_Variables():
    pass


class Material:

    name = "base"    
    constitutive_parameters = Constitutive_Parameters() # Preciso disso mesmo?
    state = State()
    # internal_variables = Internal_Variables()

    def calculate_state(self, F, **kwargs):
        """ Given a new set of kinematic state variables, calculates the material state under this new condition and returns a state"""

    def save_state(self,trial_state):
        self.state = copy.deepcopy(trial_state)

    # # def update_internal_variables(self):
    # #     pass        

#-----


