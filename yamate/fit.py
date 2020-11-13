from scipy.optimize import minimize


class Model:
    def __init__(self, error_fn):
        """Instantiates de model class

        Parameters

        error_fn: callable
            the error function to minimize. must have the following interface:
                'error_fn(parameters, true_curves) -> float'
            where true_curves is a tuple of ndarrays, which the material will be fitted to.
        """

        self.error_fn = error_fn

    def fit(
        self,
        parameters_guess: dict,
        goal_curve,
        additional_material_parameters={},
        **minimize_kwargs
    ):
        """
        Identify material parameters through minimization of the error between curves.

        Parameters

        parameters_guess: dict
            The identifiable parameters names and a initial guessed value. Parameter name and initial value pairs.

        error_fn_args: tuple
            extra arguments to the error function.

        minimize_kwargs
            extra arguments to the 'scipy.optimize.minimize' function. check scipy documentation for available options.
        """

        parameter_names = list(parameters_guess.keys())
        parameter_initial_guess = list(parameters_guess.values())

        # packs arguments for the error function
        error_fn_args = (goal_curve, parameter_names, additional_material_parameters)

        # performs the error minimization between the real and the simulated curves
        result = minimize(
            fun=self.error_fn,
            x0=parameter_initial_guess,
            args=error_fn_args,
            **minimize_kwargs
        )

        # links the value to the parameter name
        identified_parameters = {}
        for value, parameter in zip(result.x, parameter_names):
            identified_parameters[parameter] = value

        return identified_parameters
