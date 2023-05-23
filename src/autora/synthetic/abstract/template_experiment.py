"""
A template synthetic experiment.

Examples:
    >>> from autora.synthetic.abstract.template_experiment import template_experiment

    We can instantiate the experiment using the imported function
    >>> s = template_experiment()
    >>> s  # doctest: +ELLIPSIS
    SyntheticExperimentCollection(name='Template Experiment', params={'name': ...}, ...)

    >>> s.name
    'Template Experiment'

    >>> s.variables
    VariableCollection(...)

    >>> s.domain()
    array([[0],
           [1],
           [2],
           [3]])

    >>> s.ground_truth  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    functools.partial(<function template_experiment.<locals>.experiment_runner at 0x...>,
                      added_noise_=0.0)

    >>> s.ground_truth(1.)
    2.0

    >>> s.ground_truth(s.domain())
    array([[1.],
           [2.],
           [3.],
           [4.]])


    >>> s.experiment_runner  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <function template_experiment.<locals>.experiment_runner at 0x...>

    >>> s.experiment_runner(1.)
    1.8697820493137682

    >>> s.experiment_runner(s.domain())
    array([[1.01278404],
           [1.96837574],
           [2.99831988],
           [3.91469561]])

    >>> s.plotter()
    >>> plt.show()  # doctest: +SKIP

"""


from functools import partial

import numpy as np
from numpy.typing import ArrayLike

from autora.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, VariableCollection


def template_experiment(
    # Add any configurable parameters with their defaults here:
    name: str = "Template Experiment",
    added_noise: float = 0.1,
    random_state: int = 42,
):
    """
    A template for synthetic experiments.

    Parameters:
        added_noise: standard deviation of gaussian noise added to output
        random_state: seed for random number generator
    """

    params = dict(
        # Include all parameters here:
        name=name,
        added_noise=added_noise,
        random_state=random_state,
    )

    # Define variables
    x = IV(name="Intensity", allowed_values=np.arange(4))
    y = DV(name="Response")
    variables = VariableCollection(
        independent_variables=[x],
        dependent_variables=[y],
    )

    # Define experiment runner
    rng = np.random.default_rng(random_state)

    def experiment_runner(x: ArrayLike, added_noise_=added_noise):
        """A function which simulates noisy observations."""
        x_ = np.array(x)
        y = x_ + 1.0 + rng.normal(0, added_noise_, size=x_.shape)
        return y

    ground_truth = partial(experiment_runner, added_noise_=0.0)
    """A function which simulates perfect observations"""

    def domain():
        """A function which returns all possible independent variable values as a 2D array."""
        x = variables.independent_variables[0].allowed_values.reshape(-1, 1)
        return x

    def plotter(model=None):
        """A function which plots the ground truth and (optionally) a fitted model."""
        import matplotlib.pyplot as plt

        plt.figure()
        x = domain()
        plt.plot(x, ground_truth(x), label="Ground Truth")

        if model is not None:
            plt.plot(x, model.predict(x), label="Fitted Model")

        plt.xlabel(variables.independent_variables[0].name)
        plt.ylabel(variables.dependent_variables[0].name)
        plt.legend()
        plt.title(name)

    # The object which gets stored in the synthetic inventory
    collection = SyntheticExperimentCollection(
        name=name,
        variables=variables,
        experiment_runner=experiment_runner,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        closure=template_experiment,
    )
    return collection
