from functools import partial
from typing import Optional

import numpy as np

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, ValueType, VariableCollection


def stevens_power_law(
    name="Stevens' Power Law",
    resolution=100,
    proportionality_constant=1.0,
    modality_constant=0.8,
    maximum_stimulus_intensity=5.0,
    added_noise=0.01,
    random_state: Optional[int] = None,
):
    """
    Stevens' Power Law

    Args:
        name: name of the experiment
        resolution: number of allowed values for stimulus
        modality_constant: power constant
        proportionality_constant: constant multiplier
        maximum_stimulus_intensity: maximum value for stimulus
        added_noise: standard deviation of normally distributed noise added to y-values
        random_state: integer used to seed the random number generator

    """

    params = dict(
        added_noise=added_noise,
        name=name,
        resolution=resolution,
        proportionality_constant=proportionality_constant,
        modality_constant=modality_constant,
        maximum_stimulus_intensity=maximum_stimulus_intensity,
        random_state=random_state,
    )

    iv1 = IV(
        name="S",
        allowed_values=np.linspace(1 / resolution, maximum_stimulus_intensity, resolution),
        value_range=(1 / resolution, maximum_stimulus_intensity),
        units="intensity",
        variable_label="Stimulus Intensity",
        type=ValueType.REAL
    )

    dv1 = DV(
        name="perceived_intensity",
        value_range=(0, maximum_stimulus_intensity),
        units="sensation",
        variable_label="Perceived Intensity",
        type=ValueType.REAL
    )

    variables = VariableCollection(
        independent_variables=[iv1],
        dependent_variables=[dv1],
    )

    rng = np.random.default_rng(random_state)

    def experiment_runner(
        X: np.ndarray,
        std: float = 0.01,
    ):
        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            y = proportionality_constant * x[0] ** modality_constant + rng.random.normal(0, std)
            Y[idx] = y

        return Y

    ground_truth = partial(experiment_runner, std=0.0)

    def domain():
        s_values = variables.independent_variables[0].allowed_values

        X = np.array(np.meshgrid(s_values)).T.reshape(-1, 1)
        return X

    def plotter(
        model=None,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        colors = mcolors.TABLEAU_COLORS
        col_keys = list(colors.keys())
        X = domain()
        y = ground_truth(X)
        plt.plot(X, y, label="Original", c=colors[col_keys[0]])
        if model is not None:
            y = model.predict(X)
            plt.plot(X, y, label=f"Recovered", c=colors[col_keys[0]], linestyle="--")
        x_limit = [0, variables.independent_variables[0].value_range[1]]
        y_limit = [0, 4]
        x_label = "Stimulus Intensity"
        y_label = "Perceived Stimulus Intensity"

        plt.xlim(x_limit)
        plt.ylim(y_limit)
        plt.xlabel(x_label, fontsize="large")
        plt.ylabel(y_label, fontsize="large")
        plt.legend(loc=2, fontsize="medium")
        plt.title("Stevens' Power Law", fontsize="x-large")
        plt.show()

    collection = SyntheticExperimentCollection(
        name=name,
        description=stevens_power_law.__doc__,
        variables=variables,
        experiment_runner=experiment_runner,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=stevens_power_law,
    )
    return collection
