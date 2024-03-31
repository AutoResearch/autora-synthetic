from functools import partial
from typing import Optional

import numpy as np
import pandas as pd

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import VariableCollection


def data_experiment(
        data: pd.DataFrame,
        variables: VariableCollection,
        step_size: Optional[int] = 1,
        name: str = "Experiment from data",
        random_state: Optional[int] = None,
):
    """
    A synthetic experiments for preexisting experiment data.

    For preexisting data in form of a pd.DataFrame, extract rows of data and use them as experiment.

    Step_size is used when simulating the data in the same order as given, step by step.

    If step_size is set to None, we can also use conditions to extract observations. This can be
    used to simulate the experiment in a different order.

    Parameters:
        data: The data to use. Contains columns of independent and dependent variables.
        variables: VariableCollection of independent and dependent variables. Contains description
            of independent and dependent variables.
        step_size: How many observation to extract. If this is set to None, the runner will
            match conditionns instead
        name: Name of the experiment
        random_state: Seed for random number generator

    Examples:
        >>> from autora.variable import Variable

        We need to declare all variables used in the experiment.
        >>> x_1 = Variable('x_1', value_range=(0, 1))
        >>> x_2 = Variable('x_2', value_range=(0, 1))
        >>> y = Variable('y', value_range=(0, 1))

        And create a VariableCollection
        >>> v_collection = VariableCollection(
        ...     independent_variables=[x_1, x_2], dependent_variables=[y]
        ... )

        We also need preexisting experiment data. Here, we use randomly generated data. For
        reproducible results, we use a random seed.
        >>> np.random.seed(42)
        >>> data_preexisting = pd.DataFrame({
        ...     'x_1': np.random.rand(20),
        ...     'x_2': np.random.rand(20),
        ...     'y': np.random.rand(20)
        ... })
        >>> data_preexisting
                 x_1       x_2         y
        0   0.374540  0.611853  0.122038
        1   0.950714  0.139494  0.495177
        2   0.731994  0.292145  0.034389
        3   0.598658  0.366362  0.909320
        4   0.156019  0.456070  0.258780
        5   0.155995  0.785176  0.662522
        6   0.058084  0.199674  0.311711
        7   0.866176  0.514234  0.520068
        8   0.601115  0.592415  0.546710
        9   0.708073  0.046450  0.184854
        10  0.020584  0.607545  0.969585
        11  0.969910  0.170524  0.775133
        12  0.832443  0.065052  0.939499
        13  0.212339  0.948886  0.894827
        14  0.181825  0.965632  0.597900
        15  0.183405  0.808397  0.921874
        16  0.304242  0.304614  0.088493
        17  0.524756  0.097672  0.195983
        18  0.431945  0.684233  0.045227
        19  0.291229  0.440152  0.325330

        Now, we can initialize an experiment runner. The step size determines how many rows of
        the data we use each time we run the runner. Here, we also use a random seed. This seed
        will be used if we later want to add noise to the observations
        >>> runner = data_experiment(
        ...     data=data_preexisting,
        ...     variables=v_collection,
        ...     step_size=4,
        ...     random_state=42
        ... )

        To run the experiment runner, we need to set a cycle. This tells the runner which row is
        the starting row for the experiment. For example, the first cycle will return the first four
        rows since the step_size was set to 4.
        >>> runner.run(0)
                x_1       x_2         y
        0  0.374540  0.611853  0.122038
        1  0.950714  0.139494  0.495177
        2  0.731994  0.292145  0.034389
        3  0.598658  0.366362  0.909320

        If we set the cycle to 1, we will get row 4 to 7:
        >>> runner.run(1)
                x_1       x_2         y
        4  0.156019  0.456070  0.258780
        5  0.155995  0.785176  0.662522
        6  0.058084  0.199674  0.311711
        7  0.866176  0.514234  0.520068

        We can also add aditional noise:
        >>> runner.run(1, 0.01)
                x_1       x_2         y
        4  0.156019  0.456070  0.258780
        5  0.155995  0.785176  0.662522
        6  0.058084  0.199674  0.311711
        7  0.866176  0.514234  0.520068

        If we set the step_size in the runner to None, we can also use conditions and find matching
        observations.
        >>> runner_on_conditions = data_experiment(
        ...     data=data_preexisting,
        ...     variables=v_collection,
        ...     step_size=None,
        ...     random_state=42
        ... )

        Here, we select conditions from row 6 and 13
        >>> c_to_simulate = pd.DataFrame({
        ...     'x_1': data_preexisting['x_1'][[6, 13]],
        ...     'x_2': data_preexisting['x_2'][[6, 13]]
        ... })
        >>> c_to_simulate
                 x_1       x_2
        6   0.058084  0.199674
        13  0.212339  0.948886

        Then, we can run the runner on these conditions:
        >>> runner_on_conditions.run(conditions=c_to_simulate)
                x_1       x_2         y
        0  0.058084  0.199674  0.311711
        1  0.212339  0.948886  0.894827

        Warning. If conditions are used that are not in the experiment data, this will return an
        empty DataFrame
        >>> c_not_existing = pd.DataFrame({
        ...     'x_1': [0, .1, .2, .3],
        ...     'x_2': [0, .1, .2, .3]
        ... })
        >>> c_not_existing
           x_1  x_2
        0  0.0  0.0
        1  0.1  0.1
        2  0.2  0.2
        3  0.3  0.3

        >>> runner_on_conditions.run(conditions=c_not_existing)
        Empty DataFrame
        Columns: [x_1, x_2, y]
        Index: []

        If the conditions exist multiple time in the preexisting data, all the duplicates will be
        returned
        >>> data_preexisting_duplicates = pd.DataFrame({
        ...     'x_1': [.1, .2, .1, .3],
        ...     'x_2': [.1, .3, .1, .2],
        ...     'y': [.1, .2, .3, .4]
        ... })
        >>> runner_on_conditions_with_duplicates = data_experiment(
        ...     data=data_preexisting_duplicates,
        ...     variables=v_collection,
        ...     step_size=None,
        ...     random_state=42
        ... )
        >>> c_with_duplicates = pd.DataFrame({
        ...     'x_1': [.1],
        ...     'x_2': [.1]
        ... })
        >>> runner_on_conditions_with_duplicates.run(conditions=c_with_duplicates)
           x_1  x_2    y
        0  0.1  0.1  0.1
        1  0.1  0.1  0.3



    """

    params = dict(
        # Include all parameters here:
        data=data,
        variables=variables,
        step_size=step_size,
        name=name,
        random_state=random_state
    )

    # Define experiment runner
    rng = np.random.default_rng(random_state)

    def run(
            cycle: Optional[int] = None,
            conditions: Optional[pd.DataFrame] = None,
            added_noise: Optional[float] = None,
            random_state: Optional[int] = None
    ):
        """
        A function which simulates
        """
        if cycle is None and conditions is None:
            raise Exception("Either conditions or cycle has to be defined.")
        if step_size is not None:
            if cycle is None:
                raise Exception("Step_size in runner is defined, but no cycle is given.")
            _data = data.copy()[cycle * step_size: (cycle + 1) * step_size]
            if not added_noise:
                return _data
            if random_state:
                _rng = np.random.default_rng(random_state)
            else:
                _rng = rng
            for dv in variables.dependent_variables:
                _data[dv.name] += _rng.normal(0, added_noise, size=len(_data))
            return _data
        match_columns = [iv.name for iv in variables.independent_variables]
        _data = pd.merge(data, conditions[match_columns], on=match_columns, how='inner')
        return _data

    ground_truth = partial(run, added_noise=0.0)
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
        description=data_experiment.__doc__,
        variables=variables,
        run=run,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=data_experiment,
    )
    return collection
