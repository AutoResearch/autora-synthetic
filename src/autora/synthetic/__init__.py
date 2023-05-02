"""
Provides an interface for loading and saving synthetic experiments.

Examples:
    >>> from autora.synthetic import retrieve, describe
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    The registry is accessed using the `retrieve` function, optionally setting parameters. Here
    we load the Weber-Fechner law:
    >>> s = retrieve("weber_fechner", rng=np.random.default_rng(seed=180))

    Use the describe function to give information about the synthetic experiment:
    >>> describe(s) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Weber-Fechner Law...

    The synthetic experiement `s` has properties like the name of the experiment:
    >>> s.name
    'Weber-Fechner Law'

    ... a valid variables description:
    >>> s.variables  # doctest: +ELLIPSIS
    VariableCollection(...)

    ... a function to generate the full domain of the data (if possible)
    >>> x = s.domain()
    >>> x   # doctest: +ELLIPSIS
    array([[0...]])

    ... the experiment_runner runner which can be called to generate experimental results:
    >>> import numpy as np
    >>> y = s.experiment_runner(x)  # doctest: +ELLIPSIS
    >>> y
    array([[ 0.00433955],
           [ 1.79114625],
           [ 2.39473454],
           ...,
           [ 0.00397802],
           [ 0.01922405],
           [-0.00612883]])

    ... a function to plot the ground truth:
    >>> s.plotter()
    >>> plt.show()  # doctest: +SKIP

    ... against a fitted model if it exists:
    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression().fit(x, y)
    >>> s.plotter(model)
    >>> plt.show()  # doctest: +SKIP

"""

from autora.synthetic import data
from autora.synthetic.inventory import (
    Inventory,
    SyntheticExperimentCollection,
    describe,
    register,
    retrieve,
)
