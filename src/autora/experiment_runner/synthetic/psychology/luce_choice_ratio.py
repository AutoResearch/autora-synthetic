from functools import partial

import numpy as np

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, ValueType, VariableCollection


def luce_choice_ratio(
    name="Luce-Choice-Ratio",
    added_noise=0.01,
    resolution=8,
    maximum_similarity=10,
    focus=0.8,
    rng=np.random.default_rng(),
):
    """
    Luce-Choice-Ratio

    Args:
        name: name of the experiment
        added_noise: standard deviation of normally distributed noise added to y-values
        resolution: number of allowed values for stimulus DVs
        maximum_similarity: upperbound for DVs
        focus: parameter measuring participant focus
        rng: integer used to seed the random number generator

    Shepard-Luce Choice Rule according to:
        - Equation (4) in Logan, G. D., & Gordon, R. D. (2001).
        - and in Executive control of visual attention in dual-task situations.
            Psychological review, 108(2), 393.
        - Equation (5) in Luce, R. D. (1963). Detection and recognition.

    Examples:
        First we seed numpy to get replicable results:
        >>> np.random.seed(42)

        We can instantiate a Shepard-Cue Choice Experiment. We use a seed to get replicable results:
        >>> l_s_experiment = luce_choice_ratio(rng=42)

        We can look at the name of the experiment:
        >>> l_s_experiment.name
        'Luce-Choice-Ratio'

        To call the ground truth, we can use an attribute of the experiment:
        >>> l_s_experiment.ground_truth(np.array([[1,2,3,4]]))
        array([[0.21052632]])

        We can also run an experiment:
        >>> l_s_experiment.experiment_runner(np.array([[1,2,3,4]]))
        array([[0.21016246]])

        To plot the experiment use:
        >>> l_s_experiment.plotter()
        >>> plt.show()  # doctest: +SKIP

    """
    minimum_similarity = 1 / maximum_similarity

    params = dict(
        name=name,
        added_noise=added_noise,
        maximum_similarity=maximum_similarity,
        minimum_similarity=minimum_similarity,
        resolution=resolution,
        focus=focus,
        rng=rng,
    )

    similarity_category_A1 = IV(
        name="similarity_category_A1",
        allowed_values=np.linspace(minimum_similarity, maximum_similarity, resolution),
        value_range=(minimum_similarity, maximum_similarity),
        units="similarity",
        variable_label="Similarity with Category A1",
        type=ValueType.REAL,
    )

    similarity_category_A2 = IV(
        name="similarity_category_A2",
        allowed_values=np.linspace(minimum_similarity, maximum_similarity, resolution),
        value_range=(minimum_similarity, maximum_similarity),
        units="similarity",
        variable_label="Similarity with Category A2",
        type=ValueType.REAL,
    )

    similarity_category_B1 = IV(
        name="similarity_category_B1",
        allowed_values=np.linspace(minimum_similarity, maximum_similarity, resolution),
        value_range=(minimum_similarity, maximum_similarity),
        units="similarity",
        variable_label="Similarity with Category B1",
        type=ValueType.REAL,
    )

    similarity_category_B2 = IV(
        name="similarity_category_B2",
        allowed_values=np.linspace(minimum_similarity, maximum_similarity, resolution),
        value_range=(minimum_similarity, maximum_similarity),
        units="similarity",
        variable_label="Similarity with Category B2",
        type=ValueType.REAL,
    )

    choose_A1 = DV(
        name="choose_A1",
        value_range=(0, 1),
        units="probability",
        variable_label="Probability of Choosing A1",
        type=ValueType.PROBABILITY,
    )

    variables = VariableCollection(
        independent_variables=[
            similarity_category_A1,
            similarity_category_A2,
            similarity_category_B1,
            similarity_category_B2,
        ],
        dependent_variables=[choose_A1],
    )

    def experiment_runner(
        X: np.ndarray,
        focus_: float = focus,
        added_noise_: float = added_noise,
    ):
        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            similarity_A1 = x[0]
            similarity_A2 = x[1]
            similarity_B1 = x[2]
            similarity_B2 = x[3]

            y = (similarity_A1 * focus + np.random.normal(0, added_noise_)) / (
                similarity_A1 * focus
                + similarity_A2 * focus
                + similarity_B1 * (1 - focus_)
                + similarity_B2 * (1 - focus_)
            )
            # probability can't be negative or larger than 1 (the noise can make it so)
            if y <= 0:
                y = 0.0001
            elif y >= 1:
                y = 0.9999
            Y[idx] = y

        return Y

    ground_truth = partial(experiment_runner, added_noise_=0.0)

    def domain():
        similarity_A1 = variables.independent_variables[0].allowed_values
        similarity_A2 = variables.independent_variables[1].allowed_values
        similarity_B1 = variables.independent_variables[2].allowed_values
        similarity_B2 = variables.independent_variables[3].allowed_values

        X = np.array(
            np.meshgrid(
                similarity_A1,
                similarity_A2,
                similarity_B1,
                similarity_B2,
            )
        ).T.reshape(-1, 4)

        # remove all conditions from X where the focus is 0 and the similarity of A1 is 0
        # or the similarity of A2 is 0
        X = X[~((X[:, 0] == 0) & (X[:, 1] == 0) & (X[:, 2] == 0) & (X[:, 3] == 0))]
        return X

    def plotter(
        model=None,
    ):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        similarity_A1 = np.linspace(
            variables.independent_variables[0].value_range[0],
            variables.independent_variables[0].value_range[1],
            100,
        )

        similarity_A2 = 0.5  # 1 - similarity_A1

        similarity_B1_list = [0.5, 0.75, 1]
        similarity_B2 = 0

        colors = mcolors.TABLEAU_COLORS
        col_keys = list(colors.keys())
        for idx, similarity_B1 in enumerate(similarity_B1_list):
            # similarity_B2 = 1 - similarity_B1
            X = np.zeros((len(similarity_A1), 4))

            X[:, 0] = similarity_A1
            X[:, 1] = similarity_A2
            X[:, 2] = similarity_B1
            X[:, 3] = similarity_B2

            y = ground_truth(X)
            plt.plot(
                similarity_A1.reshape((len(similarity_A1), 1)),
                y,
                label=f"Similarity to B1 = {similarity_B1} (Original)",
                c=colors[col_keys[idx]],
            )

            if model is not None:
                y = model.predict(X)
                plt.plot(
                    similarity_A1,
                    y,
                    label=f"Similarity to B1 = {similarity_B1} (Recovered)",
                    c=colors[col_keys[idx]],
                    linestyle="--",
                )

        x_limit = [np.min(similarity_A1), np.max(similarity_A1)]
        y_limit = [0, 1]
        x_label = "Similarity to Category A1"
        y_label = "Probability of Selecting Category A1"

        plt.xlim(x_limit)
        plt.ylim(y_limit)
        plt.xlabel(x_label, fontsize="large")
        plt.ylabel(y_label, fontsize="large")
        plt.legend(loc=4, fontsize="medium")
        plt.title("Shepard-Luce Choice Ratio", fontsize="x-large")

    collection = SyntheticExperimentCollection(
        name=name,
        description=luce_choice_ratio.__doc__,
        variables=variables,
        experiment_runner=experiment_runner,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=luce_choice_ratio,
    )
    return collection
