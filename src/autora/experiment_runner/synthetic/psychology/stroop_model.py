from functools import partial
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from autora.experiment_runner.synthetic.utilities import SyntheticExperimentCollection
from autora.variable import DV, IV, ValueType, VariableCollection


def stroop_model(
    name="Stroop Model",
    resolution=10,
    temperature=1.0,
    random_state: Optional[int] = None,
):
    """
    Stroop Model

    Args:
        name: name of the experiment
        resolution: number of allowed values for stimulus
        temperature: choice temperature
        random_state: integer used to seed the random number generator
    """

    params = dict(
        name=name,
        resolution=resolution,
        temperature=temperature,
        random_state=random_state,
    )

    color_green = IV(
        name="color_green",
        allowed_values=np.linspace(0, 1, resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Color Green",
        type=ValueType.REAL,
    )

    color_red = IV(
        name="color_red",
        allowed_values=np.linspace(0, 1, resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Color Red",
        type=ValueType.REAL,
    )

    word_green = IV(
        name="word_green",
        allowed_values=np.linspace(0, 1, resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Word GREEN",
        type=ValueType.REAL,
    )

    word_red = IV(
        name="word_red",
        allowed_values=np.linspace(0, 1, resolution),
        value_range=(0, 1),
        units="intensity",
        variable_label="Word RED",
        type=ValueType.REAL,
    )

    task_color = IV(
        name="task_color",
        allowed_values=[0, 1],
        value_range=(0, 1),
        units="intensity",
        variable_label="Color Naming Task",
        type=ValueType.REAL,
    )

    task_word = IV(
        name="task_word",
        allowed_values=[0, 1],
        value_range=(0, 1),
        units="intensity",
        variable_label="Word Reading Task",
        type=ValueType.REAL,
    )

    response_green = DV(
        name="performance",
        value_range=(0, 1),
        units="percentage",
        variable_label="P(Green Response)",
        type=ValueType.PROBABILITY,
    )

    variables = VariableCollection(
        independent_variables=[
            color_green,
            color_red,
            word_green,
            word_red,
            task_color,
            task_word,
        ],
        dependent_variables=[response_green],
    )

    torch.manual_seed(random_state)

    class StroopModel(nn.Module):
        def __init__(self, choice_temperature, std=0.):
            super(StroopModel, self).__init__()

            self.choice_temperature = choice_temperature
            self.std = std

            # define affine transformations
            self.input_color_hidden_color = nn.Linear(2, 2, bias=False)
            self.input_word_hidden_word = nn.Linear(2, 2, bias=False)
            self.hidden_color_output = nn.Linear(2, 2, bias=False)
            self.hidden_word_output = nn.Linear(2, 2, bias=False)
            self.task_hidden_color = nn.Linear(2, 2, bias=False)
            self.task_hidden_word = nn.Linear(2, 2, bias=False)

            self.bias = Variable(torch.ones(1) * -4, requires_grad=False)
            self.input_color_hidden_color.weight.data = (
                    torch.FloatTensor([[1, -1], [-1, 1]]) * 2.2
            )
            self.hidden_color_output.weight.data = (
                    torch.FloatTensor([[1, -1], [-1, 1]]) * 1.3
            )

            self.input_word_hidden_word.weight.data = (
                    torch.FloatTensor([[1, -1], [-1, 1]]) * 2.6
            )
            self.hidden_word_output.weight.data = (
                    torch.FloatTensor([[1, -1], [-1, 1]]) * 2.5
            )

            self.task_hidden_color.weight.data = (
                    torch.FloatTensor([[1.0, 0.0], [1.0, 0]]) * 4
            )
            self.task_hidden_word.weight.data = torch.FloatTensor([[0, 1], [0, 1]]) * 4

        def forward(self, input):

            input = torch.Tensor(input)
            if len(input.shape) <= 1:
                input = input.view(1, len(input))

            # convert inputs
            color = torch.zeros(input.shape[0], 2)
            word = torch.zeros(input.shape[0], 2)
            task = torch.zeros(input.shape[0], 2)

            color[:, 0:2] = input[:, 0:2]
            word[:, 0:2] = input[:, 2:4]
            task[:, 0:2] = input[:, 4:6]

            color_hidden = torch.sigmoid(
                self.input_color_hidden_color(color)
                + self.task_hidden_color(task)
                + self.bias
            )

            word_hidden = torch.sigmoid(
                self.input_word_hidden_word(word) + self.task_hidden_word(task) + self.bias
            )

            output = self.hidden_color_output(color_hidden) + self.hidden_word_output(
                word_hidden
            )

            # add noise
            if self.std > 0:
                output += torch.randn(output.shape) * self.std

            output_softmaxed = torch.exp(output * 1 / self.choice_temperature) / (
                    torch.exp(output[:, 0] * 1 / self.choice_temperature)
                    + torch.exp(output[:, 1] * 1 / self.choice_temperature)
            )

            return output_softmaxed

    def experiment_runner(
        conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
        observation_noise: float = 0.01,
    ):
        X = np.array(conditions)
        Y = np.zeros((X.shape[0], 1))

        # Stroop Model according to
        # Cohen, J. D., Dunbar, K. M., McClelland, J. L., & Rohrer, D. (1990). On the control of automatic processes: a parallel distributed processing account of the Stroop effect. Psychological review, 97(3), 332.
        model = StroopModel(temperature, std=observation_noise)

        for idx, x in enumerate(X):
            # compute regular output
            output_net = model(x).detach().numpy()
            p_choose_A = output_net[0][0]

            Y[idx] = p_choose_A

        return Y

    ground_truth = partial(experiment_runner, observation_noise=0.0)

    def domain():
        s1_values = variables.independent_variables[0].allowed_values
        s2_values = variables.independent_variables[1].allowed_values
        X = np.array(np.meshgrid(s1_values, s2_values)).T.reshape(-1, 2)
        # remove all combinations where s1 > s2
        X = X[X[:, 0] <= X[:, 1]]
        return X

    def plotter(
        model=None,
    ):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        colors = mcolors.TABLEAU_COLORS
        col_keys = list(colors.keys())

        S0_list = [1, 2, 4]
        delta_S = np.linspace(0, 5, 100)

        for idx, S0_value in enumerate(S0_list):
            S0 = S0_value + np.zeros(delta_S.shape)
            S1 = S0 + delta_S
            X = np.array([S0, S1]).T
            y = ground_truth(X)
            plt.plot(
                delta_S,
                y,
                label=f"$S_0 = {S0_value}$ (Original)",
                c=colors[col_keys[idx]],
            )
            if model is not None:
                y = model.predict(X)
                plt.plot(
                    delta_S,
                    y,
                    label=f"$S_0 = {S0_value}$ (Recovered)",
                    c=colors[col_keys[idx]],
                    linestyle="--",
                )

        x_limit = [0, variables.independent_variables[0].value_range[1]]
        y_limit = [0, 2]
        x_label = r"Stimulus Intensity Difference $\Delta S = S_1 - S_0$"
        y_label = "Perceived Intensity of Stimulus $S_1$"

        plt.xlim(x_limit)
        plt.ylim(y_limit)
        plt.xlabel(x_label, fontsize="large")
        plt.ylabel(y_label, fontsize="large")
        plt.legend(loc=2, fontsize="medium")
        plt.title("Stroop Model", fontsize="x-large")

    collection = SyntheticExperimentCollection(
        name=name,
        description=stroop_model.__doc__,
        variables=variables,
        experiment_runner=experiment_runner,
        ground_truth=ground_truth,
        domain=domain,
        plotter=plotter,
        params=params,
        factory_function=stroop_model,
    )
    return collection
