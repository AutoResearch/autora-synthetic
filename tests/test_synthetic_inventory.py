from hypothesis import assume, given
from hypothesis import strategies as st

from autora.synthetic import SyntheticExperimentCollection, describe, register, retrieve
from autora.variable import VariableCollection

all_bundled_model_names = [
    "expected_value",
    "prospect_theory",
    "template_experiment",
    "weber_fechner",
]


@given(st.text(), st.text())
def test_model_registration_retrieval(name1, name2):
    # We can register a model and retrieve it
    assume(name1 != name2)

    register(name1, lambda: SyntheticExperimentCollection())
    empty = retrieve(name1)
    assert empty.name is None

    # We can register another model and retrieve it as well
    register(
        name2,
        lambda: SyntheticExperimentCollection(variables=VariableCollection()),
    )
    only_variables = retrieve(name2)
    assert only_variables.variables is not None

    # We can still retrieve the first model, and it is equal to the first version
    empty_copy = retrieve(name1)
    assert empty_copy == empty


@given(st.sampled_from(all_bundled_model_names))
def test_bundled_model_retrieval(name):
    retrieve(name)


@given(st.sampled_from(all_bundled_model_names))
def test_bundled_model_description(name):
    describe(retrieve(name))
