from hypothesis import assume, given
from hypothesis import strategies as st

from autora.synthetic import (
    Inventory,
    SyntheticExperimentCollection,
    describe,
    register,
    retrieve,
)

all_bundled_model_names = [
    "expected_value",
    "prospect_theory",
    "template_experiment",
    "weber_fechner",
]


@given(st.text())
def test_model_registration_retrieval_allows_any_string(name):
    model = SyntheticExperimentCollection()
    register(name, lambda: model)

    retrieved = retrieve(name)

    assert retrieved is model


@given(st.text(), st.text())
def test_model_registration_retrieval_dont_collide_with_two_models(name1, name2):
    # We can register a model and retrieve it
    assume(name1 != name2)

    model1 = SyntheticExperimentCollection()
    model2 = SyntheticExperimentCollection()
    register(name1, lambda: model1)
    retrieved1 = retrieve(name1)
    assert retrieved1 is model1

    # We can register another model and retrieve it as well
    register(name2, lambda: model2)
    retrieved2 = retrieve(name2)
    assert retrieved2 is model2

    # We can still retrieve the first model, and it is equal to the first version
    retrieved3 = retrieve(name1)
    assert retrieved3 is model1


@given(st.sampled_from(all_bundled_model_names))
def test_bundled_models_can_be_retrieved_by_name(name):
    model = retrieve(name)
    assert model is not None


@given(st.sampled_from(all_bundled_model_names))
def test_bundled_models_can_be_described_by_name(name):
    description = describe(name)
    assert isinstance(description, str)


@given(st.sampled_from(all_bundled_model_names))
def test_bundled_models_can_be_described_by_model(name):
    model = retrieve(name)
    description = describe(model)
    assert isinstance(description, str)


@given(st.sampled_from(all_bundled_model_names))
def test_model_descriptions_from_name_model_closure_are_the_same(name):
    description_from_name = describe(name)
    description_from_model = describe(retrieve(name))
    closure = Inventory[name]
    description_from_closure = describe(closure)

    assert description_from_name == description_from_model
    assert description_from_model == description_from_closure
    assert description_from_closure == description_from_name
