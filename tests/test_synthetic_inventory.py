from hypothesis import assume, given
from hypothesis import strategies as st

from autora.synthetic.utilities import SyntheticExperimentCollection, register, retrieve


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
