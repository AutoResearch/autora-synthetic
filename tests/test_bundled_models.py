from hypothesis import given
from hypothesis import strategies as st

from autora.synthetic.abstract.template_experiment import template_experiment
from autora.synthetic.economics.expected_value_theory import expected_value_theory
from autora.synthetic.economics.prospect_theory import prospect_theory
from autora.synthetic.psychophysics.weber_fechner_law import weber_fechner_law
from autora.synthetic.utilities import describe, register, retrieve

all_bundled_models = [
    ("expected_value_theory", expected_value_theory),
    ("prospect_theory", prospect_theory),
    ("template_experiment", template_experiment),
    ("weber_fechner_law", weber_fechner_law),
]

all_bundled_model_names = [b[0] for b in all_bundled_models]

for name, func in all_bundled_models:
    register(name, func)


@given(name=st.sampled_from(all_bundled_model_names))
def test_bundled_models_can_be_retrieved_by_name(name):
    model = retrieve(name)
    assert model is not None


@given(name=st.sampled_from(all_bundled_model_names))
def test_bundled_models_can_be_described_by_name(name):
    description = describe(name)
    assert isinstance(description, str)


@given(name=st.sampled_from(all_bundled_model_names))
def test_bundled_models_can_be_described_by_model(name):
    model = retrieve(name)
    description = describe(model)
    assert isinstance(description, str)


@given(name=st.sampled_from(all_bundled_model_names))
def test_model_descriptions_from_name_model_closure_are_the_same(name):
    description_from_name = describe(name)
    description_from_model = describe(retrieve(name))
    description_from_closure = describe(retrieve(name).closure)

    assert description_from_name == description_from_model
    assert description_from_model == description_from_closure
    assert description_from_closure == description_from_name
