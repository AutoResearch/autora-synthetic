# AutoRA Synthetic Experiments

A package with synthetic experiment data for testing AutoRA theorists and experimentalists.

## User Guide

You will need:

- `python` 3.8 or greater: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Install the synthetic data package:

```shell
pip install -U "autora-synthetic-data"
```

!!! success
    It is recommended to use a `python` environment manager like `virtualenv`.

Check your installation by running:
```shell
python -c "from autora.synthetic import retrieve, describe; describe(retrieve('weber_fechner'))"
```

## Developer Guide

### Get started

Clone the repository (e.g. using [GitHub desktop](https://desktop.github.com), 
or the [`gh` command line tool](https://cli.github.com)) 
and install it in "editable" mode in an isolated `python` environment, (e.g. 
with 
[virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)) as follows:

In the repository root, create a new virtual environment:
```shell
virtualenv venv
```

Activate it:
```shell
source venv/bin/activate
```

Use `pip install` to install the current project (`"."`) in editable mode (`-e`) with dev-dependencies (`[dev]`):
```shell
pip install -e ".[dev]"
```

Run the test cases:
```shell
pytest --doctest-modules
```

Activate the pre-commit hooks:
```shell
pre-commit install
```

### Add new datasets

- First, get to know the existing examples and how to use them with the documentation in 
[`src/autora/synthetic/`](`src/autora/synthetic/`).
- Duplicate the 
  [`template_experiment`](`src/autora/synthetic/data/template_experiment.py`) or another 
  existing experiment in [`src/autora/synthetic/data`](`src/autora/synthetic/data`).
- Ensure that the `register` function at the bottom of the file is updated with the experiment's 
  `id` (can't be the same as any other experiment) and the updated experiment generating 
  function.
- Make sure the file is imported in
  [`src/autora/synthetic/data/__init__.py`](`src/autora/synthetic/data/__init__.py`).
- Check that the new experiment can be retrieved using the `retrieve` function.
- Add code to the template as required. 

### Add new dependencies 

In pyproject.toml add the new dependencies under `dependencies`

Install the added dependencies:
```shell
pip install -e ".[dev]"
```

### Publish the package

Update the metadata under `project` in the pyproject.toml file to include name, description, author-name, author-email and version

- Follow the guide here: [https://packaging.python.org/en/latest/tutorials/packaging-projects/](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

Build the package using:
```shell
python -m build
```

Publish the package to PyPI using `twine`:
```shell
twine upload dist/*
```
