# AutoRA Synthetic Experiments

A package with synthetic experiment data for testing AutoRA theorists and experimentalists.

## User Guide

You will need:

- `python` 3.8 or greater: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Install the synthetic data package:

```shell
pip install -U "autora-synthetic-data"
```

> 💡We recommend using a `python` environment manager like `virtualenv`.

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

### Add a new dataset

- First, get to know the existing examples and how to use them with the documentation in 
[`src/autora/synthetic/`](src/autora/synthetic/).
- Duplicate the 
  [`template_experiment`](src/autora/synthetic/data/template_experiment.py) or another 
  existing experiment in [`src/autora/synthetic/data`](src/autora/synthetic/data).
- Ensure that the `register` function at the bottom of the file is updated with the experiment's 
  `id` (can't be the same as any other experiment) and the updated experiment generating 
  function.
- Ensure that the docstring of the constructing function is updated with a description of the 
  experiment. 
- Make sure the file is imported in
  [`src/autora/synthetic/data/__init__.py`](src/autora/synthetic/data/__init__.py).
- Check that the new experiment can be retrieved using the `retrieve` function like this:
  ```shell
  python -c "from autora.synthetic import retrieve, describe; describe(retrieve('new_experiment_name'))"
  ```
- Add code to the template as required.

New experiments can be submitted as pull requests.

### Publish the package

This package can be published using GitHub actions – create a new "Release" on the GitHub 
repository, and Actions will do the rest.