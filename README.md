# AutoRA Synthetic Data

A package with synthetic experiment data for testing AutoRA theorists and experimentalists.

## User Guide

You will need:

- `python` 3.8 or greater: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Install the synthetic data package:

```shell
pip install -U "autora[synthetic-data]" --pre
```

> 💡We recommend using a `python` environment manager like `virtualenv`.

Print a description of the prospect theory model by Kahneman and Tversky by running:
```shell
python -c "from autora.synthetic.economics.prospect_theory import prospect_theory; print(prospect_theory().description)"
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

New datasets should match existing examples in [`src/autora/synthetic/`](src/autora/synthetic/). 
> 💡A good starting point might be to duplicate an existing example.

Each experiment is described in a single file which includes a "factory function" which:
- constructs the experiment, and 
- optionally takes parameters to tune aspects of the experiment.  

New experiments fulfilling these requirements can be submitted as pull requests.

### Publish the package

This package is published using GitHub actions – create a new "Release" on the GitHub 
repository, and Actions will do the rest.
