# Synthetic Data

Synthetic experiment data for testing AutoRA theorists and experimentalists. 

## Quickstart Guide

You will need:

- `python` 3.8 or greater: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Install the synthetic data package:

```shell
pip install -U "autora"
```

!!! success
    It is recommended to use a `python` environment manager like `virtualenv`.

Print a description of the prospect theory model by Kahneman and Tversky by running:
```shell
python -c "
from autora.experiment_runner.synthetic.economics.prospect_theory import prospect_theory
study = prospect_theory()
print(study.description)
"
```
