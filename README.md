# AutoRA Synthetic Data

Synthetic experiment data for testing AutoRA theorists and experimentalists. 

## User Guide

You will need:

- `python` 3.8 or greater: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Install the synthetic data package (as part of `autora`):

```shell
pip install -U "autora"
```

> 💡We recommend using a `python` environment manager like `virtualenv`.

Print a description of the prospect theory model by Kahneman and Tversky by running:
```shell
python -c "
from autora.experiment_runner.synthetic.economics.prospect_theory import prospect_theory
study = prospect_theory()
print(study.description)
"
```

For more information, see the
[documentation](https://autoresearch.github.io/autora/user-guide/experiment-runners/synthetic/).
