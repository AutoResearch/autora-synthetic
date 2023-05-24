# AutoRA Synthetic Data

Synthetic experiment data for testing AutoRA theorists and experimentalists. 

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

