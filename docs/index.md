# Synthetic Data

Synthetic experiment data for testing AutoRA theorists and experimentalists. 

## Quickstart Guide

You will need:

- `python` 3.8 or greater: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Install the synthetic data package:

```shell
pip install -U "autora[synthetic-data]" --pre
```

!!! success
    It is recommended to use a `python` environment manager like `virtualenv`.

Check your installation by running:
```shell
python -c "from autora.synthetic import retrieve, describe; describe(retrieve('weber_fechner'))"
```
