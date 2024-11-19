# Automated Identification of Opinion Poliary

## Getting Started

To run any of the Python code in this directory, you must use Python's module package functionality.
This allows the `experiments` directory to pull in source from the `src` directory.

An example would be

```bash
python -m experiments.feature_sets
```

## Library Dependencies

To ensure expected behaviour, ensure your runtime has access to the following libraries *at minimum*.

```
pytorch
transformers
accelerate
sklearn
os
nltk
numpy
scipy
```

To conduct experiments, ensure your runtime has access to the following libraries.

```
matplotlib
```

## Directory Structure

The source code for the main deliverables of this project is all found in the `src` directory.

We have also included the `experiments` directory for those who would like to replicate our experiments and
generate the figures found in the main paper accompanying this project.