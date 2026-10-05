# Fashion ANN Pipeline

End-to-end Fashion-MNIST classification pipeline using Git, DVC, TensorFlow, and a Google Drive DVC remote.

The project trains a fully connected artificial neural network to classify 28x28 Fashion-MNIST images into 10 classes. The workflow is designed to be reproducible with:

```bash
dvc repro
```

## Project Layout

```text
src/prepare.py      # Download and save raw Fashion-MNIST arrays
src/preprocess.py   # Normalize data and create train/validation/test splits
src/train.py        # Train the ANN model from params.yaml
src/evaluate.py     # Evaluate model and write metrics
params.yaml         # Hyperparameters and paths
dvc.yaml            # Reproducible DVC pipeline stages
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc repro
```

