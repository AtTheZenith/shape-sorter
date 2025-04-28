# Shape Sorter

An neural network made to classify and sort digitally drawn shapes.

## Key Features

-   Neural Network Classification: Utilizes a trained neural network to accurately identify and categorize various shapes.
-   Interactive Interface: Provides a user-friendly interface for drawing and testing shape recognition.
-   Jupyter Notebook Demonstrations: Includes notebooks that showcase the model's capabilities and usage examples.

## Installation Guide

### Using Python 3.12

#### Step 1

```bash
    git clone https://github.com/AtTheZenith/shape-sorter/
    cd shape-sorter
```

#### Step 2

```bash
    python -m venv
    .venv/Scripts/activate
```

#### Step 3

```bash
    python install -r requirements.txt
    python ./src/model_loader.py
```

### Using [uv](https://github.com/astral-sh/uv):

#### Step 1

```bash
    git clone https://github.com/AtTheZenith/shape-sorter/
    cd shape-sorter
```

#### Step 2

```bash
    uv sync
    uv run ./src/model_loader.py
```
