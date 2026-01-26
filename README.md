## LIFT: Likelihood-Based Intervention Faithfulness Training

### Requirements

A Python environment with `pip` is required. PyTorch must already be installed on your system.

### Installation and Configuration

Clone the repository and install the dependencies:

```bash
pip -r /path/to/lift/requirements.txt
```

Before running any code, edit `config.py` to configure hyperparameters and paths. In particular, verify directory settings such as `DATA_DIR`.

NOTE: The combination `MISTRAL=True` and `LARGE=False` has not been implemented.

## Model Fine-Tuning

LIFT supports two approaches to fine-tuning: running the full pipeline or training from a pre-generated dataset.

### End-to-End LIFT Pipeline

To run the complete pipeline (this requires more compute and time), execute:

```bash
cd /path/to/lift
python pipeline.py
```

This command runs the entire LIFT process end-to-end. The final fine-tuned model will be saved to a subdirectory inside `config.CHECKPOINTS_DIR`.

### Fine-Tuning with a Pre-Generated Dataset

As an alternative, you can fine-tune using the released SFT examples. This option requires significantly less computational power, but the data-generation parameters cannot be changed.

First, set the hyperparameters in `config.py` to match the values reported in the paper. Ensure the correct model is selected. Then run:

```bash
cd /path/to/lift
python finetune-only.py
```

The resulting model will be saved to a directory under `config.CHECKPOINTS_DIR`.

## Evaluation

After producing a fine-tuned model, evaluate it by running:

```bash
python evaluation.py [OPTIONS]
```

Refer to the `main` function in `evaluation.py` for details on the available command-line options.
