# Project Overview

This project involves constructing patches, training and testing a model (GraphFormer), and providing interpretation of the results.

## Directory Structure

- `patch_construction/`: Contains code for constructing patches and generating nodes.
  - `EfficientNet/`: EfficientNet model used for feature extraction.
  - `graph_construction.py`: Code for constructing graphs from patches.
  - `node_generation.py`: Code for generating nodes.
- `exp/`: Contains experiments for training and testing the model.
- `models/`: Contains model implementations.
  - `graphformer.py`: The proposed GraphFormer model.
  - Other model variants and utilities.
- `interpretation/`: Contains code for model interpretation (e.g., Integrated Gradients).
- `utils/`: Utility functions and helper code.
- `requirements.txt`: List of dependencies.

## Workflow

1. **Patch Construction**: 
   - Use scripts in `patch_construction/` to extract features and construct graphs from raw data.
   - Outputs are saved for use in the experiments.

2. **Experiments (Training and Testing)**:
   - Located in the `exp/` directory.
   - Scripts here load the constructed patches, train the GraphFormer model (or other baselines), and evaluate performance.

3. **Model**:
   - The main model of interest is `GraphFormer` in `models/graphformer.py`.
   - Other models in `models/` are for comparison or ablation studies.
   - For detailed network parameters, see [NETWORK_PARAMETERS.md](NETWORK_PARAMETERS.md).

4. **Interpretation**:
   - The `interpretation/` directory contains scripts for explaining model predictions (e.g., using Integrated Gradients).
   - Tools include visualization and attribution methods.

5. **Utilities**:
   - The `utils/` directory contains helper functions used across the project.

## Installation

1. Clone the repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run patch construction:
   ```bash
   # Example (adjust according to actual scripts)
   cd patch_construction
   python construct.py
   ```

2. Run experiments:
   ```bash
   cd ../exp
   python train.py  # or test.py, depending on the setup
   ```

3. Run interpretation:
   ```bash
   cd ../interpretation
   python ig_cal_main.py  # or other interpretation scripts
   ```

## Notes

- Ensure that the output of each step is correctly placed for the next step (e.g., patch construction outputs are used by the experiment scripts).
- Check individual script headers for specific command-line arguments and configurations.
