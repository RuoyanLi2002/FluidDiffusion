# FluidDiffusion

## Main Scripts
- **model.py**: The diffusion model implementation.
- **train.py**: The script used for training the diffusion model.
- **rollout.py**: The script to generate rollout trajectories.
- **render.py**: Renders the rollout trajectories for visualization.

## `data`
- **data_frame.py**: Generates frame data from pickle files.
- **data_sequence.py**: Generates sequence data from pickle files.
- **t2p.py**: Converts TensorFlow data into pickle format.
  
## `recon_loss`
- **model.py**: The model implementation with MSE of reconstructed sample as the loss function.
- **train.py**: The script for training the model with MSE of reconstructed sample as the loss function.

## `InteractionNetwork`
- **model.py**: The original implementation of the Interaction Network model.
- **test.py**: A test script that verifies my implementation against the original Interaction Network model.
  
## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
