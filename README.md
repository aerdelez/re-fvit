# fact-ai

## Fixes

1. Removed package versions from env.yaml file.
2. Removed `backports-zoneinfo` (incompatible with newer Python versions) and `guided-diffusion` (must be installed locally) packages from env.yaml file.
3. Added `modules/` module from https://github.com/hila-chefer/Transformer-Explainability.

## Setup

1. Change working directory to `FViT-main` with `cd FViT-main`.

2. Crate a new Conda environment with `conda env create -f fixed_env.yml`. This may fail on WSL because `libmpich-dev` or `libopenmpi-dev` are missing; install them if a missing error message appears.

3. Activate the environment with `conda activate fvit` and install `guided-diffusion` with `pip install guided-diffusion/`.

4. Copy the `256x256_diffusion_uncond.pt` model checkpoint from https://github.com/openai/guided-diffusion to the `guided-diffusion/models` directory.
