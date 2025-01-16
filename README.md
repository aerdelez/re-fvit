# fact-ai

## Issues

1. `ViT_LRP.py` and other files import modules with prefix `baselines.ViT.` except it's located in that module. This is correct for running the notebook from the 'root' directory, but not for `imagenet_seg_eval.py` which has the same location as `ViT_LRP.py` and other files. I removed the prefix in this branch since I am trying to run `imagenet_seg_eval.py`. 

Alternative approach: I added some sys.path shenanigans in the `imagenet_set_eval.py` such that we can keep the prefixes and also execute the sctipt :).

2. No code to test classification/segmentation accuracy?

## Fixes

1. Removed package versions from env.yaml file.
2. Removed `backports-zoneinfo` (incompatible with newer Python versions) and `guided-diffusion` (must be installed locally) packages from env.yaml file.
3. Added `modules/`, `utils/` and `/data` modules from https://github.com/hila-chefer/Transformer-Explainability.
4. Added multiple packages to environmment file because of imagenet_seg_eval.py dependencies


## Setup

1. Change working directory to `FViT-main` with `cd FViT-main`.

2. Crate a new Conda environment with `conda env create -f fixed_env.yaml`. This may fail on WSL because `libmpich-dev` or `libopenmpi-dev` are missing; install them if a missing error message appears.

3. Activate the environment with `conda activate fvit` and install `guided-diffusion` with `pip install guided-diffusion/`.

4. Copy the `256x256_diffusion_uncond.pt` model checkpoint from https://github.com/openai/guided-diffusion to the `guided-diffusion/models` directory.

5. Download the ImageNet segmentation dataset (`gtsegs_ijcv.mat`) from https://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat. When running `imagenet_seg_eval.py`, specify path to this file.

## Running the Experiments

Go to the `FViT-main` directory and run `python -W ignore /root/fact/fact-ai/FViT-main/baselines/ViT/imagenet_seg_eval.py --imagenet-seg-path gtsegs_ijcv.mat --method='dds'`. `-W ignore` removes the `UndefinedWarning`s of computing the F1-score. `method='dds` makes the script utilise Denoised Diffusion Smoothing.  
