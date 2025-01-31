# fact-ai

## Code Fixes

1. Removed package versions from env.yaml file to make compatible.
2. Removed `backports-zoneinfo` (incompatible with newer Python versions) and `guided-diffusion` (must be installed locally) packages from env.yaml file.
3. Added `modules/`, `utils/` and other modules from https://github.com/hila-chefer/Transformer-Explainability since the scripts import them.
4. Added multiple packages to environmment file because of imagenet_seg_eval.py dependencies.
5. Edited tasks scripts to allow imports to work for themselves as well as the demo.
6. Small method fixes (e.g. ImageNet class use in generate_visualizations)

## Layout

`FViT-main` is a clone of the `main` branch of the https://github.com/kaustpradalab/FViT repository at commit `860b45f`. It appears that the authors obtained the `baselines/ViT` directory from https://github.com/hila-chefer/Transformer-Explainability and extended it by adding a qualitative demo `fvit-demo.ipynb`. In order to run the demo as well as the segmentation under adversarial attack and classification under perturbation tasks, we applied the aforementioned fixes. We also crated a DDS class (DDS.py) now used in both task scripts and added Attribution Rollout to the image segmentation script as another method.


## Setting up the Environment

1. Change working directory to `FViT-main` with `cd FViT-main`. This is important for correct imports of modules when running the scripts with a shell.

2. Crate a new Conda environment with `conda env create -f fixed_env.yaml`. This may fail on WSL because `libmpich-dev` or `libopenmpi-dev` are missing; install them if a missing error message appears.

3. Activate the environment with `conda activate fvit` and install `guided-diffusion` with `pip install guided-diffusion/`.

4. Copy the `256x256_diffusion_uncond.pt` model checkpoint from https://github.com/openai/guided-diffusion to the `guided-diffusion/models` directory.


## Running the Qualitative Demo

- Open `fvit-demo.ipynb` and run the notebok with the `fvit` environment.


## Running the Image Segmentation Task

1. Download the ImageNet segmentation dataset (`gtsegs_ijcv.mat`) from https://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat. When running `imagenet_seg_eval.py`, specify path to this file.

2. Go to the `FViT-main` directory and run `python -W ignore /root/fact/fact-ai/FViT-main/baselines/ViT/imagenet_seg_eval.py --imagenet-seg-path gtsegs_ijcv.mat --method=<method> --attack --attack_noise=<attack_noise>`. `-W ignore` removes the `UndefinedWarning`s of computing the F1-score. `--method` argument specifies the interpretability method used for segmentation, options include `'rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'lrp_last_layer', 'attn_last_layer', 'attn_gradcam', 'dds', 'attr_rollout', 'attr_rollout_dds'`. `--attack` flag should be included if one wishes to replicate the method's perfomance under PGD attack with possibility of setting `--attack_noise` to any float value, which defaults to Hu's default of 8/255.


## Running the Image Perturbation Task
1. Download ImageNet DevKit(?) with `wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz`.

2. Download ImageNet Validation Set from https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5 and place it in the same directory as the ImageNet Devkit; a torrent utility is necessary for this source.

3. Go to `FViT-main` and run `python baselines/ViT/generate_visualizations.py --imagenet-validation-path ~/path-from-home-to-imagenet-files/ --method chosen-method imagenet-subset-ratio subset-ratio`. This creates the visualizations needed for the perturbation task. The `--imagenet-validation-path` argument needs to specify a path to the two (compressed) ImageNet files. The `subset-ratio` specifies what portion of the ImageNet dataset is used (between 0 and 1).

4. Run the perturbation test with `python baselines/ViT/pertubation_eval_from_hdf5.py --method chosen-method --use-dds`. Other flags may be modified. `--use-dds` specifies whether to use DDS with the chosen method.
