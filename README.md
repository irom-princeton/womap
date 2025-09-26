<p align="center">

  <h1 align="center"><img src="assets/favicon.png" width="25"> WoMAP: World Models For Embodied <br /> Open-Vocabulary Object
                    Localization</h1>
  <p align="center"> 
        <span class="author-block"><a
                href="https://tenny-yinyijun.github.io/">Tenny&nbsp;Yin*</a></span>,
        <span class="author-block"><a href="https://may0mei.github.io/">Zhiting&nbsp;Mei</a></span>,
        <span class="author-block"><a href="#">Tao&nbsp;Sun</a></span>,
        <span class="author-block"><a href="https://lihzha.github.io/">Lihan&nbsp;Zha</a></span>,
        <span class="author-block"><a
                href="https://www.linkedin.com/in/zhou-emily/">Emily&nbsp;Zhou<sup>+</sup></a></span>,
        <span class="author-block"><a
                href="https://www.linkedin.com/in/jeremy-bao/">Jeremy&nbsp;Bao<sup>+</sup></a></span>,
        <span class="author-block"><a
                href="https://www.linkedin.com/in/miyu-yamane/">Miyu&nbsp;Yamane<sup>+</sup></a></span>,
        <span class="author-block"><a href="#">Ola&nbsp;Shorinwa*</a></span>,
        <span class="author-block"><a
                href="https://irom-lab.princeton.edu/majumdar/">Anirudha&nbsp;Majumdar</a></span>
  </p>
  <p align="center">
    <sup>&#42;</sup>Equal Contribution.
    <sup>+</sup>Authors contributed equally.
</p>
<p align="center">
  <a href="">
    <img src="assets/irom_princeton.png" width="80%">
  </a>
  <h3 align="center"><a href="https://robot-womap.github.io/"> Project Page</a> | <a href= "https://arxiv.org/abs/2506.01600">arXiv</a> </h3>
  <div align="center"></div>
</p>

## WoMAP
We introduce <i>World Models for Active Perception</i> (<strong>WoMAP</strong>): a recipe for training open-vocabulary object localization policies that: (i) uses a Gaussian Splatting-based real-to-sim-to-real pipeline for scalable data generation without the need for expert demonstrations, (ii) distills dense rewards signals from open-vocabulary object detectors, and (iii) leverages a latent world model for dynamics and rewards prediction to ground high-level action proposals at inference time.


## Getting Started

1. [Installation](#installation)
2. [Training the World Model](#training-the-world-model)
3. [Running Experiments](#experiments)

## Installation
1. Clone this repo.
```
git clone https://github.com/irom-princeton/womap.git
```

2. Install `wmap` as a Python package.
```
python -m pip install -e .
```

## Training the World Model

Train (with GSplat):
###### DINO, CLIP, or ViT Encoder with Frozen Weights (with Dynamics and Rewards Predictors)
```bash
python main.py --fname configs/gsplat/cfg_target_seq2.yaml \
--projname <project name, e.g., 0211-test-encoder> \
--expname <experiment name, e.g., test> \
--encoder <dino, clip, vit> --frozen
```

###### DINO, CLIP, or ViT Encoder with Frozen Weights (*without* Dynamics Predictor)
```bash
python main.py --fname configs/gsplat/cfg_target_seq2.yaml \
--projname <project name, e.g., 0211-test-encoder> \
--expname <experiment name, e.g., test> \
--encoder <dino, clip, vit> --frozen \
--ablate_rewards
```

###### DINO, CLIP, or ViT Encoder with Frozen Weights (*without* Rewards Predictor)
```bash
python main.py --fname configs/gsplat/cfg_target_seq2.yaml \
--projname <project name, e.g., 0211-test-encoder> \
--expname <experiment name, e.g., test> \
--encoder <dino, clip, vit> --frozen \
--ablate_dynamics
```

You can find bash scripts for ablating the dynamics and rewards predictors and for training the entire model in:
```
bash_scripts/ablation_dynamics.bash
```
```
bash_scripts/ablation_rewards.bash
```
```
bash_scripts/train_model.bash
```

respectively, which you can run on the terminal, e.g., via:
```
bash bash_scripts/ablation_dynamics.bash
```

To enable finetuning the encoder's weights, remove the flag `--frozen`.

Templates for running jobs via SLURM are available in the `slurm_scripts` directory. 
Please update the following fields in the shell scripts:
```bash
#SBATCH --mail-user=<princeton Net ID>@princeton.edu
#

# load modules or conda environments here
source ~/.bashrc

# activate virtual environment
micromamba activate <path to micromama environment>
# or path to conda env
# conda activate <path to micromama environment>

# run
cd <path to the project folder>
bash bash_scripts/ablation_rewards.bash
```

For example, to run an ablation experiment on the rewards predictor, run the following command:
```bash
sbatch slurm_scripts/slurm_ablation_rewards.sh
```
To run an ablation experiment on the dynamics predictor, run the following command:
```bash
sbatch slurm_scripts/slurm_ablation_dynamics.sh
```
To train the dynamics and rewards predictors, run the following command:
```bash
sbatch slurm_scripts/slurm_train_model.sh
```

The output files can be found in `slurm_outputs`.

## Experiments

### Generate experiment configurations

Generate experiment config files under `experiment_configs/` speicifying the model and experiments to run

### Run experiments

Specify relevant arguments in the bash file. The script will submit one job per experiment configuration (model + experiment):

```bash
bash bash_scripts/submit_experiments.bash 
```
### Visualize Results

Interactive script using `extract_result.py`