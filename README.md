# BeyondMimic-Style Motion Tracking

This is a project to implement BeyondMimic-style motion tracking on a Unitree G1 humanoid using a home-made asymmetric PPO. This project is for learning-purpose instead of building a high-fidelity motion tracking model. Thus we make a lot of simplifications (for example, we don't have adaptive sampling and domain randomization), and only "overfit" some specific motion clips. Note that some hyperparameters (for example, in reward function) are not the same as in the paper. (Thank Codex for the great help!! Note that this is a hobby-level implementation, it's not well-coded, and might even has bugs.)


Project Structure:
- ./AGENTS.md: Provides an overview of this project to Codex.
- ./Literatures: Three important papers to be considered. Providing details like observation, action, reward design, etc.
- ./data: Motion data and robot models
    - ./data/LAFAN1: Humanoid motion retargeted for unitree G1, downloaded from HuggingFace repo lvhaidong/LAFAN1_Retargeting_Dataset
    - ./data/g1: Unitree G1 model, downloaded from [mujoco model gallery](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
    - ./data/LAFAN1_interpolated: We will put interpolated motion data here
- ./BeyondMimic_Code: The [official implementation of BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)
- ./*.py: Core implementations

If you want to learn about BeyondMimic-Style motion tracking, use AGENTS.md, ./Literatures and ./BeyondMimic_Code, then Codex can answer almost everything!

First download data
```
python download_data.py
```

Then uncomment the 4th block in the main function of motion.py, then interpolate the downloaded motion data by running
```bash
python motion.py
```
which generates ./data/LAFAN1_interpolated/*.mat. The 5th block in the main function of motion.py let you visualize a motion clip from either original data or interpolated data.

Then train a neural network tracking policy by
```bash
python train.py train
```
For simplicity, we only train on first 500 frames (10 second) of a single motion clip. You can change the PPO hyperparameter, motion clip you train on, and the motion length in train.py. I split the training in multiple stages. It takes a **long** time to train on a M1 Macbook.

After each stage of training, you can visualize the tracking policy after each stage of training by
```bash
python train.py test
```

I trained for two motion clips and below is the result:

#### walk1_subject1 (~2e7 env steps):
<p align="center">
<img src="walk1_subject1_target.gif" width="300">
<img src="walk1_subject1_learned.gif" width="300">
</p>

<p align="center">
<b>Left:</b> Reference. <b>Right:</b> Learned Policy.
</p>

#### fight1_subject2 (~5e7 env steps):
<p align="center">
<img src="fight1_subject2_target.gif" width="300">
<img src="fight1_subject2_learned.gif" width="300">
</p>

<p align="center">
<b>Left:</b> Reference. <b>Right:</b> Learned Policy.
</p>



