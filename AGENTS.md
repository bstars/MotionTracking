# Motion Tracking

This is a project for motion tracking on a Unitree G1 humanoid. The directory structure is as follows

- ./Literatures: Three important papers to be considered. Providing details like observation, action, reward design ...
- ./data: Motion data and robot models
    - ./data/LAFAN1: Humanoid motion retargeted for unitree G1, downloaded from HuggingFace repo lvhaidong/LAFAN1_Retargeting_Dataset
    - ./data/g1: Unitree G1 model, downloaded from [mujoco model gallery](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
- ./BeyondMimic_Code: The official implementation of BeyondMimic

This project aims to reimplement BeyondMimic-style motion tracking, but with some potential modification and simplification.



## Observation

BeyondMimic uses asymmetric observations: the actor receives proprioception and a compact motion reference, while the critic additionally receives the tracked robot bodies. All terms are concatenated in the order below.

##### Policy observation

- `command`: reference joint positions and velocities at the current motion timestep, $[q_{\mathrm{ref}}, \dot q_{\mathrm{ref}}]$. For the 29-DoF G1 this has dimension $58$. It acts as the motion-phase cue; BeyondMimic does not use a separate scalar phase variable.
- `motion_anchor_pos_b`: reference anchor position relative to the current robot anchor, expressed in the robot anchor frame. Dimension $3$.
- `motion_anchor_ori_b`: reference anchor orientation relative to the current robot anchor, represented by the first two columns of its rotation matrix. Dimension $6$.
- `base_lin_vel`: current base linear velocity. Dimension $3$.
- `base_ang_vel`: current base angular velocity. Dimension $3$.
- `joint_pos`: current joint positions relative to the default joint positions. Dimension $29$.
- `joint_vel`: current joint velocities relative to the default joint velocities. Dimension $29$.
- `actions`: previous policy action. Dimension $29$.

The total G1 policy-observation dimension is $58 + 3 + 6 + 3 + 3 + 29 + 29 + 29 = 160$. Observation corruption is enabled during training. Uniform noise is applied to anchor position $[-0.25, 0.25]$, anchor orientation $[-0.05, 0.05]$, base linear velocity $[-0.5, 0.5]$, base angular velocity $[-0.2, 0.2]$, joint position $[-0.01, 0.01]$, and joint velocity $[-0.5, 0.5]$.

##### Critic observation

The critic receives all policy terms without observation noise, plus:

- `body_pos`: current tracked-body positions relative to the robot anchor. Dimension $3|B_{\mathrm{target}}|$.
- `body_ori`: current tracked-body orientations relative to the robot anchor, represented by the first two columns of each rotation matrix. Dimension $6|B_{\mathrm{target}}|$.

The official G1 configuration tracks 14 bodies, so these privileged terms have dimensions $42$ and $84$. The total critic-observation dimension is therefore $286$.

Code positions:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:113`: policy observation definition
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:135`: privileged critic observation definition
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:100`: reference command contents
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py:32`: relative tracked-body position and orientation observations
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py:60`: relative motion-anchor position and orientation observations

## Anchor Transform

BeyondMimic does not directly track every body pose in the reference motion's original global frame. Instead, it re-anchors the reference body layout around the robot's current anchor body so the robot can preserve the motion style while tolerating global $x,y$ drift.

For each body $b \in B$, the desired pose is:

$$
\hat{T}_b =
T_{\mathrm{anchor}}
T^{-1}_{b_{\mathrm{ref}},\mathrm{motion}}
T_{b,\mathrm{motion}}
$$

The middle term $T^{-1}_{b_{\mathrm{ref}},\mathrm{motion}} T_{b,\mathrm{motion}}$ expresses body $b$ in the local frame of the reference body $b_{\mathrm{ref}}$ from the motion clip. Left-multiplying by $T_{\mathrm{anchor}}$ places that same local body configuration around the robot's current anchor.

The anchor pose is $T_{\mathrm{anchor}} = (p_{\mathrm{anchor}}, R_{\mathrm{anchor}})$, where:

$$
p_{\mathrm{anchor}} =
\begin{bmatrix}
p_{b_{\mathrm{ref}},x} \\
p_{b_{\mathrm{ref}},y} \\
p_{b_{\mathrm{ref}},z,\mathrm{motion}}
\end{bmatrix}
$$

and:

$$
R_{\mathrm{anchor}} =
R_z\left(
\mathrm{yaw}
\left(
R_{b_{\mathrm{ref}}}
R^\top_{b_{\mathrm{ref}},\mathrm{motion}}
\right)
\right)
$$

This means the re-anchored target uses the robot anchor's current horizontal position, keeps the reference motion's anchor height, and applies only the yaw difference between the current robot anchor and the motion anchor. The desired body twists remain unchanged, $\hat{V}_b = V_{b,\mathrm{motion}}$.

Implementation note: the official code forms this transform by setting the target anchor $x,y$ from the robot anchor, target $z$ from the motion anchor, computing a yaw-only orientation delta, and applying that transform to all motion body poses.

Code positions:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:289`: sets the re-anchored target position base
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:291`: computes the yaw-only orientation delta
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:293`: applies the orientation transform to body orientations
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:294`: applies the position transform to body positions

## Termination

BeyondMimic uses early termination to reset rollouts when tracking has clearly failed. The following hard-failure checks are used in addition to the normal episode timeout.

- Anchor height failure
For G1, the anchor body is `torso_link`. The rollout terminates when the robot anchor height deviates from the reference motion anchor height by more than $0.25$m:
$$
|z_{\mathrm{anchor,ref}} - z_{\mathrm{anchor,robot}}| > 0.25
$$
This check is z-only. It does not terminate on horizontal $x,y$ drift, because the target motion is re-anchored around the robot's current horizontal anchor position. It catches falls, excessive crouching, or incorrect vertical motion.

- Anchor orientation failure
The rollout terminates when the target and robot anchor frames disagree too much in roll/pitch. The official code rotates the world gravity vector into the target anchor frame and the robot anchor frame, then compares the local z components:
$$
|g^z_{\mathrm{anchor,ref}} - g^z_{\mathrm{anchor,robot}}| > 0.8
$$
This is mainly a torso tilt or fall detector. It is largely insensitive to yaw, since yaw around the world vertical axis does not change projected gravity.

- End-effector height failure
The rollout terminates when any tracked end effector has vertical position error greater than $0.25$m. For G1 the checked bodies are `left_ankle_roll_link`, `right_ankle_roll_link`, `left_wrist_yaw_link`, and `right_wrist_yaw_link`:
$$
\exists b \in B_{\mathrm{ee}}:\ |z_{b,\mathrm{target}} - z_{b,\mathrm{robot}}| > 0.25
$$
This check is also z-only. It catches large foot or wrist height mismatches, such as dragging a foot that should be lifted, lifting a foot that should be planted, or losing upper-body tracking enough for wrist height to become implausible.

Code positions:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:258`: anchor height termination threshold
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:262`: anchor orientation termination threshold
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:266`: end-effector height termination threshold and body list
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py:23`: anchor z-only position failure implementation
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py:28`: projected-gravity anchor orientation failure implementation
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py:51`: body z-only position failure implementation
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py:15`: G1 anchor body name

## Adaptive Sampling

BeyondMimic uses adaptive reset-frame sampling to focus training on parts of the reference motion where the policy is currently failing. Instead of sampling every reset frame uniformly from the whole motion clip, it keeps a shared failure histogram over motion-time bins and samples future resets more often from bins with recent early terminations.

For a motion clip with $T$ frames, divide the timeline into $K$ bins. A rollout at motion frame $t$ belongs to bin:

$$
b(t)
=
\left\lfloor
\frac{tK}{T}
\right\rfloor
$$

clamped to $[0, K-1]$.

For each bin $k$, BeyondMimic maintains a smoothed failure score $F_k$. When a batch of parallel environments resets, the command term first checks which environments terminated early. If the failed environments ended at motion frames $t_i$, the current failure count for bin $k$ is:

$$
C_k
=
\sum_i
\mathbf{1}
\left[
\mathrm{terminated}_i
\land
b(t_i) = k
\right]
$$

The persistent failure score is updated as an exponential moving average:

$$
F_k
\leftarrow
\alpha C_k
+
(1-\alpha)F_k
$$

where the official default is $\alpha = 0.001$. This makes the adaptive distribution change slowly and prevents a single reset batch from dominating the curriculum.

The sampling weight for bin $k$ is the failure score plus a uniform floor:

$$
w_k
=
F_k
+
\frac{\rho}{K}
$$

where the official default is $\rho = 0.1$. The reset-bin probability is then:

$$
p_k
=
\frac{w_k}
{\sum_{j=0}^{K-1} w_j}
$$

Each resetting environment samples a bin independently from the same shared distribution:

$$
B \sim \mathrm{Categorical}(p_0, p_1, \dots, p_{K-1})
$$

Then it samples uniformly inside that bin:

$$
u \sim \mathcal{U}(0,1)
$$

and converts the sampled bin position back to a motion frame:

$$
t_{\mathrm{reset}}
=
\left\lfloor
\frac{B + u}{K}
(T - 1)
\right\rfloor
$$

The result is a piecewise-uniform distribution over the motion timeline whose bin masses increase near recently failed regions. The uniform floor keeps every part of the motion reachable, so training does not collapse onto only the hardest frames.

Implementation note: when multiple parallel environments use the same reference motion, they should share one adaptive sampling distribution for that motion. Each environment stores its own current motion frame, but the failure scores $F_k$ and reset probabilities $p_k$ are shared across all environments using the same clip. If multiple clips are trained together, keep one sampler per clip.

The official implementation also supports optional smoothing over neighboring bins. It builds a kernel:

$$
g_i
=
\frac{\lambda^i}
{\sum_{m=0}^{M-1}\lambda^m},
\quad
i = 0,\dots,M-1
$$

and convolves the raw weights with this kernel before normalization. The defaults are $M = 1$ and $\lambda = 0.8$, so smoothing is effectively disabled unless `adaptive_kernel_size` is increased.

Code positions:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:80`: computes the number of motion-time bins
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:81`: stores the shared smoothed failure count per bin
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:207`: adaptive sampling implementation
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:208`: reads which environments terminated early
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:210`: maps motion frames to failure bins
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:217`: adds the uniform sampling floor
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:227`: samples reset bins from the shared probability distribution
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:229`: samples a reset frame inside the selected bin
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:296`: updates the smoothed failure histogram
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:368`: default adaptive sampler hyperparameters

## Action

BeyondMimic uses normalized joint-position actions. The policy outputs:

$$
a \in [-1, 1]^{29}
$$

for the 29 actuated G1 joints. The action is converted into joint position setpoints:

$$
q_{\mathrm{target}}
=
q_{\mathrm{default}}
+
\alpha \odot a
$$

where $q_{\mathrm{default}}$ is the default joint pose, $\alpha$ is the per-joint action scale, and $\odot$ denotes elementwise multiplication. These setpoints are then tracked by the joint PD controller; the policy does not directly output torque.

For G1, BeyondMimic sets the action scale from the joint torque limit and stiffness:

$$
\alpha_j
=
0.25
\frac{\tau_{j,\max}}{K_{p,j}}
$$

This makes the maximum action-induced joint-position offset proportional to the joint's available torque authority. In this reimplementation, the same rule is used:

$$
\alpha_j
=
0.25
\frac{\mathrm{TORQUE\_LIM}_j}{\mathrm{STIFFNESS}_j}
$$

Implementation note: the actual MuJoCo control target should include the default joint offset. Use $q_{\mathrm{default}} + \alpha \odot a$ as the PD target, not only $\alpha \odot a$, unless the actuator interface is explicitly defined to accept offsets.

Code positions:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:106`: joint-position action with default offset enabled
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py:14`: assigns G1 action scale
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/robots/g1.py:184`: computes `G1_ACTION_SCALE`
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/robots/g1.py:195`: action-scale formula `0.25 * effort_limit / stiffness`
- `mdp.py:69`: local reimplementation of the G1 action-scale vector

## G1 PD Gain Rule

BeyondMimic computes G1 joint PD gains from a desired second-order response instead of using one global stiffness/damping value.

Code position:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/robots/g1.py:12`: defines `NATURAL_FREQ = 10 * 2.0 * pi` and `DAMPING_RATIO = 2.0`
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/robots/g1.py:15`: computes stiffness as `STIFFNESS = ARMATURE * NATURAL_FREQ**2`
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/robots/g1.py:20`: computes damping as `DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ`

Equivalent formulas:
- $K_p = I \omega_n^2$
- $K_d = 2 \zeta I \omega_n$

where $I$ is the motor/joint armature, $\omega_n$ is the desired natural frequency in rad/s, and $\zeta$ is the damping ratio. BeyondMimic uses $\omega_n = 2\pi \cdot 10$ and $\zeta = 2.0$.

## Reward

BeyondMimic uses a motion-agnostic reward with positive task-space tracking terms and a small set of regularization penalties. The tracking reward is the core imitation objective: it measures how well the robot matches the reference motion in Cartesian body space, while regularization discourages unsafe or jittery behavior.

The overall reward can be viewed as:

$$
r =
r_{\mathrm{tracking}}
+ r_{\mathrm{anchor}}
- 0.1 r_{\mathrm{action\_rate}}
- 10.0 r_{\mathrm{joint\_limit}}
- 0.1 r_{\mathrm{contact}}
$$

where the positive terms encourage motion tracking and the negative terms penalize action jitter, joint-limit violation, and undesired contacts.

##### Tracking reward

The tracking reward compares the desired target-body poses and twists against the current robot body poses and twists. For each tracked body $b \in B_{\mathrm{target}}$, BeyondMimic computes four task-space errors:

- body position error, $\hat p_b - p_b$
- body orientation error, represented as a rotation distance between $\hat R_b$ and $R_b$
- body linear velocity error, $\hat v_b - v_b$
- body angular velocity error, $\hat \omega_b - \omega_b$

For each signal $s \in \{p, R, v, \omega\}$, the squared error is averaged over all tracked bodies:

$$
\bar e_s =
\frac{1}{|B_{\mathrm{target}}|}
\sum_{b \in B_{\mathrm{target}}}
\| e_{s,b} \|^2
$$

The averaged error is then converted into a bounded Gaussian-shaped reward:

$$
r_s =
\exp\left(
-\frac{\bar e_s}{\sigma_s^2}
\right)
$$

The tracking reward is the sum of the four body terms:

$$
r_{\mathrm{tracking}}
=
r_p + r_R + r_v + r_\omega
$$

The official implementation uses:

- body position: $\sigma_p = 0.3$, weight $1.0$
- body orientation: $\sigma_R = 0.4$, weight $1.0$
- body linear velocity: $\sigma_v = 1.0$, weight $1.0$
- body angular velocity: $\sigma_\omega = 3.14$, weight $1.0$

The exponential form gives each term a maximum value of $1$ under perfect tracking and smoothly decreases toward $0$ as the error grows. This makes the four body-tracking terms bounded, comparable, and easy to combine.

The target positions and orientations used by the body tracking reward are not the raw global poses from the motion clip. They are the re-anchored targets described in the Anchor Transform section: the desired body layout is placed around the robot's current anchor body so the policy tracks body shape, timing, and style while tolerating global horizontal drift.

The code also includes optional global anchor tracking terms:

$$
r_{\mathrm{anchor}}
=
0.5 r_{p,\mathrm{anchor}}
+ 0.5 r_{R,\mathrm{anchor}}
$$

These compare the robot anchor body against the motion anchor body using the same exponential form, with $\sigma_p = 0.3$ for anchor position and $\sigma_R = 0.4$ for anchor orientation. For G1, the anchor body is `torso_link`.

Important distinction: the reference joint positions and velocities in the policy observation are used as a compact motion-phase cue, but the reward does not directly penalize joint position or joint velocity tracking error. The imitation objective is primarily Cartesian body tracking.

Code positions:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:202`: anchor position tracking reward
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:207`: anchor orientation tracking reward
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:212`: body position tracking reward
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:217`: body orientation tracking reward
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:222`: body linear velocity tracking reward
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:227`: body angular velocity tracking reward
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py:20`: anchor position exponential reward implementation
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py:26`: anchor orientation exponential reward implementation
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py:32`: body position exponential reward implementation
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py:43`: body orientation exponential reward implementation
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py:55`: body linear velocity exponential reward implementation
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py:66`: body angular velocity exponential reward implementation

##### Action-rate penalty

The action-rate term penalizes abrupt changes between consecutive policy actions. It encourages smoother target joint-position commands and reduces high-frequency jitter:

$$
r_{\mathrm{action\_rate}}
=
\| a_t - a_{t-1} \|^2
$$

The configured reward weight is negative:

$$
-0.1 r_{\mathrm{action\_rate}}
$$

In the low-frequency G1 environment variant, this weight is additionally scaled by `LOW_FREQ_SCALE` so the action-smoothness cost remains comparable when the policy update frequency changes.

Code positions:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:232`: action-rate penalty configuration
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py:47`: low-frequency action-rate weight scaling

##### Joint-limit penalty

The joint-limit term penalizes joint positions that exceed the robot's soft joint-position limits. It is a hardware-safety regularizer, not a motion-imitation term:

$$
r_{\mathrm{joint\_limit}}
=
\sum_j
\left[
\max(l_j - q_j, 0)
+
\max(q_j - u_j, 0)
\right]
$$

where $q_j$ is the current joint position, and $[l_j, u_j]$ are the soft lower and upper limits for joint $j$. The configured weight is:

$$
-10.0 r_{\mathrm{joint\_limit}}
$$

The official G1 asset uses a soft joint-position limit factor of $0.9$, so the penalty begins before the mechanical joint limits are fully reached.

Code positions:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:233`: joint-limit penalty configuration
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/robots/g1.py:61`: G1 soft joint-position limit factor

##### Contact penalty

The contact term penalizes undesired contacts. For G1, the allowed contact bodies are the two ankles and two wrists; contacts on other robot bodies are penalized when the contact force exceeds the configured threshold:

$$
r_{\mathrm{contact}}
=
\sum_{b \notin B_{\mathrm{ee}}}
\mathbf{1}
\left[
\| f_b \| > 1.0\mathrm{N}
\right]
$$

The configured weight is:

$$
-0.1 r_{\mathrm{contact}}
$$

This discourages self-collisions and unintended body-ground contacts while still permitting contacts at the tracked end effectors used by the termination logic: `left_ankle_roll_link`, `right_ankle_roll_link`, `left_wrist_yaw_link`, and `right_wrist_yaw_link`.

Code positions:
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:238`: undesired-contact penalty configuration
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:242`: contact sensor body-name filter
- `BeyondMimic_Code/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:248`: contact-force threshold
