# Reinforcement Learning for Robot Manipulation in Isaac Sim

This repository documents two manipulation policies trained from scratch with deep reinforcement
learning inside NVIDIA Isaac Sim, using GPU-parallel physics to run thousands of environments at
once and the RTX path tracer to render the final rollouts. The goal was to capture the full learning
progression of each policy, from random behaviour to a competent controller, and present it as a
timelapse together with the training curves.

Both tasks use the same learning algorithm and differ only in the robot, the observation/action
spaces and the reward. The first is an intuitive pick-and-place with a 7-DoF Franka arm; the second
is in-hand cube reorientation with a 24-DoF Shadow Hand, which is one of the harder dexterous
manipulation benchmarks and needs an order of magnitude more training.

| Franka pick-and-place | Shadow Hand reorientation |
|---|---|
| ![Franka lift](assets/franka_lift.gif) | ![Shadow Hand](assets/shadow_hand.gif) |

Full-resolution renders: [`assets/franka_lift_timelapse.mp4`](assets/franka_lift_timelapse.mp4),
[`assets/shadow_hand_timelapse.mp4`](assets/shadow_hand_timelapse.mp4).

## Method

Each task is a Markov decision process $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ in which a stochastic
policy $\pi_\theta(a\mid s)$ is optimised to maximise the expected discounted return

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^{t}\, r_t\right].$$

Training uses Proximal Policy Optimization (PPO), an on-policy actor–critic method. The actor is
updated with the clipped surrogate objective, which keeps each update close to the previous policy:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\big(\rho_t(\theta)\,\hat{A}_t,\;
\operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\right],
\qquad
\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.$$

Advantages are estimated with Generalised Advantage Estimation, trading off bias and variance through
$\lambda$:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^{l}\,\delta_{t+l},
\qquad
\delta_t = r_t + \gamma\,V_\phi(s_{t+1}) - V_\phi(s_t).$$

The critic $V_\phi$ is fit by regression to the returns, and an entropy term keeps the policy
exploratory, giving the combined loss

$$L(\theta,\phi) = -L^{\text{CLIP}}(\theta) + c_v\,\mathbb{E}_t\big[(V_\phi(s_t) - \hat{R}_t)^2\big]
- c_e\,\mathbb{E}_t\big[\mathcal{H}[\pi_\theta(\cdot\mid s_t)]\big].$$

Observations are normalised with a running mean and standard deviation, advantages are normalised per
batch, and the learning rate is adapted from the measured KL divergence between successive policies.

## Task 1 — Franka pick-and-place

A Franka Panda arm must reach a cube on a table, grasp it and bring it to a commanded target pose.
Observations are the arm and gripper state plus the cube and target poses; actions are continuous
end-effector commands. The reward is dense and shaped so that progress is rewarded before the cube is
ever lifted: a reaching term pulls the gripper to the cube, a lifting term activates once the cube
clears the table, and a tracking term rewards moving the lifted cube toward the goal,

$$r = \underbrace{1 - \tanh\!\big(\lVert p_{ee} - p_{obj}\rVert/\sigma\big)}_{\text{reach}}
\; + \; w_{l}\,\mathbf{1}\big[z_{obj} > z_{\min}\big]
\; + \; w_{g}\,\mathbf{1}[\text{lifted}]\big(1 - \tanh(\lVert p_{obj} - p_{goal}\rVert/\sigma)\big).$$

The policy converges quickly: a competent grasp-and-place emerges within a few hundred PPO epochs.
The reward and loss decay over training are shown below.

![Franka training curves](assets/franka_lift_curve.png)

## Task 2 — Shadow Hand in-hand reorientation

A fixed 24-DoF Shadow Hand holds a cube and must rotate it in-hand until its orientation matches a
randomly sampled target, shown next to the hand. When the target is reached a new one is sampled, and
the number of reorientations achieved before the cube is dropped is tracked as *consecutive
successes*. The relevant error is the geodesic distance between the current and target orientation
quaternions,

$$d_{\text{rot}} = 2\,\arcsin\!\Big(\min\big(1,\; \lVert\,\mathbf{q}_{obj} \otimes \mathbf{q}_{goal}^{-1}\,\rVert_{\text{vec}}\big)\Big),$$

and the reward rewards shrinking that error, penalises large actions, and adds a bonus on success:

$$r = \frac{c}{\,\lvert d_{\text{rot}}\rvert + \varepsilon\,} \; - \; w_a\,\lVert a_t\rVert^2
\; + \; b_{\text{succ}}\,\mathbf{1}\big[\lvert d_{\text{rot}}\rvert < \theta_{\text{th}}\big].$$

This task is far harder than the pick-and-place. Reward climbs steadily over thousands of epochs and
the consecutive-success count rises from zero to roughly thirty as the policy learns to chain
reorientations without dropping the cube.

![Shadow Hand training curves](assets/shadow_hand_curve.png)

## Training and rendering setup

The work runs on a single NVIDIA RTX 4090. Simulation and learning are GPU-resident end to end: PPO
runs 4096 (Franka) to 8192 (Shadow Hand) environments in parallel and reaches hundreds of thousands
of simulated steps per second, so the Franka policy trains in minutes and the Shadow Hand policy in
about half an hour. Rendering for the videos is done offscreen with the RTX renderer in headless mode,
one environment at a time, with the viewpoint placed for a clean cinematic shot.

The software stack is NVIDIA Isaac Sim 5.1 and Isaac Lab for the simulation, scene assets and task
definitions; the `rl_games` library for the PPO implementation; PyTorch (CUDA 12.8 build) as the
tensor and autograd backend; Python 3.11; TensorBoard for logging; and ffmpeg for assembling the
timelapse videos and these GIFs.

## Machine learning details

- Algorithm: PPO with a clipped surrogate objective, GAE advantages, a squared-error value head, an
  entropy bonus, and a KL-adaptive learning rate.
- Policy: a Gaussian actor over continuous actions with a shared-input actor–critic network; the
  Shadow Hand policy adds a recurrent (LSTM) layer to cope with the partially observed dynamics of
  in-hand manipulation.
- Scale: massively parallel on-policy data collection across thousands of vectorised environments on
  one GPU, which is what makes wall-clock training times this short.
- Inputs and stability: running observation and value normalisation, per-batch advantage
  normalisation, gradient clipping, and reward shaping. Isaac Lab's event system applies domain
  randomisation (object mass, friction, initial poses) so the policies are not overfit to a single
  configuration.

## Reproduce

Train and then record a checkpoint with the standard Isaac Lab entry points:

```bash
# train (Shadow Hand shown; swap the task id for the Franka lift)
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Repose-Cube-Shadow-Direct-v0 --headless --num_envs 8192

# render a checkpoint offscreen with the RTX renderer
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Repose-Cube-Shadow-Direct-v0 --headless --num_envs 1 \
  --checkpoint <path/to/checkpoint.pth> --video --video_length 240
```

Checkpoints are saved periodically during training; the timelapses are built by rendering a spread of
them in order and concatenating the clips.
