# Reinforcement Learning for Robot Manipulation in Isaac Sim

Two manipulation policies trained from scratch with deep reinforcement learning inside NVIDIA Isaac
Sim, using GPU-parallel physics to run thousands of environments at once and the RTX path tracer to
render the rollouts. Each timelapse below captures the full learning progression — from random
flailing to a competent controller — alongside the training curves.

The two tasks share the same learning algorithm and differ only in robot, observation/action space,
and reward: an intuitive pick-and-place with a 7-DoF Franka arm, and in-hand cube reorientation with
a 24-DoF Shadow Hand — one of the harder dexterous benchmarks, needing an order of magnitude more
training.

| Franka pick-and-place | Shadow Hand reorientation |
|---|---|
| ![Franka lift](assets/franka_lift.gif) | ![Shadow Hand](assets/shadow_hand.gif) |

Full-resolution renders: [`franka_lift_timelapse.mp4`](assets/franka_lift_timelapse.mp4),
[`shadow_hand_timelapse.mp4`](assets/shadow_hand_timelapse.mp4).

## The math

Each task is a Markov decision process $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ in which a stochastic
policy $\pi_\theta(a\mid s)$ is optimised to maximise the expected discounted return

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^{t}\, r_t\right].$$

Training uses **Proximal Policy Optimization (PPO)**, an on-policy actor–critic method. The actor is
updated with the clipped surrogate objective, which keeps each update close to the previous policy:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\big(\rho_t(\theta)\,\hat{A}_t,\;
\operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\right],
\qquad
\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.$$

Advantages use **Generalised Advantage Estimation**, trading bias against variance through $\lambda$:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^{l}\,\delta_{t+l},
\qquad
\delta_t = r_t + \gamma\,V_\phi(s_{t+1}) - V_\phi(s_t).$$

The critic $V_\phi$ is fit by regression to the returns and an entropy term keeps the policy
exploratory, giving the combined loss

$$L(\theta,\phi) = -L^{\text{CLIP}}(\theta) + c_v\,\mathbb{E}_t\big[(V_\phi(s_t) - \hat{R}_t)^2\big]
- c_e\,\mathbb{E}_t\big[\mathcal{H}[\pi_\theta(\cdot\mid s_t)]\big].$$

Observations are normalised with a running mean/std, advantages are normalised per batch, and the
learning rate is adapted from the measured KL divergence between successive policies.

## Task 1 — Franka pick-and-place

A Franka Panda arm reaches a cube on a table, grasps it, and brings it to a commanded target pose.
The reward is dense and shaped so progress is rewarded before the cube is ever lifted — a reaching
term pulls the gripper to the cube, a lifting term activates once it clears the table, and a tracking
term rewards moving the lifted cube toward the goal:

$$r = \underbrace{1 - \tanh\!\big(\lVert p_{ee} - p_{obj}\rVert/\sigma\big)}_{\text{reach}}
\; + \; w_{l}\,\mathbf{1}\big[z_{obj} > z_{\min}\big]
\; + \; w_{g}\,\mathbf{1}[\text{lifted}]\big(1 - \tanh(\lVert p_{obj} - p_{goal}\rVert/\sigma)\big).$$

The policy converges quickly — a competent grasp-and-place emerges within a few hundred PPO epochs.

![Franka training curves](assets/franka_lift_curve.png)

## Task 2 — Shadow Hand in-hand reorientation

A fixed 24-DoF Shadow Hand holds a cube and rotates it in-hand until its orientation matches a
randomly sampled target. When the target is reached a new one is sampled, and the number of
reorientations before the cube is dropped is tracked as *consecutive successes*. The relevant error
is the geodesic distance between the current and target orientation quaternions,

$$d_{\text{rot}} = 2\,\arcsin\!\Big(\min\big(1,\; \lVert\,\mathbf{q}_{obj} \otimes \mathbf{q}_{goal}^{-1}\,\rVert_{\text{vec}}\big)\Big),$$

and the reward rewards shrinking that error, penalises large actions, and adds a bonus on success:

$$r = \frac{c}{\,\lvert d_{\text{rot}}\rvert + \varepsilon\,} \; - \; w_a\,\lVert a_t\rVert^2
\; + \; b_{\text{succ}}\,\mathbf{1}\big[\lvert d_{\text{rot}}\rvert < \theta_{\text{th}}\big].$$

Far harder than the pick-and-place: reward climbs steadily over thousands of epochs and the
consecutive-success count rises from zero to roughly thirty as the policy learns to chain
reorientations without dropping the cube.

![Shadow Hand training curves](assets/shadow_hand_curve.png)

## How it was done

- **Scale.** Massively parallel on-policy data collection across thousands of vectorised
  environments on a single GPU — 4096 (Franka) to 8192 (Shadow Hand) — reaching hundreds of
  thousands of simulated steps per second. The Franka policy trains in minutes, the Shadow Hand in
  about half an hour.
- **Policy.** A Gaussian actor over continuous actions with a shared-input actor–critic network; the
  Shadow Hand adds a recurrent (LSTM) layer to cope with the partially observed dynamics of in-hand
  manipulation.
- **Stability.** Running observation/value normalisation, per-batch advantage normalisation,
  gradient clipping, and reward shaping. Domain randomisation (object mass, friction, initial poses)
  keeps the policies from overfitting a single configuration.
- **Rendering.** GPU-resident simulation and learning end to end; final rollouts rendered offscreen
  with the RTX path tracer, one environment at a time, with the viewpoint placed for a clean
  cinematic shot.

Hardware: a single NVIDIA RTX 4090.
