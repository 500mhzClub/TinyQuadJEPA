# TinyQuadJEPA: current architecture, what it is, and where it should go next

## Overview

This repository currently implements a **two-level control stack** for a Mini Pupper style quadruped:

- **System 1**: a low-level PPO locomotion controller that turns a high-level body-frame command into joint targets.
- **System 2**: an action-conditioned latent dynamics model that encodes vision + proprioception into a joint latent state and predicts the next latent under a commanded body velocity.

The current System 2 is best described as:

> **an action-conditioned JEPA-style latent world model trained with a VICReg objective**

It is **JEPA-like** because it predicts in latent space rather than reconstructing pixels. It is **not yet a true learned EBM** because the current “energy” is a hand-defined latent distance / VICReg objective, not a separately learned scalar compatibility function.

---

## High-level architecture

```text
RGB camera (64x64) ----> VisionEncoder ----\
                                             > JointEncoder -> z_t (256-d latent)
Proprio state (47-d) -> ProprioEncoder ----/

z_t + cmd_t (vx, vy, wz) -> LatentPredictor (GRUCell) -> ẑ_{t+1}

Training target:
RGB_{t+1}, proprio_{t+1} -> JointEncoder -> z_{t+1}

Loss:
VICReg(ẑ_{t+1}, z_{t+1}) = sim + var + cov
```

At deployment time:

```text
current observation -> z_t
candidate commands -> rollout predictor in latent space
pick command with lowest terminal latent cost to goal waypoint
send chosen high-level command to PPO controller
PPO produces 12 joint actions
joint position targets drive the robot
```

---

## Current modules

### 1. Vision encoder

The visual stream is a compact convolutional encoder:

- 4 convolution blocks with ELU activations
- stride-2 downsampling
- flatten + linear projection
- LayerNorm on the 128-d visual feature

Its job is not to decode images or preserve every detail. Its job is to produce a compact representation that is useful for future prediction and control.

### 2. Proprio encoder

The proprioceptive stream maps the 47-d robot state into a 128-d feature with:

- linear -> ELU -> linear
- LayerNorm on the final feature

This gives System 2 access to body state, joint state, and motion cues alongside vision.

### 3. Joint encoder

The visual and proprio features are concatenated and fused into a **256-d latent state**.

This latent is the internal “state of the world as far as the planner cares about it”.

### 4. Latent predictor

The predictor is an **action-conditioned recurrent transition model**:

- input: `[z_t ; cmd_t]`
- input projection with ELU
- `GRUCell` hidden state of size 256
- output projection back to latent space

This means the model is not only predicting “what happens next”, but “what happens next **if I command this body velocity**”.

### 5. System 1 controller

System 1 is a PPO locomotion policy that takes a 50-d observation and outputs 12 bounded actions. The action is turned into joint position targets relative to a nominal standing pose.

System 2 does **not** directly output joint torques or joint angles. It outputs a **high-level command** `(vx, vy, wz)` which System 1 executes.

That separation is good design for your setting:

- System 1 handles fast local motor control.
- System 2 handles slower latent-space planning.

---

## Training objective

The current training objective is VICReg-style prediction matching.

For each time step:

- encode `(vision_t, proprio_t)` to `z_t`
- predict `ẑ_{t+1}` from `z_t` and `cmd_t`
- encode `(vision_{t+1}, proprio_{t+1})` to `z_{t+1}`
- apply VICReg between `ẑ_{t+1}` and `z_{t+1}`

The total loss is:

```text
L = 25 * L_sim + 25 * L_var + 1 * L_cov
```

where:

- `L_sim`: MSE between predicted and target latents
- `L_var`: variance floor penalty to avoid collapse
- `L_cov`: decorrelation penalty to reduce feature redundancy

This makes the model:

- **predictive** via similarity loss
- **non-collapsed** via variance loss
- **less entangled** via covariance loss

---

## Why this is JEPA-style

A JEPA predicts a target representation from context, instead of reconstructing pixels.

Your current setup does exactly that:

- context = current latent state + command
- target = latent encoding of the next observation
- prediction is carried out entirely in latent space

That makes it a valid **JEPA-style latent predictor**.

---

## Why this is not yet a true EBM

Right now, “energy” means one of two things:

1. the weighted VICReg training loss during optimisation, or
2. a hand-crafted planning cost such as cosine distance between a predicted latent and a goal latent.

A **true learned EBM** would instead learn a scalar function such as:

```text
E(context, action_seq, goal) -> scalar
```

with the property that:

- compatible / good futures have **low energy**
- incompatible / bad futures have **high energy**

That energy would itself be a learned object, not just a distance metric chosen by hand.

So the current system is better named:

> **TinyQuadJEPA: action-conditioned latent predictor with VICReg training**

and not yet:

> **true JEPA EBM**

---

## Current strengths

- clean split between low-level motor skill and high-level planning
- multimodal latent state from vision + proprioception
- recurrent action-conditioned predictor
- no obvious representational collapse
- suitable for waypoint chasing and short-horizon planning demos
- good baseline before moving to a more canonical JEPA design

---

## Current limitations

1. **No target/EMA encoder**
   - the same encoder is used on both sides of the prediction task.

2. **No explicit masking / abstraction mechanism**
   - the model is asked to match next-step latent state directly.
   - this is useful, but less canonical than a masked predictive JEPA setup.

3. **No learned energy head**
   - planning cost is latent distance, not a trained energy network.

4. **Short-horizon control objective**
   - evaluation mostly demonstrates whether local waypoint chasing works.

5. **Planner currently searches simple command families**
   - useful for demos, but not yet a rich trajectory optimiser.

---

## Recommended evaluation philosophy for the current run

At this stage, evaluation should answer three different questions.

### A. Is the model alive?

Sanity-check metrics:

- training loss trending down
- similarity loss trending down
- variance penalty near zero
- covariance penalty steadily decreasing
- latent rollouts stay numerically stable

### B. Does it predict useful latent dynamics?

Offline checkpoint evaluation on held-out rollouts:

- 1-step latent error
- H-step latent rollout error for H in {1, 3, 5, 10, 15}
- cosine similarity to true future latents
- ranking quality of commands by physical progress vs latent score

### C. Can it produce a compelling demo?

Closed-loop simulator demos:

- latent waypoint chasing
- slalom navigation
- energy landscape plots
- side-by-side ego / 3rd person video with planning HUD

For your current repo stage, **C is acceptable for presentation**, but **A and B are what make the model development trustworthy**.

---

## Further work: how to make this a more canonical JEPA

### 1. Add a target encoder

Move from:

```text
z_target = encoder(next_obs)
```

to:

```text
z_target = target_encoder(next_obs)
```

where `target_encoder` is an EMA copy of the online encoder.

Why:

- stabilises targets
- makes training more canonical for joint-embedding methods
- reduces representational drift

### 2. Predict a goal-conditioned or masked target representation

Instead of only predicting the immediate next latent, make the task more JEPA-like by predicting a target representation from:

- partial context
- masked observations
- future offset(s)
- goal-conditioned future states

Possible extensions:

- random future offset prediction
- multi-head predictor for horizons 1, 3, 5, 10, 15
- masked visual context with proprio retained

### 3. Add a learned energy head

Introduce something like:

```text
E_theta(z_context, cmd_seq, z_goal) -> scalar
```

or

```text
E_theta(z_pred, z_goal) -> scalar
```

Train it with contrastive / ranking structure:

- positive: actual future / successful command sequence
- negatives: mismatched goals, shuffled commands, failed futures, off-trajectory samples

Loss candidates:

- hinge / margin ranking loss
- InfoNCE style contrastive objective
- binary logistic energy discrimination

This is the main step that would justify the **EBM** label.

### 4. Separate predictable from unpredictable content

A canonical JEPA should not be forced to model every pixel-level detail.

You can push the latent to represent controllable, task-relevant structure by:

- predicting only a projector head rather than raw latent
- using stop-gradient target branches
- using uncertainty-aware or latent bottleneck heads
- adding invariance augmentations

### 5. Upgrade the planner

After the model is stable:

- move from constant command rollouts to command sequences
- use CEM / MPPI over body-frame commands
- optionally learn a value head over latent-goal compatibility
- evaluate how latent score correlates with true task success

### 6. Add held-out offline evaluation

This is the most important missing benchmark.

Create a proper validation script that:

- loads held-out rollouts
- computes teacher-forced and free-running latent rollout error
- logs results per checkpoint
- plots horizon-vs-error curves

---

## Suggested roadmap

### Phase 1: finish the baseline cleanly

- finish the current 20-epoch run
- save the best checkpoint(s)
- generate polished demo videos
- collect a small set of fixed evaluation plots
- document exactly what the current model is

This gives you a stable baseline and a clean “version 0” story.

### Phase 2: canonical JEPA pass

Fork a new branch and add:

- EMA target encoder
- improved evaluation suite
- optional multi-horizon predictor heads
- stronger offline metrics

### Phase 3: true JEPA EBM pass

Add:

- learned energy head
- positives / negatives / ranking loss
- goal-conditioned energy planning
- command-sequence planning

At that point, the system can honestly be presented as a **JEPA + learned energy-based planner**.

---

## Practical answer to “what should I do next?”

Yes: **finish the current run, make the demos, and then move to a more canonical JEPA**.

Why this is the right move:

- you already have healthy enough metrics to justify completing the run
- you need a stable baseline before changing the method family
- demos are valuable for motivation, presentations, and debugging
- switching architectures mid-run would muddy the story and remove your reference point

The current version is already good enough to be a strong:

- proof of concept
- baseline latent planner
- demo platform
- stepping stone to a more canonical JEPA / EBM design

---

## One-line project description

> **TinyQuadJEPA is a two-level quadruped control stack in which a PPO locomotion controller executes high-level body commands selected by an action-conditioned multimodal latent predictor trained with a VICReg-style JEPA objective.**
