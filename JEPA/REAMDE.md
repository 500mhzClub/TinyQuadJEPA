# JEPA World Model (System 2)

This is the "Cortex" of the robot. It does not control the motors directly. It predicts the consequences of actions.

## Architecture
We are using a **Joint-Embedding Predictive Architecture** (I-JEPA style), adapted for robotics.

### Components
1. **Context Encoder:**
   * **Vision:** ResNet-18 backbone (taking 128x128 images).
   * **Proprioception:** A small MLP encoding the servo feedback (Torque/Position).
   * **Fusion:** Concatenates both into a single Latent State vector $z_t$.
2. **Predictor:**
   * A 4-layer MLP that takes $(z_t, action)$ and predicts the *next* latent state $z_{t+1}$.
   * It learns physics: "If I step forward (Action) and see a wall (State), the next State involves a torque spike (Collision)."

## The "Dreamer" Loop (MPC)
On the Jetson, we run this at 10Hz-30Hz:
1. **Encode:** Get current $z_t$.
2. **Dream:** The Predictor generates 50 "Tentacles" (random action sequences) into the future.
3. **Evaluate:** We score the tentacles.
   * *Bad:* High torque spike (Collision), highly tilted IMU (Falling).
   * *Good:* Smooth velocity, upright.
4. **Act:** We send the best action sequence to the System 1 Walking Policy.

## Training & Fine-Tuning
1. **Pre-training:** Done entirely on the dataset from `sim/`.
2. **Sim-to-Real Alignment:** The real camera is noisy and the lens distorts.
   * **Step:** Record 30 mins of the real robot walking around the house.
   * **Fine-tune:** Run 5 epochs on the Radeon to align the simulated latents with real-world latents.