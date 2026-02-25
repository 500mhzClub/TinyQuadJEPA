# Simulation Environment (Genesis)

We are using **Genesis** because Isaac Lab locks us into CUDA. Genesis supports Vulkan and runs natively on the Radeon Pro 9700.

## The Reality Gap
Simulations usually fail because they assume motors are magic. Real motors suck. They have latency, friction, and they get weaker as the battery dies.

This folder contains the `STS3215_Actuator` Python class. It injects:
* **Latency:** A 20ms delay buffer to mimic UART comms.
* **Voltage Sag:** Torque drops as the simulated LiPo voltage drops.
* **Torque-Speed Curve:** Physics limits based on back-EMF.

**Do not turn these off.** If you train on perfect motors, the real robot may struggle.

## Workflow

### 1. System 1 Training (The Spinal Cord)
We use the Ryzen 9950X3D to spawn 1,024 parallel robots.
* **Goal:** Train a "Blind Walker" policy (PPO).
* **Input:** Joint angles + IMU (No vision).
* **Output:** Motor velocity commands.
* **Domain Randomization:** We vary friction (ice to carpet) and payload mass to make the policy robust.

### 2. Data Generation (For JEPA)
Once the robot can walk blindly, we turn on the cameras.
* **Script:** `generate_unseen_worlds.py`
* **Process:** Procedurally generates infinite rooms with random walls, stairs, and textures.
* **Output:** Saves tuples of `(Image_t, Proprioception_t, Action_t, Image_t+1)`.
* **Volume:** We need about 500k frames. This leverages the 32GB VRAM on the Radeon to batch render high-res textures.