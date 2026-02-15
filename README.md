# TinyQuadJEPA: Autonomous World Models on the Edge

This is the repo for TinyQuadJEPA

The goal: build a 3D-printed, 12-DOF quadruped that  *understands* physics using a JEPA (Joint-Embedding Predictive Architecture) World Model.

1. **System 1 (Reflex):** A blind, fast PPO policy that keeps the dog upright.
2. **System 2 (Reasoning):** A visual JEPA World Model that predicts the future in latent space to handle navigation.

## The Hardware Stack

* **Workstation:** AMD Ryzen 9 9950X3D (Physics) + Radeon AI Pro 9700 32GB (Vision Training).
* **Robot Brain:** NVIDIA Jetson Orin Nano 8GB (Running TensorRT).
* **Body:** Custom 3D-printed chassis (Mini Pupper v2 style) with STS3215 Serial Bus Servos.

## Repository Structure

* `sim/`: The Genesis-based simulation environment. This is where we train the blind walker and generate massive visual datasets.
* `JEPA/`: The World Model architecture. PyTorch code for the Context Encoder, Predictor, and the MPC "Dreamer" loop.
* `prints/`: STL files and slicing profiles for the Bambu Lab A1.
* `hardware/`: Bill of Materials, wiring diagrams, and the "Hacker" build guide.
