import torch

class STS3215_Actuator:
    def __init__(self, num_envs, device, dt=0.02):
        """
        Simulates a batch of STS3215 Serial Bus Servos.
        Batch-optimized for Genesis (runs on GPU).
        """
        self.device = device
        self.num_envs = num_envs

        # --- STS3215 Physical Specs (11.1V 3S LiPo) ---
        self.stall_torque = 3.0          # ~Nm
        self.no_load_speed = 6.0         # rad/s

        # --- Reality Gap Parameters ---
        self.dt = dt
        self.latency_steps = int(0.02 / dt)

        # Command history: (env, motor, history)
        self.history_len = self.latency_steps + 1
        self.command_queue = torch.zeros(
            (num_envs, 12, self.history_len),
            device=device,
            dtype=torch.float32,
        )

        # PD gains
        self.kp = 25.0
        self.kd = 0.5

    def step(self, target_pos, current_pos, current_vel, voltage=11.1):
        """
        target_pos:  (num_envs, 12) - policy command (may require grad)
        current_pos: (num_envs, 12) - joint positions (usually detached)
        current_vel: (num_envs, 12) - joint velocities (usually detached)
        """
        # --- 1) Update history buffer WITHOUT building autograd graphs ---
        with torch.no_grad():
            # shift left: [1..end] -> [0..end-1]
            if self.history_len > 1:
                self.command_queue[:, :, :-1].copy_(self.command_queue[:, :, 1:])
            # write new command into last slot (store detached)
            self.command_queue[:, :, -1].copy_(target_pos.detach())

            delayed_target = self.command_queue[:, :, 0]

        # --- 2) Voltage sag model (scalar) ---
        voltage_factor = float(max(0.0, min(1.0, float(voltage) / 11.1)))
        effective_stall_torque = self.stall_torque * voltage_factor

        # --- 3) PD controller ---
        torque = self.kp * (delayed_target - current_pos) - self.kd * current_vel

        # --- 4) Torque-speed limit ---
        torque_limit = effective_stall_torque * (1.0 - torch.abs(current_vel) / self.no_load_speed)
        torque_limit = torch.clamp(torque_limit, min=0.0)

        real_torque = torch.clamp(torque, -torque_limit, torque_limit)
        return real_torque

    def reset(self):
        with torch.no_grad():
            self.command_queue.zero_()
