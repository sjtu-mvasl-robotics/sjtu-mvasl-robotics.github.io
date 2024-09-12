    def _reward_tracking_lin_vel(self):
        env = self.env
        lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1)
        tracking_sigma = 0.1  # Temperature parameter for normalization
        return torch.exp(-lin_vel_error / tracking_sigma) * 0.5  # Adjust scaling
    
    # Tracking angular velocity
    def _reward_tracking_ang_vel(self):
        env = self.env
        ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
        tracking_sigma = 0.1  # Temperature parameter for normalization
        return torch.exp(-ang_vel_error / tracking_sigma) * 0.5  # Adjust scaling
    
    # Orientation Stability
    def _reward_orientation_stability(self):
        env = self.env
        projected_gravity = env.projected_gravity
        stability_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        stability_sigma = 0.1  # Temperature parameter for normalization
        return torch.exp(-stability_error / stability_sigma) * 0.2  # Adjust scaling
    
    # Action Rate
    def _reward_action_rate(self):
        env = self.env
        action_rate_penalty = torch.sum(torch.square(env.last_actions - env.actions), dim=1)
        return -0.01 * action_rate_penalty  # Adjust scaling
    
    # Base Height
    def _reward_base_height(self):
        env = self.env
        desired_base_height = 0.62
        base_height_error = torch.abs(self._get_base_height() - desired_base_height)
        height_sigma = 0.1  # Temperature parameter for normalization
        return torch.exp(-base_height_error / height_sigma) * 0.1  # Adjust scaling
    
    # Feet Height
    def _reward_feet_height(self):
        env = self.env
        feet_height_error = torch.sum((env.current_max_feet_height - env.last_max_feet_height).abs(), dim=1)
        feet_height_sigma = 0.1  # Temperature parameter for normalization
        return torch.exp(-feet_height_error / feet_height_sigma) * 0.1  # Adjust scaling
    
    # Feet Balance
    def _reward_feet_balance(self):
        env = self.env
        feet_balance_penalty = torch.var(env.last_feet_air_time, dim=-1)
        return -0.01 * feet_balance_penalty  # Adjust scaling
    
    # Survival
    def _reward_survival(self):
        env = self.env
        return self._survival() * 0.1  # Adjust scaling
    
    # Torque Limits
    def _reward_torque_limits(self):
        env = self.env
        torque_penalty = torch.sum(
            (torch.abs(env.torques) - env.torque_limits * env.cfg.rewards.soft_torque_limit).clamp(min=0.), dim=1)
        return -0.01 * torque_penalty  # Adjust scaling
    
    # Joint Limits
    def _reward_joint_limits(self):
        env = self.env
        joint_limit_penalty = torch.sum(
            (torch.abs(env.dof_pos) - env.dof_pos_limits[:, 1]).clamp(min=0.), dim=1)
        joint_limit_penalty += torch.sum(
            (env.dof_pos_limits[:, 0] - torch.abs(env.dof_pos)).clamp(min=0.), dim=1)
        return -0.01 * joint_limit_penalty  # Adjust scaling
    
    # Collision Avoidance
    def _reward_collision_avoidance(self):
        env = self.env
        collision_penalty = torch.sum(1. * (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1),
                                      dim=1)
        return -0.01 * collision_penalty  # Adjust scaling
