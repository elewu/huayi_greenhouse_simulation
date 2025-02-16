import rldev

import random
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

from apep.utils.transformation_tools import global_to_local, local_to_global
from apep.data.format import ModelOutputDim, ObjectTensorOutputSize
from apep.model.template import ModelWrapper, compute_objectives
from apep.objective import rl_objectives
from apep.metric import rl_metrics, safety_metrics, planning_metrics



class Planner(ModelWrapper):
    def __init__(
        self,
        model_params: rldev.BaseData,
        feature_params,
        target_params,
        meta_cfg,
    ):
        self.random_sample = model_params.random_sample

        self.objective_value = rl_objectives.PPOValueObjective(weight=model_params.weight_value)
        # self.objective_cls = planning_objectives.TrajectoryWTAClsObjective()

        objectives_il = [
            # planning_objectives.TrajectoryImitationObjective(),
        ]
        objectives_rl = [
            rl_objectives.PPOPolicyObjective(),
            rl_objectives.PPOEntropyObjective(),
        ]
        self.objectives_il = {objective.name: objective for objective in objectives_il}
        self.objectives_rl = {objective.name: objective for objective in objectives_rl}
        # self.loss_collision = safety_objectives.meanEgoCollisionRate(meta_cfg.cfg)


        metrics = [
            # planning_metrics.MeanAcceleration(),
            # planning_metrics.MeanJerk(),

            planning_metrics.TrajectoryVariance(),
        ]
        timewise_metrics = [
            # safety_metrics.meanEgoOutMapBoundaryV2(meta_cfg.cfg, interpolate_scale=self.interpolate_scale, topk=1, fix_mode=True),
            # safety_metrics.meanEgoCollisionRate(meta_cfg.cfg, interpolate_scale=self.interpolate_scale, topk=1, fix_mode=True),
        ]

        quality_metrics = [
            safety_metrics.ProgressTowardsGoal(meta_cfg.cfg),
            safety_metrics.FinalApproachesGoal(meta_cfg.cfg),
            safety_metrics.InBound(meta_cfg.cfg, interpolate_scale=model_params.interpolate_scale, pad=False),
            safety_metrics.NearCenter(meta_cfg.cfg, interpolate_scale=model_params.interpolate_scale, pad=False),
            # safety_metrics.meanEgoCollisionRate(meta_cfg.cfg, interpolate_scale=self.interpolate_scale, pad=False),
            # safety_metrics.meanEgoOutMapBoundary(meta_cfg.cfg, interpolate_scale=self.interpolate_scale, pad=False),
            # # safety_metrics.meanEgoRouteAlign(meta_cfg.cfg),
            # safety_metrics.meanEgoSpeed(meta_cfg.cfg),
        ]

        if model_params.rl_opt:
            metrics.append(rl_metrics.PPORatio())
            metrics.append(rl_metrics.PPORatioRange())
        
        ModelWrapper.__init__(self, model_params, feature_params, target_params, meta_cfg, [], metrics, timewise_metrics, quality_metrics)

        self.detach_ar = model_params.detach_ar
        self.il_opt = model_params.il_opt
        self.rl_opt = model_params.rl_opt
        # self.use_critic = model_params.use_critic

        self.d_model = model_params.context_encoder['D_MODEL']
        self.num_future_poses = target_params.future_trajectory_sampling.num_poses

        influence_time = 2.0 ## seconds
        # self.gamma = np.power(0.1, 1/(influence_time / target_params.future_trajectory_sampling.interval_length))
        self.gamma = model_params.gamma
        self.lamda = model_params.lamda

        self.policy = GaussianPolicy(model_params, feature_params, target_params, meta_cfg)
        return


    def forward(self, data: rldev.Data, scenarios=None, rl_opt=True):
        res = self.forward_training(data, scenarios)
        return res



    def forward_training(self, data: rldev.Data, scenarios):
        data = data.copy()

        ### for random sampling
        num_modes = 1
        if not self.training and self.random_sample > 0:
            num_modes = self.random_sample

        ##############################################
        ### state0
        pose = data.pose.unsqueeze(ObjectTensorOutputSize.mode).repeat_interleave(num_modes, dim=ObjectTensorOutputSize.mode)
        goal = data.goal.unsqueeze(ObjectTensorOutputSize.mode).repeat_interleave(num_modes, dim=ObjectTensorOutputSize.mode)
        map_polylines = data.map_polylines.unsqueeze(ObjectTensorOutputSize.mode).repeat_interleave(num_modes, dim=ObjectTensorOutputSize.mode)

        num_future_timestamps = 8  ### todo:

        ##############################################

        states = []
        actions_data = []
        poses = []
        poses_mask = []
        for time_idx in range(num_future_timestamps):
            state = rldev.Data(
                pose=pose, map_polylines=map_polylines,
                goal=goal,
            )
            action_data = self.select_action(state)
            ego_next_pose = action_data.global_pose
            ego_next_pose_mask = action_data.pose_mask

            states.append(state)
            actions_data.append(action_data)
            poses.append(ego_next_pose)
            poses_mask.append(ego_next_pose_mask)

            ##############################################
            ### prepare for next stamp
            pose = ego_next_pose
            if self.detach_ar:  ### should be the same
                pose = pose.detach()
        state = rldev.Data(
            pose=pose, map_polylines=map_polylines,
            goal=goal,
        )
        states.append(state)


        traj = torch.stack(poses, dim=-2)
        trajectory = rldev.Data(data=traj, mask=torch.stack(poses_mask, dim=-1))


        res = rldev.Data(
            trajectory=trajectory,

            data=data,
            states=states,
            actions_data=actions_data,
        )
        return res




    def compute_objectives(self, input_data, predictions, experience, targets, scenarios, global_step):
        batch_size = predictions['trajectory'].data.shape[0]

        ### data
        state_old = experience.state
        action_old = experience.action_data.action
        num_timesteps = action_old.shape[1]
        ### forward evaluate
        action_data = self.policy(
            state_old.flatten(end_dim=1),
            action=action_old.flatten(end_dim=1),
        ).unflatten(dim=0, sizes=(batch_size, -1))
        predictions.update(action_data=action_data)

        # self.loss_collision.compute(predictions['data'], predictions, targets)

        objectives = {}

        objectives.update(**compute_objectives([self.objective_value], action_data, experience, scenarios))
        if self.rl_opt:
            objectives.update(**compute_objectives(self.objectives_rl.values(), action_data, experience, scenarios))

        # print(f'global_step [{global_step}]: ', rldev.Data(**objectives).mean())
        return objectives





    def select_action(self, state):
        shape = state.pose.shape[:-1]
        device = state.pose.device
        if self.training:
            z = self.policy.get_noise((*shape, self.policy.ActionDim.dim), device)
        else:
            if self.random_sample > 0:
                z = self.policy.get_noise((*shape, self.policy.ActionDim.dim), device)
            else:
                z = torch.zeros((*shape, self.policy.ActionDim.dim), device=device)
        action_data = self.policy(state, z=z)
        return action_data







class GaussianPolicy(nn.Module):
    def __init__(
        self,
        model_params: rldev.BaseData,
        feature_params,
        target_params,
        meta_cfg,
    ):
        super().__init__()
        self.d_model = model_params.context_encoder['D_MODEL']
        # self.control_space = model_params.control_space
        self.independent_gaussian = model_params.independent_gaussian
        self.normalize_mean = model_params.normalize_mean
        self.normalize_std = model_params.normalize_std
        self.separate_decoder = model_params.separate_decoder
        self.vehicle_dynamics = model_params.vehicle_dynamics

        if self.vehicle_dynamics == 'delta':
            dynamics_func = IdentityModel
        elif self.vehicle_dynamics == 'unicycle':
            dynamics_func = UnicycleModel
        else:
            raise RuntimeError(f'Known func: {self.vehicle_dynamics}')
        self.dynamics = dynamics_func(model_params.ego_vehicle_parameters, target_params.future_trajectory_sampling)
        self.ActionDim = self.dynamics.ActionDim

        self.logstd_min = -5
        # self.logstd_max = 0
        self.max_action = self.dynamics.max_action
        self.logstd_max = self.dynamics.max_action.log()

        if model_params.encoder_type == 'v1':
            self.encoder = Encoder(model_params)
        else:
            raise NotImplementedError
        self.dim_feature = self.encoder.dim_feature

        if self.separate_decoder:
            self.decoder_mean = nn.Sequential(
                nn.Linear(self.dim_feature, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, self.ActionDim.dim),
            )
            self.decoder_std = nn.Sequential(
                nn.Linear(self.dim_feature, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, self.ActionDim.dim),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.dim_feature, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, self.ActionDim.dim *2),
            )
        self.value_func = nn.Sequential(
            nn.Linear(self.dim_feature, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )
        return


    def forward(self, state, z=None, action=None, dropout_rate={}):
        ego_feature, ego_mask, out_dropout_rate, out_attentions = self.encoder(state, dropout_rate=dropout_rate)
        if self.separate_decoder:
            mean = self.decoder_mean(ego_feature)
            logstd = self.decoder_std(ego_feature)
        else:
            mean, logstd = self.decoder(ego_feature).chunk(2, dim=-1)
        assert mean.shape[-1] == self.ActionDim.dim
        assert logstd.shape[-1] == self.ActionDim.dim
        value = self.value_func(ego_feature)
        if self.normalize_mean:
            mean = torch.tanh(mean) * self.max_action[None,None].to(mean)
        if self.normalize_std:
            logstd = torch.tanh(logstd)
            logstd_max = self.logstd_max[None,None].to(logstd)
            logstd = (logstd_max-self.logstd_min) * logstd + (logstd_max + self.logstd_min)
            logstd *= 0.5
        var = logstd.exp().square()
        # sigma = logstd

        with torch.no_grad():
            ego_pose = state.pose.detach()
        # if not self.control_space:
        #     mean, var = pose_distribution_local_to_global(mean, var, ego_pose)
        std = var.sqrt()
        sigma = logstd = std.log()

        # dist = MultivariateNormal(mean, torch.diag_embed(var))
        z_dist = self.base_distribution((mean.shape[0], self.ActionDim.dim), mean.device)
        if z != None: ## direct
            action = mean + std * z
        elif action != None: ## inverse
            z = (action - mean) * (-sigma).exp()
        else:
            raise NotImplementedError

        ####################################################################
        ### method 1
        # logprob = dist.log_prob(action).unsqueeze(-1)

        ### method 2
        if self.independent_gaussian:
            log_prob_z = z_dist.log_prob(z.transpose(0,1)).sum(dim=-1).transpose(0,1)
        else:
            log_prob_z = z_dist.log_prob(z.transpose(0,1)).transpose(0,1)
        log_det = sigma.sum(dim=-1)
        logprob = (log_prob_z - log_det).unsqueeze(-1)
        
        ####################################################################
        global_pose = self.dynamics(ego_pose, action)

        return rldev.Data(
            action=action, global_pose=global_pose, logprob=logprob, mu=mean, var=var, value=value,
            pose_mask=ego_mask,
            # agent_feature=agent_feature, agent_mask=agent_mask,
            # dropout_rate=rldev.Data(**out_dropout_rate).unflatten(0, (-1, self.num_modes)),
            # out_attentions=out_attentions,
        )
    

    def base_distribution(self, shape, device):
        zeros = torch.zeros(shape, device=device)
        ones = torch.ones(shape, device=device)
        if self.independent_gaussian:
            z_dist = Normal(zeros, ones)
        else:
            z_dist = MultivariateNormal(zeros, torch.diag_embed(ones))    
        return z_dist


    def get_noise(self, shape, device):
        z_dist = self.base_distribution(shape, device)
        z = z_dist.sample()
        return z




class Encoder(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.dim_feature = model_params.context_encoder['D_MODEL']

        self.map_encoder = PointNet(3, 64, self.dim_feature)
        self.goal_encoder = nn.Linear(3, self.dim_feature)
    
    def forward(self, state, dropout_rate={}):

        goal = global_to_local(state.goal, state.pose)
        goal_feature = self.goal_encoder(goal)

        map_polylines = global_to_local(state.map_polylines, state.pose[..., None,None,:])
        map_polylines_mask = torch.ones_like(map_polylines[..., 0], dtype=bool)
        batch_size, num_modes = map_polylines.shape[:2]
        map_feature = self.map_encoder(map_polylines.flatten(end_dim=1), map_polylines_mask.flatten(end_dim=1)).unflatten(0, (batch_size, num_modes))
        
        ego_feature = goal_feature + map_feature.sum(dim=-2)
        ego_mask = torch.ones_like(ego_feature[..., 0], dtype=bool)
        return ego_feature, ego_mask, {}, rldev.Data()





class PointNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.first_mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels),
        )


    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.first_mlp(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid


        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.second_mlp(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        return feature_buffers





class IdentityModel(nn.Module):
    StateDim = ModelOutputDim
    ActionDim = ModelOutputDim
    # class ActionDim:
    #     x = 0
    #     y = 1
    #     heading = 2
    #     dim = 3
    max_action = torch.tensor([1, 1, 1], dtype=torch.float32)
    max_action_inner = torch.tensor([5, 2, 1.57], dtype=torch.float32)

    def __init__(self, vehicle_parameters, future_trajectory_sampling):
        super().__init__()
        return


    def forward(self, state, action):
        action = action.clamp(-1, 1) * self.max_action_inner[None,None].to(action)
        return local_to_global(action, state)





class UnicycleModel(nn.Module):
    StateDim = ModelOutputDim
    class ActionDim:
        v = 0
        w = 1
        dim = 2

    max_action = torch.tensor([1, 1], dtype=torch.float32)
    max_action_inner = torch.tensor([2, 1], dtype=torch.float32)

    def __init__(self, vehicle_parameters, future_trajectory_sampling):
        super().__init__()

        self.vehicle_parameters = vehicle_parameters
        self.max_omega = 1
        self.min_velocity = 0.0
        self.max_velocity = 22.0 # 22.0

        self.dt = 0.1
        self.delta_t = future_trajectory_sampling.interval_length
        self.num_iters = int(self.delta_t / self.dt)
        return


    def forward(self, state, action):
        action = action.clamp(-1, 1) * self.max_action_inner[None,None].to(action)
        v = action[..., self.ActionDim.v]
        w = action[..., self.ActionDim.w]

        x = state[..., self.StateDim.x]
        y = state[..., self.StateDim.y]
        heading = state[..., self.StateDim.heading]

        for _ in range(self.num_iters):
            next_x = x + self.dt *v * torch.cos(heading)
            next_y = y + self.dt *v * torch.sin(heading)
            next_heading = rldev.pi2pi_tensor(heading + self.dt * w)

            x, y, heading = next_x, next_y, next_heading

        next_state = torch.stack([next_x, next_y, next_heading], dim=-1)
        return next_state


