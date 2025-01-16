import rldev, rllib

import numpy as np
import torch

from apep.env import EnvGazebo


def run_one_episode(i_episode, config, writer, env, method):
    running_reward = 0
    state = env.reset()
    done = False
    while not done:
        # action_data = rldev.Data(action=env.action_space.sample())
        action_data = method.select_action( torch.from_numpy(state).unsqueeze(0).float() )
        next_state, reward, done, _ = env.step(action_data.squeeze(0).numpy())
        experience = rllib.template.Experience(
                state=torch.from_numpy(state).float().unsqueeze(0),
                next_state=torch.from_numpy(next_state).float().unsqueeze(0),
                action_data=action_data, reward=reward, done=done)
        method.store(experience)

        state = next_state
        running_reward += reward
    
    method.update_parameters()
    
    writer.add_scalar('env/reward', running_reward, i_episode)
    return



def init(config):
    ### common param
    rldev.setup_seed(config.seed)
    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    ### env param
    from apep.state_space import StateSpaceGoalReach
    from apep.action_space import ActionSpaceDeltaPose, ActionSpaceVelocity
    from apep.reward_func import RewardFuncDistance

    # state_space = StateSpaceGoalReach()
    # action_space = ActionSpaceDeltaPose()
    # reward_func = RewardFuncDistance()
    # max_steps = 50

    state_space = StateSpaceGoalReach()
    action_space = ActionSpaceVelocity()
    reward_func = RewardFuncDistance()
    max_steps = 100

    env = EnvGazebo(state_space=state_space, action_space=action_space, reward_func=reward_func, max_steps=max_steps)

    config.set('env_name', "GazeboReachGoal")
    config.set('dim_state', env.state_space.dim_state)
    config.set('dim_action', env.action_space.dim_action)
    config.set('continuous_action_space', env.action_space.continuous)

    ### method param
    method_name = config.method.upper()
    if method_name == 'PPO_SIMPLE':
        from rllib.ppo_simple import PPOSimple as Method
        if config.continuous_action_space:
            config.set('net_ac', rllib.ppo_simple.ActorCriticContinuous)
        else:
            config.set('net_ac', rllib.ppo_simple.ActorCriticDiscrete)

    elif method_name == 'A1C':
        from rllib.a1c import A1C as Method
        if config.continuous_action_space:
            config.set('net_ac', rllib.a1c.ActorCriticContinuous)
        else:
            config.set('net_ac', rllib.a1c.ActorCriticDiscrete)
    elif method_name == 'PPO':
        from rllib.ppo import PPO as Method
        if config.continuous_action_space:
            config.set('net_ac', rllib.ppo.ActorCriticContinuous)
        else:
            config.set('net_ac', rllib.ppo.ActorCriticDiscrete)
    elif method_name == 'PPG':
        from rllib.ppg import PPG as Method
        if config.continuous_action_space:
            config.set('net_ac', rllib.ppg.ActorCriticContinuous)
        else:
            config.set('net_ac', rllib.ppg.ActorCriticDiscrete)
    
    elif method_name == 'SAC':
        from rllib.sac import SAC as Method

    else:
        raise NotImplementedError('Not support this method.')

    model_name = f'{config.env_name}/{Method.__name__}'
    writer = rllib.basic.create_dir(config, model_name)
    method = Method(config, writer)
    return writer, env, method



def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', dest='description', default='Nothing', help='[Method] description.')


    argparser.add_argument('-m', '--method', default='ppo', type=str, help='[Method] Method to use.')

    argparser.add_argument('--model-dir', default='None', type=str, help='[Model] dir contains model (default: False)')
    argparser.add_argument('--model-num', default=-1, type=str, help='[Model] model-num to use.')

    argparser.add_argument('--seed', default=0, type=int, help='seed.')
    argparser.add_argument('--render', action='store_true', help='render the env (default: False)')

    ### method params
    argparser.add_argument('--batch-size', default=128, type=int, help='[Method Param]')
    argparser.add_argument('--buffer-size', default=2000, type=int, help='[Method Param]')

    argparser.add_argument('--weight-value', default=0.5, type=float, help='[Method Param] available: PPO')
    argparser.add_argument('--weight-entropy', default=0.001, type=float, help='[Method Param] available: PPO')

    argparser.add_argument('--gamma', default=0.9, type=float, help='[Method Param] available: PPO')

    args = argparser.parse_args()
    return args



def main():
    config = rllib.basic.YamlConfig()
    args = generate_args()
    config.update(args)

    writer, env, method = init(config)
    try:
        for i_episode in range(10000):
            run_one_episode(i_episode, config, writer, env, method)
    finally:
        writer.close()


if __name__ == '__main__':
    main()






