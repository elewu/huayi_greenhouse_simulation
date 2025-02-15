import rldev

import os
import importlib
import copy
import pickle

import numpy as np

from apep.tools import global_to_local
from apep.env import EnvGazebo
from apep.data_collector import DataCollector


def run_one_episode(config, env: EnvGazebo, data_collector: DataCollector):
    data_collector.reset()
    env.reset()
    env.gazebo_state.unpause(1)
    env.gazebo_state.pause()
    done = False
    save_data = {}
    while not done:
        # next_state, reward, done, info = env.step(rldev.Data(action=env.action_space.sample()))
        next_state, reward, done, info = env.step(rldev.Data(action=global_to_local(env.goal_pose, env.current_pose)))
        save_data[env.time_step] = copy.deepcopy(data_collector.data_dict)
    save_path = os.path.join(config.path_pack.output_path, f'episode_{env.step_reset}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    return



def init(config):
    ### common param
    rldev.setup_seed(config.seed)

    ### env param
    from apep.state_space import StateSpaceGoalReach
    from apep.action_space import ActionSpaceDeltaPose, ActionSpaceVelocity
    from apep.reward_func import RewardFuncDistance
    from apep.metric_func import MetricFuncDistanceToGoal
    from apep.scenario_random_func import ScenarioRandomFuncGreenhouse

    state_space = StateSpaceGoalReach()
    action_space = ActionSpaceDeltaPose()
    reward_func = RewardFuncDistance()
    metric_func = MetricFuncDistanceToGoal()
    scenario_random_func = ScenarioRandomFuncGreenhouse()
    max_steps = 100

    # state_space = StateSpaceGoalReach()
    # action_space = ActionSpaceVelocity()
    # reward_func = RewardFuncDistance()
    # metric_func = MetricFuncDistanceToGoal()
    # scenario_random_func = ScenarioRandomFuncGreenhouse()
    # max_steps = 100

    launch_file = os.path.join(os.path.dirname(importlib.util.find_spec('apep').origin), '../src/mybot_description/launch/green_house.launch')

    env = EnvGazebo(
        state_space=state_space, action_space=action_space,
        reward_func=reward_func, metric_func=metric_func,
        scenario_random_func=scenario_random_func,
        max_steps=max_steps, launch_file=launch_file
    )

    config.set('env_name', "GazeboCollectData")
    config.set('dim_state', env.state_space.dim_state)
    config.set('dim_action', env.action_space.dim_action)
    config.set('continuous_action_space', env.action_space.continuous)

    model_name = f'{config.env_name}'
    writer = rldev.create_dir(config, model_name)

    data_collector = DataCollector()
    return writer, env, data_collector



def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', dest='description', default='Nothing', help='[Method] description.')

    argparser.add_argument('--seed', default=0, type=int, help='seed.')
    argparser.add_argument('--render', action='store_true', help='render the env (default: False)')

    args = argparser.parse_args()
    return args



def main():
    config = rldev.YamlConfig()
    args = generate_args()
    config.update(args)

    writer, env, data_collector = init(config)
    try:
        for _ in range(1000):
            run_one_episode(config, env, data_collector)
    finally:
        writer.close()


if __name__ == '__main__':
    main()






