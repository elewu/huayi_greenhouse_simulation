import rldev

import numpy as np

from apep.env import EnvGazebo


def run_one_episode(env):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(rldev.Data(action=action))
        # print(f'next_state: {next_state}, reward: {reward}, done: {done}')

    return



def main():
    from apep.state_space import StateSpaceGoalReach
    from apep.action_space import ActionSpaceDeltaPose
    from apep.reward_func import RewardFuncDistance

    state_space = StateSpaceGoalReach()
    action_space = ActionSpaceDeltaPose()
    reward_func = RewardFuncDistance()

    env = EnvGazebo(state_space=state_space, action_space=action_space, reward_func=reward_func)
    run_one_episode(env)



if __name__ == '__main__':
    main()






