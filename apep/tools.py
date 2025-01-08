
import functools

import numpy as np
import torch



def global_to_local(poses, pose0):
    if isinstance(poses, np.ndarray):
        framework = np
        stack_func = functools.partial(np.stack, axis=-1)
    elif isinstance(poses, torch.Tensor):
        framework = torch
        stack_func = functools.partial(torch.stack, dim=-1)
    else:
        raise NotImplementedError


    x_world, y_world, theta_world = poses[...,0], poses[...,1], poses[...,2]
    x0, y0, theta0 = pose0[...,0], pose0[...,1], pose0[...,2]

    x_local = (x_world-x0) *framework.cos(theta0) + (y_world-y0) *framework.sin(theta0)
    y_local =-(x_world-x0) *framework.sin(theta0) + (y_world-y0) *framework.cos(theta0)
    delta_theta = pi2pi(theta_world - theta0)
    return stack_func([x_local, y_local, delta_theta])



def local_to_global(poses, pose0):
    if isinstance(poses, np.ndarray):
        framework = np
        stack_func = functools.partial(np.stack, axis=-1)
    elif isinstance(poses, torch.Tensor):
        framework = torch
        stack_func = functools.partial(torch.stack, dim=-1)
    else:
        raise NotImplementedError
    
    x_local, y_local, theta_local = poses[...,0], poses[...,1], poses[...,2]
    x0, y0, theta0 = pose0[...,0], pose0[...,1], pose0[...,2]

    x_world = x0 + x_local *framework.cos(theta0) - y_local *framework.sin(theta0)
    y_world = y0 + x_local *framework.sin(theta0) + y_local *framework.cos(theta0)
    theta_world = pi2pi(theta_local + theta0)

    return stack_func([x_world, y_world, theta_world])





def pi2pi(theta):
    if isinstance(theta, np.ndarray) or isinstance(theta, float):
        return pi2pi_numpy(theta)
    elif isinstance(theta, torch.Tensor):
        return pi2pi_torch(theta)
    else:
        raise NotImplementedError
    

def pi2pi_numpy(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def pi2pi_torch(theta):
    TWO_PI = 2 * np.pi
    theta = torch.fmod(torch.fmod(theta, TWO_PI) + TWO_PI, TWO_PI)
    return torch.where(theta > np.pi, theta - TWO_PI, theta)


