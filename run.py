import os
import time
import torch
import numpy as np

from config import Config, build_env
from agent import ReplayBuffer

def train_agent(args: Config):
    args.init_before_training()

    env = 

def render_agent(env_class, env_args: dict, net_dims: [int], agent_class, actor_path: str, render_times: int = 8):
    pass

class Evaluator:
    pass

def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):  # cumulative_rewards and episode_steps
    pass

def draw_learning_curve_using_recorder(cwd: str):
    pass

