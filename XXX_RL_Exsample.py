from .agent import AgentPPO
from env import PendulumEnv
from config import Config, get_gym_env_args
from run import train_agent


def train_ppo_for_pendulm():
    agent_class = AgentPPO # DRL algorithm name
    env_class = PendulumEnv # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env = PendulumEnv(), if_print = True)  # return env_args

    args = Config(agent_class, env_class, env_args)
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small

    train_agent(args)

if __name__ == "__main__":
    train_ppo_for_pendulum()