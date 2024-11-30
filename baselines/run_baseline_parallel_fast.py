from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    ep_length = 2048 * 10
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': True, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
        'explore_weight': 3,  # 2.5
        'curriculum_stage': 1  # Start with stage 1
    }

    print(env_config)

    num_cpu = 16  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                             name_prefix='poke')
    # env_checker.check_env(env)
    # learn_steps_1 = 5
    # learn_steps_2 = 15
    # learn_steps_3 = 20
    # put a checkpoint here you want to start from
    file_name = 'session_e41c9eff/poke_38207488_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, gamma=0.998, batch_size=128)

    curriculum_stages = 13

    for stage in range(1, curriculum_stages + 1):
        print(f"\nStarting curriculum stage {stage}\n")
        env_config['curriculum_stage'] = stage
        env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
        model.set_env(env)
        if stage == 1:
            learn_steps = 5
        if stage == 2:
            learn_steps = 15
        if stage == 3:
            learn_steps = 20
        for i in range(learn_steps):
            try:
                model.learn(total_timesteps=(ep_length) * num_cpu * 1000, callback=checkpoint_callback)
            except Exception as e:
                print(f"Error during learning at stage {stage}, step {i}: {e}")
