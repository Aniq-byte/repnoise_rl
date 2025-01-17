import os
import time
import argparse
import logging
import json
import subprocess
import numpy as np

import torch

from pathlib import Path
import pickle

from machiavelli.agent.memory import Transition

from machiavelli.game.machiavelli_env import MachiavelliEnv
from machiavelli.game.machiavelli_env import Trajectory
import machiavelli.agent.logger as logger
from random import choice
from collections import defaultdict
from tqdm import tqdm

from machiavelli.agent.power_regularizer import PowerRegularizer
from machiavelli.agent.drrn_agent import DRRN_Agent, build_state
from machiavelli.agent.memory import Transition

from pathlib import Path


logging.getLogger().setLevel(logging.CRITICAL)


def configure_logger(log_dir, suffix, add_tb=1, add_wb=1, args=None):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    log_types = [logger.make_output_format('log', log_dir, log_suffix=suffix),
                 logger.make_output_format('json', log_dir, log_suffix=suffix),
                 logger.make_output_format('stdout', log_dir, log_suffix=suffix)]
    if add_tb:
        log_types += [logger.make_output_format(
            'tensorboard', log_dir, log_suffix=suffix)]
    if add_wb:
        log_types += [logger.make_output_format(
            'wandb', log_dir, log_suffix=suffix, args=args)]
    tb = logger.Logger(log_dir, log_types)
    global log
    log = logger.log


def train(agent, lm, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, args):
    start = time.time()
    obs, rewards, dones, infos, transitions, trajectories = [], [], [], [], [], []
    env_steps, max_score = 0, 0

    # Initialize trajectories for each environment
    for env in envs:
        ob, info = env.reset(seed=args.seed)
        obs.append(ob)
        rewards.append(0)
        dones.append(False)
        infos.append(info)
        transitions.append([])
        trajectories.append(Trajectory(game_name=args.game, agent_name=str(agent), env_seed=args.seed))

    states = build_state(agent._tokenize, obs, infos)
    valid_ids = [[agent._tokenize(a) for a in info['game_state']['choice_texts']] for info in infos]

    for step in tqdm(range(1, max_steps + 1)):
        try:
            # ACT: Agent chooses an action
            action_ids, action_idxs, _ = agent.act(states, valid_ids, lm=lm, eps=args.eps,
                                                   alpha=args.alpha, beta=args.beta, k=args.eps_top_k)
            action_strs = [info['game_state']['choice_texts'][idx]
                           for info, idx in zip(infos, action_idxs)]

            # STEP: Environment responds to the action
            next_obs, next_rewards, next_dones, next_infos = [], [], [], []
            for i, (env, action_str, action_idx) in enumerate(zip(envs, action_strs, action_idxs)):
                ob, reward, done, info = env.step(action_idx)
                trajectories[i].add(obs[i], action_idx, reward, info)  # Add step to trajectory

                if env.curr_points() >= max_score:
                    max_score = env.curr_points()
                    agent.memory.clear_alpha()
                if done:
                    tb.logkv_mean('EpisodeScore', info['game_state']['points'])
                    if env.curr_points() >= max_score:
                        for transition in transitions[i]:
                            agent.observe(transition, is_prior=True)
                    ob, info = env.reset(seed=args.seed)
                    trajectories[i] = Trajectory(game_name=args.game, agent_name=str(agent), env_seed=args.seed)  # Reset trajectory
                next_obs.append(ob)
                next_rewards.append(0 if done else reward)
                next_dones.append(False if done else done)
                next_infos.append(info)

            # Update observations
            rewards, dones, infos = next_rewards, next_dones, next_infos

            # LOG: Environment and agent state
            log(f'>> Action{step}: {action_strs[0]}')
            log(f"Reward{step}: {rewards[0]}, Score {infos[0]['game_state']['points']}, "
                f"Num ach {len(infos[0]['game_state']['unlocked_achievements'])}, Done {dones[0]}\n")

            # UPDATE: State, valid actions, and transitions
            next_states = build_state(agent._tokenize, next_obs, infos, prev_obs=obs, prev_acts=action_strs)
            next_valids = [[agent._tokenize(a) for a in info['game_state']['choice_texts']] for info in infos]
            for state, act, rew, next_state, valids, done, transition in zip(states, action_ids, rewards, next_states,
                                                                             next_valids, dones, transitions):
                if len(act) > 0:  # not [] (i.e. reset)
                    transition.append(Transition(
                        state, act, rew, next_state, valids, done))
                    agent.observe(transition[-1])
            obs, states, valid_ids = next_obs, next_states, next_valids

            # Periodic operations
            if step % log_freq == 0:
                tb.logkv("FPS", round(
                    (step * args.num_envs) / (time.time() - start), 2))
                tb.logkv("Max score seen", max_score)
                tb.dumpkvs()

            if step % update_freq == 0:
                agent.update()

            # Save model and trajectories at checkpoints
            if step % checkpoint_freq == 0:
                savedir = Path('./checkpoints_utility/')
                savedir.mkdir(parents=True, exist_ok=True)
                # torch.save(
                #    agent, savedir / f'{args.game}_thresh{args.logit_threshold}_alpha{args.alpha}_'
                #                     f'beta{args.beta}_weights{args.reg_weights}_seed{args.seed}.pt')

                # Save trajectories
                traj_dir = savedir / 'trajectories' / args.game
                traj_dir.mkdir(parents=True, exist_ok=True)
                for i, traj in enumerate(trajectories):
                    traj.save(traj_dir / f'traj_env{i}_step{step}.pkl')

        except RuntimeError as e:
            if "All Tensors given to nested_tensor must have the same dimension" in str(e):
                print(f"Step {step}: Skipping due to tensor dimension mismatch: {e}")
                continue  # Skip the problematic step
            else:
                raise  # Re-raise other unexpected errors

        # Manage memory
        if step % 1000 == 0:
            print("empty cache")
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./machiavelli/game/agent/logs/drrn')
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=20000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=1000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=2500, type=int)
    parser.add_argument('--priority_fraction', default=0.5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    # logger
    parser.add_argument('--tensorboard', default=0, type=int)
    parser.add_argument('--wandb', default=0, type=int)
    parser.add_argument('--wandb_project', default='machiavelli_drrn_agent', type=str)

    # language model
    # microsoft/deberta-v3-small
    parser.add_argument('--lm_name', default='gpt2', type=str)
    parser.add_argument('--regularizer_lm_name',
                        default='gpt2', type=str)
    parser.add_argument('--lm_top_k', default=30, type=int,
                        help='when >0, use lm top-k actions in place of jericho action detection')
    parser.add_argument('--lm_type', default='gpt', help='gpt | ngram')
    parser.add_argument('--lm_path', default='gpt2')
    parser.add_argument('--lm_dict', default='')

    # exploration
    parser.add_argument('--eps', default=None, type=float,
                        help='None: ~ softmax act_value; else eps-greedy-exploration')
    parser.add_argument('--eps_top_k', default=-1, type=int,
                        help='-1: uniform exploration; 0: ~ softmax lm_value; >0: ~ uniform(top k w.r.t. lm_value)')
    parser.add_argument('--alpha', default=0, type=float,
                        help='act_value = -alpha * bert_value + q_value; used to penalize powerful actions')
    parser.add_argument('--logit_threshold', default=0.5, type=float,
                        help='threshold for logit to be considered as a high utility action')
    parser.add_argument('--reg_weights', default='0,0,0', type=str, help='comma-separated string of weights for the model regularizer')
    parser.add_argument('--beta', default=1, type=float,
                        help='temperature for softmax')

    return parser.parse_args()


def main():
    os.environ['WANDB_DISABLE_SERVICE'] = 'True'
    from machiavelli.utils import random_seed
    args = parse_args()
    random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    weights = [float(w) for w in args.reg_weights.split(',')]
    if weights == [0, 0, 0]:
        print("No regularization")
        lm = None
    else:
        lm = PowerRegularizer(args.regularizer_lm_name, 'models/model_weights.pth', args.reg_weights, utility_thresh=args.logit_threshold)

    with open('machiavelli/agent/game2beta.json') as f:
        game2beta = json.load(f)
    args.game = args.game.strip().lower()
    if args.game not in game2beta:
        return
    args.beta = game2beta[args.game]
    print(args)
    suffix = '{}_beta{}_weights{}'.format(args.game, args.beta, args.reg_weights)
    configure_logger(args.output_dir, suffix,
                     args.tensorboard, args.wandb, args)

    envs = [MachiavelliEnv(game=args.game,
                    desired_achievements=None,
                    )
            for _ in range(args.num_envs)]
    args.stats_dim = len(envs[0].player.user_vars)
    agent = DRRN_Agent(args)
    train(agent, lm, envs, args.max_steps, args.update_freq,
          args.eval_freq, args.checkpoint_freq, args.log_freq, args)


if __name__ == "__main__":
    main()