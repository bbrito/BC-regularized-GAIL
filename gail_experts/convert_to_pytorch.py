import argparse
import os
import sys
import pickle
import h5py
import numpy as np
import torch
from imitation.data import rollout

def main():
    parser = argparse.ArgumentParser(
        'Converts expert trajectories from h5 to pt format.')
    parser.add_argument(
        '--h5-file',
        default=None,
        help='input h5 file',
        type=str)
    parser.add_argument(
        '--pkl-file',
        default='trajs_feedbackgame.pkl',
        help='input pkl file',
        type=str)
    parser.add_argument(
        '--pt-file',
        default=None,
        help='output pt file, by default replaces file extension with pt',
        type=str)
    args = parser.parse_args()

    if args.pt_file is None:
        args.pt_file = os.path.splitext(args.pkl_file)[0] + '.pt'

    with open(args.pkl_file,"rb") as f:
        # This is a list of `imitation.data.types.Trajectory`, where
        # every instance contains observations and actions for a single expert
        # demonstration.
        trajectories = pickle.load(f)

    transitions = rollout.flatten_trajectories(trajectories)

    actions = np.expand_dims(trajectories[0].acts,axis=0)
    states = np.expand_dims(trajectories[0].obs,axis=0)
    rewards = np.expand_dims(trajectories[0].rews,axis=0)
    lens = np.ones((1))* actions.shape[1]
    for traj in trajectories[1:]:
        actions = np.concatenate((actions,np.expand_dims(traj.acts,axis=0)),axis=0)
        states = np.concatenate((states,np.expand_dims(traj.obs,axis=0)),axis=0)
        rewards = np.concatenate((rewards,np.expand_dims(traj.rews,axis=0)),axis=0)
        lens = np.concatenate((lens,np.array([traj.acts.shape[0]])),axis=0)

    states = torch.from_numpy(states).float()
    actions = torch.from_numpy(actions).float()
    rewards = torch.from_numpy(rewards).float()
    lens = torch.from_numpy(lens).long()

    if args.h5_file is not None:

        with h5py.File(args.h5_file, 'r') as f:
            dataset_size = f['obs_B_T_Do'].shape[0]  # full dataset size

            states = f['obs_B_T_Do'][:dataset_size, ...][...]
            actions = f['a_B_T_Da'][:dataset_size, ...][...]
            rewards = f['r_B_T'][:dataset_size, ...][...]
            lens = f['len_B'][:dataset_size, ...][...]

            states = torch.from_numpy(states).float()
            actions = torch.from_numpy(actions).float()
            rewards = torch.from_numpy(rewards).float()
            lens = torch.from_numpy(lens).long()

    data = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'lengths': lens
    }

    torch.save(data, args.pt_file)


if __name__ == '__main__':
    main()
