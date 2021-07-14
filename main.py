import copy
import glob
import os
import time
from collections import deque
import imitating_games
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from imitation.util import util

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs, VecPyTorch, VecNormalize
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def record_trajectories():
    args = get_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Append the model name
    log_dir = os.path.expanduser(args.log_dir)
    log_dir = os.path.join(log_dir, args.model_name, str(args.seed))

    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, 1,
                         args.gamma, log_dir, device, True, training=False, red=True)

    # Take activation for carracing
    print("Loaded env...")
    activation = None
    if args.env_name == 'CarRacing-v0' and args.use_activation:
        activation = torch.tanh
    print(activation)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy, 'env': args.env_name},
        activation=activation,
    )
    actor_critic.to(device)

    # Load from previous model
    if args.load_model_name and args.model_name != 'random':
        loaddata = torch.load(os.path.join(args.save_dir, args.load_model_name, args.load_model_name + '_{}.pt'.format(args.seed)))
        state = loaddata[0]
        try:
            #print(envs.venv, loaddata[2])
            if not args.load_model_name.endswith('BC'):
                obs_rms, ret_rms = loaddata[1:]
                envs.venv.ob_rms = obs_rms
                envs.venv.ret_rms = None
                #envs.venv.ret_rms = ret_rms
                print(obs_rms)
                print("Loaded obs rms")
            else:
                envs.venv.ob_rms = None
                envs.venv.ret_rms = None
                print("Its BC, no normalization from env")
        except:
            print("Couldnt load obsrms")
            obs_rms = ret_rms = None
        try:
            actor_critic.load_state_dict(state)
        except:
            actor_critic = state
    elif args.load_model_name and args.model_name == 'random':
        print("Using random policy...")
        envs.venv.ret_rms = None
        envs.venv.ob_rms = None
    else:
        raise NotImplementedError

    # Record trajectories
    actions = []
    rewards = []
    observations = []
    episode_starts = []
    total_rewards = []

    last_length = 0

    for eps in range(args.num_episodes):
        obs = envs.reset()
        # Init variables for storing
        episode_starts.append(True)
        reward = 0
        while True:
            # Take action
            act = actor_critic.act(obs, None, None, None)[1]
            next_state, rew, done, info = envs.step(act)
            #print(obs.shape, act.shape, rew.shape, done)
            reward += rew
            # Add the current observation and act
            observations.append(obs.data.cpu().numpy()[0]) # [C, H, W]
            actions.append(act.data.cpu().numpy()[0]) # [A]
            rewards.append(rew[0, 0].data.cpu().numpy())
            if done[0]:
                break
            episode_starts.append(False)
            obs = next_state + 0
        print("Total reward: {}".format(reward[0, 0].data.cpu().numpy()))
        print("Total length: {}".format(len(observations) - last_length))
        last_length = len(observations)
        total_rewards.append(reward[0, 0].data.cpu().numpy())

    # Save these values
    '''
    if len(envs.observation_space.shape) == 3:
        save_trajectories_images(observations, actions, rewards, episode_starts)
    else:
        save_trajectories(observations, actions, rewards, episode_starts)
    '''
    pathname = args.load_model_name if args.model_name != 'random' else 'random'
    prefix = 'length' if args.savelength else ''
    with open(os.path.join(args.save_dir, args.load_model_name, pathname + '_{}{}.txt'.format(prefix, args.seed)), 'wt') as f:
        if not args.savelength:
            for rew in total_rewards:
                f.write('{}\n'.format(rew))
        else:
            avg_length = len(observations) / args.num_episodes
            f.write("{}\n".format(avg_length))


def save_trajectories(obs, acts, rews, eps):
    raise NotImplementedError


def save_trajectories_images(obs, acts, rews, eps):
    # Save images only
    args = get_args()
    obs_path = []
    acts = np.array(acts)
    rews = np.array(rews)
    eps = np.array(eps)
    print(acts.shape, rews.shape, eps.shape)

    # Get image dir to save
    save_dir = os.path.join(args.save_dir, args.load_model_name, )
    image_dir = os.path.join(save_dir, 'images')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    image_id = 0
    for ob in obs:
        # Scaled image from [0, 1]
        path = os.path.join(image_dir, str(image_id))
        obimg = (ob * 255).astype(np.uint8).transpose(1, 2, 0)  # [H, W, C]
        # Save image and record image path
        np.save(path, obimg)
        obs_path.append(path)
        image_id += 1

    expert_dict = {
            'obs': obs_path,
            'actions': acts,
            'rewards': rews,
            'episode_starts': eps,
    }

    torch.save(expert_dict, os.path.join(save_dir, 'expert_data.pkl'))
    print("Saved")

def main():
    args = get_args()

    # Record trajectories
    if args.record_trajectories:
        record_trajectories()
        return

    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Append the model name
    log_dir = os.path.expanduser(args.log_dir)
    log_dir = os.path.join(log_dir, args.model_name, str(args.seed))

    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    #envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                     args.gamma, log_dir, device, False)

    game = imitating_games.GuidanceFeedbackGame()

    envs = util.make_vec_env("FeedbackGame-v0", n_envs=1)

    envs = VecNormalize(envs, ob=True, gamma=args.gamma)

    envs = VecPyTorch(envs, device)

    envs.num_envs = 1
    expert_policy = imitating_games.GameSolverExpertPolicy(envs.envs[0].env.env)

    obs_shape = len(envs.observation_space.shape)

    # Take activation for carracing
    print("Loaded env...")
    activation = None
    if args.env_name == 'CarRacing-v0' and args.use_activation:
        activation = torch.tanh
    print(activation)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy, 'env': args.env_name},
        activation=activation
    )
    actor_critic.to(device)
    # Load from previous model
    if args.load_model_name:
        state = torch.load(os.path.join(args.save_dir, args.load_model_name, args.load_model_name + '_{}.pt'.format(args.seed)))[0]
        try:
            actor_critic.load_state_dict(state)
        except:
            actor_critic = state

    # If BCGAIL, then decay factor and gamma should be float
    if args.bcgail:
        assert type(args.decay) == float
        assert type(args.gailgamma) == float
        if args.decay < 0:
            args.decay = 1
        elif args.decay > 1:
            args.decay = 0.5**(1./args.decay)

        print('Gamma: {}, decay: {}'.format(args.gailgamma, args.decay))
        print('BCGAIL used')

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            gamma=args.gailgamma,
            decay=args.decay,
            act_space=envs.action_space,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        if len(envs.observation_space.shape) == 1:

            # Load RED here
            red = None
            if args.red:
                red = gail.RED(envs.observation_space.shape[0] + envs.action_space.shape[0],
                        100, device, args.redsigma, args.rediters)

            discr = gail.Discriminator(
                envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
                device, red=red, sail=args.sail, learn=args.learn)
            file_name = os.path.join(
                args.gail_experts_dir, "trajs_{}.pt".format(
                    args.env_name.split('-')[0].lower()))

            expert_dataset = gail.ExpertDataset(
                file_name, num_trajectories=args.num_traj, subsample_frequency=1)

            args.gail_batch_size = min(args.gail_batch_size, len(expert_dataset))
            drop_last = len(expert_dataset) > args.gail_batch_size
            gail_train_loader = torch.utils.data.DataLoader(
                dataset=expert_dataset,
                batch_size=args.gail_batch_size,
                shuffle=True,
                drop_last=drop_last)
            print("Data loader size", len(expert_dataset))
        else:
            # env observation shape is 3 => its an image
            assert len(envs.observation_space.shape) == 3
            discr = gail.CNNDiscriminator(
                    envs.observation_space.shape, envs.action_space, 100,
                    device)
            file_name = os.path.join(
                args.gail_experts_dir, 'expert_data.pkl')

            expert_dataset = gail.ExpertImageDataset(file_name, act=envs.action_space)
            gail_train_loader = torch.utils.data.DataLoader(
                dataset=expert_dataset,
                batch_size=args.gail_batch_size,
                shuffle=True,
                drop_last = len(expert_dataset) > args.gail_batch_size,
            )
            print('Dataloader size', len(gail_train_loader))

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    print(num_updates)
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                try:
                    envs.venv.eval()
                except:
                    pass

            gail_epoch = args.gail_epoch
            if j < 10 and obs_shape == 1:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                if obs_shape == 1:
                    discr.update(gail_train_loader, rollouts,
                                 utils.get_vec_normalize(envs)._obfilt)
                else:
                    discr.update(gail_train_loader, rollouts,
                                 None)
            if obs_shape == 3:
                obfilt = None
            else:
                obfilt = utils.get_vec_normalize(envs)._rev_obfilt

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step], obfilt) # The reverse function is passed down for RED to receive unnormalized obs which it is trained on

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if args.bcgail:
            if obs_shape == 3:
                obfilt = None
            else:
                obfilt = utils.get_vec_normalize(envs)._obfilt
            value_loss, action_loss, dist_entropy = agent.update(rollouts, gail_train_loader, obfilt)
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.model_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic.state_dict(),
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None),
                getattr(utils.get_vec_normalize(envs), 'ret_rms', None)
            ], os.path.join(save_path, args.model_name + "_{}.pt".format(args.seed)))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
