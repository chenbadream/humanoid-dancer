import os
import torch
import numpy as np
from loguru import logger

from legged_gym.envs.h1.h1_phc import H1PHC
from legged_gym.envs.h1.h1_phc_config import PHCConfig
from legged_gym.agents.amp_agent import AMPAgent
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, get_args

def train(args):
    # Create environment
    env = H1PHC(
        cfg=class_to_dict(PHCConfig()),
        sim_params=parse_sim_params(args),
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless
    )
    
    # Create agent
    agent_config = {
        'obs_dim': env.obs_dim,
        'action_dim': env.action_dim,
        'amp_obs_dim': env.amp_obs_dim,
        'device': args.sim_device,
        'gamma': 0.99,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'disc_lr': 3e-4,
        'action_noise': 0.1,
        'amp_obs_demo_buffer_size': 10000,
        'amp_replay_buffer_size': 10000,
        'disc_coef': 0.1,
        'disc_logit_reg': 0.01,
        'disc_grad_penalty': 0.1,
        'disc_weight_decay': 0.0,
        'disc_reward_scale': 1.0
    }
    agent = AMPAgent(agent_config)
    
    # Set AMP buffers in environment
    env.set_amp_buffers(
        agent.amp_obs_demo_buffer,
        agent.amp_replay_buffer
    )
    
    # Training loop
    num_epochs = args.num_epochs
    steps_per_epoch = args.steps_per_epoch
    batch_size = args.batch_size
    
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_lengths = []
        
        for step in range(steps_per_epoch):
            # Reset environment
            obs = env.reset()
            
            # Run episode
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Get action from agent
                action = agent.get_action(obs)
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                
                # Store transition
                agent.amp_replay_buffer.store({
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'next_obs': next_obs,
                    'done': done,
                    'amp_obs': env.get_amp_observations()
                })
                
                # Update observation
                obs = next_obs
                
                # Update episode stats
                episode_reward += reward
                episode_length += 1
                
                # Update agent if enough samples
                if len(agent.amp_replay_buffer) >= batch_size:
                    batch = agent.amp_replay_buffer.sample(batch_size)
                    agent.update(batch)
            
            # Store episode stats
            epoch_rewards.append(episode_reward)
            epoch_lengths.append(episode_length)
        
        # Log epoch stats
        mean_reward = np.mean(epoch_rewards)
        mean_length = np.mean(epoch_lengths)
        logger.info(f"Epoch {epoch}: Mean Reward = {mean_reward:.2f}, Mean Length = {mean_length:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'mean_reward': mean_reward
            }
            torch.save(checkpoint, os.path.join(args.log_dir, f'checkpoint_{epoch}.pt'))

if __name__ == '__main__':
    args = get_args()
    train(args) 