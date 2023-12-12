#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import matplotlib.pyplot as plt

import draw
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.ppo_agent import PPO
from LSTM_Attention import LSTM_Attention

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    ############## Hyperparameters for the experiments ##############
    env_name = "AlphaRTC"
    max_num_episodes = 5      # maximal episodes

    update_interval = 4000      # update policy every update_interval timesteps
    save_interval = 2          # save model every save_interval episode
    exploration_param = 0.05    # the std var of action distribution
    K_epochs = 37               # update policy for K_epochs
    ppo_clip = 0.2              # clip parameter of PPO
    gamma = 0.99                # discount factor
    seq_len = 100               # LSTM input

    lr = 3e-5                 # Adam parameters
    betas = (0.9, 0.999)
    state_dim = 5
    action_dim = 1
    data_path = f'./data/' # Save model and reward curve here
    #############################################

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    env = GymEnv()
    storage = Storage() # used for storing data
    ppo = PPO(state_dim, action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)

    record_episode_reward = []
    episode_reward  = 0
    time_step = 0
    
    lstm_attention = LSTM_Attention()
    state_dict = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LSTM_Attention.pth'))
    lstm_attention.load_state_dict(state_dict)

    # training loop
    for episode in range(max_num_episodes):
        while time_step < update_interval:
            done = False            
            env.reset()
            throughput_seq = []
            state = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0])
            while not done and time_step < update_interval:
                action = ppo.select_action(state, storage)
                _state, reward, done, _ = env.step(action)
                throughput_seq.append(_state[0]) # 收集吞吐量序列
                
                if len(throughput_seq) > seq_len: 
                    throughput_seq.pop(0) # 维持序列长度
                    X = torch.Tensor(throughput_seq).unsqueeze(0).unsqueeze(2).to(torch.float32)
                    # 用lstm-attention得到预测值
                    pred, attention = lstm_attention(X) # X:[batch_size=1, seq_len, embedding_dim=1]
                    pred  = torch.squeeze(pred, 0)
                    # 将预测值补充到state中
                    state = torch.cat([torch.Tensor(_state), pred], dim=0)
                else:
                    state = torch.cat([torch.Tensor(_state), torch.Tensor([_state[0]])], dim=0)
                
                # Collect data for update
                storage.rewards.append(reward)
                storage.is_terminals.append(done)
                time_step += 1
                episode_reward += reward

        next_value = ppo.get_value(state)
        storage.compute_returns(next_value, gamma)

        # update
        policy_loss, val_loss = ppo.update(storage, state)
        storage.clear_storage()
        episode_reward /= time_step
        record_episode_reward.append(episode_reward)
        print('Episode {} \t Average policy loss, value loss, reward {}, {}, {}'.format(episode, policy_loss, val_loss, episode_reward))

        if episode > 0 and not (episode % save_interval):
            ppo.save_model(data_path)
            plt.plot(range(len(record_episode_reward)), record_episode_reward)
            plt.xlabel('Episode')
            plt.ylabel('Averaged episode reward')
            plt.savefig('%sreward_record.jpg' % (data_path))

        episode_reward = 0
        time_step = 0

    # draw.draw_module(ppo.policy, data_path)


if __name__ == '__main__':
    main()
