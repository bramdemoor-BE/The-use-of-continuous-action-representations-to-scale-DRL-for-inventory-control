# Illustration of method proposed in Vanvuchelen et al. (2024): 'The Use of Continuous Action Representation to 
# Scale Deep Reinforcement Learning for Inventory Control' (Availabel at SSRN: 4253600)
# Based on minimal PPO implementation from Barhate, N. (2021). Minimal pytorch implementation of proximal policy optimization.https://github.com/nikhilbarhate99/PPO-PyTorch


import os
import glob
import time
import math
from datetime import datetime
import torch
import numpy as np
from JRP_env import jrp_env
from PPO import PPO

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "jrp_test"

    n_prod = 2              # number of products
    dem = [20,10]           # Poisson demand rate
    c_hold = [1,1]          # holding cost
    c_back = [19,19]        # backlog cost
    c_minor = [10,10]       # minor order cost
    c_major = 75            # major order cost
    horizon = 1000          # length of one game


    # we scale the reward for improved training
    reward_scale = c_major
    for i in range(n_prod):
        reward_scale += (c_hold[i] * dem[i])
        reward_scale += (c_back[i] * dem[i])
        reward_scale += c_minor[i]

    min_inv = -40
    max_inv = 100

    S_min = 0   # minimum order-up-to level
    S_max = 66  # maximum order-up-to level                      

    a_min = -2
    a_max = 2


    has_continuous_action_space = True  # continuous action space; else discrete
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
    print_freq = horizon        # print avg reward in the interval (in num timesteps)
    log_freq = horizon           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.8                   # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 1           # factor that decays action_std every action_std_decay_freq until min_action_std       
    min_action_std = 0.0                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(5*1000)  # action_std decay frequency (in num timesteps)


    ################ PPO hyperparameters ################
    update_timestep =  1000      # update policy every n timesteps
    K_epochs = 10               # update policy for K epochs in one PPO update
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr_actor = 3e-4       # learning rate for actor network
    lr_critic = 3e-4       # learning rate for critic network
    entropy_loss = 1e-2
    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)
    env = jrp_env(n_prod, dem, c_hold, c_back, c_minor, c_major, min_inv, max_inv, horizon)

    # state space dimension
    state_dim = env.n_prod

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.n_prod
    else:
        action_dim = env.max_inv ** env.n_prod

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", horizon)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, entropy_loss)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')


    log_running_reward = 0

    time_step = 0
    game = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0
        done = False

        while not done:

            # select action with policy
            scale_state = np.array([0 for _ in range(env.n_prod)])  #we rescale the state between [0,1]
            for i in range(env.n_prod):
                scale_state[i] = (state[i] - env.min_inv) / (env.max_inv - env.min_inv)
            action = ppo_agent.select_action(scale_state)

            for i in range(env.n_prod):
                
                #clip the action in [a_min, a_max]
                action[i] = np.clip(action[i], a_min, a_max)
                
                #tailored mapping function
                action[i] = math.ceil(((action[i] - a_min) / (a_max - a_min)) * (S_max - S_min) + S_min)

                #pass the correct order to the environment (environment works with normal orders, we optimize order-up-to levels)
                if action[i] > state[i]:
                    action[i] = action[i] - state[i]
                else:
                    action[i] = 0

            state, reward, done, _ = env.step(action)

            # saving (scaled) reward and is_terminals
            ppo_agent.buffer.rewards.append(reward / reward_scale)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_freq
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(game, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = current_ep_reward / print_freq
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Cost per Period : {}".format(game, time_step, print_avg_reward))


            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")


        log_running_reward += current_ep_reward

        game += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
