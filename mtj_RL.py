import os
import random
import numpy as np
import argparse

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from mtj_config_RL import mtj_run



class MTJ_Env(Env):
  def __init__(self):
    # alpha_vals   = [0.01, 0.03, 0.05, 0.07, 0.1]            # damping constant
    # Ki_vals      = [0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3, 1e-3]   # anistrophy energy    
    # Ms_vals      = [0.3e6, 0.7e6, 1.2e6, 1.6e6, 2e6]        # saturation magnetization
    # Rp_vals      = [500, 1000, 5000, 25000, 50000]          # parallel resistance
    # TMR_vals     = [0.3, 0.5, 2, 4, 6]                      # tunneling magnetoresistance ratio
    # eta_vals     = [0.1, 0.2, 0.4, 0.6, 0.8]                # spin hall angle
    # J_she_vals   = [0.01e12, 0.1e12, 0.25e12, 0.5e12, 1e12] # current density
    # d_vals       = [50]                                     # free layer diameter
    # tf_vals      = [1.1]                                    # free layer thickness

    # Initial parameter values
    self.alpha = 0.05
    self.Ki = 0.6e-3
    self.Ms = 1.2e6
    self.Rp = 5000
    self.TMR = 2
    self.eta = 0.4
    self.J_she = 0.25e12
    self.d = 50
    self.tf = 1.1

    self.dev_samples = 100

    # Parameter ranges
    self.alpha_range = [0.01, 0.1]
    self.Ki_range = [0.2e-3, 1e-3]
    self.Ms_range = [0.3e6, 2e6]
    self.Rp_range = [500, 50000]
    self.TMR_range = [0.3, 6]
    self.eta_range = [0.1, 0.8]
    self.J_she_range = [0.01e12, 1e12]

    # Parameter step sizes
    self.alpha_step = 0.01
    self.Ki_step = 0.1e-3
    self.Ms_step = 0.1e6
    self.Rp_step = 500
    self.TMR_step = 0.5
    self.eta_step = 0.4
    self.J_she_step = 0.25e12

    # Actions: increase/decrease the 7 parameters
    self.action_space = Discrete(14)
    
    # Observation array
    self.observation_space = Box(low=np.array([-1000000]), high=np.array([1000000])) # TODO: change this
    
    # Set initial state
    chi2, bitstream, energy_avg, countData, bitData, magTheta, magPhi = mtj_run(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, run=0, samples=self.dev_samples)
    self.state = np.mean(energy_avg)
    self.prev_state = self.state

    # self.state = np.array([2.98751468577824e-13], dtype=float)
    # self.prev_state = self.state
    
    # Set episode length
    self.episode_length = 1
  

  def apply_action(self, action):
    if action == 0:
      temp = self.alpha + self.alpha_step
      self.alpha = min(temp, self.alpha_range[1])
    elif action == 1:
      temp = self.alpha - self.alpha_step
      self.alpha = max(temp, self.alpha_range[0])

    elif action == 2:
      temp = self.Ki + self.Ki_step
      self.Ki = min(temp, self.Ki_range[1])
    elif action == 3:
      temp = self.Ki - self.Ki_step
      self.Ki = max(temp, self.Ki_range[0])
    
    elif action == 4:
      temp = self.Ms + self.Ms_step
      self.Ms = min(temp, self.Ms_range[1])
    elif action == 5:
      temp = self.Ms - self.Ms_step
      self.Ms = max(temp, self.Ms_range[0])
    
    elif action == 6:
      temp = self.Rp + self.Rp_step
      self.Rp = min(temp, self.Rp_range[1])
    elif action == 7:
      temp = self.Rp - self.Rp_step
      self.Rp = max(temp, self.Rp_range[0])
    
    elif action == 8:
      temp = self.TMR + self.TMR_step
      self.TMR = min(temp, self.TMR_range[1])
    elif action == 9:
      temp = self.TMR - self.TMR_step
      self.TMR = max(temp, self.TMR_range[0])
    
    elif action == 10:
      temp = self.eta + self.eta_step
      self.eta = min(temp, self.eta_range[1])
    elif action == 11:
      temp = self.eta - self.eta_step
      self.eta = max(temp, self.eta_range[0])
    
    elif action == 12:
      temp = self.J_she + self.J_she_step
      self.J_she = min(temp, self.J_she_range[1])
    elif action == 13:
      temp = self.J_she - self.J_she_step
      self.J_she = max(temp, self.J_she_range[0])


  def reward_function(self):
    if self.state < self.prev_state:
      reward = 1 
    else: 
      reward = -1 
    return reward


  def step(self, action):
    # Apply action
    self.apply_action(action)
    
    # Sample new configuration
    chi2, bitstream, energy_avg, countData, bitData, magTheta, magPhi = mtj_run(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, run=0, samples=self.dev_samples)
    self.prev_state = self.state
    self.state = np.array([np.mean(energy_avg)], dtype=float)
    
    # Reduce episode length
    self.episode_length -= 1 
    
    # Calculate reward
    reward = self.reward_function()
    
    # Check if episode is done
    if self.episode_length <= 0: 
      done = True
    else:
      done = False
    
    # Set placeholder for info
    self.info = {'alpha': self.alpha,
                'Ki'    : self.Ki,
                'Ms'    : self.Ms,
                'Rp'    : self.Rp,
                'TMR'   : self.TMR,
                'eta'   : self.eta,
                'J_she' : self.J_she}
    
    # Return step information
    return self.state, reward, done, self.info


  def render(self):
    # Implement viz
    pass
  

  def reset(self):
    # Initial parameter values
    self.alpha = 0.05
    self.Ki = 0.6e-3
    self.Ms = 1.2e6
    self.Rp = 5000
    self.TMR = 2
    self.eta = 0.4
    self.J_she = 0.25e12
    self.d = 50
    self.tf = 1.1
    
    self.episode_length = 60
    chi2, bitstream, energy_avg, countData, bitData, magTheta, magPhi = mtj_run(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, run=0, samples=self.dev_samples)
    self.state = np.mean(energy_avg)
    self.prev_state = self.state

    # self.state = np.array([2.98751468577824e-13], dtype=float)
    # self.prev_state = self.state
    
    return self.state
  

def Test_Env():
  env = MTJ_Env()

  episodes = 5
  for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
      # env.render()
      action = env.action_space.sample()
      n_state, reward, done, info = env.step(action)
      score += reward

    print('Episode:{} Score:{}'.format(episode, score))

  env.close()


def Train_Model(model_name, training_timesteps:int, eval:bool=True):
  env = MTJ_Env()
  # env = ShowerEnv()
  log_path = os.path.join('Training', 'Logs')
  model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
  model.learn(total_timesteps=training_timesteps)

  # Save model
  model_path = os.path.join('Training', 'Saved_Models', model_name)
  model.save(model_path)

  if eval == True:
    evaluation = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(evaluation)


def Test_Model(model_path:str, episodes:int):
  model = PPO.load(model_path)
  env = MTJ_Env()

  for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    infos = []
    energies = []
    
    while not done:
      # env.render()
      action, _states = model.predict(state)
      n_state, reward, done, info = env.step(action)
      score += reward

      infos.append(info)
      energies.append(env.state[0])

    best_config = infos[np.argmin(energies)]
    print('Episode:{} Score:{}'.format(episode, score))
    print('Best Config: ', best_config)

  env.close()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--activity", "-a", required=True, type=str, choices=["TrainModel", "TestModel", "TestEnv"], help="activity to perform")
  args = parser.parse_args()

  if (args.activity == "TrainModel"):
    model_name = "mtj_model_PPO"
    training_timesteps = 1
    eval = False
    Train_Model(model_name, training_timesteps, eval)

  if (args.activity == "TestModel"):
    model_path = "Training/Saved_Models/mtj_model_PPO"
    Test_Model(model_path, episodes=10)

  if (args.activity == "TestEnv"):
    Test_Env()

  # J_she = 50
  # Ms = 1
  # Ki = 1
  # TMR   = 1  # TMR ratio at V=0,120%  
  # Rp    = 1      # Magenetoresistance at parallel state, 8000 Ohm
  # a     = 1              # Width of the MTJ in m
  # b     = 1              # Length of the MTJ in m
  # tf    = 1             # Thickness of the freelayer in m                           
  # alpha = 1               # Gilbert damping damping factor
  # eta   = 1                # Spin hall angle
  # d     = 1               # Width,length and thichness of beta-W strip (heavy metal layer)
  # # self.params_set_flag = True

  # mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, run=0, samples=100)