# Illustration of method proposed in Vanvuchelen et al. (2024): 'The Use of Continuous Action Representation to 
# Scale Deep Reinforcement Learning for Inventory Control' (Availabel at SSRN: 4253600)
# Based on minimal PPO implementation from Barhate, N. (2021). Minimal pytorch implementation of proximal policy optimization.https://github.com/nikhilbarhate99/PPO-PyTorch

import numpy as np

class jrp_env(object):

	def __init__(self, n_prod, dem, c_hold, c_back, c_minor, c_major, min_inv, max_inv, horizon):

		self.n_prod = n_prod
		self.dem = dem
		self.c_hold = c_hold
		self.c_back = c_back
		self.c_minor = c_minor
		self.c_major = c_major
		self.min_inv = min_inv
		self.max_inv = max_inv
		self.horizon = horizon
		self.state = np.array([0 for i in range(n_prod)])
		self.time = 0


	def step(self, action):

		demand = [np.random.poisson(self.dem[i]) for i in range(self.n_prod)]
		self.state = np.array([max(self.min_inv, min(self.max_inv, self.state[i] + action[i]) - demand[i]) for i in range(self.n_prod)])
		cost = 0
		order = False
		for i in range(self.n_prod):
			if self.state[i] > 0:
				cost -= (self.c_hold[i] * self.state[i])
			else:
				cost += (self.c_back[i] * self.state[i])
			if action[i] > 0:
				cost -= self.c_minor[i]
				order = True
		if order:
			cost -= self.c_major
		self.time += 1

		return self.state, cost, self.isFinished(), None


	def isFinished(self):
		
		return self.time == self.horizon


	def reset(self):
		
		self.state = np.array([0 for i in range(self.n_prod)])
		self.time = 0

		return self.state