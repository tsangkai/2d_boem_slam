###
### The implementation of block online EM SLAM
### 
###

# motion model
# S_{t+1} = S_t + u + w_t
# 
# observation model
# O_t = \lambda - S_t + v_t



import numpy as np
import matplotlib.pyplot as plt
import math

import config
import em_algorithm




landmark_pos = config.landmark_pos

# input parameters
bar_u = 0.4
delta_u = 0.1

sigma_Q = config.sigma_Q
Q = sigma_Q * sigma_Q * config.I_2

sigma_R = config.sigma_R
R = sigma_R * sigma_R * config.I_2


num_trial = config.num_trial
num_block = config.num_block  # block size

total_T = config.total_T


landmark_error_array = np.zeros((num_trial, num_block))

traj_error_array = np.zeros((num_trial, total_T))



for trial in range(num_trial):


	state = config.initial_state_est
	current_est = config.initial_state_est
	current_cov = config.initial_state_cov


	landmark_hat = np.matrix([[np.random.uniform(22,28)],
		[np.random.uniform(-3,3)]])
	

	sum_T = 0


	for n in range(num_block):

		T = int(math.ceil(math.pow(1.2, n)))


		exteroceptive_history = [0] * T
		proprioceptive_history = [0] * T

		gt_traj = [0] * T

		# system dynamics and observation
		for t in range(T):

			# system transition
			odometry_input = np.matrix([[np.random.uniform(bar_u-delta_u, bar_u+delta_u)],
				[np.random.uniform(bar_u-delta_u, bar_u+delta_u)]])

			proprioceptive_history[t] = odometry_input.copy()

			state = state + odometry_input + np.matrix([[np.random.normal(0, sigma_Q)],
				[np.random.normal(0, sigma_Q)]])

			gt_traj[t] = state.copy()

			# observation
			observation = landmark_pos - state + np.matrix([[np.random.normal(0, sigma_R)],
				[np.random.normal(0, sigma_R)]])

			exteroceptive_history[t] = observation.copy()



		# EM step
		[state_est, state_cov, landmark_hat] = em_algorithm.EM_HMM(current_est, current_cov, landmark_hat, proprioceptive_history, exteroceptive_history)

		current_est = state_est[-1]
		current_cov = state_cov[-1]

		landmark_error = math.sqrt((landmark_hat[0,0]-landmark_pos[0,0])**2 +  (landmark_hat[1,0]-landmark_pos[1,0])**2)
		landmark_error_array[trial, n] = landmark_error

		for t in range(T):
			traj_error = math.sqrt((gt_traj[t][0] - state_est[t][0])**2 + (gt_traj[t][1] - state_est[t][1])**2 )
			traj_error_array[trial, sum_T + t] = traj_error


		sum_T += T


landmark_error_mean = np.mean(landmark_error_array, axis=0)
landmark_error_std = np.std(landmark_error_array, axis=0)

boem_file = open('block_online_em_slam_landmark.txt', 'w')

sum_T = 0
for n in range(num_block):
	T = int(math.ceil(math.pow(1.2, n)))
	sum_T += T
	boem_file.write(str(sum_T) + ', ' + str(landmark_error_mean[n]) + ', ' + str(landmark_error_std[n]) + '\n')

boem_file.close()



traj_error_mean = np.mean(traj_error_array, axis=0)
traj_error_std = np.std(traj_error_array, axis=0)


boem_file = open('block_online_em_slam_traj.txt', 'w')

for t in range(total_T):
	boem_file.write(str(t) + ', ' + str(traj_error_mean[t]) + ', ' + str(traj_error_std[t]) + '\n')

boem_file.close()

