import numpy as np
import math



I_2 = np.matrix([[1.0, 0], [0, 1]])


landmark_pos = np.matrix([[25.0], 
		[0.0]])

initial_state = np.matrix([[1.0], [0.0]])
initial_state_est = np.matrix([[1.0], [0.0]])
initial_state_cov = 0.00001 * I_2




u = np.matrix([[0.4], 
		[0.4]])



sigma_R = 0.5
sigma_Q = 0.2



num_trial = 100



# Block online EM SLAM

num_block = 23
alpha = 1.2 # the ratio of the block increase



total_T = 0

for n in range(num_block):
	total_T += int(math.ceil(math.pow(1.2, n)))