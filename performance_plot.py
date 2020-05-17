

import numpy as np
import matplotlib.pyplot as plt

import config


alpha_value = 0.35

color_red = '#DC4C46'
color_blue = '#2F4A92'


T = config.total_T


plt.figure(1)
plt.subplot(211)



# batch EM
em_t = np.empty(T)
em_error = np.empty(T)
em_std = np.empty(T)

em_file = open('batch_em_slam_traj.txt', 'r')

line_count = 0
for line in em_file:
	data = line.split(", ")

	em_t[line_count] = float(data[0])
	em_error[line_count] = float(data[1])
	em_std[line_count] = float(data[2])

	line_count += 1
em_file.close()

plt.plot(em_t, em_error, label = 'batch EM', linewidth = 1.6, color = color_red)
plt.fill_between(em_t, em_error-em_std, em_error+em_std, color = color_red, alpha = alpha_value)



# block online EM

boem_t = np.empty(T)
boem_error = np.empty(T)
boem_std = np.empty(T)

boem_file = open('block_online_em_slam_traj.txt', 'r')

line_count = 0
for line in boem_file:
	data = line.split(", ")

	boem_t[line_count] = float(data[0])
	boem_error[line_count] = float(data[1])
	boem_std[line_count] = float(data[2])

	line_count += 1
boem_file.close()

plt.plot(boem_t, boem_error, label = 'block online EM', linewidth = 1.6, color = color_blue)
plt.fill_between(boem_t, boem_error-boem_std, boem_error+boem_std, color = color_blue, alpha = alpha_value)


# plotting setting

# plt.xscale('log')
plt.ylabel('trajectory error [m]')
plt.axis([1, T, 0, 5])
plt.legend()










### landmark 



plt.subplot(212)

# batch EM

em_file = open('batch_em_slam_landmark.txt', 'r')

em_line = em_file.readline()

batch_error_mean = float(em_line.split(", ")[0])
batch_error_std =  float(em_line.split(", ")[1])

em_file.close()

plt.plot([1,T],[batch_error_mean, batch_error_mean], label = 'batch EM', linewidth = 1.6, color = color_red)
plt.fill_between([1,T], [batch_error_mean-0.5*batch_error_std, batch_error_mean-0.5*batch_error_std], [batch_error_mean+0.5*batch_error_std, batch_error_mean+0.5*batch_error_std], color = color_red, alpha = alpha_value)



# block online EM

boem_t = np.empty(config.num_block)
boem_error = np.empty(config.num_block)
boem_std = np.empty(config.num_block)


boem_file = open('block_online_em_slam_landmark.txt', 'r')

line_count = 0
for line in boem_file:
	data = line.split(", ")

	boem_t[line_count] = float(data[0])
	boem_error[line_count] = float(data[1])
	boem_std[line_count] = float(data[2])

	line_count += 1

boem_file.close()


plt.plot(boem_t, boem_error, label = 'block online EM', linewidth = 1.6, color = color_blue)
plt.fill_between(boem_t, boem_error-boem_std, boem_error+boem_std, color = color_blue, alpha = alpha_value)



# plotting setting

# plt.xscale('log')
plt.xlabel('time [s]')
plt.ylabel('landmark error [m]')
plt.axis([1, T, 0, 5])
plt.show()

plt.savefig('performance.png')