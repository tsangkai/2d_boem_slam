# EM algorithm for HMM

import numpy as np

import parameter


Q = parameter.sigma_Q * parameter.sigma_Q * parameter.I_2
R = parameter.sigma_R * parameter.sigma_R * parameter.I_2

def EM_HMM(init_est, init_cov, init_lambda, proprioceptive_arr, exteroceptive_arr):


	T = len(proprioceptive_arr)
	state_est = init_est
	state_cov = init_cov
	lambda_est = init_lambda

	forward_kf_est = [0] * T
	forward_kf_cov = [0] * T



	# forward Kalman filtering
	for t in range(T):
		state_est = state_est + proprioceptive_arr[t]
		state_cov = state_cov + Q

		state_est = state_est - state_cov * (state_cov + R).getI() * (exteroceptive_arr[t] - lambda_est + state_est)
		state_cov = state_cov - state_cov * (state_cov + R).getI() * state_cov

		forward_kf_est[t] = state_est.copy()
		forward_kf_cov[t] = state_cov.copy()


	# backward RTS smoothing
	backward_rts_est = [0] * T
	backward_rts_cov = [0] * T
		
	backward_rts_est[T-1] = forward_kf_est[T-1]
	backward_rts_cov[T-1] = forward_kf_cov[T-1]

	for t in reversed(range(T-1)):

		c_t = forward_kf_cov[t] * (forward_kf_cov[t] + Q).getI()
		
		backward_rts_est[t] = forward_kf_est[t] + c_t * (backward_rts_est[t+1] - forward_kf_est[t] - proprioceptive_arr[t])
		backward_rts_cov[t] = forward_kf_cov[t] + c_t * (backward_rts_cov[t+1] - forward_kf_cov[t] - Q) * c_t.getT()


	# calculating new lambda
	# this step will be replaced by the NLS optimizers
	new_lambda_est = np.matrix([[0.0], [0.0]])

	for t in range(T):
		new_lambda_est = new_lambda_est + (1.0/T) * (backward_rts_est[t] + exteroceptive_arr[t])
		

	final_est = forward_kf_est[T-1].copy()
	final_cov = forward_kf_cov[T-1].copy()


	return [backward_rts_est, backward_rts_cov, new_lambda_est]
	# return [final_est, final_cov, new_lambda_est]