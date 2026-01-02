import numpy as np
import time
import csv
import concurrent.futures
import sys
from scipy.stats import norm

def american_option_price(S0, K, T, r, sigma, option_type, num_paths=10000, num_steps=100):
	# Implementation of Gustafsson's method for American option pricing
	# with controlled variate using European option price
	# Parameters:
	# S0: initial stock price
	# K: strike price
	# T: time to expiration (in years)
	# r: risk-free rate
	# sigma: volatility
	# option_type: 'call' or 'put'
	# num_paths: number of Monte Carlo paths
	# num_steps: number of time steps

	# Time increment
	dt = T / num_steps
	discount = np.exp(-r * dt) # Discount factor per time step

	# Initialize stock prices at t = T
	# Directly through Geometric Normal Distribution, not step-by-step
	Z = np.random.standard_normal(num_paths)
	X = (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z
	S = S0 * np.exp(X)

	# Calculate payoff at expiration
	if option_type.lower() == 'call':
		V = np.maximum(S - K, 0)
	elif option_type.lower() == 'put':
		V = np.maximum(K - S, 0)
	
	european_V0 = np.mean(V) * np.exp(-r * T)
	
	Zeros = np.zeros(num_paths)

	# Used to plot exercise boundary
	exercise_S = []
	times = []

	for t_step in range(num_steps - 1, 0, -1):
		# Discount cash flows back one period
		V *= discount

		# Stock prices at time t
		X = sample_brownian_bridge_step(X, Zeros, t_step * dt, sigma, dt, num_paths)
		S = S0 * np.exp(X)

		# In-the-money paths
		if option_type.lower() == 'call':
			in_the_money = S > K
		else:  # put
			in_the_money = S < K
		
		# Out-of-the-money cases are not exercised
		# They just carry the discounted value forward

		if np.sum(in_the_money) > 0:
			# Immediate exercise value
			if option_type.lower() == 'call':
				exercise_V = np.maximum(S[in_the_money] - K, 0)
			else:  # put
				exercise_V = np.maximum(K - S[in_the_money], 0)

			# Continuation value (discounted future cash flows)
			continuation_V = V[in_the_money]
			S_itm = S[in_the_money]
			diff = exercise_V - continuation_V

			# Regression con continuation value using Laguerre polynomials
			fit_diff = lin_reg_Laguerre(S_itm, diff)
			# fit_continuation =  exercise_V - fit_diff

			# Return a snapshot of the situation at this step for plotting
			# if t_step == int(num_steps/2):
			# 	exercise_where = np.where(fit_diff > 0)[0]
			# 	return S_itm, continuation_V, fit_continuation, exercise_V, exercise_where

			# Decide whether to exercise or continue
			# V[in_the_money] = np.where(fit_diff > 0, exercise_V, continuation_V)
			V[in_the_money] = np.where(fit_diff > 0, exercise_V, continuation_V)
			
			# Estimate exercise price as the least itm S where exercise_V > fit_continuation
			# exercise_where = np.where(fit_diff > 0)[0]
			# if len(exercise_where) > 0:
			# 	S_exercise = S[in_the_money][exercise_where]
			# 	exercise_S.append(S_exercise[np.argmin(np.abs(S_exercise - K))])
			# 	times.append(t_step * dt)
	
	V0 = discount*np.mean(V)
	if option_type.lower() == 'call':
		V0 = max(V0, S0 - K)
	else:  # put
		V0 = max(V0, K - S0)
	return V0 + european_option_price(S0, K, T, r, sigma, option_type) - european_V0#, times, exercise_S # Uncomment for exercise boundary
	
def sample_brownian_bridge_step(Xi, Xf, T, sigma, dt, size):
	# X(t) = xi + (xf - xi) * (t/T) + σ(W(t) - (t/T)W(T))
	# = xi + (xf - xi) * (t/T) + σ((1-t/T)W(t) - (t/T)W(T-t))
	avg = Xi + (Xf - Xi) * (dt/T)
	sigma1 = sigma * (1 - dt/T) * np.sqrt(dt)
	sigma2 = sigma * (dt/T) * np.sqrt(T - dt)

	# Returns a sampler of Brownian bridge at time dt
	# May simplify the algebra to avoid squares and square roots
	return avg + np.sqrt(sigma1**2 + sigma2**2) * np.random.standard_normal(size)

def lin_reg_Laguerre(S, V):
	Laguerre_basis = np.column_stack([np.ones_like(S),
		np.ones_like(S) - S,
		(2 * np.ones_like(S) - 4 * S + S**2) / 2
		, (6 * np.ones_like(S) - 18 * S + 9 * S**2 - S**3) / 6
		])
	beta = np.linalg.lstsq(Laguerre_basis, V, rcond=None)[0]
	return Laguerre_basis @ beta

def european_option_price(S0, K, T, r, sigma, option_type):
	# European option price using Black-Scholes formula
	d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
	d2 = d1 - sigma * np.sqrt(T)
	if option_type.lower() == 'call':
		price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
	else:  # put
		price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
	return price

def collect_data(num_steps):
	print(f"Running for {num_steps} time steps...")

	# Parameters
	# S0 = 90      # Initial stock price 90, 100, 110
	K = 100       # Strike price
	T = 1.0       # Time to expiration
	r = 0.03      # Risk-free rate
	sigma = 0.15   # Volatility

	S0, V_known = 90, 10.726486710094511
	# S0, V_known = 100, 4.820608184813253
	# S0, V_known = 110, 1.828207584020458

	repeat = 20  # Number of repetitions per step

	start_time = time.time()
	prices = [american_option_price(S0, K, T, r, sigma, "put", num_paths=1000000, num_steps=num_steps) for _ in range(repeat)]
	# prices = [1,2,3,4,5] # Placeholder for testing
	run_time = (time.time() - start_time) / repeat
	price_mean = np.mean(prices)
	rel_err = abs(price_mean - V_known) / V_known
	std_err = np.std(prices, ddof=1) / np.sqrt(len(prices))

	print(f"Computation finished for {num_steps} time steps.")
	return (num_steps, price_mean, run_time, rel_err, std_err)

# if __name__ == "__main__": # Simple single run
# 	# Parameters
# 	S0 = 90
# 	K = 100
# 	T = 1.0
# 	r = 0.03
# 	sigma = 0.15
	
# 	price = american_option_price(S0, K, T, r, sigma, 'put', num_paths=10000, num_steps=200)
# 	print(f"American Put Option Price: ${price:.2f}")

if __name__ == "__main__": # Collect prices_avg, run_times, rel_errs, std_errs with varying num_steps

	number_steps = list(range(10, 205, 5))

	# Parallel
	# with concurrent.futures.ProcessPoolExecutor() as executor:
	# 	results = list(executor.map(collect_data, number_steps))

	# Sequential
	results = []
	for steps in number_steps:
		results.append(collect_data(steps))

	numbers_steps, prices_avg, run_times, rel_errs, std_errs = zip(*results)
	
	with open("gustafsson_results/S=90_3bases.csv", "w", newline="") as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(["num_steps", "price_mean", "run_time", "rel_err", "std_err"])
		for n, v, t, e, s in zip(numbers_steps, prices_avg, run_times, rel_errs, std_errs):
			writer.writerow([n, v, t, e, s])