import numpy as np

def american_option_price(S0, K, T, r, sigma, option_type, num_paths=10000, num_steps=100):
	# Implementation of Gustafsson's method for American option pricing
	# with modification
	# namely, for each X(t), average over the two reflections at X(t+dt) for each single sampler
	# Sofar does not work better than the original Gustafsson
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
	V = get_payoff(S, K, option_type)

	Zeros = np.zeros(num_paths)
	beta = None # Fitting coefficients for regression, will be reused
	S_solution = None # Estimated optimal exercise boundary
	exercise_boundary_times = []
	exercise_boundary_prices = []

	for t_step in range(num_steps - 1, 0, -1):
		# Discount cash flows back one period
		V *= discount

		# Stock prices at time t
		X, X_reflect = sample_brownian_bridge_step_reflection(X, Zeros, t_step * dt, sigma, dt, num_paths, (r - 0.5 * sigma**2))
		S = S0 * np.exp(X)
		S_reflect = S0 * np.exp(X_reflect)

		# Regress a little out-of-the-money
		# Already discounted to time t
		# a_little_out_of_the_money = find_where_a_little_out_of_the_money(S, K, option_type, rel_margin=0.05)
		# if np.sum(a_little_out_of_the_money) > 0:
		# 	Taylor_basis = generate_Taylor_basis(S[a_little_out_of_the_money], K)
		# 	beta_a_little_otm = np.linalg.lstsq(Taylor_basis, V[a_little_out_of_the_money], rcond=None)[0]
		# else: beta_a_little_otm = None

		# In-the-money paths
		in_the_money = find_where_in_the_money(S, K, option_type)
		
		# Out-of-the-money cases are not exercised
		# They just carry the discounted value forward

		if np.sum(in_the_money) > 0:

			# Continuation value (averaged over the two reflections)
			if beta is None: # Use discounted expiration payoff
				V_reflection = get_payoff(S_reflect[in_the_money], K, option_type) * discount**(num_steps-t_step)
			else: # Use max of regression on reflection and exercise value
				exercise_V_reflection = get_payoff(S_reflect[in_the_money], K, option_type)
				fit_continuation_reflection = (generate_Laguerre_basis(S_reflect[in_the_money]) @ beta)
				if S_solution is None:
					V_reflection = np.where(exercise_V_reflection > fit_continuation_reflection, exercise_V_reflection, fit_continuation_reflection)
				else:
					V_reflection = np.where(np.abs(S_reflect[in_the_money]-K) >= np.abs(S_solution-K), exercise_V_reflection, fit_continuation_reflection)
				V_reflection *= discount
			
			# reflection_in_the_money = find_where_in_the_money(S_reflect[in_the_money], K, option_type)

			# continuation_V = V[in_the_money]
			continuation_V = (V[in_the_money] + V_reflection) / 2
			# if False: continuation_V = V[in_the_money] # Avoid using reflection too close to maturity
			# if beta_a_little_otm is not None: # Fit already discounted
			# 	continuation_V[reflection_in_the_money == False] = (V[in_the_money][reflection_in_the_money == False] + (generate_Taylor_basis(S_reflect[in_the_money][reflection_in_the_money == False], K) @ beta_a_little_otm)) / 2
			
			# Immediate exercise value
			exercise_V = get_payoff(S[in_the_money], K, option_type)

			# Regression con continuation value using Laguerre polynomials
			Laguerre_basis = generate_Laguerre_basis(S[in_the_money])
			beta = np.linalg.lstsq(Laguerre_basis, continuation_V, rcond=None)[0]
			fit_continuation = Laguerre_basis @ beta

			# Estimate S_solution as the S value closest to K where diff is positive
			diff = exercise_V - fit_continuation
			S_solution = None
			positive_indices = np.where(diff > 0)[0]
			if len(positive_indices) > 0:
				S_candidates = S[in_the_money][positive_indices]
				S_solution = S_candidates[np.argmin(np.abs(S_candidates - K))]

			# Decide whether to exercise or continue
			# V[in_the_money] = np.where(exercise_V > fit_continuation, exercise_V, fit_continuation)
			if S_solution is None:
				V[in_the_money] = np.where(exercise_V > fit_continuation, exercise_V, fit_continuation)
			else:
				V[in_the_money] = np.where(np.abs(S[in_the_money]-K) >= np.abs(S_solution-K), exercise_V, fit_continuation)
				exercise_boundary_times.append(t_step * dt)
				exercise_boundary_prices.append(S_solution)

	V0 = discount*np.mean(V)
	if option_type.lower() == 'call':
		V0 = max(V0, S0 - K)
	else:  # put
		V0 = max(V0, K - S0)
	return V0

def get_payoff(S, K, option_type):
	if option_type.lower() == 'call':
		return np.maximum(S - K, 0)
	elif option_type.lower() == 'put':
		return np.maximum(K - S, 0)

def find_where_in_the_money(S, K, option_type):
	if option_type.lower() == 'call':
		return S > K
	else:  # put
		return S < K

def find_where_near_the_money(S, K, rel_margin=0.1):
	return np.abs(S - K) <= rel_margin * K

def find_where_a_little_out_of_the_money(S, K, option_type, rel_margin=0.1):
	if option_type.lower() == 'call':
		return (S <= K) & (S >= K * (1 - rel_margin))
	else:  # put
		return (S >= K) & (S <= K * (1 + rel_margin))

def sample_brownian_bridge_step_reflection(Xi, Xf, T, sigma, dt, size, mu):
	# First return item: perform the standard Brownian bridge sampling at time dt
	# Second return item: from this sample, generate the reflection Xi' at time t=0,
	# which is a reflection of Xi w.r.t. the sample at time dt
	# X(t) = xi + (xf - xi) * (t/T) + σ(W(t) - (t/T)W(T))
	# = xi + (xf - xi) * (t/T) + σ((1-t/T)W(t) - (t/T)W(T-t))
	avg = Xi + (Xf - Xi) * (dt/T)
	sigma1 = sigma * (1 - dt/T) * np.sqrt(dt)
	sigma2 = sigma * (dt/T) * np.sqrt(T - dt)
	sigma = np.sqrt(sigma1**2 + sigma2**2)
	standard_normal = np.random.standard_normal(size)

	# Returns a sampler of Brownian bridge at time dt
	# May simplify the algebra to avoid squares and square roots
	X_dt_sample = avg + sigma * standard_normal
	Xi_reflect = 2 * (X_dt_sample + mu * dt) - Xi
	return (X_dt_sample, Xi_reflect)

def generate_Laguerre_basis(S):
	# Generate Laguerre polynomial basis functions up to degree 3
	return np.column_stack([np.ones_like(S),
		np.ones_like(S) - S,
		(2 * np.ones_like(S) - 4 * S + S**2) / 2,
		(6 * np.ones_like(S) - 18 * S + 9 * S**2 - S**3) / 6#,
		# (24 * np.ones_like(S) - 96 * S + 72 * S**2 - 16 * S**3 + S**4) / 24,
		# (120 * np.ones_like(S) - 600 * S + 600 * S**2 - 200 * S**3 + 25 * S**4 - S**5) / 120
		])

def generate_Taylor_basis(S, K):
	# Generate up to polynomial basis around K
	return np.column_stack([np.ones_like(S),
		S - np.ones_like(S) * K,
		S**2 - 2 * K * S + np.ones_like(S) * K**2,
		S**3 - 3 * K * S**2 + 3 * K**2 * S - np.ones_like(S) * K**3
		])
