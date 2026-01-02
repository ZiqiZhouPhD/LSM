import numpy as np
import time

def american_option_price(S0, K, T, r, sigma, option_type, num_paths=10000, num_steps=100):
	"""
	Price American options using Monte Carlo simulation with Longstaff-Schwartz method
	Naive implementation, partially vibe coded
	
	Parameters:
	S0: initial stock price
	K: strike price
	T: time to expiration (in years)
	r: risk-free rate
	sigma: volatility
	option_type: 'call' or 'put'
	num_paths: number of Monte Carlo paths
	num_steps: number of time steps
	"""
	# time.sleep(1.0)
	
	dt = T / num_steps
	discount = np.exp(-r * dt)
	
	# Generate stock price paths using Geometric Brownian Motion
	Z = np.random.standard_normal((num_paths, num_steps))
	stock_paths = np.zeros((num_paths, num_steps + 1))
	stock_paths[:, 0] = S0
	
	for t in range(1, num_steps + 1):
		stock_paths[:, t] = stock_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
	
	# Calculate payoff at expiration
	if option_type.lower() == 'call':
		payoff = np.maximum(stock_paths[:, -1] - K, 0)
	elif option_type.lower() == 'put':
		payoff = np.maximum(K - stock_paths[:, -1], 0)
	
	# Backward induction for American option
	values = payoff.copy()

	# time.sleep(1.0)
	
	for t in range(num_steps - 1, 0, -1):
		# Discount cash flows back one period
		values *= discount
		
		# In-the-money paths
		if option_type.lower() == 'call':
			in_the_money = stock_paths[:, t] > K
		else:  # put
			in_the_money = stock_paths[:, t] < K
		
		if np.sum(in_the_money) > 0:
			# Immediate exercise value
			if option_type.lower() == 'call':
				exercise_value = np.maximum(stock_paths[in_the_money, t] - K, 0)
			else:  # put
				exercise_value = np.maximum(K - stock_paths[in_the_money, t], 0)
			
			# Continuation value (discounted future cash flows)
			continuation_value = values[in_the_money]
			
			# Simple regression (using polynomial of stock price)
			X = stock_paths[in_the_money, t]
			Y = continuation_value
			
			# Polynomial basis functions [1, X, X^2, X^3]
			X_basis = np.column_stack([np.ones_like(X), X, X**2, X**3])
			
			# Linear regression
			coefficients = np.linalg.lstsq(X_basis, Y, rcond=None)[0]
			expected_continuation = X_basis @ coefficients
			exercise = exercise_value > expected_continuation
			values[in_the_money] = np.where(exercise, exercise_value, continuation_value)
	
	V0 = discount*np.mean(values)
	if option_type.lower() == 'call':
		V0 = max(V0, S0 - K)
	else:  # put
		V0 = max(V0, K - S0)
	return V0

# Example usage
if __name__ == "__main__":
	# Parameters
	S0 = 90	  # Initial stock price
	K = 100	   # Strike price
	T = 1.0	   # Time to expiration (1 year)
	r = 0.03	  # Risk-free rate (5%)
	sigma = 0.15   # Volatility (20%)
		
	# Run the pricing several times and collect results
	prices = []
	for _ in range(20):
		prices.append(american_option_price(S0, K, T, r, sigma, 'put', num_paths=10000, num_steps=100))

	prices = np.array(prices)
	print(f"American Put Option Price: avg=${prices.mean():.2f}, std error=${prices.std(ddof=1)/np.sqrt(len(prices)):.4f}")