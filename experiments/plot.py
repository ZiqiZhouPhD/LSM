import matplotlib.pyplot as plt
import numpy as np
import LSM_simple
import LSM_Gustafsson
import LSM_Gustafsson_CIR
import LSM_Gustafsson_CV
import LSM_Gustafsson_dynamic
import LSM_Gustafsson_sym

def read_data(f):
	data = {}
	with open(f, 'r', encoding='utf-8') as f:
		lines = f.readlines()
		headers = lines[0].strip().split('\t')
		for header in headers:
			data[header] = []
		for line in lines[1:]:
			values = line.strip().split('\t')
			for header, value in zip(headers, values):
				data[header].append(float(value))
	return data

def plot_rel_errs(files, labels, export_path):
	plt.figure(figsize=(4, 3))
	for f, label in zip(files, labels):
		data = read_data(f)
		plt.plot(data['num_steps'], data['rel_err'], label=label)
	plt.xlabel('Number of Time Steps')
	plt.ylabel('Relative Error')
	plt.legend()
	plt.tight_layout()
	plt.savefig(export_path, format='pdf')
	plt.close()

def plot_run_time(files, labels, export_path):
	plt.figure(figsize=(4, 3))
	for f, label in zip(files, labels):
		data = read_data(f)
		plt.plot(data['num_steps'], data['run_time'], label=label)
	plt.xlabel('Number of Time Steps')
	plt.ylabel('Run Time (seconds)')
	plt.legend()
	plt.tight_layout()
	plt.savefig(export_path, format='pdf')
	plt.close()

def plot_price(file, true_value, export_path):
	repeat = 20
	plt.figure(figsize=(4, 3))
	data = read_data(file)
	prices = np.array(data['price_mean'])
	prices_upper = prices + np.array(data['std_err']) * np.sqrt(repeat)
	prices_lower = prices - np.array(data['std_err']) * np.sqrt(repeat)
	plt.plot(data['num_steps'], prices, label='Computed Value')
	plt.plot(data['num_steps'], prices_upper, linestyle='dotted', color='red', label='std err')
	plt.plot(data['num_steps'], prices_lower, linestyle='dotted', color='red')
	plt.axhline(y=true_value, linestyle='dashed', color='gray', label='True Value')
	plt.xlabel('Number of Time Steps')
	plt.ylabel('Computed Price')
	plt.legend()
	plt.tight_layout()
	plt.savefig(export_path, format='pdf')
	plt.close()

def code_to_compare():
	# Parameters
	S0, V_known = 90, 10.726486710094511
	K = 100
	T = 1.0
	r = 0.03
	sigma = 0.15
	num_paths = 10000
	num_steps = 100
	repeat = 100
	
	LSM_prices_list = []
	for Module in [LSM_simple, LSM_Gustafsson, LSM_Gustafsson_CIR, LSM_Gustafsson_CV, LSM_Gustafsson_dynamic, LSM_Gustafsson_sym]:
		# Run the pricing several times and collect results
		prices = []
		for _ in range(repeat):
			prices.append(Module.american_option_price(S0, K, T, r, sigma, 'put', num_paths=num_paths, num_steps=num_steps))
		prices = np.array(prices)
		LSM_prices_list.append(prices)

	plt.figure(figsize=(8, 6))
	plt.boxplot(LSM_prices_list, tick_labels=['Simple', 'Gustafsson', 'Isotonic', 'Controlled Var.', 'Dynamic', 'Symmetric'])
	plt.axhline(y=V_known, color='gray', linestyle='dashed', label='True Value')
	plt.ylabel('Price')
	plt.legend()
	plt.tight_layout()
	plt.savefig('fig/boxplot_compare2.pdf', format='pdf')
	plt.close()

def code_to_compare_european():
	# Have to make LSM European, namely skip value update
	# Parameters
	S0 = 90
	K = 100
	T = 1.0
	r = 0.03
	sigma = 0.15
	num_paths = 10000
	num_steps = 100
	repeat = 1000

	V_known = LSM_Gustafsson.european_option_price(S0, K, T, r, sigma, 'put')
	
	prices = []
	for _ in range(repeat):
		prices.append(LSM_Gustafsson.american_option_price(S0, K, T, r, sigma, 'put', num_paths=num_paths, num_steps=num_steps))
	prices = np.array(prices)

	plt.figure(figsize=(4, 6))
	plt.boxplot([prices], tick_labels=['European'])
	plt.axhline(y=V_known, color='gray', linestyle='dashed', label='True Value')
	plt.ylabel('Price')
	plt.legend()
	plt.tight_layout()
	plt.savefig('fig/boxplot_european.pdf', format='pdf')
	plt.close()

def code_to_plot_exercise_boundary():
	# Need to activate the return of exercise boundary in LSM implementation first
	# Parameters
	S0 = 100
	K = 100
	T = 1.0
	r = 0.03
	sigma = 0.15
	num_paths = 1000000
	num_steps = 100

	_, times, exercise_S = LSM_Gustafsson.american_option_price(S0, K, T, r, sigma, 'call', num_paths=num_paths, num_steps=num_steps)

	plt.figure(figsize=(5, 3))
	plt.plot(times, exercise_S)
	plt.axhline(y=K, color='gray', linestyle='dashed', label='Strike')
	plt.xlabel('time')
	plt.ylabel('Stock Price')
	plt.legend()
	plt.tight_layout()
	plt.savefig('fig/call_exercise_boundary_4bases.pdf', format='pdf')
	plt.close()

def code_to_plot_step_situation():
	# Need to activate the return of (itm) stock prices, continuation fit, continuation values, exercise values, exercise_where in LSM implementation first
	# Parameters
	S0 = 80
	K = 100
	T = 1.0
	r = 0.03
	sigma = 0.15
	num_paths = 1000
	num_steps = 100

	S_itm, diff, fit_diff, _, exercise_where = LSM_Gustafsson.american_option_price(S0, K, T, r, sigma, 'put', num_paths=num_paths, num_steps=num_steps)

	plt.figure(figsize=(5, 3))
	plt.scatter(S_itm, diff, color='black', label='not exercised')
	plt.scatter(S_itm[exercise_where], diff[exercise_where], color='red', label='exercised')
	order = np.argsort(S_itm)
	S_itm_ordered = S_itm[order]
	fit_diff_ordered = fit_diff[order]
	plt.plot(S_itm_ordered, fit_diff_ordered, color='green', linestyle='--', label='fit')
	plt.xlabel('Stock Price')
	plt.ylabel('Difference Value')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.savefig('fig/situation_Laguerre.pdf', format='pdf')
	plt.close()

def plot_memory():
	# Need to delay the LSM implementations a bit to get stable memory readings
	from memory_profiler import memory_usage
	S0 = 90
	K = 100
	T = 1.0
	r = 0.03
	sigma = 0.15
	num_paths = 100000
	
	num_steps = list(range(10, 105, 10))
	memory_usages_simple = []
	for n in num_steps:
		print(f"Measuring memory for LSM_simple with {n} steps...")
		mem_usage = memory_usage((LSM_simple.american_option_price, (S0, K, T, r, sigma, "put"), {'num_paths':num_paths, 'num_steps':n}))
		memory_usages_simple.append(max(mem_usage)-min(mem_usage))
		print(f"Memory usage: {memory_usages_simple[-1]} MiB")
	
	memory_usages_gustafsson = []
	for n in num_steps:
		print(f"Measuring memory for LSM_Gustafsson with {n} steps...")
		mem_usage = memory_usage((LSM_Gustafsson.american_option_price, (S0, K, T, r, sigma, "put"), {'num_paths':num_paths, 'num_steps':n}))
		memory_usages_gustafsson.append(max(mem_usage)-min(mem_usage))
		print(f"Memory usage: {memory_usages_gustafsson[-1]} MiB")
	
	plt.figure(figsize=(10, 6))
	plt.plot(num_steps, memory_usages_simple, label='LSM Regular')
	plt.plot(num_steps, memory_usages_gustafsson, label='LSM Gustafsson')
	plt.xlabel('Number of Time Steps')
	plt.ylabel('Memory Usage (MiB)')
	plt.legend()
	plt.savefig('fig/memory.pdf', format='pdf')
	plt.close()

if __name__ == "__main__":
	# data_files = ["gustafsson_results/S=110_2bases.csv", "gustafsson_results/S=110_3bases.csv", "gustafsson_results/S=110_4bases.csv"]
	# labels = ["2 bases", "3 bases", "4 bases"]
	# # true_value = 10.726486710094511
	# # true_value = 4.820608184813253
	# # true_value = 1.828207584020458
	# true_value = 7.485088 # For S0=100, call option
	# filename_stem = "fig/call_S=100"
	# plot_rel_errs(data_files, labels, filename_stem + '_relerr.pdf')
	# plot_run_time(data_files, labels, filename_stem + '_runtime.pdf')
	# plot_price(data_files[0], true_value, filename_stem + '_price.pdf')

	code_to_compare()

	# code_to_plot_exercise_boundary()

	# code_to_plot_step_situation()

	# plot_memory()
	
	# pass