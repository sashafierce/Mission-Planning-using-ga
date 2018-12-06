# script that reads results file and plot error bars
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fields = ['Approach', 'Mean', 'Std']

df = pd.read_csv("compare.csv", skipinitialspace=True, usecols=fields)

experiments = df.Approach
x_pos = np.arange(len(experiments))
means = df.Mean
sds = df.Std

fig, ax = plt.subplots()
ax.bar(x_pos, means, yerr=sds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Mean time taken (in sec)')
ax.set_xlabel('Approach used')
ax.set_xticks(x_pos)
ax.set_xticklabels(experiments)
ax.set_title('Error bars - comparing with baseline approaches (31 mission sites)')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars_baseline.png')
plt.show()
