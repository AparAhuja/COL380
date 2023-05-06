import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

fig, ax = plt.subplots()

# for thread in data['#threads'].unique():
#     ax.plot(data[data['#threads'] == thread]['#procs'], data[data['#threads'] == thread]['#time'], label=f'{thread} threads')

for proc in data['#procs'].unique():
    ax.plot(data[data['#procs'] == proc]['#threads'], data[data['#procs'] == proc]['#time'], label=f'{proc} procs')

base = data[(data['#procs'] == 1) & (data['#threads'] == 1)]['#time'][0]
data['speedup'] = base / data['#time']
data['efficiency'] = data['speedup'] / data['#procs']
data['seq fraction'] = (1/data['efficiency'] - 1) / (data['#procs'] - 1)

data.to_csv('data.csv', index=False)

ax.set_xlabel('#threads')
ax.set_ylabel('time (ms)')
ax.set_title('Varying number of threads and processes with time')
ax.legend()

plt.savefig('plot2.png')