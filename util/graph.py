from rlcard.utils import plot_curve
from rlcard.utils import Logger

import os
import csv
import matplotlib.pyplot as plt

# log_dir='results/dqn'

paths = [
    'ddqn-mc',
    'nfsp-ddqn-td',
    'nfsp-ddqn-mc',
    # 'nfsp-ddqn',
    # 'nfsp-ddqn-mc'
]

save_path = '../data/sum'


def get_curve(csv_path, length=None):
    label = csv_path
    csv_path = '../results/' + csv_path + '/performance.csv'
    csv_file = open(csv_path)
    reader = csv.DictReader(csv_file)
    x = []
    y = []
    for row in reader:
        x.append(int(row['timestep']))
        y.append(float(row['reward']))
    if length is None:
        return x[1:], y[1:], label
    return x[1:length], y[1:length], label


fig, ax = plt.subplots()

for path in paths:
    x1, y1, label = get_curve(path,70)
    ax.plot(x1, y1, label=label)

# ax.hlines(0.35,0,6000000,colors='red',label='random')
ax.plot()
ax.set(xlabel='timestep', ylabel='reward')
ax.legend()
ax.grid()

save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fig.savefig(save_path)
