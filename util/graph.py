from rlcard.utils import plot_curve
from rlcard.utils import Logger

import os
import csv
import matplotlib.pyplot as plt

# log_dir='results/dqn'
csv_path1 = '../results/testlargestep/performance.csv'
csv_path2 = '../results/largestep/performance.csv'
csv_path3 = '../results/sdfs/performance.csv'
save_path = '../results/vslambda'


def get_curve(csv_path,length=49):
    csv_file = open(csv_path)
    reader = csv.DictReader(csv_file)
    x = []
    y = []
    for row in reader:
        x.append(int(row['timestep']))
        y.append(float(row['reward']))
    return x[1:length], y[1:length]


fig, ax = plt.subplots()

x1, y1 = get_curve(csv_path1,98)
x2, y2 = get_curve(csv_path2,98)
x3, y3 = get_curve(csv_path3,98)

ax.plot(x1, y1, label='MC')
ax.plot(x2, y2, label='TD')
ax.plot(x3, y3, label='TD(lambda)')

# ax.hlines(0.35,0,6000000,colors='red',label='random')

ax.plot()
ax.set(xlabel='timestep', ylabel='reward')
ax.legend()
ax.grid()

save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fig.savefig(save_path)
