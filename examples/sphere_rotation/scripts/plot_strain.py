# Plots the strain histories of probe_strain.py in the format of the
# measured E_r_theta figure: one panel per CSV file, one line per radius,
# with a dot at the first negative peak of each.
#    pvpython plot_strain.py strain.png dt_1e-4/E_rphi.csv [more CSV files]
#       [--labels "dt = 1e-4 s" ...] [--ylabel "E_{r\theta}"]
import argparse
import csv
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('png', help='the figure to write')
parser.add_argument('csv', nargs='+', help='CSV files of probe_strain.py, one panel each')
parser.add_argument('--labels', nargs='+', help='a title per panel')
parser.add_argument('--ylabel', default=r'E_{r\theta}', help='LaTeX of the y label')
parser.add_argument('--ylim', type=float, nargs=2, default=[-0.2, 0.2])
args = parser.parse_args()

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'font.size': 13, 'axes.linewidth': 0.8})
# The colors of the measured figure, by radius in mm; others take the cycle.
colors = {20.0: (0.50, 0.50, 0.50), 40.0: (0.93, 0.49, 0.19),
          60.0: (0.46, 0.17, 0.53), 80.0: (0.96, 0.10, 0.96)}


def draw(axis, csv_file, title):
    rows = list(csv.reader(open(csv_file)))
    radii = [float(re.search(r'r([0-9.]+)mm', name).group(1)) for name in rows[0][1:]]
    data = np.array([[float(value) for value in row] for row in rows[1:]])
    time = data[:, 0]
    for column, radius in enumerate(radii, 1):
        axis.plot(time, data[:, column], color=colors.get(radius), linewidth=1.4,
                  label=r'$r = %.1f$ mm' % radius)
    # A dot at the first negative peak, the first local minimum below a third
    # of the lowest value of the first 100 ms.
    for column, line in enumerate(axis.get_lines()[:len(radii)], 1):
        values = data[:, column]
        threshold = values[time <= 100.0].min() / 3.0
        for ii in range(1, len(values) - 1):
            if values[ii] < threshold and values[ii] <= values[ii - 1] and values[ii] <= values[ii + 1]:
                axis.plot(time[ii], values[ii], 'o', color=line.get_color(), markersize=9)
                break
    axis.set_xlim(0, 50.0 * np.ceil((time.max() + 1.0) / 50.0))
    axis.set_ylim(*args.ylim)
    axis.grid(True, color=(0.9, 0.9, 0.9), linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel('Time (ms)')
    axis.set_ylabel('$' + args.ylabel + '$')
    # Two columns filled by columns, as in the measured legend.
    legend = axis.legend(ncol=2, loc='upper right', bbox_to_anchor=(1.0, 1.22), frameon=True,
                         fancybox=False, edgecolor='black', handlelength=2.5,
                         columnspacing=1.0, fontsize=12)
    legend.get_frame().set_linewidth(0.8)
    if title:
        axis.set_title(title, loc='left', fontsize=13)


titles = args.labels if args.labels else [None] * len(args.csv)
figure, axes = plt.subplots(len(args.csv), 1, figsize=(10, 3.6 * len(args.csv)), squeeze=False)
for axis, csv_file, title in zip(axes[:, 0], args.csv, titles):
    draw(axis, csv_file, title)
figure.tight_layout(h_pad=2.5)
figure.savefig(args.png, dpi=300)
print('saved', args.png)
