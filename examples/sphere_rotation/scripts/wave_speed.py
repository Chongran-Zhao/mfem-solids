# Shear wave propagation diagram from the VTU series of vtu_writer: the
# arrival time of the first negative peak of a spherical strain component
# (E_r_phi by default, the in-plane shear E_r_theta of the measurements) at
# radii on a ray of the equatorial plane, a linear fit of the radius against
# it over the fit range, whose slope is the shear wave speed, and the 95%
# prediction band of the fit. The arrival time is refined by the parabola
# through the lowest sample and its neighbours, so the VTU series should be
# fine in time, about 1 ms. Writes the figure and a CSV file of the arrival
# times, and prints the wave speed.
#    pvpython wave_speed.py results_vms_dynamic_vtu/results_vms_dynamic_vtu.pvd wave_speed.png
#       [--angle 60] [--radii 10 80 5] [--fit 10 80] [--component r_phi]
import argparse
import csv
import math

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from probe_tools import get_history, get_ray_points, tensor_components

parser = argparse.ArgumentParser()
parser.add_argument('pvd', help='the .pvd file of vtu_writer')
parser.add_argument('png', help='the figure to write; the CSV file takes its name')
parser.add_argument('--angle', type=float, default=60.0, help='degrees from the x axis')
parser.add_argument('--radii', type=float, nargs=3, default=[10.0, 80.0, 5.0],
                    metavar=('FIRST', 'LAST', 'STEP'), help='radii in mm')
parser.add_argument('--fit', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                    help='radii in mm of the fit range, all by default')
parser.add_argument('--component', default='r_phi', choices=tensor_components)
args = parser.parse_args()

first_mm, last_mm, step_mm = args.radii
radii_mm = np.arange(first_mm, last_mm + 0.5 * step_mm, step_mm)
times_ms, strain = get_history(args.pvd, get_ray_points(radii_mm, args.angle), args.component)

# The first negative peak: the first local minimum below a third of the
# lowest value of the first 100 ms, refined by the parabola through the
# minimum and its neighbours.
arrival_ms = []
for column in range(len(radii_mm)):
    values = strain[:, column]
    threshold = values[times_ms <= 100.0].min() / 3.0
    for ii in range(1, len(values) - 1):
        if values[ii] < threshold and values[ii] <= values[ii - 1] and values[ii] <= values[ii + 1]:
            left, middle, right = values[ii - 1], values[ii], values[ii + 1]
            shift = 0.5 * (left - right) / (left - 2.0 * middle + right)
            arrival_ms.append(times_ms[ii] + shift * (times_ms[ii + 1] - times_ms[ii]))
            break
arrival_ms = np.array(arrival_ms)

# r = a + b t over the fit range; the wave speed is -b, in mm/ms = m/s.
fit_min, fit_max = args.fit if args.fit else (radii_mm.min(), radii_mm.max())
in_fit = (radii_mm >= fit_min - 1e-9) & (radii_mm <= fit_max + 1e-9)
all_radii_mm, all_arrival_ms = radii_mm, arrival_ms
radii_mm, arrival_ms = radii_mm[in_fit], arrival_ms[in_fit]
fit = stats.linregress(arrival_ms, radii_mm)
wave_speed = -fit.slope
num_points = len(radii_mm)
residual_std = math.sqrt(np.sum((radii_mm - (fit.intercept + fit.slope * arrival_ms)) ** 2)
                         / (num_points - 2))
t_quantile = stats.t.ppf(0.975, num_points - 2)
time_mean = arrival_ms.mean()
time_spread = np.sum((arrival_ms - time_mean) ** 2)
time_fine = np.linspace(arrival_ms.min() - 1.0, arrival_ms.max() + 1.5, 200)
radius_fine = fit.intercept + fit.slope * time_fine
band = t_quantile * residual_std * np.sqrt(1.0 + 1.0 / num_points
                                           + (time_fine - time_mean) ** 2 / time_spread)

with open(args.png.rsplit('.', 1)[0] + '.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['radius_mm', 'arrival_ms'])
    writer.writerows(zip(all_radii_mm, all_arrival_ms))
print('c_wave = %.3f m/s, R^2 = %.4f' % (wave_speed, fit.rvalue ** 2))
for radius, arrival in zip(all_radii_mm, all_arrival_ms):
    print('  r = %4.0f mm  arrival %6.2f ms' % (radius, arrival))

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'font.size': 13, 'axes.linewidth': 0.8})
red = (0.75, 0.22, 0.17)
figure, axis = plt.subplots(figsize=(7.0, 3.6))
axis.fill_between(time_fine, radius_fine - band, radius_fine + band, color=(0.96, 0.80, 0.80),
                  alpha=0.6, linewidth=0, label='95% PI')
axis.plot(time_fine, radius_fine, color=red, linewidth=1.4, label='Linear fit')
axis.plot(arrival_ms, radii_mm, 'o', color=(0.20, 0.38, 0.72), markeredgecolor=(0.10, 0.18, 0.40),
          markersize=7, label='Peak (fit range)')
outside = ~in_fit
if outside.any():
    axis.plot(all_arrival_ms[outside], all_radii_mm[outside], 'o', color='white',
              markeredgecolor=(0.10, 0.18, 0.40), markersize=7)
axis.text(0.18, 0.40, r'$c_{\mathrm{wave}} = %.1f$ m/s' % wave_speed + '\n' + r'$R^2 = %.4f$'
          % fit.rvalue ** 2, transform=axis.transAxes, color=red, fontsize=14, ha='center',
          va='center')
axis.set_xlabel('Arrival Time (ms)')
axis.set_ylabel('Radial Position (mm)')
axis.set_title('R-T Shear Wave Propagation Diagram')
axis.set_ylim(0, 100)
axis.set_xlim(5.0 * math.floor(time_fine.min() / 5.0), 5.0 * math.ceil(time_fine.max() / 5.0) + 5.0)
axis.grid(True, color=(0.9, 0.9, 0.9), linewidth=0.8)
axis.set_axisbelow(True)
handles, labels = axis.get_legend_handles_labels()
legend = axis.legend([handles[2], handles[0], handles[1]], [labels[2], labels[0], labels[1]],
                     loc='upper right', fancybox=False, edgecolor='black', fontsize=12)
legend.get_frame().set_linewidth(0.8)
figure.tight_layout()
figure.savefig(args.png, dpi=300)
print('saved', args.png)
