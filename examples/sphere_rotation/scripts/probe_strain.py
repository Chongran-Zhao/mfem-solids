# Writes the history of a spherical Green-Lagrange strain component at radii
# on a ray of the equatorial plane z = 0 to a CSV file, one column per radius,
# from the VTU series of vtu_writer. The default is E_r_phi, the in-plane
# shear E_r_theta of the measurements, at 20, 40, 60, 80 mm on the measured
# ray, 60 degrees from the x axis.
#    pvpython probe_strain.py results_vms_dynamic_vtu/results_vms_dynamic_vtu.pvd E_rphi.csv
#       [--angle 60] [--radii 20 40 60 80] [--component r_phi]
import argparse
import csv

from probe_tools import get_history, get_ray_points, tensor_components

parser = argparse.ArgumentParser()
parser.add_argument('pvd', help='the .pvd file of vtu_writer')
parser.add_argument('csv', help='the CSV file to write')
parser.add_argument('--angle', type=float, default=60.0, help='degrees from the x axis')
parser.add_argument('--radii', type=float, nargs='+', default=[20.0, 40.0, 60.0, 80.0],
                    help='radii in mm')
parser.add_argument('--component', default='r_phi', choices=tensor_components)
args = parser.parse_args()

times_ms, values = get_history(args.pvd, get_ray_points(args.radii, args.angle), args.component)
with open(args.csv, 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['time_ms'] + ['r%gmm' % radius for radius in args.radii])
    for time, row in zip(times_ms, values):
        writer.writerow([time] + list(row))
print('saved', args.csv)
