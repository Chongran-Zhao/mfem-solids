# Writes the history of a spherical displacement component, in mm, at radii
# on a ray of the equatorial plane z = 0 to a CSV file, one column per radius,
# from the VTU series of vtu_writer. The default is u_phi, the circumferential
# displacement U_theta of the measurements, at their radii 11.12, 32.31 and
# 53.38 mm on the measured ray, 60 degrees from the x axis.
#    pvpython probe_disp.py results_vtu/results_vtu.pvd U_phi.csv
#       [--angle 60] [--radii 11.12 32.31 53.38] [--component phi]
import argparse
import csv

from probe_tools import get_history, get_ray_points, vector_components

parser = argparse.ArgumentParser()
parser.add_argument('pvd', help='the .pvd file of vtu_writer')
parser.add_argument('csv', help='the CSV file to write')
parser.add_argument('--angle', type=float, default=60.0, help='degrees from the x axis')
parser.add_argument('--radii', type=float, nargs='+', default=[11.12, 32.31, 53.38],
                    help='radii in mm')
parser.add_argument('--component', default='phi', choices=vector_components)
args = parser.parse_args()

times_ms, values = get_history(args.pvd, get_ray_points(args.radii, args.angle), args.component,
                               array='displacement_sph')
with open(args.csv, 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['time_ms'] + ['r%gmm' % radius for radius in args.radii])
    for time, row in zip(times_ms, values * 1e3):
        writer.writerow([time] + list(row))
print('saved', args.csv)
