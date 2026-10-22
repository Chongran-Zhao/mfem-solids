# Tools of the post-processing scripts of the sphere: points on a ray of the
# equatorial plane, and the history of a spherical strain component at given
# points, sampled from the VTU series of vtu_writer. Run by pvpython.
import math

import numpy as np

from paraview import servermanager
from paraview.simple import PVDReader, ProgrammableSource, ResampleWithDataset, UpdatePipeline

# The components of Green_Lagrange_strain_sph and the other tensors of
# VTK_Tools_Sphere, theta from the z axis and phi about it, and those of
# displacement_sph and the other vectors.
tensor_components = ['rr', 'theta_theta', 'phi_phi', 'r_theta', 'theta_phi', 'r_phi']
vector_components = ['r', 'theta', 'phi']


def get_ray_points(radii_mm, angle_deg):
    """Points at radii_mm on the ray of z = 0 at angle_deg from the x axis, in m."""
    angle = math.radians(angle_deg)
    return [(radius * 1e-3 * math.cos(angle), radius * 1e-3 * math.sin(angle), 0.0)
            for radius in radii_mm]


def get_history(pvd_file, points, component, array='Green_Lagrange_strain_sph'):
    """Times in ms and the component of array, a tensor or a vector of
    VTK_Tools_Sphere, at the points, one row per time."""
    reader = PVDReader(FileName=pvd_file)
    source = ProgrammableSource(OutputDataSetType='vtkPolyData')
    source.Script = '\n'.join(
        ['import vtk', 'points = vtk.vtkPoints()']
        + ['points.InsertNextPoint(%r, %r, %r)' % tuple(point) for point in points]
        + ['output = self.GetPolyDataOutput()', 'output.SetPoints(points)'])
    probe = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=source)
    components = vector_components if component in vector_components else tensor_components
    index = components.index(component)

    times_ms, values = [], []
    for time in reader.TimestepValues:
        UpdatePipeline(time=time, proxy=probe)
        data = servermanager.Fetch(probe).GetPointData().GetArray(array)
        times_ms.append(time * 1e3)
        values.append([data.GetComponent(ii, index) for ii in range(len(points))])
    return np.array(times_ms), np.array(values)
