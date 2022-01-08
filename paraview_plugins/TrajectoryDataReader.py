# Trajectory data Paraview reader

import h5py
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, smhint
from paraview.vtk.util import numpy_support as vtknp
from paraview import vtk
import logging
logger = logging.getLogger(__name__)


@smproxy.reader(label="Trajectory Data Reader",
                extensions="h5",
                file_description="HDF5 files")
class TrajectoryDataReader(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
                                        nInputPorts=0,
                                        nOutputPorts=1,
                                        outputType='vtkPolyData')

    @smproperty.stringvector(name="File")
    @smdomain.filelist()
    @smhint.filechooser(extensions="h5", file_description="HDF5 files")
    def SetFile(self, value):
        self._filename = value
        self.Modified()

    @smproperty.stringvector(name="Subfile",
                             default_values="/")
    def SetSubfile(self, value):
        self._subfile = value
        self.Modified()

    @smproperty.stringvector(name="CoordinatesDataset",
                             default_values="CoordCenterInertial.dat")
    def SetCoordinatesDataset(self, value):
        self._coords_dataset = value
        self.Modified()

    @smproperty.doublevector(name="RadialScale", default_values=1.)
    def SetRadialScale(self, value):
        self._radial_scale = value
        self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        logger.debug("Requesting data...")
        output = dsa.WrapDataObject(vtkPolyData.GetData(outInfo))
        
        logger.warning(self._subfile)
        logger.warning(self._filename)
        
        with h5py.File(self._filename, 'r') as trajectory_file:
            subfile = trajectory_file[self._subfile]
            coords = np.array(subfile[self._coords_dataset])
        coords[:, 1:] *= self._radial_scale
        logger.debug("Loaded coordinates with shape {}.".format(coords.shape))

        # Construct a line of points
        points_vtk = vtk.vtkPoints()
        # Each ID is composed of (1) the order of the point in the line and (2)
        # the index in the `vtkPoints` constructed above
        line_vtk = vtk.vtkPolyLine()
        point_ids = line_vtk.GetPointIds()
        point_ids.SetNumberOfIds(len(coords))
        for i, point in enumerate(coords):
            points_vtk.InsertPoint(i, *point[1:])
            point_ids.SetId(i, i)
        output.SetPoints(points_vtk)
        # Set the line ordering as "cell data"
        output.Allocate(1, 1)
        output.InsertNextCell(line_vtk.GetCellType(), line_vtk.GetPointIds())

        # Add time data to the points
        time = vtknp.numpy_to_vtk(coords[:, 0])
        time.SetName('Time')
        output.GetPointData().AddArray(time)

        # Add remaining datasets from file to trajectory points
        with h5py.File(self._filename, 'r') as trajectory_file:
            subfile = trajectory_file[self._subfile]
            for dataset in subfile:
                if dataset == self._coords_dataset:
                    continue
                dataset_vtk = vtknp.numpy_to_vtk(subfile[dataset][:, 1:])
                dataset_vtk.SetName(dataset.replace('.dat', ''))
                output.GetPointData().AddArray(dataset_vtk)
        return 1
