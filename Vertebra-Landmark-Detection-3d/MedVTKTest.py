import vtk
filepath = "E:\\PycharmProject\\Testing1\\BAIBDDW1\\"

def ShowDicomVtk3D(dicompath):
    render = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    ir = vtk.vtkRenderWindowInteractor()
    ir.SetRenderWindow(renWin)
    renWin.AddRenderer(render)

    style = vtk.vtkInteractorStyleTrackballCamera()
    ir.SetInteractorStyle(style)

    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicompath)
    #   reader.Update()

    contourfilter = vtk.vtkContourFilter()
    contourfilter.SetInputConnection(reader.GetOutputPort())
    contourfilter.SetValue(0, 500)

    normal = vtk.vtkPolyDataNormals()
    normal.SetInputConnection(contourfilter.GetOutputPort())
    normal.SetFeatureAngle(60)

    conMapper = vtk.vtkPolyDataMapper()
    conMapper.SetInputConnection(normal.GetOutputPort())
    conMapper.ScalarVisibilityOff()

    conActor = vtk.vtkActor()
    conActor.SetMapper(conMapper)
    render.AddActor(conActor)

    boxFilter = vtk.vtkOutlineFilter()
    boxFilter.SetInputConnection(reader.GetOutputPort())

    boxMapper = vtk.vtkPolyDataMapper()
    boxMapper.SetInputConnection(boxFilter.GetOutputPort())

    boxActor = vtk.vtkActor()
    boxActor.SetMapper(boxMapper)
    boxActor.GetProperty().SetColor(255, 255, 255)
    render.AddActor(boxActor)

    camera = vtk.vtkCamera()
    camera.SetViewUp(0, 0, -1)
    camera.SetPosition(0, 1, 0)
    camera.SetFocalPoint(0, 0, 0)
    camera.ComputeViewPlaneNormal()
    camera.Dolly(1.5)

    render.SetActiveCamera(camera)
    render.ResetCamera()

    ir.Initialize()
    ir.Start()

ShowDicomVtk3D(filepath)