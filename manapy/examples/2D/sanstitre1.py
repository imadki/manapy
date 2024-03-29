import vtk

# Create a render window
renderWindow = vtk.vtkRenderWindow()
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Create a renderer
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)

# Create a VTK file reader
reader = vtk.vtkUnstructuredGridReader()

# Create a mapper and actor
mapper = vtk.vtkDataSetMapper()
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Add the actor to the renderer
renderer.AddActor(actor)

# Initialize the interactor and start the rendering loop
renderWindowInteractor.Initialize()

# Loop over the VTK files
for i in range(42):
    # Update the file name
    reader.SetFileName(f"results/visu0-{i}.vtk")
    reader.Update()

    # Update the mapper's input
    mapper.SetInputData(reader.GetOutput())

    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()
