import gmsh

def generate_square_geo_file(filename, lc):
    gmsh.initialize()
    gmsh.model.add("square")

    # Define the square domain
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)

    # Define the lines
    l1 = gmsh.model.geo.addLine(p4, p3)
    l2 = gmsh.model.geo.addLine(p3, p2)
    l3 = gmsh.model.geo.addLine(p2, p1)
    l4 = gmsh.model.geo.addLine(p1, p4)

    # Define the line loop
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    # Define the surface
    gmsh.model.geo.addPlaneSurface([loop])

    # Define physical entities (optional)
    gmsh.model.addPhysicalGroup(1, [l1], 1)
    gmsh.model.addPhysicalGroup(1, [l2], 2)
    gmsh.model.addPhysicalGroup(1, [l3], 3)
    gmsh.model.addPhysicalGroup(1, [l4], 4)
    gmsh.model.addPhysicalGroup(2, [1], 1)

    # Generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Write the .geo file
    gmsh.write(filename)

    gmsh.finalize()

# Define the characteristic length scale
lc = 0.002

# Generate the .geo file
generate_square_geo_file("square_larger.msh", lc)
