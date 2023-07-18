#!/usr/bin/env julia
# Has to be at least Julia v1.9.0.

# Running in parallel:
#   ${JULIA_DEPOT_PATH}/.julia/bin/mpiexecjl --project=. -n 3 julia hybrid-t8code-mesh.jl
#
# More information: https://juliaparallel.org/MPI.jl/stable/usage/
using OrdinaryDiffEq
using Trixi

using MPI
using T8code
using T8code.Libt8: SC_LP_ESSENTIAL
using T8code.Libt8: SC_LP_PRODUCTION

# Directly ported from: `src/t8_cmesh/t8_cmesh_examples.c: t8_cmesh_new_periodic_hybrid`.
function cmesh_new_periodic_hybrid(comm, n_dims)::t8_cmesh_t
	vertices = [  # Just all vertices of all trees. partly duplicated
		0, 0, 0,                    # tree 0, triangle
		0.5, 0, 0,
		0.5, 0.5, 0,
		0, 0, 0,                    # tree 1, triangle
		0.5, 0.5, 0,
		0, 0.5, 0,
		0.5, 0, 0,                  # tree 2, quad
		1, 0, 0,
		0.5, 0.5, 0,
		1, 0.5, 0,
		0, 0.5, 0,                  # tree 3, quad
		0.5, 0.5, 0,
		0, 1, 0,
		0.5, 1, 0,
		0.5, 0.5, 0,                # tree 4, triangle
		1, 0.5, 0,
		1, 1, 0,
		0.5, 0.5, 0,                # tree 5, triangle
		1, 1, 0,
		0.5, 1, 0,
	]

	# Generally, one can define other geometries. But besides linear the other
	# geometries in t8code do not have C interface yet.
	linear_geom = t8_geometry_linear_new(n_dims)

	#
	# This is how the cmesh looks like. The numbers are the tree numbers:
	#
	#   +---+---+
	#   |   |5 /|
	#   | 3 | / |
	#   |   |/ 4|
	#   +---+---+
	#   |1 /|   |
	#   | / | 2 |
	#   |/0 |   |
	#   +---+---+
	#

	cmesh_ref = Ref(t8_cmesh_t())
	t8_cmesh_init(cmesh_ref)
	cmesh = cmesh_ref[]

	# Use linear geometry
	t8_cmesh_register_geometry(cmesh, linear_geom)
	t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
	t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)
	t8_cmesh_set_tree_class(cmesh, 2, T8_ECLASS_QUAD)
	t8_cmesh_set_tree_class(cmesh, 3, T8_ECLASS_QUAD)
	t8_cmesh_set_tree_class(cmesh, 4, T8_ECLASS_TRIANGLE)
	t8_cmesh_set_tree_class(cmesh, 5, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1 +  0:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  9:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[1 + 18:end]), 4)
    t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[1 + 30:end]), 4)
    t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[1 + 42:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 5, @views(vertices[1 + 51:end]), 3)

	t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
	t8_cmesh_set_join(cmesh, 0, 2, 0, 0, 0)
	t8_cmesh_set_join(cmesh, 0, 3, 2, 3, 0)

	t8_cmesh_set_join(cmesh, 1, 3, 0, 2, 1)
	t8_cmesh_set_join(cmesh, 1, 2, 1, 1, 0)

	t8_cmesh_set_join(cmesh, 2, 4, 3, 2, 0)
	t8_cmesh_set_join(cmesh, 2, 5, 2, 0, 1)

	t8_cmesh_set_join(cmesh, 3, 5, 1, 1, 0)
	t8_cmesh_set_join(cmesh, 3, 4, 0, 0, 0)

	t8_cmesh_set_join(cmesh, 4, 5, 1, 2, 0)

	t8_cmesh_commit(cmesh, comm)

	return cmesh
end

function cmesh_new_periodic_tri(comm, n_dims)::t8_cmesh_t
	vertices = [ # Just all vertices of all trees. partly duplicated
		0, 0, 0,                    # tree 0, triangle
		1.0, 0, 0,
		1.0, 1.0, 0,
		0, 0, 0,                    # tree 1, triangle
		1.0, 1.0, 0,
		0, 1.0, 0,
	]

	# Generally, one can define other geometries. But besides linear the other
	# geometries in t8code do not have C interface yet.
	linear_geom = t8_geometry_linear_new(n_dims)

	#
	# This is how the cmesh looks like. The numbers are the tree numbers:
	#
	#   +---+
	#   |1 /|
	#   | / |
	#   |/0 |
	#   +---+
	#

	cmesh_ref = Ref(t8_cmesh_t())
	t8_cmesh_init(cmesh_ref)
	cmesh = cmesh_ref[]

	# /* Use linear geometry */
	t8_cmesh_register_geometry(cmesh, linear_geom)
	t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
	t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1 +  0:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  9:end]), 3)

	t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
	t8_cmesh_set_join(cmesh, 0, 1, 0, 1, 0)
	t8_cmesh_set_join(cmesh, 0, 1, 2, 0, 0)

	t8_cmesh_commit(cmesh, comm)

	return cmesh
end

function cmesh_new_periodic_quad(comm, n_dims)::t8_cmesh_t
	vertices = [ # Just all vertices of all trees. partly duplicated
		0, 0, 0,                    # tree 0, quad
		1.0, 0, 0,
		0, 1.0, 0,
		1.0, 1.0, 0,
	]

	# Generally, one can define other geometries. But besides linear the other
	# geometries in t8code do not have C interface yet.
	linear_geom = t8_geometry_linear_new(n_dims)

	#
	# This is how the cmesh looks like. The numbers are the tree numbers:
	#
	#   +---+
	#   |   |
	#   | 0 |
	#   |   |
	#   +---+
	#

	cmesh_ref = Ref(t8_cmesh_t())
	t8_cmesh_init(cmesh_ref)
	cmesh = cmesh_ref[]

	# Use linear geometry
	t8_cmesh_register_geometry(cmesh, linear_geom)
	t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_QUAD)
	# t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_QUAD)

	t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1+0:end]), 4)
	# t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  12:end]), 4)

	t8_cmesh_set_join(cmesh, 0, 0, 0, 1, 0)
	t8_cmesh_set_join(cmesh, 0, 0, 2, 3, 0)

	t8_cmesh_commit(cmesh, comm)

	return cmesh
end

function build_forest(comm, n_dims, level, case)
	# More information and meshes: https://github.com/DLR-AMR/t8code/blob/main/src/t8_cmesh/t8_cmesh_examples.c#L1481
	if case == 1
		# Periodic mesh of quads.
		cmesh = cmesh_new_periodic_quad(comm, n_dims)
	elseif case == 2
		# Periodic mesh of triangles.
		cmesh = cmesh_new_periodic_tri(comm, n_dims)
	elseif case == 3
		# Periodic mesh of quads and triangles.
		# cmesh = t8_cmesh_new_periodic_hybrid(comm) # The `t8code` version does exactly the same.
		cmesh = cmesh_new_periodic_hybrid(comm, n_dims)
	else
		error("case = $case not allowed.")
	end
	scheme = t8_scheme_new_default_cxx()

	let do_face_ghost = 1
		forest = t8_forest_new_uniform(cmesh, scheme, level, do_face_ghost, comm)
		return forest
	end
end

#
# Initialization.
#

# Initialize MPI. This has to happen before we initialize sc or t8code.
# mpiret = MPI.Init()

# We will use MPI_COMM_WORLD as a communicator.
comm = MPI.COMM_WORLD

# Initialize the sc library, has to happen before we initialize t8code.
T8code.Libt8.sc_init(comm, 1, 1, C_NULL, SC_LP_ESSENTIAL)
# Initialize t8code with log level SC_LP_PRODUCTION. See sc.h for more info on the log levels.
t8_init(SC_LP_PRODUCTION)

# Initialize an adapted forest with periodic boundaries.
n_dims = 2
refinement_level = 4
# cases: 1 = quad, 2 = triangles, 3 = hybrid
forest = build_forest(comm, n_dims, refinement_level, 3)

max_number_faces = 4

number_trees = t8_forest_get_num_local_trees(forest)
println("rank $(MPI.Comm_rank(comm)): #trees $number_trees, #elements $(t8_forest_get_local_num_elements(forest)), #ghost_elements $(t8_forest_get_num_ghosts(forest))")

if MPI.Comm_rank(comm) == 0
	println("#global elements $(t8_forest_get_global_num_elements(forest))")
end

mesh = T8codeMesh{n_dims}(forest, max_number_faces)

####################################################

advection_velocity = (0.2, 0.1)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

function initial_condition_test(x, t, equations::LinearScalarAdvectionEquation2D)
	x1 = x[1] - equations.advection_velocity[1] * t
	x2 = x[2] - equations.advection_velocity[2] * t

	sigma = 0.05
	x1c = 0.5
	x2c = 0.5
	return SVector(exp(-0.5 * ((x1 - x1c)^2 + (x2 - x2c)^2) / sigma))
end
initial_condition = initial_condition_test

solver = FVMuscl(surface_flux = flux_lax_friedrichs)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

# fv_method_first_order(semi)
ode = semidiscretize(semi, (0.0, 10.0));

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=5,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback, save_solution, analysis_callback, alive_callback, stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, Euler(),
            dt=5.0e-2, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback()

#
# Clean-up
#

t8_forest_unref(Ref(forest))
T8code.Libt8.sc_finalize()
# MPI.Finalize()
