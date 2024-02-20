
using OrdinaryDiffEq
using Trixi
using Smesh

####################################################

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test
# initial_condition = initial_condition_constant

# boundary_condition = BoundaryConditionDirichlet(initial_condition)
# boundary_conditions = Dict(:x_neg => boundary_condition,
#                            :x_pos => boundary_condition,
#                            :y_neg => boundary_condition,
#                            :y_pos => boundary_condition)

solver = FV(surface_flux = flux_lax_friedrichs)

# TODO: Refinement with TriangularMesh by Smesh.jl? Does it work? And if yes, how?

coordinates_min = [-1.0, -1.0]
coordinates_max = [1.0, 1.0]

initial_refinement_level = 3
n_points_x = 2^initial_refinement_level
n_points_y = 2^initial_refinement_level
data_points = mesh_basic(coordinates_min, coordinates_max, n_points_x, n_points_y)
mesh = TriangularMesh(data_points)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
# boundary_conditions = boundary_conditions)

ode = semidiscretize(semi, (0.0, 1.0));

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)#, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, Euler(),# CarpenterKennedy2N54(williamson_condition=false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, saveat = 0.1, callback = callbacks)
summary_callback()

# using Plots; pyplot()
# anim = @animate for i in eachindex(sol.u)
#     surface(semi.cache.midpoints[1, :], semi.cache.midpoints[2, :], sol.u[i],
#                     #=zaxis=[1.8, 2.2],=# xlabel="x", ylabel="y", title="t=$(sol.t[i])")
# end
# gif(anim, "anim_fps15.gif", fps = 15)
# # @info "" semi.cache.midpoints[1, :]
# # @info "" semi.cache.midpoints[2, :]
# # @info "" sol.u[end]
# plt = display(surface(semi.cache.midpoints[1, :], semi.cache.midpoints[2, :], sol.u[1]))
# plt = display(surface(semi.cache.midpoints[1, :], semi.cache.midpoints[2, :], sol.u[end]))
# scatter(semi.cache.data_points[1, :], semi.cache.data_points[2, :], markercolor=:red, label="points")
# scatter!(semi.cache.midpoints[1, :], semi.cache.midpoints[2, :], markercolor=:blue, label="midpoints")
