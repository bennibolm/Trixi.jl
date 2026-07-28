using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

"""
    initial_condition_density_pulse(x, t, equations::CompressibleEulerEquations3D)

A Gaussian pulse in the density with constant velocity and pressure; reduces the
compressible Euler equations to the linear advection equations.
"""
function initial_condition_density_pulse(x, t, equations::CompressibleEulerEquations3D)
    rho_0 = 0.01
    rho = rho_0 + exp(-(x[1]^2 + x[2]^2 + x[3]^2)) / 2
    v1 = 1
    v2 = 1
    v3 = 1
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_v3 = rho * v3
    p = 1
    rho_e = p / (equations.gamma - 1) + 1 / 2 * rho * (v1^2 + v2^2 + v3^2)
    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e)
end
initial_condition = initial_condition_density_pulse

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = [],
                                positivity_variables_nonlinear = [],
                                local_twosided_variables_cons = [],
                                local_onesided_variables_nonlinear = [],
                                bar_states = false)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)

mortar = MortarIDP(equations, basis, limiter_idp;
                   pure_low_order = false)
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

coordinates_min = (-5.0, -5.0, -5.0)
coordinates_max = (5.0, 5.0, 5.0)
trees_per_dimension = (4, 4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 1, initial_refinement_level = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true)

# Note: Workaround to add a refinement patch after mesh is constructed
# Refine bottom left quadrant of each tree to level 1
function refine_fn(p8est, which_tree, quadrant)
    quadrant_obj = unsafe_load(quadrant)
    if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.z == 0 &&
       quadrant_obj.level < 2
        # return true (refine)
        return Cint(1)
    else
        # return false (don't refine)
        return Cint(0)
    end
end

# Refine recursively until each bottom left quadrant of a tree has level 1
# The mesh will be rebalanced before the simulation starts
refine_fn_c = @cfunction(refine_fn, Cint,
                         (Ptr{Trixi.p8est_t}, Ptr{Trixi.p4est_topidx_t},
                          Ptr{Trixi.p8est_quadrant_t}))
Trixi.refine_p4est!(mesh.p4est, true, refine_fn_c, C_NULL)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,),
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

stepsize_callback = StepsizeCallback(cfl = 0.8, bar_states = false)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        LimitingAnalysisCallback(interval = 100),
                        stepsize_callback);

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
