
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on Flash
- http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
    r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    # r0 = 0.5 # = more reasonable setup
    E = 1.0
    p0_inner = 3 * (equations.gamma - 1) * E / (3 * pi * r0^2)
    p0_outer = 1.0e-5 # = true Sedov setup
    # p0_outer = 1.0e-3 # = more reasonable setup

    # Calculate primitive variables
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_chandrashekar
basis = LobattoLegendreBasis(3)
limiter_mcl = SubcellLimiterMCL(equations, basis;
                                density_limiter = true,
                                density_coefficient_for_all = false,
                                sequential_limiter = true,
                                conservative_limiter = false,
                                positivity_limiter_density = false,
                                positivity_limiter_pressure = true,
                                positivity_limiter_pressure_exact = true,
                                entropy_limiter_semidiscrete = true,
                                smoothness_indicator = false,
                                Plotting = true)
volume_integral = VolumeIntegralSubcellLimiting(limiter_mcl;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.9)

limiting_analysis_callback = LimitingAnalysisCallback(output_directory = "out",
                                                      interval = 1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback, limiting_analysis_callback,
                        save_solution)
###############################################################################
# run the simulation

output_directory = "out"
stage_callbacks = (BoundsCheckCallback(save_errors = true, interval = 100,
                                       output_directory = output_directory),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
