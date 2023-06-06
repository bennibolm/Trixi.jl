
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_density_wave_highdensity

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorIDP(equations, basis;
                            density_tvd=false,
                            positivity=true,
                            spec_entropy=false,
                            positivity_correction_factor=0.1, max_iterations_newton=10,
                            newton_tolerances=(1.0e-12, 1.0e-14),
                            bar_states=true,
                            smoothness_indicator=false)

volume_integral = VolumeIntegralShockCapturingSubcell(indicator_sc;
                                                      volume_flux_dg=volume_flux,
                                                      volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback,
                        save_solution)
###############################################################################
# run the simulation

stage_callbacks = (AntidiffusiveStage(), BoundsCheckCallback(save_errors=true))

sol = Trixi.solve(ode, alg=Trixi.SimpleSSPRK33(stage_callbacks=stage_callbacks);
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
