
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant

surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorMCL(equations, basis;
                            DensityLimiter=false,
                            DensityAlphaForAll=false,
                            SequentialLimiter=false,
                            ConservativeLimiter=false,
                            PressurePositivityLimiterKuzmin=true, PressurePositivityLimiterKuzminExact=true,
                            DensityPositivityLimiter=true,
                            SemiDiscEntropyLimiter=false,
                            indicator_smooth=false,
                            IDPCheckBounds=true,
                            Plotting=true)

volume_integral = VolumeIntegralShockCapturingSubcell(indicator_sc;
                                                      volume_flux_dg=volume_flux,
                                                      volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Mapping as described in https://arxiv.org/abs/2012.12040 but reduced to 2D.
# This particular mesh is unstructured in the yz-plane, but extruded in x-direction.
# Apply the warping mapping in the yz-plane to get a curved 2D mesh that is extruded
# in x-direction to ensure free stream preservation on a non-conforming mesh.
# See https://doi.org/10.1007/s10915-018-00897-9, Section 6.

# Mapping as described in https://arxiv.org/abs/2012.12040, but reduced to 2D
function mapping(xi_, eta_)
  # Transform input variables between -1 and 1 onto [0,3]
  xi = 1.5 * xi_ + 1.5
  eta = 1.5 * eta_ + 1.5

  y = eta + 3/8 * (cos(1.5 * pi * (2 * xi - 3)/3) *
                   cos(0.5 * pi * (2 * eta - 3)/3))

  x = xi + 3/8 * (cos(0.5 * pi * (2 * xi - 3)/3) *
                  cos(2 * pi * (2 * y - 3)/3))

  return SVector(x, y)
end

cells_per_dimension = (32, 32)
mesh = StructuredMesh(cells_per_dimension, mapping, periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10000,
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

sol = Trixi.solve(ode;
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
