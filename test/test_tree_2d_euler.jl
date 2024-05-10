module TestExamples2DEuler

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "Compressible Euler" begin
#! format: noindent

@trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        l2=[
                            9.321181253186009e-7,
                            1.4181210743438511e-6,
                            1.4181210743487851e-6,
                            4.824553091276693e-6,
                        ],
                        linf=[
                            9.577246529612893e-6,
                            1.1707525976012434e-5,
                            1.1707525976456523e-5,
                            4.8869615580926506e-5,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms_sc_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_sc_subcell.jl"),
                        l2=[
                            2.0633069593983843e-6,
                            1.9337331005472223e-6,
                            1.9337331005227536e-6,
                            5.885362117543159e-6,
                        ],
                        linf=[
                            1.636984098429828e-5,
                            1.5579038690871627e-5,
                            1.557903868998345e-5,
                            5.260532107742577e-5,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_convergence_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence_pure_fv.jl"),
                        l2=[
                            0.026440292358506527,
                            0.013245905852168414,
                            0.013245905852168479,
                            0.03912520302609374,
                        ],
                        linf=[
                            0.042130817806361964,
                            0.022685499230187034,
                            0.022685499230187922,
                            0.06999771202145322,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_convergence_IDP.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence_IDP.jl"),
                        l2=[
                            0.1289984161854359,
                            0.012899841618543363,
                            0.025799683237087086,
                            0.003224960404636081,
                        ],
                        linf=[
                            0.9436588685021441,
                            0.0943658868502173,
                            0.1887317737004306,
                            0.02359147170911058,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
                        l2=[
                            0.0010600778457964775,
                            0.00010600778457634275,
                            0.00021201556915872665,
                            2.650194614399671e-5,
                        ],
                        linf=[
                            0.006614198043413566,
                            0.0006614198043973507,
                            0.001322839608837334,
                            0.000165354951256802,
                        ],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=[
                            2.259440511766445e-6,
                            2.318888155713922e-6,
                            2.3188881557894307e-6,
                            6.3327863238858925e-6,
                        ],
                        linf=[
                            1.498738264560373e-5,
                            1.9182011928187137e-5,
                            1.918201192685487e-5,
                            6.0526717141407005e-5,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.061751715597716854,
                            0.05018223615408711,
                            0.05018989446443463,
                            0.225871559730513,
                        ],
                        linf=[
                            0.29347582879608825,
                            0.31081249232844693,
                            0.3107380389947736,
                            1.0540358049885143,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_ec.jl with flux_kennedy_gruber" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.03481471610306124,
                            0.027694280613944234,
                            0.027697905866996532,
                            0.12932052501462554,
                        ],
                        linf=[
                            0.31052098400669004,
                            0.3481295959664616,
                            0.34807152194137336,
                            1.1044947556170719,
                        ],
                        maxiters=10,
                        surface_flux=flux_kennedy_gruber,
                        volume_flux=flux_kennedy_gruber)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_ec.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.03481122603050542,
                            0.027662840593087695,
                            0.027665658732350273,
                            0.12927455860656786,
                        ],
                        linf=[
                            0.3110089578739834,
                            0.34888111987218107,
                            0.3488278669826813,
                            1.1056349046774305,
                        ],
                        maxiters=10,
                        surface_flux=flux_chandrashekar,
                        volume_flux=flux_chandrashekar)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
                        l2=[
                            0.05380629130119074,
                            0.04696798008325309,
                            0.04697067787841479,
                            0.19687382235494968,
                        ],
                        linf=[
                            0.18527440131928286,
                            0.2404798030563736,
                            0.23269573860381076,
                            0.6874012187446894,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shockcapturing_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_shockcapturing_subcell.jl"),
                        l2=[
                            0.08508152653623638,
                            0.04510301725066843,
                            0.04510304668512745,
                            0.6930705064715306,
                        ],
                        linf=[
                            0.31136518019691406,
                            0.5617651935473419,
                            0.5621200790240503,
                            2.8866869108596056,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_shockcapturing_subcell.jl (fixed time step)" begin
    # Testing local SSP method without stepsize callback
    # Additionally, tests combination with SaveSolutionCallback using time interval
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_shockcapturing_subcell.jl"),
                        dt=2.0e-3,
                        tspan=(0.0, 0.25),
                        save_solution=SaveSolutionCallback(dt = 0.1 + 1.0e-8),
                        callbacks=CallbackSet(summary_callback, save_solution,
                                              analysis_callback, alive_callback),
                        l2=[
                            0.05624855363458103,
                            0.06931288786158463,
                            0.06931283188960778,
                            0.6200535829842072,
                        ],
                        linf=[
                            0.29029967648805566,
                            0.6494728865862608,
                            0.6494729363533714,
                            3.0949621505674787,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
                        l2=[
                            0.14170569763947993,
                            0.11647068900798814,
                            0.11647072556898294,
                            0.3391989213659599,
                        ],
                        linf=[
                            1.6544204510794196,
                            1.35194638484646,
                            1.3519463848472744,
                            1.831228461662809,
                        ],
                        maxiters=30)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_blast_wave_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_pure_fv.jl"),
                        l2=[
                            0.39957047631960346,
                            0.21006912294983154,
                            0.21006903549932,
                            0.6280328163981136,
                        ],
                        linf=[
                            2.20417889887697,
                            1.5487238480003327,
                            1.5486788679247812,
                            2.4656795949035857,
                        ],
                        tspan=(0.0, 0.5),
                        # Let this test run longer to cover some lines in flux_hllc
                        coverage_override=(maxiters = 10^5, tspan = (0.0, 0.1)))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_amr.jl"),
                        l2=[
                            0.6835576416907511,
                            0.2839963955262972,
                            0.28399565983676,
                            0.7229447806293277,
                        ],
                        linf=[
                            3.0969614882801393,
                            1.7967947300740248,
                            1.7967508302506658,
                            3.040149575567518,
                        ],
                        tspan=(0.0, 1.0),
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_blast_wave_sc_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_sc_subcell.jl"),
                        l2=[
                            0.30785094769124677,
                            0.17599603017990473,
                            0.17594201496603085,
                            0.614120201076276,
                        ],
                        linf=[
                            1.2971828380703805,
                            1.1057475500114755,
                            1.105770653844522,
                            2.4364101844067916,
                        ],
                        tspan=(0.0, 0.5),
                        initial_refinement_level=4,
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_blast_wave_sc_subcell_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_blast_wave_sc_subcell_nonperiodic.jl"),
                        l2=[
                            0.3221177942225801,
                            0.1798478357478982,
                            0.1798364616438908,
                            0.6136884131056267,
                        ],
                        linf=[
                            1.343766644801395,
                            1.1749593109683463,
                            1.1747613085307178,
                            2.4216006041018785,
                        ],
                        tspan=(0.0, 0.5),
                        initial_refinement_level=4,
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_blast_wave_MCL.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_MCL.jl"),
                        l2=[
                            0.32716628280821736,
                            0.17711362716405113,
                            0.17710881738119433,
                            0.6192141753914343,
                        ],
                        linf=[
                            1.3147680231795071,
                            1.1313232952582144,
                            1.1308868661560831,
                            2.4962119219206,
                        ],
                        tspan=(0.0, 0.5),
                        initial_refinement_level=4,
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
                        l2=[
                            0.4866953770742574,
                            0.1673477470091984,
                            0.16734774700934,
                            0.6184367248923149,
                        ],
                        linf=[
                            2.6724832723962053,
                            1.2916089288910635,
                            1.2916089289001427,
                            6.474699399394252,
                        ],
                        tspan=(0.0, 1.0),
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave_sc_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_sedov_blast_wave_sc_subcell.jl"),
                        l2=[
                            0.47651273561515994,
                            0.16605194156429376,
                            0.16605194156447747,
                            0.6184646142923547,
                        ],
                        linf=[
                            2.559717182592356,
                            1.3594817545576394,
                            1.3594817545666105,
                            6.451896959781657,
                        ],
                        tspan=(0.0, 1.0),
                        initial_refinement_level=4,
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave_MCL.jl" begin
    rm("out/deviations.txt", force = true)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave_MCL.jl"),
                        l2=[
                            0.4740321851943766,
                            0.15889871334104985,
                            0.15889871334104988,
                            0.6190405536267991,
                        ],
                        linf=[
                            4.011954283668753,
                            1.8527131099524292,
                            1.8527131099524277,
                            6.465833729130187,
                        ],
                        tspan=(0.0, 1.0),
                        initial_refinement_level=4,
                        coverage_override=(maxiters = 6,),
                        save_errors=true,
                        output_directory="out")
    lines = readlines("out/deviations.txt")
    @test lines[1] ==
          "# iter, simu_time, rho_min, rho_max, rho_v1_min, rho_v1_max, rho_v2_min, rho_v2_max, rho_e_min, rho_e_max, pressure_min"
    @test startswith(lines[end], "349") || startswith(lines[end], "1")
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave.jl (HLLE)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
                        l2=[
                            0.352405949321075,
                            0.17207721487429464,
                            0.17207721487433883,
                            0.6263024434020885,
                        ],
                        linf=[
                            2.760997358628186,
                            1.8279186132509326,
                            1.8279186132502805,
                            6.251573757093399,
                        ],
                        tspan=(0.0, 0.5),
                        callbacks=CallbackSet(summary_callback,
                                              analysis_callback, alive_callback,
                                              stepsize_callback),
                        surface_flux=flux_hlle),
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_positivity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_positivity.jl"),
                        l2=[
                            0.48862067511841695,
                            0.16787541578869494,
                            0.16787541578869422,
                            0.6184319933114926,
                        ],
                        linf=[
                            2.6766520821013002,
                            1.2910938760258996,
                            1.2910938760258899,
                            6.473385481404865,
                        ],
                        tspan=(0.0, 1.0),
                        coverage_override=(maxiters = 3,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_blob_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_mortar.jl"),
                        l2=[
                            0.22271619518391986,
                            0.6284824759323494,
                            0.24864213447943648,
                            2.9591811489995474,
                        ],
                        linf=[
                            9.15245400430106,
                            24.96562810334389,
                            10.388109127032374,
                            101.20581544156934,
                        ],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_blob_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_amr.jl"),
                        l2=[
                            0.2086261501910662,
                            1.2118352377894666,
                            0.10255333189606497,
                            5.296238138639236,
                        ],
                        linf=[
                            14.829071984498198,
                            74.12967742435727,
                            6.863554388300223,
                            303.58813147491134,
                        ],
                        tspan=(0.0, 0.12),
                        # Let this test run longer to cover the ControllerThreeLevelCombined lines
                        coverage_override=(maxiters = 10^5,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_kelvin_helmholtz_instability_fjordholm_etal.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_kelvin_helmholtz_instability_fjordholm_etal.jl"),
                        l2=[
                            0.1057230211245312,
                            0.10621112311257341,
                            0.07260957505339989,
                            0.11178239111065721,
                        ],
                        linf=[
                            2.998719417992662,
                            2.1400285015556166,
                            1.1569648700415078,
                            1.8922492268110913,
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_kelvin_helmholtz_instability.jl"),
                        l2=[
                            0.055691508271624536,
                            0.032986009333751655,
                            0.05224390923711999,
                            0.08009536362771563,
                        ],
                        linf=[
                            0.24043622527087494,
                            0.1660878796929941,
                            0.12355946691711608,
                            0.2694290787257758,
                        ],
                        tspan=(0.0, 0.2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_kelvin_helmholtz_instability_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_kelvin_helmholtz_instability_amr.jl"),
                        l2=[
                            0.05569452733654995,
                            0.033107109983417926,
                            0.05223609622852158,
                            0.08007777597488817,
                        ],
                        linf=[
                            0.2535807803900303,
                            0.17397028249895308,
                            0.12321616095649354,
                            0.269046666668995,
                        ],
                        tspan=(0.0, 0.2),
                        coverage_override=(maxiters = 2,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_kelvin_helmholtz_instability_sc_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_kelvin_helmholtz_instability_sc_subcell.jl"),
                        l2=[
                            0.055703165296633834,
                            0.032987233605927,
                            0.05224472051711956,
                            0.08011565264331237,
                        ],
                        linf=[
                            0.24091018397460595,
                            0.1660190071332282,
                            0.12356154893467916,
                            0.2695167937393226,
                        ],
                        tspan=(0.0, 0.2),
                        initial_refinement_level=5,
                        coverage_override=(maxiters = 2,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_kelvin_helmholtz_instability_MCL.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_kelvin_helmholtz_instability_MCL.jl"),
                        l2=[
                            0.055703165296633834,
                            0.032987233605927,
                            0.05224472051711956,
                            0.08011565264331237,
                        ],
                        linf=[
                            0.24091018397460595,
                            0.1660190071332282,
                            0.12356154893467916,
                            0.2695167937393226,
                        ],
                        tspan=(0.0, 0.2),
                        initial_refinement_level=5,
                        coverage_override=(maxiters = 2,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_colliding_flow.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_colliding_flow.jl"),
                        l2=[
                            0.007237139090503349,
                            0.044887582765386916,
                            1.0453570959003603e-6,
                            0.6627307840935432,
                        ],
                        linf=[
                            0.19437260992446315,
                            0.5554343646648533,
                            5.943891455255412e-5,
                            15.188919846360125,
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_colliding_flow_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_colliding_flow_amr.jl"),
                        l2=[
                            0.006768801432802192,
                            0.032184992228603666,
                            6.923887797276484e-7,
                            0.6784222932398366,
                        ],
                        linf=[
                            0.2508663007713608,
                            0.4097017076529792,
                            0.0003528986458217968,
                            22.435474993016918,
                        ],
                        tspan=(0.0, 0.1),
                        coverage_override=(maxiters = 2,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_astro_jet_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_astro_jet_amr.jl"),
                        l2=[
                            0.011338365293662804,
                            10.09743543555765,
                            0.00392429463200361,
                            4031.7811487690506,
                        ],
                        linf=[
                            3.3178633141984193,
                            2993.6445033486402,
                            8.031723414357423,
                            1.1918867260293828e6,
                        ],
                        tspan=(0.0, 1.0e-7),
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_astro_jet_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_astro_jet_subcell.jl"),
                        l2=[
                            0.4186473232186195,
                            341.42386623555944,
                            12.913743102619245,
                            135260.31735534978,
                        ],
                        linf=[
                            6.594617349637199,
                            5225.251243383396,
                            417.4788228266706,
                            2.0263599311276933e6,
                        ],
                        initial_refinement_level=5,
                        tspan=(0.0, 1.0e-4),
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_astro_jet_MCL.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_astro_jet_MCL.jl"),
                        l2=[
                            0.4142490642847159,
                            339.10045752248817,
                            12.41716316125269,
                            134277.32794840127,
                        ],
                        linf=[
                            5.649893737038036,
                            4628.887032664001,
                            373.39317079274724,
                            1.8133961097673306e6,
                        ],
                        initial_refinement_level=5,
                        tspan=(0.0, 1.0e-4),
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_euler_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
                        l2=[
                            0.00013492249515826863,
                            0.006615696236378061,
                            0.006782108219800376,
                            0.016393831451740604,
                        ],
                        linf=[
                            0.0020782600954247776,
                            0.08150078921935999,
                            0.08663621974991986,
                            0.2829930622010579,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_vortex_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                        # Expected errors are exactly the same as in the parallel test!
                        l2=[
                            0.0017208369388227673,
                            0.09628684992237334,
                            0.09620157717330868,
                            0.1758809552387432,
                        ],
                        linf=[
                            0.021869936355319086,
                            0.9956698009442038,
                            1.0002507727219028,
                            2.223249697515648,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_vortex_mortar_split.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
                        l2=[
                            0.0017203323613648241,
                            0.09628962878682261,
                            0.09621241164155782,
                            0.17585995600340926,
                        ],
                        linf=[
                            0.021740570456931674,
                            0.9938841665880938,
                            1.004140123355135,
                            2.224108857746245,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_vortex_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_shockcapturing.jl"),
                        l2=[
                            0.0017158367642679273,
                            0.09619888722871434,
                            0.09616432767924141,
                            0.17553381166255197,
                        ],
                        linf=[
                            0.021853862449723982,
                            0.9878047229255944,
                            0.9880191167111795,
                            2.2154030488035588,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_vortex_mortar_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_vortex_mortar_shockcapturing.jl"),
                        l2=[
                            0.0017203324051381415,
                            0.09628962899999398,
                            0.0962124115572114,
                            0.1758599596626405,
                        ],
                        linf=[
                            0.021740568112562086,
                            0.9938841624655501,
                            1.0041401179009877,
                            2.2241087041100798,
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_vortex_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_amr.jl"),
                        # Expected errors are exactly the same as in the parallel test!
                        l2=[
                            5.051719943432265e-5,
                            0.0022574259317084747,
                            0.0021755998463189713,
                            0.004346492398617521,
                        ],
                        linf=[
                            0.0012880114865917447,
                            0.03857193149447702,
                            0.031090457959835893,
                            0.12125130332971423,
                        ],
                        # Let this test run longer to cover some lines in the AMR indicator
                        coverage_override=(maxiters = 10^5, tspan = (0.0, 10.5)))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_ec.jl with boundary_condition_slip_wall" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.03341239373099515,
                            0.026673245711492915,
                            0.026678871434568822,
                            0.12397486476145089,
                        ],
                        linf=[
                            0.3290981764688339,
                            0.3812055782309788,
                            0.3812041851225023,
                            1.168251216556933,
                        ],
                        periodicity=false,
                        boundary_conditions=boundary_condition_slip_wall,
                        cfl=0.3, tspan=(0.0, 0.1)) # this test is sensitive to the CFL factor
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_warm_bubble.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_warm_bubble.jl"),
                        l2=[
                            0.0001379946769624388,
                            0.02078779689715382,
                            0.033237241571263176,
                            31.36068872331705,
                        ],
                        linf=[
                            0.0016286690573188434,
                            0.15623770697198225,
                            0.3341371832270615,
                            334.5373488726036,
                        ],
                        tspan=(0.0, 10.0),
                        initial_refinement_level=4)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 100
    end
end

# Coverage test for all initial conditions
@testset "Compressible Euler: Tests for initial conditions" begin
    @trixi_testset "elixir_euler_vortex.jl one step with initial_condition_constant" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
                            l2=[
                                1.1790213022362371e-16,
                                8.580657423476384e-17,
                                1.3082387431804115e-16,
                                1.6182739965672862e-15,
                            ],
                            linf=[
                                3.3306690738754696e-16,
                                2.220446049250313e-16,
                                5.273559366969494e-16,
                                3.552713678800501e-15,
                            ],
                            maxiters=1,
                            initial_condition=initial_condition_constant)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_sedov_blast_wave.jl one step" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
                            l2=[
                                0.0021196114178949396,
                                0.010703549234544042,
                                0.01070354923454404,
                                0.10719124037195142,
                            ],
                            linf=[
                                0.11987270645890724,
                                0.7468615461136827,
                                0.7468615461136827,
                                3.910689155287799,
                            ],
                            maxiters=1)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end
end
end

end # module
