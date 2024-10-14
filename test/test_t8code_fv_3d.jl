module TestExamplesT8codeMesh2D

using Test
using Trixi

include("test_trixi.jl")

# I added this temporary test file for constantly testing while developing.
# The tests have to be adapted at the end.
EXAMPLES_DIR = joinpath(examples_dir(), "t8code_3d_fv")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)
mkdir(outdir)

@testset "T8codeMesh3D" begin
#! format: noindent

# @trixi_testset "test save_mesh_file" begin
#     @test_throws Exception begin
#         # Save mesh file support will be added in the future. The following
#         # lines of code are here for satisfying code coverage.

#         # Create dummy mesh.
#         mesh = T8codeMesh((1, 1), polydeg = 1,
#                           mapping = Trixi.coordinates2mapping((-1.0, -1.0), (1.0, 1.0)),
#                           initial_refinement_level = 1)

#         # This call throws an error.
#         Trixi.save_mesh_file(mesh, "dummy")
#     end
# end

# @trixi_testset "test check_for_negative_volumes" begin
#     @test_warn "Discovered negative volumes" begin
#         # Unstructured mesh with six cells which have left-handed node ordering.
#         mesh_file = Trixi.download("https://gist.githubusercontent.com/jmark/bfe0d45f8e369298d6cc637733819013/raw/cecf86edecc736e8b3e06e354c494b2052d41f7a/rectangle_with_negative_volumes.msh",
#                                    joinpath(EXAMPLES_DIR,
#                                             "rectangle_with_negative_volumes.msh"))

#         # This call should throw a warning about negative volumes detected.
#         mesh = T8codeMesh(mesh_file, 2)
#     end
# end

# NOTE: Since I use 2x2x2 tree instead of 8x8x8, I need to increase the resolution 2 times by the factor of 2 -> +2
@trixi_testset "elixir_advection_basic.jl" begin
    @trixi_testset "first-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            order=1,
                            initial_refinement_level=2+2,
                            l2=[0.2848617953369851],
                            linf=[0.3721898718954475])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end
    @trixi_testset "second-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            initial_refinement_level=2+2,
                            l2=[0.10381089565603231],
                            linf=[0.13787405651527007])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end
    @trixi_testset "second-order FV, extended reconstruction stencil" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            initial_refinement_level=1+2,
                            extended_reconstruction_stencil=true,
                            l2=[0.3282177575292713],
                            linf=[0.39002345444858333])
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

@trixi_testset "elixir_advection_gauss.jl" begin
    @trixi_testset "first-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_gauss.jl"),
                            order=1,
                            l2=[0.1515258539168874],
                            linf=[0.43164936150417055])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end
    @trixi_testset "second-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_gauss.jl"),
                            l2=[0.04076672839289378],
                            linf=[0.122537463101035582])
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

@trixi_testset "elixir_advection_basic_hybrid.jl" begin
    @trixi_testset "first-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_hybrid.jl"),
                            order=1,
                            l2=[0.20282363730327146],
                            linf=[0.28132446651281295])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end
    @trixi_testset "second-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_hybrid.jl"),
                            l2=[0.02153993127089835],
                            linf=[0.039109618097251886])
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

@trixi_testset "elixir_advection_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
                        l2=[0.022202106950138526],
                        linf=[0.0796166790338586])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms.jl" begin
    @trixi_testset "first-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                            order=1,
                            l2=[
                                0.050763790354290725,
                                0.0351299673616484,
                                0.0351299673616484,
                                0.03512996736164839,
                                0.1601847269543808,
                            ],
                            linf=[
                                0.07175521415072939,
                                0.04648499338897771,
                                0.04648499338897816,
                                0.04648499338897816,
                                0.2235470564880404,
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
    @trixi_testset "second-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                            order=2,
                            l2=[
                                0.012308219704695382,
                                0.010791416898840429,
                                0.010791416898840464,
                                0.010791416898840377,
                                0.036995680347196136,
                            ],
                            linf=[
                                0.01982294164697862,
                                0.01840725612418126,
                                0.01840725612418148,
                                0.01840725612418148,
                                0.05736595182767079,
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
    @trixi_testset "second-order FV, extended reconstruction stencil" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                            order=2,
                            extended_reconstruction_stencil=true,
                            l2=[
                                0.05057867333486591,
                                0.03596196296013507,
                                0.03616867188152877,
                                0.03616867188152873,
                                0.14939041550302212,
                            ],
                            linf=[
                                0.07943789383956079,
                                0.06389365911606859,
                                0.06469291944863809,
                                0.0646929194486372,
                                0.23507781748792533,
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
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
