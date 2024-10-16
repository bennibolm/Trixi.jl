module TestExamplesP4estMesh3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "P4estMesh3D" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [0.00016263963870641478],
      linf = [0.0014537194925779984])
  end

  @trixi_testset "elixir_advection_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_curved.jl"),
      l2   = [0.0004750004258546538],
      linf = [0.026527551737137167])
  end

  @trixi_testset "elixir_advection_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
      l2   = [0.00253595715323843],
      linf = [0.016486952252155795])
  end

  @trixi_testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [9.773852895157622e-6],
      linf = [0.0005853874124926162])
  end

  @trixi_testset "elixir_advection_amr_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_unstructured_curved.jl"),
      l2   = [1.6236411810065552e-5],
      linf = [0.0010554006923731395],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_advection_cubed_sphere.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_cubed_sphere.jl"),
      l2   = [0.002006918015656413],
      linf = [0.027655117058380085])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [0.002590388934758452],
      linf = [0.01840757696885409])
  end

  @trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonconforming_unstructured_curved.jl"),
      l2   = [4.070355207909268e-5, 4.4993257426833716e-5, 5.10588457841744e-5, 5.102840924036687e-5, 0.00019986264001630542],
      linf = [0.0016987332417202072, 0.003622956808262634, 0.002029576258317789, 0.0024206977281964193, 0.008526972236273522],
      tspan = (0.0, 0.01))
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [0.0015106060984283647, 0.0014733349038567685, 0.00147333490385685, 0.001473334903856929, 0.0028149479453087093],
      linf = [0.008070806335238156, 0.009007245083113125, 0.009007245083121784, 0.009007245083102688, 0.01562861968368434],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      l2   = [5.162664597942288e-15, 1.941857343642486e-14, 2.0232366394187278e-14, 2.3381518645408552e-14, 7.083114561232324e-14],
      linf = [7.269740365245525e-13, 3.289868377720495e-12, 4.440087186807773e-12, 3.8686831516088205e-12, 9.412914891981927e-12],
      tspan = (0.0, 0.03))
  end

  @trixi_testset "elixir_euler_free_stream_extruded.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_extruded.jl"),
      l2   = [8.444868392439035e-16, 4.889826056731442e-15, 2.2921260987087585e-15, 4.268460455702414e-15, 1.1356712092620279e-14],
      linf = [7.749356711883593e-14, 2.8792246364872653e-13, 1.1121659149182506e-13, 3.3228975127030935e-13, 9.592326932761353e-13],
      tspan=(0.0, 0.1))
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.010380390326164493, 0.006192950051354618, 0.005970674274073704, 0.005965831290564327, 0.02628875593094754],
      linf = [0.3326911600075694, 0.2824952141320467, 0.41401037398065543, 0.45574161423218573, 0.8099577682187109],
      tspan = (0.0, 0.2))
  end

  @trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
      l2   = [7.82070951e-02, 4.33260474e-02, 4.33260474e-02, 4.33260474e-02, 3.75260911e-01],
      linf = [7.45329845e-01, 3.21754792e-01, 3.21754792e-01, 3.21754792e-01, 4.76151527e+00],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_mhd_alfven_wave_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_nonconforming.jl"),
      l2   = [0.00019018725889431733, 0.0006523517707148006, 0.0002401595437705759, 0.0007796920661427565,
              0.0007095787460334334, 0.0006558819731628876, 0.0003565026134076906, 0.0007904654548841712,
              9.437300326448332e-7],
      linf = [0.0012482306861187897, 0.006408776208178299, 0.0016845452099629663, 0.0068711236542984555,
              0.004626581522263695, 0.006614624811393632, 0.0030068344747734566, 0.008277825749754025,
              1.3475027166309006e-5],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_mhd_shockcapturing_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shockcapturing_amr.jl"),
      l2   = [0.006298541670176575, 0.0064368506652601265, 0.007108729762852636, 0.006530420607206385,
              0.02061185869237284, 0.005562033787605515, 0.007571716276627825, 0.005571862660453231,
              3.909755063709152e-6],
      linf = [0.20904054009050665, 0.18622917151105936, 0.2347957890323218, 0.19432508025509926,
              0.6858860133405615, 0.15172116633332622, 0.22432820727833747, 0.16805989780225183,
              0.000535219040687628],
      tspan = (0.0, 0.04))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # module
