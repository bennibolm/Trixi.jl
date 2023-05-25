# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# everything related to a DG semidiscretization in 2D,
# currently limited to Lobatto-Legendre nodes

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::TreeMesh{2}, equations,
                      dg::DG, RealT, uEltype)
  # Get cells for which an element needs to be created (i.e. all leaf cells)
  leaf_cell_ids = local_leaf_cells(mesh.tree)

  elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

  interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

  boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

  mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

  cache = (; elements, interfaces, boundaries, mortars)

  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
  cache = (;cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

  return cache
end


# The methods below are specialized on the volume integral type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                      equations, volume_integral::VolumeIntegralFluxDifferencing, dg::DG, uEltype)
  NamedTuple()
end


function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}}, equations,
                      volume_integral::VolumeIntegralShockCapturingHG, dg::DG, uEltype)
  element_ids_dg   = Int[]
  element_ids_dgfv = Int[]

  cache = create_cache(mesh, equations,
                       VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                       dg, uEltype)

  A3dp1_x = Array{uEltype, 3}
  A3dp1_y = Array{uEltype, 3}

  fstar1_L_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fstar1_R_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fstar2_L_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]
  fstar2_R_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]

  return (; cache..., element_ids_dg, element_ids_dgfv,
          fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded)
end


function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}}, equations,
                      volume_integral::VolumeIntegralPureLGLFiniteVolume, dg::DG, uEltype)

  A3dp1_x = Array{uEltype, 3}
  A3dp1_y = Array{uEltype, 3}

  fstar1_L_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fstar1_R_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fstar2_L_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]
  fstar2_R_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]

  return (; fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded)
end


function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}}, equations,
                      volume_integral::VolumeIntegralShockCapturingSubcell, dg::DG, uEltype)

  cache = create_cache(mesh, equations,
                       VolumeIntegralPureLGLFiniteVolume(volume_integral.volume_flux_fv),
                       dg, uEltype)
  if volume_integral.indicator.indicator_smooth
    element_ids_dg   = Int[]
    element_ids_dgfv = Int[]
    cache = (; cache..., element_ids_dg, element_ids_dgfv)
  end

  A3dp1_x = Array{uEltype, 3}
  A3dp1_y = Array{uEltype, 3}
  A3d = Array{uEltype, 3}

  fhat1_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fhat2_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]
  flux_temp_threaded = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg)) for _ in 1:Threads.nthreads()]

  ContainerAntidiffusiveFlux2D = Trixi.ContainerAntidiffusiveFlux2D{uEltype}(0, nvariables(equations), nnodes(dg))

  return (; cache..., ContainerAntidiffusiveFlux2D, fhat1_threaded, fhat2_threaded, flux_temp_threaded)
end


# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                      equations, mortar_l2::LobattoLegendreMortarL2, uEltype)
  # TODO: Taal performance using different types
  MA2d = MArray{Tuple{nvariables(equations), nnodes(mortar_l2)}, uEltype, 2, nvariables(equations) * nnodes(mortar_l2)}
  fstar_upper_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]
  fstar_lower_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]

  # A2d = Array{uEltype, 2}
  # fstar_upper_threaded = [A2d(undef, nvariables(equations), nnodes(mortar_l2)) for _ in 1:Threads.nthreads()]
  # fstar_lower_threaded = [A2d(undef, nvariables(equations), nnodes(mortar_l2)) for _ in 1:Threads.nthreads()]

  (; fstar_upper_threaded, fstar_lower_threaded)
end


# TODO: Taal discuss/refactor timer, allowing users to pass a custom timer?

function rhs!(du, u, t,
              mesh::Union{TreeMesh{2}, P4estMesh{2}}, equations,
              initial_condition, boundary_conditions, source_terms::Source,
              dg::DG, cache) where {Source}
  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache, t, boundary_conditions)

  # Prolong solution to interfaces
  @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Calculate interface fluxes
  @trixi_timeit timer() "interface flux" calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.surface_integral, dg, cache)

  # Prolong solution to boundaries
  @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
    cache, t, boundary_conditions, mesh, equations, dg.surface_integral, dg)

  # Prolong solution to mortars
  @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
    cache, u, mesh, equations, dg.mortar, dg.surface_integral, dg)

  # Calculate mortar fluxes
  @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.mortar, dg.surface_integral, dg, cache)

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" calc_surface_integral!(
    du, u, mesh, equations, dg.surface_integral, dg, cache)

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @trixi_timeit timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_volume_integral!(du, u, mesh,
                               nonconservative_terms, equations,
                               volume_integral::AbstractVolumeIntegral,
                               dg, cache, t, boundary_conditions)

  calc_volume_integral!(du, u, mesh,
                        nonconservative_terms, equations,
                        volume_integral, dg, cache)

  return nothing
end


function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)

  @threaded for element in eachelement(dg, cache)
    weak_form_kernel!(du, u, element, mesh,
                      nonconservative_terms, equations,
                      dg, cache)
  end

  return nothing
end

@inline function weak_form_kernel!(du, u,
                                   element, mesh::TreeMesh{2},
                                   nonconservative_terms::False, equations,
                                   dg::DGSEM, cache, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_dhat = dg.basis

  # Calculate volume terms in one element
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    flux1 = flux(u_node, 1, equations)
    for ii in eachnode(dg)
      multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], flux1, equations, dg, ii, j, element)
    end

    flux2 = flux(u_node, 2, equations)
    for jj in eachnode(dg)
      multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], flux2, equations, dg, i, jj, element)
    end
  end

  return nothing
end


# flux differencing volume integral. For curved meshes averaging of the
# mapping terms, stored in `cache.elements.contravariant_vectors`, is peeled apart
# from the evaluation of the physical fluxes in each Cartesian direction
function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM, cache)
  @threaded for element in eachelement(dg, cache)
    flux_differencing_kernel!(du, u, element, mesh,
                              nonconservative_terms, equations,
                              volume_integral.volume_flux, dg, cache)
  end
end

@inline function flux_differencing_kernel!(du, u,
                                           element, mesh::TreeMesh{2},
                                           nonconservative_terms::False, equations,
                                           volume_flux, dg::DGSEM, cache, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_split = dg.basis

  # Calculate volume integral in one element
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.

    # x direction
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      flux1 = volume_flux(u_node, u_node_ii, 1, equations)
      multiply_add_to_node_vars!(du, alpha * derivative_split[i, ii], flux1, equations, dg, i,  j, element)
      multiply_add_to_node_vars!(du, alpha * derivative_split[ii, i], flux1, equations, dg, ii, j, element)
    end

    # y direction
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      flux2 = volume_flux(u_node, u_node_jj, 2, equations)
      multiply_add_to_node_vars!(du, alpha * derivative_split[j, jj], flux2, equations, dg, i, j,  element)
      multiply_add_to_node_vars!(du, alpha * derivative_split[jj, j], flux2, equations, dg, i, jj, element)
    end
  end
end

@inline function flux_differencing_kernel!(du, u,
                                           element, mesh::TreeMesh{2},
                                           nonconservative_terms::True, equations,
                                           volume_flux, dg::DGSEM, cache, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_split = dg.basis
  symmetric_flux, nonconservative_flux = volume_flux

  # Apply the symmetric flux as usual
  flux_differencing_kernel!(du, u, element, mesh, False(), equations, symmetric_flux, dg, cache, alpha)

  # Calculate the remaining volume terms using the nonsymmetric generalized flux
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # The diagonal terms are zero since the diagonal of `derivative_split`
    # is zero. We ignore this for now.

    # x direction
    integral_contribution = zero(u_node)
    for ii in eachnode(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      noncons_flux1 = nonconservative_flux(u_node, u_node_ii, 1, equations)
      integral_contribution = integral_contribution + derivative_split[i, ii] * noncons_flux1
    end

    # y direction
    for jj in eachnode(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      noncons_flux2 = nonconservative_flux(u_node, u_node_jj, 2, equations)
      integral_contribution = integral_contribution + derivative_split[j, jj] * noncons_flux2
    end

    # The factor 0.5 cancels the factor 2 in the flux differencing form
    multiply_add_to_node_vars!(du, alpha * 0.5, integral_contribution, equations, dg, i, j, element)
  end
end


# TODO: Taal dimension agnostic
function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGSEM, cache)
  @unpack element_ids_dg, element_ids_dgfv = cache
  @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

  # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
  alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg, cache)

  # Determine element ids for DG-only and blended DG-FV volume integral
  pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)

  # Loop over pure DG elements
  @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
    element = element_ids_dg[idx_element]
    flux_differencing_kernel!(du, u, element, mesh,
                              nonconservative_terms, equations,
                              volume_flux_dg, dg, cache)
  end

  # Loop over blended DG-FV elements
  @trixi_timeit timer() "blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
    element = element_ids_dgfv[idx_element]
    alpha_element = alpha[element]

    # Calculate DG volume integral contribution
    flux_differencing_kernel!(du, u, element, mesh,
                              nonconservative_terms, equations,
                              volume_flux_dg, dg, cache, 1 - alpha_element)

    # Calculate FV volume integral contribution
    fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
               dg, cache, element, alpha_element)
  end

  return nothing
end

# TODO: Taal dimension agnostic
function calc_volume_integral!(du, u,
                               mesh::TreeMesh{2},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralPureLGLFiniteVolume,
                               dg::DGSEM, cache)
  @unpack volume_flux_fv = volume_integral

  # Calculate LGL FV volume integral
  @threaded for element in eachelement(dg, cache)
    fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
               dg, cache, element, true)
  end

  return nothing
end


@inline function fv_kernel!(du, u,
                            mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                            nonconservative_terms, equations,
                            volume_flux_fv, dg::DGSEM, cache, element, alpha=true)
  @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache
  @unpack inverse_weights = dg.basis

  # Calculate FV two-point fluxes
  fstar1_L = fstar1_L_threaded[Threads.threadid()]
  fstar2_L = fstar2_L_threaded[Threads.threadid()]
  fstar1_R = fstar1_R_threaded[Threads.threadid()]
  fstar2_R = fstar2_R_threaded[Threads.threadid()]
  calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
               nonconservative_terms, equations, volume_flux_fv, dg, element, cache)

  # Calculate FV volume integral contribution
  for j in eachnode(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      du[v, i, j, element] += ( alpha *
                                (inverse_weights[i] * (fstar1_L[v, i+1, j] - fstar1_R[v, i, j]) +
                                 inverse_weights[j] * (fstar2_L[v, i, j+1] - fstar2_R[v, i, j])) )
    end
  end

  return nothing
end



#     calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u_leftright,
#                  nonconservative_terms::False, equations,
#                  volume_flux_fv, dg, element)
#
# Calculate the finite volume fluxes inside the elements (**without non-conservative terms**).
#
# # Arguments
# - `fstar1_L::AbstractArray{<:Real, 3}`
# - `fstar1_R::AbstractArray{<:Real, 3}`
# - `fstar2_L::AbstractArray{<:Real, 3}`
# - `fstar2_R::AbstractArray{<:Real, 3}`
@inline function calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u::AbstractArray{<:Any,4},
                              mesh::TreeMesh{2}, nonconservative_terms::False, equations,
                              volume_flux_fv, dg::DGSEM, element, cache)

  fstar1_L[:, 1,            :] .= zero(eltype(fstar1_L))
  fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
  fstar1_R[:, 1,            :] .= zero(eltype(fstar1_R))
  fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

  for j in eachnode(dg), i in 2:nnodes(dg)
    u_ll = get_node_vars(u, equations, dg, i-1, j, element)
    u_rr = get_node_vars(u, equations, dg, i,   j, element)
    flux = volume_flux_fv(u_ll, u_rr, 1, equations) # orientation 1: x direction
    set_node_vars!(fstar1_L, flux, equations, dg, i, j)
    set_node_vars!(fstar1_R, flux, equations, dg, i, j)
  end

  fstar2_L[:, :, 1           ] .= zero(eltype(fstar2_L))
  fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
  fstar2_R[:, :, 1           ] .= zero(eltype(fstar2_R))
  fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

  for j in 2:nnodes(dg), i in eachnode(dg)
    u_ll = get_node_vars(u, equations, dg, i, j-1, element)
    u_rr = get_node_vars(u, equations, dg, i, j,   element)
    flux = volume_flux_fv(u_ll, u_rr, 2, equations) # orientation 2: y direction
    set_node_vars!(fstar2_L, flux, equations, dg, i, j)
    set_node_vars!(fstar2_R, flux, equations, dg, i, j)
  end

  return nothing
end

#     calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u_leftright,
#                  nonconservative_terms::True, equations,
#                  volume_flux_fv, dg, element)
#
# Calculate the finite volume fluxes inside the elements (**with non-conservative terms**).
#
# # Arguments
# - `fstar1_L::AbstractArray{<:Real, 3}`:
# - `fstar1_R::AbstractArray{<:Real, 3}`:
# - `fstar2_L::AbstractArray{<:Real, 3}`:
# - `fstar2_R::AbstractArray{<:Real, 3}`:
# - `u_leftright::AbstractArray{<:Real, 4}`
@inline function calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u::AbstractArray{<:Any,4},
                              mesh::TreeMesh{2}, nonconservative_terms::True, equations,
                              volume_flux_fv, dg::DGSEM, element, cache)
  volume_flux, nonconservative_flux = volume_flux_fv

  # Fluxes in x
  fstar1_L[:, 1,            :] .= zero(eltype(fstar1_L))
  fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
  fstar1_R[:, 1,            :] .= zero(eltype(fstar1_R))
  fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

  for j in eachnode(dg), i in 2:nnodes(dg)
    u_ll = get_node_vars(u, equations, dg, i-1, j, element)
    u_rr = get_node_vars(u, equations, dg, i,   j, element)

    # Compute conservative part
    f1 = volume_flux(u_ll, u_rr, 1, equations) # orientation 1: x direction

    # Compute nonconservative part
    # Note the factor 0.5 necessary for the nonconservative fluxes based on
    # the interpretation of global SBP operators coupled discontinuously via
    # central fluxes/SATs
    f1_L = f1 + 0.5 * nonconservative_flux(u_ll, u_rr, 1, equations)
    f1_R = f1 + 0.5 * nonconservative_flux(u_rr, u_ll, 1, equations)

    # Copy to temporary storage
    set_node_vars!(fstar1_L, f1_L, equations, dg, i, j)
    set_node_vars!(fstar1_R, f1_R, equations, dg, i, j)
  end

  # Fluxes in y
  fstar2_L[:, :, 1           ] .= zero(eltype(fstar2_L))
  fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
  fstar2_R[:, :, 1           ] .= zero(eltype(fstar2_R))
  fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

  # Compute inner fluxes
  for j in 2:nnodes(dg), i in eachnode(dg)
    u_ll = get_node_vars(u, equations, dg, i, j-1, element)
    u_rr = get_node_vars(u, equations, dg, i, j,   element)

    # Compute conservative part
    f2 = volume_flux(u_ll, u_rr, 2, equations) # orientation 2: y direction

    # Compute nonconservative part
    # Note the factor 0.5 necessary for the nonconservative fluxes based on
    # the interpretation of global SBP operators coupled discontinuously via
    # central fluxes/SATs
    f2_L = f2 + 0.5 * nonconservative_flux(u_ll, u_rr, 2, equations)
    f2_R = f2 + 0.5 * nonconservative_flux(u_rr, u_ll, 2, equations)

    # Copy to temporary storage
    set_node_vars!(fstar2_L, f2_L, equations, dg, i, j)
    set_node_vars!(fstar2_R, f2_R, equations, dg, i, j)
  end

  return nothing
end


function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingSubcell,
                               dg::DGSEM, cache, t, boundary_conditions)
  @unpack indicator = volume_integral

  # Calculate lambdas and bar states
  @trixi_timeit timer() "calc_lambdas_bar_states!" calc_lambdas_bar_states!(u, t, mesh,
      nonconservative_terms, equations, indicator, dg, cache, boundary_conditions)
  # Calculate boundaries
  @trixi_timeit timer() "calc_var_bounds!" calc_var_bounds!(u, mesh, nonconservative_terms, equations, indicator, dg, cache)

  if indicator.indicator_smooth
    @unpack element_ids_dg, element_ids_dgfv = cache
    # Calculate element-wise blending factors α
    alpha_element = @trixi_timeit timer() "element-wise blending factors" indicator.IndicatorHG(u, mesh, equations, dg, cache)

    # Determine element ids for DG-only and subcell-wise blended DG-FV volume integral
    pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha_element, dg, cache)

    # Loop over pure DG elements
    @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
      element = element_ids_dg[idx_element]
      flux_differencing_kernel!(du, u, element, mesh,
                                nonconservative_terms, equations,
                                volume_integral.volume_flux_dg, dg, cache)
    end

    # Loop over blended DG-FV elements
    @trixi_timeit timer() "subcell-wise blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
      element = element_ids_dgfv[idx_element]
      subcell_limiting_kernel!(du, u, element, mesh,
                               nonconservative_terms, equations,
                               volume_integral, indicator,
                               dg, cache)
    end
  else # indicator.indicator_smooth == false
    # Loop over all elements
    @trixi_timeit timer() "subcell-wise blended DG-FV" @threaded for element in eachelement(dg, cache)
      subcell_limiting_kernel!(du, u, element, mesh,
                               nonconservative_terms, equations,
                               volume_integral, indicator,
                               dg, cache)
    end
  end
end

@inline function subcell_limiting_kernel!(du, u,
                                          element, mesh::Union{TreeMesh{2}, StructuredMesh{2}},
                                          nonconservative_terms::False, equations,
                                          volume_integral, indicator::IndicatorIDP,
                                          dg::DGSEM, cache)
  @unpack inverse_weights = dg.basis
  @unpack volume_flux_dg, volume_flux_fv = volume_integral

  # high-order DG fluxes
  @unpack fhat1_threaded, fhat2_threaded = cache

  fhat1 = fhat1_threaded[Threads.threadid()]
  fhat2 = fhat2_threaded[Threads.threadid()]
  calcflux_fhat!(fhat1, fhat2, u, mesh,
      nonconservative_terms, equations, volume_flux_dg, dg, element, cache)

  # low-order FV fluxes
  @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache

  fstar1_L = fstar1_L_threaded[Threads.threadid()]
  fstar2_L = fstar2_L_threaded[Threads.threadid()]
  fstar1_R = fstar1_R_threaded[Threads.threadid()]
  fstar2_R = fstar2_R_threaded[Threads.threadid()]
  calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
      nonconservative_terms, equations, volume_flux_fv, dg, element, cache)

  # antidiffusive flux
  calcflux_antidiffusive!(fhat1, fhat2, fstar1_L, fstar2_L, u, mesh,
      nonconservative_terms, equations, indicator, dg, element, cache)

  # Calculate volume integral contribution of low-order FV flux
  for j in eachnode(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      du[v, i, j, element] += inverse_weights[i] * (fstar1_L[v, i+1, j] - fstar1_R[v, i, j]) +
                              inverse_weights[j] * (fstar2_L[v, i, j+1] - fstar2_R[v, i, j])

    end
  end

  return nothing
end

@inline function subcell_limiting_kernel!(du, u,
                                          element, mesh::Union{TreeMesh{2},StructuredMesh{2}},
                                          nonconservative_terms::False, equations,
                                          volume_integral, indicator::IndicatorMCL,
                                          dg::DGSEM, cache)
  @unpack inverse_weights = dg.basis
  @unpack volume_flux_dg, volume_flux_fv = volume_integral

  # high-order DG fluxes
  @unpack fhat1_threaded, fhat2_threaded = cache
  fhat1 = fhat1_threaded[Threads.threadid()]
  fhat2 = fhat2_threaded[Threads.threadid()]
  calcflux_fhat!(fhat1, fhat2, u, mesh,
      nonconservative_terms, equations, volume_flux_dg, dg, element, cache)

  # low-order FV fluxes
  @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache
  fstar1_L = fstar1_L_threaded[Threads.threadid()]
  fstar2_L = fstar2_L_threaded[Threads.threadid()]
  fstar1_R = fstar1_R_threaded[Threads.threadid()]
  fstar2_R = fstar2_R_threaded[Threads.threadid()]
  calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
      nonconservative_terms, equations, volume_flux_fv, dg, element, cache)

  # antidiffusive flux
  calcflux_antidiffusive!(fhat1, fhat2, fstar1_L, fstar2_L,
      u, mesh, nonconservative_terms, equations, indicator, dg, element, cache)

  # limit antidiffusive flux
  calcflux_antidiffusive_limited!(u, mesh, nonconservative_terms, equations, indicator, dg, element, cache,
                                  fstar1_L, fstar2_L)

  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D
  for j in eachnode(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      du[v, i, j, element] += inverse_weights[i] * (fstar1_L[v, i+1, j] - fstar1_R[v, i, j]) +
                              inverse_weights[j] * (fstar2_L[v, i, j+1] - fstar2_R[v, i, j])

      du[v, i, j, element] += inverse_weights[i] * (-antidiffusive_flux1[v, i+1, j, element] + antidiffusive_flux1[v, i, j, element]) +
                              inverse_weights[j] * (-antidiffusive_flux2[v, i, j+1, element] + antidiffusive_flux2[v, i, j, element])
    end
  end

  return nothing
end


#     calcflux_fhat!(fhat1, fhat2, u, mesh,
#                    nonconservative_terms, equations, volume_flux_dg, dg, element, cache)
#
# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**without non-conservative terms**).
#
# # Arguments
# - `fhat1::AbstractArray{<:Real, 3}`
# - `fhat2::AbstractArray{<:Real, 3}`
@inline function calcflux_fhat!(fhat1, fhat2, u,
                                mesh::TreeMesh{2}, nonconservative_terms::False, equations,
                                volume_flux, dg::DGSEM, element, cache)

  @unpack weights, derivative_split = dg.basis
  @unpack flux_temp_threaded = cache

  flux_temp = flux_temp_threaded[Threads.threadid()]

  # The FV-form fluxes are calculated in a recursive manner, i.e.:
  # fhat_(0,1)   = w_0 * FVol_0,
  # fhat_(j,j+1) = fhat_(j-1,j) + w_j * FVol_j,   for j=1,...,N-1,
  # with the split form volume fluxes FVol_j = -2 * sum_i=0^N D_ji f*_(j,i).

  # To use the symmetry of the `volume_flux`, the split form volume flux is precalculated
  # like in `calc_volume_integral!` for the `VolumeIntegralFluxDifferencing`
  # and saved in in `flux_temp`.

  # Split form volume flux in orientation 1: x direction
  flux_temp .= zero(eltype(flux_temp))

  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      flux1 = volume_flux(u_node, u_node_ii, 1, equations)
      multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1, equations, dg, i,  j)
      multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1, equations, dg, ii, j)
    end
  end

  # FV-form flux `fhat` in x direction
  fhat1[:, 1,            :] .= zero(eltype(fhat1))
  fhat1[:, nnodes(dg)+1, :] .= zero(eltype(fhat1))

  for j in eachnode(dg), i in 1:nnodes(dg)-1, v in eachvariable(equations)
    fhat1[v, i+1, j] = fhat1[v, i, j] + weights[i] * flux_temp[v, i, j]
  end

  # Split form volume flux in orientation 2: y direction
  flux_temp .= zero(eltype(flux_temp))

  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      flux2 = volume_flux(u_node, u_node_jj, 2, equations)
      multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2, equations, dg, i, j)
      multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2, equations, dg, i, jj)
    end
  end

  # FV-form flux `fhat` in y direction
  fhat2[:, :, 1           ] .= zero(eltype(fhat2))
  fhat2[:, :, nnodes(dg)+1] .= zero(eltype(fhat2))

  for j in 1:nnodes(dg)-1, i in eachnode(dg), v in eachvariable(equations)
    fhat2[v, i, j+1] = fhat2[v, i, j] + weights[j] * flux_temp[v, i, j]
  end

  return nothing
end

@inline function calcflux_antidiffusive!(fhat1, fhat2, fstar1, fstar2, u, mesh,
                                         nonconservative_terms, equations, indicator::IndicatorIDP, dg, element, cache)
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D

  for j in eachnode(dg), i in 2:nnodes(dg)
    for v in eachvariable(equations)
      antidiffusive_flux1[v, i, j, element] = fhat1[v, i, j] - fstar1[v, i, j]
    end
  end
  for j in 2:nnodes(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      antidiffusive_flux2[v, i, j, element] = fhat2[v, i, j] - fstar2[v, i, j]
    end
  end

  antidiffusive_flux1[:, 1,            :, element] .= zero(eltype(antidiffusive_flux1))
  antidiffusive_flux1[:, nnodes(dg)+1, :, element] .= zero(eltype(antidiffusive_flux1))

  antidiffusive_flux2[:, :, 1,            element] .= zero(eltype(antidiffusive_flux2))
  antidiffusive_flux2[:, :, nnodes(dg)+1, element] .= zero(eltype(antidiffusive_flux2))

  return nothing
end

@inline function calcflux_antidiffusive!(fhat1, fhat2, fstar1, fstar2, u, mesh,
                                         nonconservative_terms, equations, indicator::IndicatorMCL, dg, element, cache)
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D

  for j in eachnode(dg), i in 2:nnodes(dg)
    for v in eachvariable(equations)
      antidiffusive_flux1[v, i, j, element] = -(fhat1[v, i, j] - fstar1[v, i, j])
    end
  end
  for j in 2:nnodes(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      antidiffusive_flux2[v, i, j, element] = -(fhat2[v, i, j] - fstar2[v, i, j])
    end
  end

  antidiffusive_flux1[:, 1,            :, element] .= zero(eltype(antidiffusive_flux1))
  antidiffusive_flux1[:, nnodes(dg)+1, :, element] .= zero(eltype(antidiffusive_flux1))

  antidiffusive_flux2[:, :, 1,            element] .= zero(eltype(antidiffusive_flux2))
  antidiffusive_flux2[:, :, nnodes(dg)+1, element] .= zero(eltype(antidiffusive_flux2))

  return nothing
end

@inline function calc_lambdas_bar_states!(u, t, mesh::TreeMesh,
    nonconservative_terms, equations, indicator, dg, cache, boundary_conditions; calcBarStates=true)

  if indicator isa IndicatorIDP && !indicator.BarStates
    return nothing
  end
  @unpack lambda1, lambda2, bar_states1, bar_states2 = indicator.cache.ContainerBarStates

  # Calc lambdas and bar states inside elements
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in 2:nnodes(dg)
      u_node     = get_node_vars(u, equations, dg, i,   j, element)
      u_node_im1 = get_node_vars(u, equations, dg, i-1, j, element)
      lambda1[i, j, element] = max_abs_speed_naive(u_node_im1, u_node, 1, equations)

      !calcBarStates && continue

      flux1     = flux(u_node,     1, equations)
      flux1_im1 = flux(u_node_im1, 1, equations)
      for v in eachvariable(equations)
        bar_states1[v, i, j, element] = 0.5 * (u_node[v] + u_node_im1[v]) - 0.5 * (flux1[v] - flux1_im1[v]) / lambda1[i, j, element]
      end
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
      u_node     = get_node_vars(u, equations, dg, i, j  , element)
      u_node_jm1 = get_node_vars(u, equations, dg, i, j-1, element)
      lambda2[i, j, element] = max_abs_speed_naive(u_node_jm1, u_node, 2, equations)

      !calcBarStates && continue

      flux2     = flux(u_node,     2, equations)
      flux2_jm1 = flux(u_node_jm1, 2, equations)
      for v in eachvariable(equations)
        bar_states2[v, i, j, element] = 0.5 * (u_node[v] + u_node_jm1[v]) - 0.5 * (flux2[v] - flux2_jm1[v]) / lambda2[i, j, element]
      end
    end
  end

  # Calc lambdas and bar states at interfaces and periodic boundaries
  @threaded for interface in eachinterface(dg, cache)
    # Get neighboring element ids
    left_id  = cache.interfaces.neighbor_ids[1, interface]
    right_id = cache.interfaces.neighbor_ids[2, interface]

    orientation = cache.interfaces.orientations[interface]

    if orientation == 1
      for j in eachnode(dg)
        u_left  = get_node_vars(u, equations, dg, nnodes(dg), j, left_id)
        u_right = get_node_vars(u, equations, dg, 1,          j, right_id)
        lambda = max_abs_speed_naive(u_left, u_right, orientation, equations)

        lambda1[nnodes(dg)+1, j, left_id]  = lambda
        lambda1[1,            j, right_id] = lambda

        !calcBarStates && continue

        flux_left  = flux(u_left,  orientation, equations)
        flux_right = flux(u_right, orientation, equations)
        bar_state = 0.5 * (u_left + u_right) - 0.5 * (flux_right - flux_left) / lambda
        for v in eachvariable(equations)
          bar_states1[v, nnodes(dg)+1, j, left_id]  = bar_state[v]
          bar_states1[v, 1,            j, right_id] = bar_state[v]
        end
      end
    else # orientation == 2
      for i in eachnode(dg)
        u_left  = get_node_vars(u, equations, dg, i, nnodes(dg), left_id)
        u_right = get_node_vars(u, equations, dg, i, 1,          right_id)
        lambda = max_abs_speed_naive(u_left, u_right, orientation, equations)

        lambda2[i, nnodes(dg)+1, left_id]  = lambda
        lambda2[i,            1, right_id] = lambda

        !calcBarStates && continue

        flux_left  = flux(u_left,  orientation, equations)
        flux_right = flux(u_right, orientation, equations)
        bar_state = 0.5 * (u_left + u_right) - 0.5 * (flux_right - flux_left) / lambda
        for v in eachvariable(equations)
          bar_states2[v, i, nnodes(dg)+1, left_id]  = bar_state[v]
          bar_states2[v, i,            1, right_id] = bar_state[v]
        end
      end
    end
  end

  # Calc lambdas and bar states at physical boundaries
  @threaded for boundary in eachboundary(dg, cache)
    element = cache.boundaries.neighbor_ids[boundary]
    orientation = cache.boundaries.orientations[boundary]
    neighbor_side = cache.boundaries.neighbor_sides[boundary]

    if orientation == 1
      if neighbor_side == 2 # Element is on the right, boundary on the left
        for j in eachnode(dg)
          u_inner = get_node_vars(u, equations, dg, 1, j, element)
          u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[1], orientation, 1,
                                             equations, dg, 1, j, element)
          lambda1[1, j, element] = max_abs_speed_naive(u_inner, u_outer, orientation, equations)

          !calcBarStates && continue

          flux_inner = flux(u_inner, orientation, equations)
          flux_outer = flux(u_outer, orientation, equations)
          bar_state = 0.5 * (u_inner + u_outer) - 0.5 * (flux_inner - flux_outer) / lambda1[1, j, element]
          for v in eachvariable(equations)
            bar_states1[v, 1, j, element] = bar_state[v]
          end
        end
      else # Element is on the left, boundary on the right
        for j in eachnode(dg)
          u_inner = get_node_vars(u, equations, dg, nnodes(dg), j, element)
          u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[2], orientation, 2,
                                             equations, dg, nnodes(dg), j, element)
          lambda1[nnodes(dg)+1, j, element] = max_abs_speed_naive(u_inner, u_outer, orientation, equations)

          !calcBarStates && continue

          flux_inner = flux(u_inner, orientation, equations)
          flux_outer = flux(u_outer, orientation, equations)
          bar_state = 0.5 * (u_inner + u_outer) - 0.5 * (flux_outer - flux_inner) / lambda1[nnodes(dg)+1, j, element]
          for v in eachvariable(equations)
            bar_states1[v, nnodes(dg)+1, j, element] = bar_state[v]
          end
        end
      end
    else # orientation == 2
      if neighbor_side == 2 # Element is on the right, boundary on the left
        for i in eachnode(dg)
          u_inner = get_node_vars(u, equations, dg, i, 1, element)
          u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[3], orientation, 3,
                                             equations, dg, i, 1, element)
          lambda2[i, 1, element] = max_abs_speed_naive(u_inner, u_outer, orientation, equations)

          !calcBarStates && continue

          flux_inner = flux(u_inner, orientation, equations)
          flux_outer = flux(u_outer, orientation, equations)
          bar_state = 0.5 * (u_inner + u_outer) - 0.5 * (flux_inner - flux_outer) / lambda2[i, 1, element]
          for v in eachvariable(equations)
            bar_states2[v, i, 1, element] = bar_state[v]
          end
        end
      else # Element is on the left, boundary on the right
        for i in eachnode(dg)
          u_inner = get_node_vars(u, equations, dg, i, nnodes(dg), element)
          u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[4], orientation, 4,
                                             equations, dg, i, nnodes(dg), element)
          lambda2[i, nnodes(dg)+1, element] = max_abs_speed_naive(u_inner, u_outer, orientation, equations)

          !calcBarStates && continue

          flux_inner = flux(u_inner, orientation, equations)
          flux_outer = flux(u_outer, orientation, equations)
          bar_state = 0.5 * (u_inner + u_outer) - 0.5 * (flux_outer - flux_inner) / lambda2[i, nnodes(dg)+1, element]
          for v in eachvariable(equations)
            bar_states2[v, i, nnodes(dg)+1, element] = bar_state[v]
          end
        end
      end
    end
  end

  return nothing
end

@inline function calc_var_bounds!(u, mesh, nonconservative_terms, equations, indicator::IndicatorIDP, dg, cache)
  if !indicator.BarStates
    return nothing
  end
  @unpack var_bounds = indicator.cache.ContainerShockCapturingIndicator
  @unpack bar_states1, bar_states2 = indicator.cache.ContainerBarStates

  counter = 1
  # Density
  if indicator.IDPDensityTVD
    rho_min = var_bounds[1]
    rho_max = var_bounds[2]
    @threaded for element in eachelement(dg, cache)
      rho_min[:, :, element] .= typemax(eltype(rho_min))
      rho_max[:, :, element] .= typemin(eltype(rho_max))
      for j in eachnode(dg), i in eachnode(dg)
        rho_min[i, j, element] = min(rho_min[i, j, element], u[1, i, j, element])
        rho_max[i, j, element] = max(rho_max[i, j, element], u[1, i, j, element])
        # TODO: Add source term!
        # - xi direction
        rho_min[i, j, element] = min(rho_min[i, j, element], bar_states1[1, i, j, element])
        rho_max[i, j, element] = max(rho_max[i, j, element], bar_states1[1, i, j, element])
        # + xi direction
        rho_min[i, j, element] = min(rho_min[i, j, element], bar_states1[1, i+1, j, element])
        rho_max[i, j, element] = max(rho_max[i, j, element], bar_states1[1, i+1, j, element])
        # - eta direction
        rho_min[i, j, element] = min(rho_min[i, j, element], bar_states2[1, i, j, element])
        rho_max[i, j, element] = max(rho_max[i, j, element], bar_states2[1, i, j, element])
        # + eta direction
        rho_min[i, j, element] = min(rho_min[i, j, element], bar_states2[1, i, j+1, element])
        rho_max[i, j, element] = max(rho_max[i, j, element], bar_states2[1, i, j+1, element])
      end
    end
    counter += 2
  end
  # Pressure
  if indicator.IDPPressureTVD
    p_min = var_bounds[counter]
    p_max = var_bounds[counter+1]
    @threaded for element in eachelement(dg, cache)
      p_min[:, :, element] .= typemax(eltype(p_min))
      p_max[:, :, element] .= typemin(eltype(p_max))
      for j in eachnode(dg), i in eachnode(dg)
        p = pressure(get_node_vars(u, equations, dg, i, j, element), equations)
        p_min[i, j, element] = min(p_min[i, j, element], p)
        p_max[i, j, element] = max(p_max[i, j, element], p)
        # - xi direction
        p = pressure(get_node_vars(bar_states1, equations, dg, i, j, element), equations)
        p_min[i, j, element] = min(p_min[i, j, element], p)
        p_max[i, j, element] = max(p_max[i, j, element], p)
        # + xi direction
        p = pressure(get_node_vars(bar_states1, equations, dg, i+1, j, element), equations)
        p_min[i, j, element] = min(p_min[i, j, element], p)
        p_max[i, j, element] = max(p_max[i, j, element], p)
        # - eta direction
        p = pressure(get_node_vars(bar_states2, equations, dg, i, j, element), equations)
        p_min[i, j, element] = min(p_min[i, j, element], p)
        p_max[i, j, element] = max(p_max[i, j, element], p)
        # + eta direction
        p = pressure(get_node_vars(bar_states2, equations, dg, i, j+1, element), equations)
        p_min[i, j, element] = min(p_min[i, j, element], p)
        p_max[i, j, element] = max(p_max[i, j, element], p)
      end
    end
    counter += 2
  end
  # Specific Entropy
  if indicator.IDPSpecEntropy
    s_min = var_bounds[counter]
    @threaded for element in eachelement(dg, cache)
      s_min[:, :, element] .= typemax(eltype(s_min))
      for j in eachnode(dg), i in eachnode(dg)
        s = entropy_spec(get_node_vars(u, equations, dg, i, j, element), equations)
        s_min[i, j, element] = min(s_min[i, j, element], s)
        # TODO: Add source?
        # - xi direction
        s = entropy_spec(get_node_vars(bar_states1, equations, dg, i, j, element), equations)
        s_min[i, j, element] = min(s_min[i, j, element], s)
        # + xi direction
        s = entropy_spec(get_node_vars(bar_states1, equations, dg, i+1, j, element), equations)
        s_min[i, j, element] = min(s_min[i, j, element], s)
        # - eta direction
        s = entropy_spec(get_node_vars(bar_states2, equations, dg, i, j, element), equations)
        s_min[i, j, element] = min(s_min[i, j, element], s)
        # + eta direction
        s = entropy_spec(get_node_vars(bar_states2, equations, dg, i, j+1, element), equations)
        s_min[i, j, element] = min(s_min[i, j, element], s)
      end
    end
    counter += 1
  end
  # Mathematical entropy
  if indicator.IDPMathEntropy
    s_max = var_bounds[counter]
    @threaded for element in eachelement(dg, cache)
      s_max[:, :, element] .= typemin(eltype(s_max))
      for j in eachnode(dg), i in eachnode(dg)
        s = entropy_math(get_node_vars(u, equations, dg, i, j, element), equations)
        s_max[i, j, element] = max(s_max[i, j, element], s)
        # - xi direction
        s = entropy_math(get_node_vars(bar_states1, equations, dg, i, j, element), equations)
        s_max[i, j, element] = max(s_max[i, j, element], s)
        # + xi direction
        s = entropy_math(get_node_vars(bar_states1, equations, dg, i+1, j, element), equations)
        s_max[i, j, element] = max(s_max[i, j, element], s)
        # - eta direction
        s = entropy_math(get_node_vars(bar_states2, equations, dg, i, j, element), equations)
        s_max[i, j, element] = max(s_max[i, j, element], s)
        # + eta direction
        s = entropy_math(get_node_vars(bar_states2, equations, dg, i, j+1, element), equations)
        s_max[i, j, element] = max(s_max[i, j, element], s)
      end
    end
  end

  return nothing
end

@inline function calc_var_bounds!(u, mesh, nonconservative_terms, equations, indicator::IndicatorMCL, dg, cache)
  @unpack var_min, var_max = indicator.cache.ContainerShockCapturingIndicator
  @unpack bar_states1, bar_states2, lambda1, lambda2 = indicator.cache.ContainerBarStates

  @threaded for element in eachelement(dg, cache)
    for v in eachvariable(equations)
      var_min[v, :, :, element] .= typemax(eltype(var_min))
      var_max[v, :, :, element] .= typemin(eltype(var_max))
    end

    if indicator.DensityLimiter
      for j in eachnode(dg), i in eachnode(dg)
        # Previous solution
        var_min[1, i, j, element] = min(var_min[1, i, j, element], u[1, i, j, element])
        var_max[1, i, j, element] = max(var_max[1, i, j, element], u[1, i, j, element])
        # - xi direction
        bar_state_rho = bar_states1[1, i, j, element]
        var_min[1, i, j, element] = min(var_min[1, i, j, element], bar_state_rho)
        var_max[1, i, j, element] = max(var_max[1, i, j, element], bar_state_rho)
        # + xi direction
        bar_state_rho = bar_states1[1, i+1, j, element]
        var_min[1, i, j, element] = min(var_min[1, i, j, element], bar_state_rho)
        var_max[1, i, j, element] = max(var_max[1, i, j, element], bar_state_rho)
        # - eta direction
        bar_state_rho = bar_states2[1, i, j, element]
        var_min[1, i, j, element] = min(var_min[1, i, j, element], bar_state_rho)
        var_max[1, i, j, element] = max(var_max[1, i, j, element], bar_state_rho)
        # + eta direction
        bar_state_rho = bar_states2[1, i, j+1, element]
        var_min[1, i, j, element] = min(var_min[1, i, j, element], bar_state_rho)
        var_max[1, i, j, element] = max(var_max[1, i, j, element], bar_state_rho)
      end
    end #indicator.DensityLimiter

    if indicator.SequentialLimiter
      for j in eachnode(dg), i in eachnode(dg)
        # Previous solution
        for v in 2:nvariables(equations)
          phi = u[v, i, j, element] / u[1, i, j, element]
          var_min[v, i, j, element] = min(var_min[v, i, j, element], phi)
          var_max[v, i, j, element] = max(var_max[v, i, j, element], phi)
        end
        # - xi direction
        bar_state_rho = bar_states1[1, i, j, element]
        for v in 2:nvariables(equations)
          bar_state_phi = bar_states1[v, i, j, element] / bar_state_rho
          var_min[v, i, j, element] = min(var_min[v, i, j, element], bar_state_phi)
          var_max[v, i, j, element] = max(var_max[v, i, j, element], bar_state_phi)
        end
        # + xi direction
        bar_state_rho = bar_states1[1, i+1, j, element]
        for v in 2:nvariables(equations)
          bar_state_phi = bar_states1[v, i+1, j, element] / bar_state_rho
          var_min[v, i, j, element] = min(var_min[v, i, j, element], bar_state_phi)
          var_max[v, i, j, element] = max(var_max[v, i, j, element], bar_state_phi)
        end
        # - eta direction
        bar_state_rho = bar_states2[1, i, j, element]
        for v in 2:nvariables(equations)
          bar_state_phi = bar_states2[v, i, j, element] / bar_state_rho
          var_min[v, i, j, element] = min(var_min[v, i, j, element], bar_state_phi)
          var_max[v, i, j, element] = max(var_max[v, i, j, element], bar_state_phi)
        end
        # + eta direction
        bar_state_rho = bar_states2[1, i, j+1, element]
        for v in 2:nvariables(equations)
          bar_state_phi = bar_states2[v, i, j+1, element] / bar_state_rho
          var_min[v, i, j, element] = min(var_min[v, i, j, element], bar_state_phi)
          var_max[v, i, j, element] = max(var_max[v, i, j, element], bar_state_phi)
        end
      end
    elseif indicator.ConservativeLimiter
      for j in eachnode(dg), i in eachnode(dg)
        # Previous solution
        for v in 2:nvariables(equations)
          var_min[v, i, j, element] = min(var_min[v, i, j, element], u[v, i, j, element])
          var_max[v, i, j, element] = max(var_max[v, i, j, element], u[v, i, j, element])
        end
        # - xi direction
        for v in 2:nvariables(equations)
          bar_state_rho = bar_states1[v, i, j, element]
          var_min[v, i, j, element] = min(var_min[v, i, j, element], bar_state_rho)
          var_max[v, i, j, element] = max(var_max[v, i, j, element], bar_state_rho)
        end
        # + xi direction
        for v in 2:nvariables(equations)
          bar_state_rho = bar_states1[v, i+1, j, element]
          var_min[v, i, j, element] = min(var_min[v, i, j, element], bar_state_rho)
          var_max[v, i, j, element] = max(var_max[v, i, j, element], bar_state_rho)
        end
        # - eta direction
        for v in 2:nvariables(equations)
          bar_state_rho = bar_states2[v, i, j, element]
          var_min[v, i, j, element] = min(var_min[v, i, j, element], bar_state_rho)
          var_max[v, i, j, element] = max(var_max[v, i, j, element], bar_state_rho)
        end
        # + eta direction
        for v in 2:nvariables(equations)
          bar_state_rho = bar_states2[v, i, j+1, element]
          var_min[v, i, j, element] = min(var_min[v, i, j, element], bar_state_rho)
          var_max[v, i, j, element] = max(var_max[v, i, j, element], bar_state_rho)
        end
      end
    end
  end

  return nothing
end

@inline function calcflux_antidiffusive_limited!(u, mesh, nonconservative_terms, equations, indicator, dg, element, cache,
                                                 fstar1, fstar2)
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D
  @unpack var_min, var_max = indicator.cache.ContainerShockCapturingIndicator
  @unpack bar_states1, bar_states2, lambda1, lambda2 = indicator.cache.ContainerBarStates

  # The antidiffuse flux can have very small absolute values. This can lead to values of f_min which are zero up to machine accuracy.
  # To avoid further calculations with these values, we replace them by 0.
  # It can also happen that the limited flux changes its sign (for instance to -1e-13).
  # This does not really make sense in theory and causes problems for the visualization.
  # Therefore we make sure that the flux keeps its sign during limiting.

  if indicator.Plotting
    # TODO: Allocations!!!
    P = zeros(eltype(u), 4, nnodes(dg), nnodes(dg))
    Q = zeros(eltype(u), 4, nnodes(dg), nnodes(dg))

    @unpack alpha_mean, alpha_mean_pressure, alpha_mean_entropy = indicator.cache.ContainerShockCapturingIndicator
    for j in eachnode(dg), i in eachnode(dg)
      alpha_mean[:, i, j, element] .= zero(eltype(alpha_mean))
      alpha_mean_pressure[i, j, element] = zero(eltype(alpha_mean_pressure))
      alpha_mean_entropy[i, j, element] = zero(eltype(alpha_mean_entropy))
    end
  end

  # Density limiter
  if indicator.DensityLimiter
    for j in eachnode(dg), i in 2:nnodes(dg)
      lambda = lambda1[i, j, element]
      bar_state_rho = bar_states1[1, i, j, element]

      # Limit density
      if antidiffusive_flux1[1, i, j, element] > 0
        f_max = lambda * min(var_max[1, i-1, j, element] - bar_state_rho,
                            bar_state_rho - var_min[1, i, j, element])
        f_max = isapprox(f_max, 0.0, atol=eps()) ? 0.0 : f_max
        flux_limited = min(antidiffusive_flux1[1, i, j, element], max(f_max, 0.0))
      else
        f_min = lambda * max(var_min[1, i-1, j, element] - bar_state_rho,
                            bar_state_rho - var_max[1, i, j, element])
        f_min = isapprox(f_min, 0.0, atol=eps()) ? 0.0 : f_min
        flux_limited = max(antidiffusive_flux1[1, i, j, element], min(f_min, 0.0))
      end

      if indicator.Plotting || indicator.DensityAlphaForAll
        if isapprox(antidiffusive_flux1[1, i, j, element], 0.0, atol=eps())
          coefficient = 1.0 # flux_limited is zero as well
        else
          coefficient = min(1, (flux_limited + sign(flux_limited) * eps()) / (antidiffusive_flux1[1, i, j, element] + sign(flux_limited) * eps()))
        end

        if indicator.Plotting
          # left node
          aux = abs(lambda * (bar_state_rho - u[1, i-1, j, element]))
          P[1, i-1, j] += aux + abs(flux_limited)
          Q[1, i-1, j] += aux + abs(antidiffusive_flux1[1, i, j, element])
          # right node
          aux = abs(lambda * (bar_state_rho - u[1, i, j, element]))
          P[1, i, j] += aux + abs(flux_limited)
          Q[1, i, j] += aux + abs(antidiffusive_flux1[1, i, j, element])

          @unpack alpha, alpha_mean = indicator.cache.ContainerShockCapturingIndicator
          alpha[1, i-1, j, element] = min(alpha[1, i-1, j, element], coefficient)
          alpha[1, i,   j, element] = min(alpha[1, i,   j, element], coefficient)
          alpha_mean[1, i-1, j, element] += coefficient
          alpha_mean[1, i  , j, element] += coefficient
        end
      end
      antidiffusive_flux1[1, i, j, element] = flux_limited

      #Limit all quantities with the same alpha
      if indicator.DensityAlphaForAll
        for v in 2:nvariables(equations)
          antidiffusive_flux1[v, i, j, element] = coefficient * antidiffusive_flux1[v, i, j, element]
        end
      end
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
      lambda = lambda2[i, j, element]
      bar_state_rho = bar_states2[1, i, j, element]

      # Limit density
      if antidiffusive_flux2[1, i, j, element] > 0
        f_max = lambda * min(var_max[1, i, j-1, element] - bar_state_rho,
                            bar_state_rho - var_min[1, i, j, element])
        f_max = isapprox(f_max, 0.0, atol=eps()) ? 0.0 : f_max
        flux_limited = min(antidiffusive_flux2[1, i, j, element], max(f_max, 0.0))
      else
        f_min = lambda * max(var_min[1, i, j-1, element] - bar_state_rho,
                            bar_state_rho - var_max[1, i, j, element])
        f_min = isapprox(f_min, 0.0, atol=eps()) ? 0.0 : f_min
        flux_limited = max(antidiffusive_flux2[1, i, j, element], min(f_min, 0.0))
      end

      if indicator.Plotting || indicator.DensityAlphaForAll
        if isapprox(antidiffusive_flux2[1, i, j, element], 0.0, atol=eps())
          coefficient = 1.0 # flux_limited is zero as well
        else
          coefficient = min(1, (flux_limited + sign(flux_limited) * eps()) / (antidiffusive_flux2[1, i, j, element] + sign(flux_limited) * eps()))
        end

        if indicator.Plotting
          # left node
          aux = abs(lambda * (bar_state_rho - u[1, i, j-1, element]))
          P[1, i, j-1] += aux + abs(flux_limited)
          Q[1, i, j-1] += aux + abs(antidiffusive_flux2[1, i, j, element])
          # right node
          aux = abs(lambda * (bar_state_rho - u[1, i, j, element]))
          P[1, i, j] += aux + abs(flux_limited)
          Q[1, i, j] += aux + abs(antidiffusive_flux2[1, i, j, element])

          @unpack alpha, alpha_mean = indicator.cache.ContainerShockCapturingIndicator
          alpha[1, i, j-1, element] = min(alpha[1, i, j-1, element], coefficient)
          alpha[1, i,   j, element] = min(alpha[1, i,   j, element], coefficient)
          alpha_mean[1, i, j-1, element] += coefficient
          alpha_mean[1, i, j,   element] += coefficient
        end
      end
      antidiffusive_flux2[1, i, j, element] = flux_limited

      #Limit all quantities with the same alpha
      if indicator.DensityAlphaForAll
        for v in 2:nvariables(equations)
          antidiffusive_flux2[v, i, j, element] = coefficient * antidiffusive_flux2[v, i, j, element]
        end
      end
    end
  end # if indicator.DensityLimiter

  # Sequential limiter
  if indicator.SequentialLimiter
    for j in eachnode(dg), i in 2:nnodes(dg)
      lambda = lambda1[i, j, element]
      bar_state_rho = bar_states1[1, i, j, element]

      # Limit velocity and total energy
      rho_limited_iim1 = lambda * bar_state_rho - antidiffusive_flux1[1, i, j, element]
      rho_limited_im1i = lambda * bar_state_rho + antidiffusive_flux1[1, i, j, element]
      for v in 2:nvariables(equations)
        bar_state_phi = bar_states1[v, i, j, element]

        phi = bar_state_phi / bar_state_rho

        g = antidiffusive_flux1[v, i, j, element] + (lambda * bar_state_phi - rho_limited_im1i * phi)

        if g > 0
          g_max = min(rho_limited_im1i * (var_max[v, i-1, j, element] - phi),
                      rho_limited_iim1 * (phi - var_min[v, i, j, element]))
          g_max = isapprox(g_max, 0.0, atol=eps()) ? 0.0 : g_max
          g_limited = min(g, max(g_max, 0.0))
        else
          g_min = max(rho_limited_im1i * (var_min[v, i-1, j, element] - phi),
                      rho_limited_iim1 * (phi - var_max[v, i, j, element]))
          g_min = isapprox(g_min, 0.0, atol=eps()) ? 0.0 : g_min
          g_limited = max(g, min(g_min, 0.0))
        end
        if indicator.Plotting
          # left node
          aux = abs(lambda * (bar_state_phi - u[v, i-1, j, element]))
          P[v, i-1, j] += aux + abs(g_limited)
          Q[v, i-1, j] += aux + abs(g)
          # right node
          aux = abs(lambda * (bar_state_phi - u[v, i, j, element]))
          P[v, i, j] += aux + abs(g_limited)
          Q[v, i, j] += aux + abs(g)

          if isapprox(g, 0.0, atol=eps())
            coefficient = 1.0 # g_limited is zero as well
          else
            coefficient = min(1, (g_limited + sign(g_limited) * eps()) / (g + sign(g_limited) * eps()))
          end
          @unpack alpha, alpha_mean = indicator.cache.ContainerShockCapturingIndicator
          alpha[v, i-1, j, element] = min(alpha[v, i-1, j, element], coefficient)
          alpha[v, i,   j, element] = min(alpha[v, i,   j, element], coefficient)
          alpha_mean[v, i-1, j, element] += coefficient
          alpha_mean[v, i  , j, element] += coefficient
        end
        antidiffusive_flux1[v, i, j, element] = (rho_limited_im1i * phi - lambda * bar_state_phi) + g_limited
      end
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
      lambda = lambda2[i, j, element]
      bar_state_rho = bar_states2[1, i, j, element]

      # Limit velocity and total energy
      rho_limited_jjm1 = lambda * bar_state_rho - antidiffusive_flux2[1, i, j, element]
      rho_limited_jm1j = lambda * bar_state_rho + antidiffusive_flux2[1, i, j, element]
      for v in 2:nvariables(equations)
        bar_state_phi = bar_states2[v, i, j, element]

        phi = bar_state_phi / bar_state_rho

        g = antidiffusive_flux2[v, i, j, element] + (lambda * bar_state_phi - rho_limited_jm1j * phi)

        if g > 0
          g_max = min(rho_limited_jm1j * (var_max[v, i, j-1, element] - phi),
                      rho_limited_jjm1 * (phi - var_min[v, i, j, element]))
          g_max = isapprox(g_max, 0.0, atol=eps()) ? 0.0 : g_max
          g_limited = min(g, max(g_max, 0.0))
        else
          g_min = max(rho_limited_jm1j * (var_min[v, i, j-1, element] - phi),
                      rho_limited_jjm1 * (phi - var_max[v, i, j, element]))
          g_min = isapprox(g_min, 0.0, atol=eps()) ? 0.0 : g_min
          g_limited = max(g, min(g_min, 0.0))
        end
        if indicator.Plotting
          # left node
          aux = abs(lambda * (bar_state_phi - u[v, i, j-1, element]))
          P[v, i, j-1] += aux + abs(g_limited)
          Q[v, i, j-1] += aux + abs(g)
          # right node
          aux = abs(lambda * (bar_state_phi - u[v, i, j, element]))
          P[v, i, j] += aux + abs(g_limited)
          Q[v, i, j] += aux + abs(g)

          if isapprox(g, 0.0, atol=eps())
            coefficient = 1.0 # g_limited is zero as well
          else
            coefficient = min(1, (g_limited + sign(g_limited) * eps()) / (g + sign(g_limited) * eps()))
          end
          @unpack alpha, alpha_mean = indicator.cache.ContainerShockCapturingIndicator
          alpha[v, i, j-1, element] = min(alpha[v, i, j-1, element], coefficient)
          alpha[v, i,   j, element] = min(alpha[v, i,   j, element], coefficient)
          alpha_mean[v, i, j-1, element] += coefficient
          alpha_mean[v, i, j,   element] += coefficient
        end

        antidiffusive_flux2[v, i, j, element] = (rho_limited_jm1j * phi - lambda * bar_state_phi) + g_limited
      end
    end
  # Conservative limiter
  elseif indicator.ConservativeLimiter
    for j in eachnode(dg), i in 2:nnodes(dg)
      lambda = lambda1[i, j, element]
      for v in 2:nvariables(equations)
        bar_state_phi = bar_states1[v, i, j, element]
        # Limit density
        if antidiffusive_flux1[v, i, j, element] > 0
          f_max = lambda * min(var_max[v, i-1, j, element] - bar_state_phi,
                               bar_state_phi - var_min[v, i, j, element])
          f_max = isapprox(f_max, 0.0, atol=eps()) ? 0.0 : f_max
          flux_limited = min(antidiffusive_flux1[v, i, j, element], max(f_max, 0.0))
        else
          f_min = lambda * max(var_min[v, i-1, j, element] - bar_state_phi,
                               bar_state_phi - var_max[v, i, j, element])
          f_min = isapprox(f_min, 0.0, atol=eps()) ? 0.0 : f_min
          flux_limited = max(antidiffusive_flux1[v, i, j, element], min(f_min, 0.0))
        end

        if indicator.Plotting
          # left node
          aux = abs(lambda * (bar_state_phi - u[v, i-1, j, element]))
          P[v, i-1, j] += aux + abs(flux_limited)
          Q[v, i-1, j] += aux + abs(antidiffusive_flux1[v, i, j, element])
          # right node
          aux = abs(lambda * (bar_state_phi - u[v, i, j, element]))
          P[v, i, j] += aux + abs(flux_limited)
          Q[v, i, j] += aux + abs(antidiffusive_flux1[v, i, j, element])

          if isapprox(antidiffusive_flux1[v, i, j, element], 0.0, atol=eps())
            coefficient = 1.0 # flux_limited is zero as well
          else
            coefficient = min(1, (flux_limited + sign(flux_limited) * eps()) / (antidiffusive_flux1[v, i, j, element] + sign(flux_limited) * eps()))
          end
          @unpack alpha, alpha_mean = indicator.cache.ContainerShockCapturingIndicator
          alpha[v, i-1, j, element] = min(alpha[v, i-1, j, element], coefficient)
          alpha[v, i,   j, element] = min(alpha[v, i,   j, element], coefficient)
          alpha_mean[v, i-1, j, element] += coefficient
          alpha_mean[v, i,   j, element] += coefficient
        end
        antidiffusive_flux1[v, i, j, element] = flux_limited
      end
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
      lambda = lambda2[i, j, element]
      for v in 2:nvariables(equations)
        bar_state_phi = bar_states2[v, i, j, element]
        # Limit density
        if antidiffusive_flux2[v, i, j, element] > 0
          f_max = lambda * min(var_max[v, i, j-1, element] - bar_state_phi,
                               bar_state_phi - var_min[v, i, j, element])
          f_max = isapprox(f_max, 0.0, atol=eps()) ? 0.0 : f_max
          flux_limited = min(antidiffusive_flux2[v, i, j, element], max(f_max, 0.0))
        else
          f_min = lambda * max(var_min[v, i, j-1, element] - bar_state_phi,
                               bar_state_phi - var_max[v, i, j, element])
          f_min = isapprox(f_min, 0.0, atol=eps()) ? 0.0 : f_min
          flux_limited = max(antidiffusive_flux2[v, i, j, element], min(f_min, 0.0))
        end

        if indicator.Plotting
          # left node
          aux = abs(lambda * (bar_state_phi - u[v, i, j-1, element]))
          P[v, i, j-1] += aux + abs(flux_limited)
          Q[v, i, j-1] += aux + abs(antidiffusive_flux2[v, i, j, element])
          # right node
          aux = abs(lambda * (bar_state_phi - u[v, i, j, element]))
          P[v, i, j] += aux + abs(flux_limited)
          Q[v, i, j] += aux + abs(antidiffusive_flux2[v, i, j, element])

          if isapprox(antidiffusive_flux2[v, i, j, element], 0.0, atol=eps())
            coefficient = 1.0 # flux_limited is zero as well
          else
            coefficient = min(1, (flux_limited + sign(flux_limited) * eps()) / (antidiffusive_flux2[v, i, j, element] + sign(flux_limited) * eps()))
          end
          @unpack alpha, alpha_mean = indicator.cache.ContainerShockCapturingIndicator
          alpha[v, i, j-1, element] = min(alpha[v, i, j-1, element], coefficient)
          alpha[v, i,   j, element] = min(alpha[v, i,   j, element], coefficient)
          alpha_mean[v, i, j-1, element] += coefficient
          alpha_mean[v, i, j,   element] += coefficient
        end
        antidiffusive_flux2[v, i, j, element] = flux_limited
      end
    end
  end # indicator.SequentialLimiter and indicator.ConservativeLimiter

  # Compute "effective" alpha using P and Q
  if indicator.Plotting
    @unpack alpha_eff = indicator.cache.ContainerShockCapturingIndicator
    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        alpha_eff[v, i, j, element] = P[v, i, j] / (Q[v, i, j] + eps())
      end
    end
  end

  # Density positivity limiter
  if indicator.DensityPositivityLimiter
    beta = indicator.DensityPositivityCorrelationFactor
    for j in eachnode(dg), i in 2:nnodes(dg)
      lambda = lambda1[i, j, element]
      bar_state_rho = bar_states1[1, i, j, element]
      # Limit density
      if antidiffusive_flux1[1, i, j, element] > 0
        f_max = (1 - beta) * lambda * bar_state_rho
        f_max = isapprox(f_max, 0.0, atol=eps()) ? 0.0 : f_max
        flux_limited = min(antidiffusive_flux1[1, i, j, element], max(f_max, 0.0))
      else
        f_min = - (1 - beta) * lambda * bar_state_rho
        f_min = isapprox(f_min, 0.0, atol=eps()) ? 0.0 : f_min
        flux_limited = max(antidiffusive_flux1[1, i, j, element], min(f_min, 0.0))
      end

      if indicator.Plotting || indicator.DensityAlphaForAll
        if isapprox(antidiffusive_flux1[1, i, j, element], 0.0, atol=eps())
          coefficient = 1.0  # flux_limited is zero as well
        else
          coefficient = flux_limited / antidiffusive_flux1[1, i, j, element]
        end

        if indicator.Plotting
          @unpack alpha, alpha_mean = indicator.cache.ContainerShockCapturingIndicator
          alpha[1, i-1, j, element] = min(alpha[1, i-1, j, element], coefficient)
          alpha[1, i,   j, element] = min(alpha[1, i,   j, element], coefficient)
          if !indicator.DensityLimiter
            alpha_mean[1, i-1, j, element] += coefficient
            alpha_mean[1, i,   j, element] += coefficient
          end
        end
      end
      antidiffusive_flux1[1, i, j, element] = flux_limited

      #Limit all quantities with the same alpha
      if indicator.DensityAlphaForAll
        for v in 2:nvariables(equations)
          antidiffusive_flux1[v, i, j, element] = coefficient * antidiffusive_flux1[v, i, j, element]
        end
      end
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
      lambda = lambda2[i, j, element]
      bar_state_rho = bar_states2[1, i, j, element]
      # Limit density
      if antidiffusive_flux2[1, i, j, element] > 0
        f_max = (1 - beta) * lambda * bar_state_rho
        f_max = isapprox(f_max, 0.0, atol=eps()) ? 0.0 : f_max
        flux_limited = min(antidiffusive_flux2[1, i, j, element], max(f_max, 0.0))
      else
        f_min = - (1 - beta) * lambda * bar_state_rho
        f_min = isapprox(f_min, 0.0, atol=eps()) ? 0.0 : f_min
        flux_limited = max(antidiffusive_flux2[1, i, j, element], min(f_min, 0.0))
      end

      if indicator.Plotting || indicator.DensityAlphaForAll
        if isapprox(antidiffusive_flux2[1, i, j, element], 0.0, atol=eps())
          coefficient = 1.0  # flux_limited is zero as well
        else
          coefficient = flux_limited / antidiffusive_flux2[1, i, j, element]
        end

        if indicator.Plotting
          @unpack alpha, alpha_mean = indicator.cache.ContainerShockCapturingIndicator
          alpha[1, i, j-1, element] = min(alpha[1, i, j-1, element], coefficient)
          alpha[1, i,   j, element] = min(alpha[1, i,   j, element], coefficient)
          if !indicator.DensityLimiter
            alpha_mean[1, i, j-1, element] += coefficient
            alpha_mean[1, i, j,   element] += coefficient
          end
        end
      end
      antidiffusive_flux2[1, i, j, element] = flux_limited

      #Limit all quantities with the same alpha
      if indicator.DensityAlphaForAll
        for v in 2:nvariables(equations)
          antidiffusive_flux2[v, i, j, element] = coefficient * antidiffusive_flux2[v, i, j, element]
        end
      end
    end
  end #if indicator.DensityPositivityLimiter

  # Divide alpha_mean by number of additions
  if indicator.Plotting
    @unpack alpha_mean = indicator.cache.ContainerShockCapturingIndicator
    # Interfaces contribute with 1.0
    if indicator.DensityLimiter || indicator.DensityPositivityLimiter
      for i in eachnode(dg)
        alpha_mean[1, i, 1, element] += 1.0
        alpha_mean[1, i, nnodes(dg), element] += 1.0
        alpha_mean[1, 1, i, element] += 1.0
        alpha_mean[1, nnodes(dg), i, element] += 1.0
      end
      for j in eachnode(dg), i in eachnode(dg)
        alpha_mean[1, i, j, element] /= 4
      end
    end
    if indicator.SequentialLimiter || indicator.ConservativeLimiter
      for v in 2:nvariables(equations)
        for i in eachnode(dg)
          alpha_mean[v, i, 1, element] += 1.0
          alpha_mean[v, i, nnodes(dg), element] += 1.0
          alpha_mean[v, 1, i, element] += 1.0
          alpha_mean[v, nnodes(dg), i, element] += 1.0
        end
        for j in eachnode(dg), i in eachnode(dg)
          alpha_mean[v, i, j, element] /= 4
        end
      end
    end
  end

  # Limit pressure à la Kuzmin
  if indicator.PressurePositivityLimiterKuzmin
    @unpack alpha_pressure, alpha_mean_pressure = indicator.cache.ContainerShockCapturingIndicator
    for j in eachnode(dg), i in 2:nnodes(dg)
      bar_state_velocity = bar_states1[2, i, j, element]^2 + bar_states1[3, i, j, element]^2
      flux_velocity = antidiffusive_flux1[2, i, j, element]^2 + antidiffusive_flux1[3, i, j, element]^2

      Q = lambda1[i, j, element]^2 * (bar_states1[1, i, j, element] * bar_states1[4, i, j, element] -
                                      0.5 * bar_state_velocity)

      if indicator.PressurePositivityLimiterKuzminExact
        # exact calculation of max(R_ij, R_ji)
        R_max = lambda1[i, j, element] *
                  abs(bar_states1[2, i, j, element] * antidiffusive_flux1[2, i, j, element] +
                      bar_states1[3, i, j, element] * antidiffusive_flux1[3, i, j, element] -
                      bar_states1[1, i, j, element] * antidiffusive_flux1[4, i, j, element] -
                      bar_states1[4, i, j, element] * antidiffusive_flux1[1, i, j, element])
        R_max += max(0, 0.5 * flux_velocity -
                        antidiffusive_flux1[4, i, j, element] * antidiffusive_flux1[1, i, j, element])
      else
        # approximation R_max
        R_max = lambda1[i, j, element] *
                  (sqrt(bar_state_velocity * flux_velocity) +
                  abs(bar_states1[1, i, j, element] * antidiffusive_flux1[4, i, j, element]) +
                  abs(bar_states1[4, i, j, element] * antidiffusive_flux1[1, i, j, element]))
        R_max += max(0, 0.5 * flux_velocity -
                        antidiffusive_flux1[4, i, j, element] * antidiffusive_flux1[1, i, j, element])
      end
      alpha = 1 # Initialize alpha for plotting
      if R_max > Q
        alpha = Q / R_max
        for v in eachvariable(equations)
          antidiffusive_flux1[v, i, j, element] *= alpha
        end
      end
      if indicator.Plotting
        alpha_pressure[i-1, j, element] = min(alpha_pressure[i-1, j, element], alpha)
        alpha_pressure[i,   j, element] = min(alpha_pressure[i,   j, element], alpha)
        alpha_mean_pressure[i-1, j, element] += alpha
        alpha_mean_pressure[i,   j, element] += alpha
      end
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
      bar_state_velocity = bar_states2[2, i, j, element]^2 + bar_states2[3, i, j, element]^2
      flux_velocity = antidiffusive_flux2[2, i, j, element]^2 + antidiffusive_flux2[3, i, j, element]^2

      Q = lambda2[i, j, element]^2 * (bar_states2[1, i, j, element] * bar_states2[4, i, j, element] -
                                      0.5 * bar_state_velocity)

      if indicator.PressurePositivityLimiterKuzminExact
        # exact calculation of max(R_ij, R_ji)
        R_max = lambda2[i, j, element] *
                  abs(bar_states2[2, i, j, element] * antidiffusive_flux2[2, i, j, element] +
                      bar_states2[3, i, j, element] * antidiffusive_flux2[3, i, j, element] -
                      bar_states2[1, i, j, element] * antidiffusive_flux2[4, i, j, element] -
                      bar_states2[4, i, j, element] * antidiffusive_flux2[1, i, j, element])
        R_max += max(0, 0.5 * flux_velocity -
                        antidiffusive_flux2[4, i, j, element] * antidiffusive_flux2[1, i, j, element])
      else
        # approximation R_max
        R_max = lambda2[i, j, element] *
                  (sqrt(bar_state_velocity * flux_velocity) +
                  abs(bar_states2[1, i, j, element] * antidiffusive_flux2[4, i, j, element]) +
                  abs(bar_states2[4, i, j, element] * antidiffusive_flux2[1, i, j, element]))
        R_max += max(0, 0.5 * flux_velocity -
                        antidiffusive_flux2[4, i, j, element] * antidiffusive_flux2[1, i, j, element])
      end
      alpha = 1 # Initialize alpha for plotting
      if R_max > Q
        alpha = Q / R_max
        for v in eachvariable(equations)
          antidiffusive_flux2[v, i, j, element] *= alpha
        end
      end
      if indicator.Plotting
        alpha_pressure[i, j-1, element] = min(alpha_pressure[i, j-1, element], alpha)
        alpha_pressure[i,   j, element] = min(alpha_pressure[i,   j, element], alpha)
        alpha_mean_pressure[i, j-1, element] += alpha
        alpha_mean_pressure[i,   j, element] += alpha
      end
    end
    if indicator.Plotting
      @unpack alpha_mean_pressure = indicator.cache.ContainerShockCapturingIndicator
      # Interfaces contribute with 1.0
      for i in eachnode(dg)
        alpha_mean_pressure[i, 1, element] += 1.0
        alpha_mean_pressure[i, nnodes(dg), element] += 1.0
        alpha_mean_pressure[1, i, element] += 1.0
        alpha_mean_pressure[nnodes(dg), i, element] += 1.0
      end
      for j in eachnode(dg), i in eachnode(dg)
        alpha_mean_pressure[i, j, element] /= 4
      end
    end
  end

  # Limit entropy
  # TODO: This is a very inefficient function. We compute the entropy four times at each node.
  # TODO: For now, this only works for Cartesian meshes.
  if indicator.SemiDiscEntropyLimiter
    for j in eachnode(dg), i in 2:nnodes(dg)
      antidiffusive_flux_local = get_node_vars(antidiffusive_flux1, equations, dg, i, j, element)
      u_local    = get_node_vars(u, equations, dg, i,   j, element)
      u_local_m1 = get_node_vars(u, equations, dg, i-1, j, element)

      # Using mathematic entropy
      v_local    = cons2entropy(u_local,    equations)
      v_local_m1 = cons2entropy(u_local_m1, equations)

      q_local    = u_local[2]    / u_local[1]    * entropy(u_local, equations)
      q_local_m1 = u_local_m1[2] / u_local_m1[1] * entropy(u_local_m1, equations)

      f_local    = flux(u_local,    1, equations)
      f_local_m1 = flux(u_local_m1, 1, equations)

      psi_local    = dot(v_local, f_local)       - q_local
      psi_local_m1 = dot(v_local_m1, f_local_m1) - q_local_m1

      delta_v = v_local - v_local_m1
      delta_psi = psi_local - psi_local_m1

      entProd_FV = dot(delta_v, fstar1[:, i, j]) - delta_psi
      delta_entProd = dot(delta_v, antidiffusive_flux_local)

      alpha = 1 # Initialize alpha for plotting
      if (entProd_FV + delta_entProd > 0.0) && (delta_entProd != 0.0)
        alpha = min(1.0, (abs(entProd_FV)+eps()) / (abs(delta_entProd)+eps()))
        for v in eachvariable(equations)
          antidiffusive_flux1[v, i, j, element] = alpha * antidiffusive_flux1[v, i, j, element]
        end
      end
      if indicator.Plotting
        @unpack alpha_entropy, alpha_mean_entropy = indicator.cache.ContainerShockCapturingIndicator
        alpha_entropy[i-1, j, element] = min(alpha_entropy[i-1, j, element], alpha)
        alpha_entropy[i,   j, element] = min(alpha_entropy[i,   j, element], alpha)
        alpha_mean_entropy[i-1, j, element] += alpha
        alpha_mean_entropy[i,   j, element] += alpha
      end
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
      antidiffusive_flux_local = get_node_vars(antidiffusive_flux2, equations, dg, i, j, element)
      u_local    = get_node_vars(u, equations, dg, i,   j, element)
      u_local_m1 = get_node_vars(u, equations, dg, i, j-1, element)

      # Using mathematic entropy
      v_local    = cons2entropy(u_local,    equations)
      v_local_m1 = cons2entropy(u_local_m1, equations)

      q_local    = u_local[3]    / u_local[1]    * entropy(u_local, equations)
      q_local_m1 = u_local_m1[3] / u_local_m1[1] * entropy(u_local_m1, equations)

      f_local    = flux(u_local,    2, equations)
      f_local_m1 = flux(u_local_m1, 2, equations)

      psi_local    = dot(v_local, f_local)       - q_local
      psi_local_m1 = dot(v_local_m1, f_local_m1) - q_local_m1

      delta_v = v_local - v_local_m1
      delta_psi = psi_local - psi_local_m1

      entProd_FV = dot(delta_v, fstar2[:, i, j]) - delta_psi
      delta_entProd = dot(delta_v, antidiffusive_flux_local)

      alpha = 1 # Initialize alpha for plotting
      if (entProd_FV + delta_entProd > 0.0) && (delta_entProd != 0.0)
        alpha = min(1.0, (abs(entProd_FV)+eps()) / (abs(delta_entProd)+eps()))
        for v in eachvariable(equations)
          antidiffusive_flux2[v, i, j, element] = alpha * antidiffusive_flux2[v, i, j, element]
        end
      end
      if indicator.Plotting
        @unpack alpha_entropy, alpha_mean_entropy = indicator.cache.ContainerShockCapturingIndicator
        alpha_entropy[i, j-1, element] = min(alpha_entropy[i, j-1, element], alpha)
        alpha_entropy[i,   j, element] = min(alpha_entropy[i,   j, element], alpha)
        alpha_mean_entropy[i, j-1, element] += alpha
        alpha_mean_entropy[i,   j, element] += alpha
      end
    end
    if indicator.Plotting
      @unpack alpha_mean_entropy = indicator.cache.ContainerShockCapturingIndicator
      # Interfaces contribute with 1.0
      for i in eachnode(dg)
        alpha_mean_entropy[i, 1, element] += 1.0
        alpha_mean_entropy[i, nnodes(dg), element] += 1.0
        alpha_mean_entropy[1, i, element] += 1.0
        alpha_mean_entropy[nnodes(dg), i, element] += 1.0
      end
      for j in eachnode(dg), i in eachnode(dg)
        alpha_mean_entropy[i, j, element] /= 4
      end
    end
  end

  return nothing
end

@inline function get_boundary_outer_state(u_inner, cache, t, boundary_condition, orientation_or_normal, direction, equations, dg, indices...)
  if boundary_condition == boundary_condition_slip_wall #boundary_condition_reflecting_euler_wall
    if orientation_or_normal isa AbstractArray
      u_rotate = rotate_to_x(u_inner, orientation_or_normal, equations)

      return SVector(u_inner[1],
                     u_inner[2] - 2.0 * u_rotate[2],
                     u_inner[3] - 2.0 * u_rotate[3],
                     u_inner[4])
    else # orientation_or_normal isa Integer
      return SVector(u_inner[1], -u_inner[2], -u_inner[3], u_inner[4])
    end
  elseif boundary_condition == boundary_condition_mixed_dirichlet_wall
    x = get_node_coords(cache.elements.node_coordinates, equations, dg, indices...)
    if x[1] < 1 / 6 # BoundaryConditionCharacteristic
      u_outer = Trixi.characteristic_boundary_value_function(initial_condition_double_mach_reflection,
                                                             u_inner, orientation_or_normal, direction, x, t, equations)

      return u_outer
    else # x[1] >= 1 / 6 # boundary_condition_slip_wall
      if orientation_or_normal isa AbstractArray
        u_rotate = rotate_to_x(u_inner, orientation_or_normal, equations)

        return SVector(u_inner[1],
                       u_inner[2] - 2.0 * u_rotate[2],
                       u_inner[3] - 2.0 * u_rotate[3],
                       u_inner[4])
      else # orientation_or_normal isa Integer
        return SVector(u_inner[1], -u_inner[2], -u_inner[3], u_inner[4])
      end
    end
  end

  return u_inner
end

@inline function get_boundary_outer_state(u_inner, cache, t, boundary_condition::BoundaryConditionDirichlet, orientation_or_normal, direction, equations, dg, indices...)
  @unpack node_coordinates = cache.elements

  x = get_node_coords(node_coordinates, equations, dg, indices...)
  u_outer = boundary_condition.boundary_value_function(x, t, equations)

  return u_outer
end

@inline function get_boundary_outer_state(u_inner, cache, t, boundary_condition::BoundaryConditionCharacteristic, orientation_or_normal, direction, equations, dg, indices...)
  @unpack node_coordinates = cache.elements

  x = get_node_coords(node_coordinates, equations, dg, indices...)
  u_outer = boundary_condition.boundary_value_function(boundary_condition.outer_boundary_value_function, u_inner, orientation_or_normal, direction, x, t, equations)

  return u_outer
end


@inline function antidiffusive_stage!(u_ode, t, dt, semi, indicator::IndicatorIDP)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)

  @trixi_timeit timer() "alpha calculation" semi.solver.volume_integral.indicator(u, semi, solver, t, dt)

  perform_IDP_correction(u, dt, mesh, equations, solver, cache)

  return nothing
end

@inline function perform_IDP_correction(u, dt, mesh::TreeMesh2D, equations, dg, cache)
  @unpack inverse_weights = dg.basis
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D
  @unpack alpha1, alpha2 = dg.volume_integral.indicator.cache.ContainerShockCapturingIndicator
  if dg.volume_integral.indicator.indicator_smooth
    elements = cache.element_ids_dgfv
  else
    elements = eachelement(dg, cache)
  end

  # Loop over blended DG-FV elements
  @threaded for element in elements
    inverse_jacobian = -cache.elements.inverse_jacobian[element]

    for j in eachnode(dg), i in eachnode(dg)
      # Note: antidiffusive_flux1[v, i, xi, element] = antidiffusive_flux2[v, xi, i, element] = 0 for all i in 1:nnodes and xi in {1, nnodes+1}
      alpha_flux1     = (1.0 - alpha1[i,   j, element]) * get_node_vars(antidiffusive_flux1, equations, dg, i,   j, element)
      alpha_flux1_ip1 = (1.0 - alpha1[i+1, j, element]) * get_node_vars(antidiffusive_flux1, equations, dg, i+1, j, element)
      alpha_flux2     = (1.0 - alpha2[i,   j, element]) * get_node_vars(antidiffusive_flux2, equations, dg, i,   j, element)
      alpha_flux2_jp1 = (1.0 - alpha2[i, j+1, element]) * get_node_vars(antidiffusive_flux2, equations, dg, i, j+1, element)

      for v in eachvariable(equations)
        u[v, i, j, element] += dt * inverse_jacobian * (inverse_weights[i] * (alpha_flux1_ip1[v] - alpha_flux1[v]) +
                                                        inverse_weights[j] * (alpha_flux2_jp1[v] - alpha_flux2[v]) )
      end
    end
  end

  return nothing
end

@inline function antidiffusive_stage!(u_ode, t, dt, semi, indicator::IndicatorMCL)

  return nothing
end


# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache, u,
                             mesh::TreeMesh{2}, equations, surface_integral, dg::DG)
  @unpack interfaces = cache
  @unpack orientations = interfaces

  @threaded for interface in eachinterface(dg, cache)
    left_element  = interfaces.neighbor_ids[1, interface]
    right_element = interfaces.neighbor_ids[2, interface]

    if orientations[interface] == 1
      # interface in x-direction
      for j in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j, left_element]
        interfaces.u[2, v, j, interface] = u[v,          1, j, right_element]
      end
    else # if orientations[interface] == 2
      # interface in y-direction
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg), left_element]
        interfaces.u[2, v, i, interface] = u[v, i,          1, right_element]
      end
    end
  end

  return nothing
end

function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{2},
                              nonconservative_terms::False, equations,
                              surface_integral, dg::DG, cache)
  @unpack surface_flux = surface_integral
  @unpack u, neighbor_ids, orientations = cache.interfaces

  @threaded for interface in eachinterface(dg, cache)
    # Get neighboring elements
    left_id  = neighbor_ids[1, interface]
    right_id = neighbor_ids[2, interface]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    left_direction  = 2 * orientations[interface]
    right_direction = 2 * orientations[interface] - 1

    for i in eachnode(dg)
      # Call pointwise Riemann solver
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
      flux = surface_flux(u_ll, u_rr, orientations[interface], equations)

      # Copy flux to left and right element storage
      for v in eachvariable(equations)
        surface_flux_values[v, i, left_direction,  left_id]  = flux[v]
        surface_flux_values[v, i, right_direction, right_id] = flux[v]
      end
    end
  end

  return nothing
end

function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{2},
                              nonconservative_terms::True, equations,
                              surface_integral, dg::DG, cache)
  surface_flux, nonconservative_flux = surface_integral.surface_flux
  @unpack u, neighbor_ids, orientations = cache.interfaces

  @threaded for interface in eachinterface(dg, cache)
    # Get neighboring elements
    left_id  = neighbor_ids[1, interface]
    right_id = neighbor_ids[2, interface]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    left_direction  = 2 * orientations[interface]
    right_direction = 2 * orientations[interface] - 1

    for i in eachnode(dg)
      # Call pointwise Riemann solver
      orientation = orientations[interface]
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
      flux = surface_flux(u_ll, u_rr, orientation, equations)

      # Compute both nonconservative fluxes
      noncons_left  = nonconservative_flux(u_ll, u_rr, orientation, equations)
      noncons_right = nonconservative_flux(u_rr, u_ll, orientation, equations)

      # Copy flux to left and right element storage
      for v in eachvariable(equations)
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        surface_flux_values[v, i, left_direction,  left_id]  = flux[v] + 0.5 * noncons_left[v]
        surface_flux_values[v, i, right_direction, right_id] = flux[v] + 0.5 * noncons_right[v]
      end
    end
  end

  return nothing
end


function prolong2boundaries!(cache, u,
                             mesh::TreeMesh{2}, equations, surface_integral, dg::DG)
  @unpack boundaries = cache
  @unpack orientations, neighbor_sides = boundaries

  @threaded for boundary in eachboundary(dg, cache)
    element = boundaries.neighbor_ids[boundary]

    if orientations[boundary] == 1
      # boundary in x-direction
      if neighbor_sides[boundary] == 1
        # element in -x direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[1, v, l, boundary] = u[v, nnodes(dg), l, element]
        end
      else # Element in +x direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[2, v, l, boundary] = u[v, 1,          l, element]
        end
      end
    else # if orientations[boundary] == 2
      # boundary in y-direction
      if neighbor_sides[boundary] == 1
        # element in -y direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[1, v, l, boundary] = u[v, l, nnodes(dg), element]
        end
      else
        # element in +y direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[2, v, l, boundary] = u[v, l, 1,          element]
        end
      end
    end
  end

  return nothing
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::TreeMesh{2}, equations, surface_integral, dg::DG)
  @assert isempty(eachboundary(dg, cache))
end

function calc_boundary_flux!(cache, t, boundary_conditions::NamedTuple,
                             mesh::TreeMesh{2}, equations, surface_integral, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[1],
                                   have_nonconservative_terms(equations),
                                   equations, surface_integral, dg, cache,
                                   1, firsts[1], lasts[1])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[2],
                                   have_nonconservative_terms(equations),
                                   equations, surface_integral, dg, cache,
                                   2, firsts[2], lasts[2])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[3],
                                   have_nonconservative_terms(equations),
                                   equations, surface_integral, dg, cache,
                                   3, firsts[3], lasts[3])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[4],
                                   have_nonconservative_terms(equations),
                                   equations, surface_integral, dg, cache,
                                   4, firsts[4], lasts[4])
end

function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,4}, t,
                                          boundary_condition, nonconservative_terms::False, equations,
                                          surface_integral ,dg::DG, cache,
                                          direction, first_boundary, last_boundary)
  @unpack surface_flux = surface_integral
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

  @threaded for boundary in first_boundary:last_boundary
    # Get neighboring element
    neighbor = neighbor_ids[boundary]

    for i in eachnode(dg)
      # Get boundary flux
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, boundary)
      if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
        u_inner = u_ll
      else # Element is on the right, boundary on the left
        u_inner = u_rr
      end
      x = get_node_coords(node_coordinates, equations, dg, i, boundary)
      flux = boundary_condition(u_inner, orientations[boundary], direction, x, t, surface_flux,
                                equations)

      # Copy flux to left and right element storage
      for v in eachvariable(equations)
        surface_flux_values[v, i, direction, neighbor] = flux[v]
      end
    end
  end

  return nothing
end

function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,4}, t,
                                          boundary_condition, nonconservative_terms::True, equations,
                                          surface_integral, dg::DG, cache,
                                          direction, first_boundary, last_boundary)
  surface_flux, nonconservative_flux = surface_integral.surface_flux
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

  @threaded for boundary in first_boundary:last_boundary
  # Get neighboring element
    neighbor = neighbor_ids[boundary]

    for i in eachnode(dg)
      # Get boundary flux
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, boundary)
      if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
        u_inner = u_ll
      else # Element is on the right, boundary on the left
        u_inner = u_rr
      end
      x = get_node_coords(node_coordinates, equations, dg, i, boundary)
      flux = boundary_condition(u_inner, orientations[boundary], direction, x, t, surface_flux,
                                equations)
      noncons_flux = boundary_condition(u_inner, orientations[boundary], direction, x, t, nonconservative_flux,
                                        equations)

      # Copy flux to left and right element storage
      for v in eachvariable(equations)
        surface_flux_values[v, i, direction, neighbor] = flux[v] + 0.5 * noncons_flux[v]
      end
    end
  end

  return nothing
end


function prolong2mortars!(cache, u,
                          mesh::TreeMesh{2}, equations,
                          mortar_l2::LobattoLegendreMortarL2, surface_integral, dg::DGSEM)

  @threaded for mortar in eachmortar(dg, cache)

    large_element = cache.mortars.neighbor_ids[3, mortar]
    upper_element = cache.mortars.neighbor_ids[2, mortar]
    lower_element = cache.mortars.neighbor_ids[1, mortar]

    # Copy solution small to small
    if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        for l in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper[2, v, l, mortar] = u[v, 1, l, upper_element]
            cache.mortars.u_lower[2, v, l, mortar] = u[v, 1, l, lower_element]
          end
        end
      else
        # L2 mortars in y-direction
        for l in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper[2, v, l, mortar] = u[v, l, 1, upper_element]
            cache.mortars.u_lower[2, v, l, mortar] = u[v, l, 1, lower_element]
          end
        end
      end
    else # large_sides[mortar] == 2 -> small elements on left side
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        for l in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper[1, v, l, mortar] = u[v, nnodes(dg), l, upper_element]
            cache.mortars.u_lower[1, v, l, mortar] = u[v, nnodes(dg), l, lower_element]
          end
        end
      else
        # L2 mortars in y-direction
        for l in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper[1, v, l, mortar] = u[v, l, nnodes(dg), upper_element]
            cache.mortars.u_lower[1, v, l, mortar] = u[v, l, nnodes(dg), lower_element]
          end
        end
      end
    end

    # Interpolate large element face data to small interface locations
    if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
      leftright = 1
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        u_large = view(u, :, nnodes(dg), :, large_element)
        element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright, mortar, u_large)
      else
        # L2 mortars in y-direction
        u_large = view(u, :, :, nnodes(dg), large_element)
        element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright, mortar, u_large)
      end
    else # large_sides[mortar] == 2 -> large element on right side
      leftright = 2
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        u_large = view(u, :, 1, :, large_element)
        element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright, mortar, u_large)
      else
        # L2 mortars in y-direction
        u_large = view(u, :, :, 1, large_element)
        element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright, mortar, u_large)
      end
    end
  end

  return nothing
end

@inline function element_solutions_to_mortars!(mortars, mortar_l2::LobattoLegendreMortarL2, leftright, mortar,
                                               u_large::AbstractArray{<:Any,2})
  multiply_dimensionwise!(view(mortars.u_upper, leftright, :, :, mortar), mortar_l2.forward_upper, u_large)
  multiply_dimensionwise!(view(mortars.u_lower, leftright, :, :, mortar), mortar_l2.forward_lower, u_large)
  return nothing
end


function calc_mortar_flux!(surface_flux_values,
                           mesh::TreeMesh{2},
                           nonconservative_terms::False, equations,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DG, cache)
  @unpack surface_flux = surface_integral
  @unpack u_lower, u_upper, orientations = cache.mortars
  @unpack fstar_upper_threaded, fstar_lower_threaded = cache

  @threaded for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    orientation = orientations[mortar]
    calc_fstar!(fstar_upper, equations, surface_flux, dg, u_upper, mortar, orientation)
    calc_fstar!(fstar_lower, equations, surface_flux, dg, u_lower, mortar, orientation)

    mortar_fluxes_to_elements!(surface_flux_values,
                               mesh, equations, mortar_l2, dg, cache,
                               mortar, fstar_upper, fstar_lower)
  end

  return nothing
end

function calc_mortar_flux!(surface_flux_values,
                           mesh::TreeMesh{2},
                           nonconservative_terms::True, equations,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DG, cache)
  surface_flux, nonconservative_flux = surface_integral.surface_flux
  @unpack u_lower, u_upper, orientations, large_sides = cache.mortars
  @unpack fstar_upper_threaded, fstar_lower_threaded = cache

  @threaded for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    orientation = orientations[mortar]
    calc_fstar!(fstar_upper, equations, surface_flux, dg, u_upper, mortar, orientation)
    calc_fstar!(fstar_lower, equations, surface_flux, dg, u_lower, mortar, orientation)

    # Add nonconservative fluxes.
    # These need to be adapted on the geometry (left/right) since the order of
    # the arguments matters, based on the global SBP operator interpretation.
    # The same interpretation (global SBP operators coupled discontinuously via
    # central fluxes/SATs) explains why we need the factor 0.5.
    # Alternatively, you can also follow the argumentation of Bohm et al. 2018
    # ("nonconservative diamond flux")
    if large_sides[mortar] == 1 # -> small elements on right side
      for i in eachnode(dg)
        # Pull the left and right solutions
        u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, equations, dg, i, mortar)
        u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, equations, dg, i, mortar)
        # Call pointwise nonconservative term
        noncons_upper = nonconservative_flux(u_upper_ll, u_upper_rr, orientation, equations)
        noncons_lower = nonconservative_flux(u_lower_ll, u_lower_rr, orientation, equations)
        # Add to primary and secondary temporary storage
        multiply_add_to_node_vars!(fstar_upper, 0.5, noncons_upper, equations, dg, i)
        multiply_add_to_node_vars!(fstar_lower, 0.5, noncons_lower, equations, dg, i)
      end
    else # large_sides[mortar] == 2 -> small elements on the left
      for i in eachnode(dg)
        # Pull the left and right solutions
        u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, equations, dg, i, mortar)
        u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, equations, dg, i, mortar)
        # Call pointwise nonconservative term
        noncons_upper = nonconservative_flux(u_upper_rr, u_upper_ll, orientation, equations)
        noncons_lower = nonconservative_flux(u_lower_rr, u_lower_ll, orientation, equations)
        # Add to primary and secondary temporary storage
        multiply_add_to_node_vars!(fstar_upper, 0.5, noncons_upper, equations, dg, i)
        multiply_add_to_node_vars!(fstar_lower, 0.5, noncons_lower, equations, dg, i)
      end
    end

    mortar_fluxes_to_elements!(surface_flux_values,
                               mesh, equations, mortar_l2, dg, cache,
                               mortar, fstar_upper, fstar_lower)
  end

  return nothing
end


@inline function calc_fstar!(destination::AbstractArray{<:Any,2}, equations,
                             surface_flux, dg::DGSEM,
                             u_interfaces, interface, orientation)

  for i in eachnode(dg)
    # Call pointwise two-point numerical flux function
    u_ll, u_rr = get_surface_node_vars(u_interfaces, equations, dg, i, interface)
    flux = surface_flux(u_ll, u_rr, orientation, equations)

    # Copy flux to left and right element storage
    set_node_vars!(destination, flux, equations, dg, i)
  end

  return nothing
end

@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::TreeMesh{2}, equations,
                                            mortar_l2::LobattoLegendreMortarL2,
                                            dg::DGSEM, cache,
                                            mortar, fstar_upper, fstar_lower)
  large_element = cache.mortars.neighbor_ids[3, mortar]
  upper_element = cache.mortars.neighbor_ids[2, mortar]
  lower_element = cache.mortars.neighbor_ids[1, mortar]

  # Copy flux small to small
  if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 1
    else
      # L2 mortars in y-direction
      direction = 3
    end
  else # large_sides[mortar] == 2 -> small elements on left side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 2
    else
      # L2 mortars in y-direction
      direction = 4
    end
  end
  surface_flux_values[:, :, direction, upper_element] .= fstar_upper
  surface_flux_values[:, :, direction, lower_element] .= fstar_lower

  # Project small fluxes to large element
  if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 2
    else
      # L2 mortars in y-direction
      direction = 4
    end
  else # large_sides[mortar] == 2 -> large element on right side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 1
    else
      # L2 mortars in y-direction
      direction = 3
    end
  end

  # TODO: Taal performance
  # for v in eachvariable(equations)
  #   # The code below is semantically equivalent to
  #   # surface_flux_values[v, :, direction, large_element] .=
  #   #   (mortar_l2.reverse_upper * fstar_upper[v, :] + mortar_l2.reverse_lower * fstar_lower[v, :])
  #   # but faster and does not allocate.
  #   # Note that `true * some_float == some_float` in Julia, i.e. `true` acts as
  #   # a universal `one`. Hence, the second `mul!` means "add the matrix-vector
  #   # product to the current value of the destination".
  #   @views mul!(surface_flux_values[v, :, direction, large_element],
  #               mortar_l2.reverse_upper, fstar_upper[v, :])
  #   @views mul!(surface_flux_values[v, :, direction, large_element],
  #               mortar_l2.reverse_lower,  fstar_lower[v, :], true, true)
  # end
  # The code above could be replaced by the following code. However, the relative efficiency
  # depends on the types of fstar_upper/fstar_lower and dg.l2mortar_reverse_upper.
  # Using StaticArrays for both makes the code above faster for common test cases.
  multiply_dimensionwise!(
    view(surface_flux_values, :, :, direction, large_element), mortar_l2.reverse_upper, fstar_upper,
                                                               mortar_l2.reverse_lower, fstar_lower)

  return nothing
end


function calc_surface_integral!(du, u, mesh::Union{TreeMesh{2}, StructuredMesh{2}},
                                equations, surface_integral::SurfaceIntegralWeakForm,
                                dg::DG, cache)
  @unpack boundary_interpolation = dg.basis
  @unpack surface_flux_values = cache.elements

  # Note that all fluxes have been computed with outward-pointing normal vectors.
  # Access the factors only once before beginning the loop to increase performance.
  # We also use explicit assignments instead of `+=` to let `@muladd` turn these
  # into FMAs (see comment at the top of the file).
  factor_1 = boundary_interpolation[1,          1]
  factor_2 = boundary_interpolation[nnodes(dg), 2]
  @threaded for element in eachelement(dg, cache)
    for l in eachnode(dg)
      for v in eachvariable(equations)
        # surface at -x
        du[v, 1,          l, element] = (
          du[v, 1,          l, element] - surface_flux_values[v, l, 1, element] * factor_1)

        # surface at +x
        du[v, nnodes(dg), l, element] = (
          du[v, nnodes(dg), l, element] + surface_flux_values[v, l, 2, element] * factor_2)

        # surface at -y
        du[v, l, 1,          element] = (
          du[v, l, 1,          element] - surface_flux_values[v, l, 3, element] * factor_1)

        # surface at +y
        du[v, l, nnodes(dg), element] = (
          du[v, l, nnodes(dg), element] + surface_flux_values[v, l, 4, element] * factor_2)
      end
    end
  end

  return nothing
end


function apply_jacobian!(du, mesh::TreeMesh{2},
                         equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    factor = -cache.elements.inverse_jacobian[element]

    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        du[v, i, j, element] *= factor
      end
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_sources!(du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{2}, dg::DG, cache)
  return nothing
end

function calc_sources!(du, u, t, source_terms,
                       equations::AbstractEquations{2}, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      x_local = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      du_local = source_terms(u_local, x_local, t, equations)
      add_to_node_vars!(du, du_local, equations, dg, i, j, element)
    end
  end

  return nothing
end


end # @muladd
