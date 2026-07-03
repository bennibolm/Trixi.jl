# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache_subcell_limiting(mesh::Union{TreeMesh{3}, P4estMesh{3}},
                                       equations,
                                       volume_integral::VolumeIntegralSubcellLimiting,
                                       dg::DG, cache_containers, uEltype)
    cache = NamedTuple()

    fhat1_L_threaded, fhat1_R_threaded,
    fhat2_L_threaded, fhat2_R_threaded,
    fhat3_L_threaded, fhat3_R_threaded = create_f_threaded(mesh, equations, dg, uEltype)

    A4d = Array{uEltype, 4}
    flux_temp_threaded = A4d[A4d(undef, nvariables(equations),
                                 nnodes(dg), nnodes(dg), nnodes(dg))
                             for _ in 1:Threads.maxthreadid()]
    fhat_temp_threaded = A4d[A4d(undef, nvariables(equations),
                                 nnodes(dg), nnodes(dg), nnodes(dg))
                             for _ in 1:Threads.maxthreadid()]

    n_elements = nelements(cache_containers.elements)
    antidiffusive_fluxes = ContainerAntidiffusiveFlux3D{uEltype}(n_elements,
                                                                 nvariables(equations),
                                                                 nnodes(dg))

    if have_nonconservative_terms(equations) == true
        A5d = Array{uEltype, 5}
        # Extract the nonconservative flux as a dispatch argument for `n_nonconservative_terms`
        _, volume_flux_noncons = volume_integral.volume_flux_dg

        flux_nonconservative_temp_threaded = A5d[A5d(undef, nvariables(equations),
                                                     n_nonconservative_terms(volume_flux_noncons),
                                                     nnodes(dg), nnodes(dg),
                                                     nnodes(dg))
                                                 for _ in 1:Threads.maxthreadid()]
        fhat_nonconservative_temp_threaded = A5d[A5d(undef, nvariables(equations),
                                                     n_nonconservative_terms(volume_flux_noncons),
                                                     nnodes(dg), nnodes(dg),
                                                     nnodes(dg))
                                                 for _ in 1:Threads.maxthreadid()]
        phi_threaded = A5d[A5d(undef, nvariables(equations),
                               n_nonconservative_terms(volume_flux_noncons),
                               nnodes(dg), nnodes(dg), nnodes(dg))
                           for _ in 1:Threads.maxthreadid()]
        cache = (; cache..., flux_nonconservative_temp_threaded,
                 fhat_nonconservative_temp_threaded, phi_threaded)
    end

    # The limiter cache was created with 0 elements
    resize_subcell_limiter_cache!(volume_integral.limiter, n_elements)
    precompute_n_mortars_per_nodes!(volume_integral, dg, cache_containers, mesh)

    return (; cache..., antidiffusive_fluxes,
            fhat1_L_threaded, fhat1_R_threaded,
            fhat2_L_threaded, fhat2_R_threaded,
            fhat3_L_threaded, fhat3_R_threaded,
            flux_temp_threaded, fhat_temp_threaded)
end

function calc_mortar_weights(equations::AbstractEquations{3},
                             basis::LobattoLegendreBasis, RealT)
    n_nodes = nnodes(basis)
    mortar_weights = zeros(RealT, n_nodes, n_nodes, n_nodes, n_nodes, 4) # [node_i (large), node_j (large), node_i (small), node_j (small), small element]
    mortar_weights_sums = zeros(RealT, n_nodes, n_nodes, 2) # [node_i, node_j, small (1) / large (2) element]

    calc_mortar_weights!(equations, mortar_weights, n_nodes, RealT)

    # Sums of mortar weights for normalization
    for j in eachnode(basis), i in eachnode(basis)
        for l in eachnode(basis), k in eachnode(basis)
            # Add weights from large element to small element
            # Sums for all small elements are equal due to symmetry
            mortar_weights_sums[i, j, 1] += mortar_weights[k, l, i, j, 1]
            # Add weights from small element to large element
            for small_element in 1:4
                mortar_weights_sums[i, j, 2] += mortar_weights[i, j, k, l,
                                                               small_element]
            end
        end
    end

    return mortar_weights, mortar_weights_sums
end

function calc_mortar_weights!(equations::AbstractEquations{3}, mortar_weights, n_nodes,
                              RealT)
    _, weights = gauss_lobatto_nodes_weights(n_nodes, RealT)

    # Local mortar weights are of the form: `w_(ij, kl) = int_S psi_(ij) phi_(kl) ds`,
    # where `psi_(ij)` are the basis functions of the large element and `phi_(kl)` are the basis
    # functions of the small element. `S` is the face connecting both elements.
    # We use piecewise constant basis functions on the LGL subgrid. So, only focus on interval,
    # where both basis functions are non-zero. `interval = [left_bound_x, right_bound_x] x [left_bound_y, right_bound_y]`.
    # `w_(ij, kl) = int_S psi_(ij) phi_(kl) ds = int_(interval) ds = (right_bound_x - left_bound_x) * (right_bound_y - left_bound_y)`.
    # The bounds in each direction are independent and can be computed separately analogously to the 2D case:
    # `right_bound = min(left_bound_large, left_bound_small)`
    # `left_bound = max(right_bound_large, right_bound_small)`
    # If `right_bound <= left_bound`, i.e., both intervals don't overlap, then `w_ij = 0`.

    # Due to the LGL subgrid, the interval bounds are cumulative LGL quadrature weights.
    cum_weights_large = [zero(RealT); cumsum(weights)] .- 1 # on [-1, 1]
    cum_weights_lower = 0.5f0 * cum_weights_large .- 0.5f0  # on [-1, 0]
    cum_weights_upper = cum_weights_lower .+ 1              # on [0, 1]
    # So, for `w_(ij, kl)` we have
    # `right_bound_x = min(cum_weights_large[i], cum_weights_small[k])`
    # `left_bound_x = max(cum_weights_large[i+1], cum_weights_small[k+1])`
    # `right_bound_y = min(cum_weights_large[j], cum_weights_small[l])`
    # `left_bound_y = max(cum_weights_large[j+1], cum_weights_small[l+1])`

    # Illustration of the positions in 3D, where ξ and η are the local coordinates
    # of the mortar element, which are precisely the local coordinates that span
    # the surface of the smaller side.
    # Note that the orientation in the physical space is completely irrelevant here.
    #   ┌─────────────┬─────────────┐  ┌───────────────────────────┐
    #   │             │             │  │                           │
    #   │    small    │    small    │  │                           │
    #   │      3      │      4      │  │                           │
    #   │             │             │  │           large           │
    #   ├─────────────┼─────────────┤  │             5             │
    # η │             │             │  │                           │
    #   │    small    │    small    │  │                           │
    # ↑ │      1      │      2      │  │                           │
    # │ │             │             │  │                           │
    # │ └─────────────┴─────────────┘  └───────────────────────────┘
    # │
    # ⋅────> ξ

    for j in 1:n_nodes, i in 1:n_nodes
        for l in 1:n_nodes, k in 1:n_nodes
            # 1st small and large element element
            left_x = max(cum_weights_large[i], cum_weights_lower[k])
            right_x = min(cum_weights_large[i + 1], cum_weights_lower[k + 1])
            left_y = max(cum_weights_large[j], cum_weights_lower[l])
            right_y = min(cum_weights_large[j + 1], cum_weights_lower[l + 1])

            # Local weight of 0 if intervals do not overlap, i.e., `right <= left`
            if right_x > left_x && right_y > left_y
                mortar_weights[i, j, k, l, 1] = (right_x - left_x) * (right_y - left_y)
            end

            # 2nd small and large element
            left_x = max(cum_weights_large[i], cum_weights_upper[k])
            right_x = min(cum_weights_large[i + 1], cum_weights_upper[k + 1])
            left_y = max(cum_weights_large[j], cum_weights_lower[l])
            right_y = min(cum_weights_large[j + 1], cum_weights_lower[l + 1])
            if right_x > left_x && right_y > left_y
                mortar_weights[i, j, k, l, 2] = (right_x - left_x) * (right_y - left_y)
            end

            # 3rd small and large element
            left_x = max(cum_weights_large[i], cum_weights_lower[k])
            right_x = min(cum_weights_large[i + 1], cum_weights_lower[k + 1])
            left_y = max(cum_weights_large[j], cum_weights_upper[l])
            right_y = min(cum_weights_large[j + 1], cum_weights_upper[l + 1])
            if right_x > left_x && right_y > left_y
                mortar_weights[i, j, k, l, 3] = (right_x - left_x) * (right_y - left_y)
            end

            # 4th small and large element
            left_x = max(cum_weights_large[i], cum_weights_upper[k])
            right_x = min(cum_weights_large[i + 1], cum_weights_upper[k + 1])
            left_y = max(cum_weights_large[j], cum_weights_upper[l])
            right_y = min(cum_weights_large[j + 1], cum_weights_upper[l + 1])
            if right_x > left_x && right_y > left_y
                mortar_weights[i, j, k, l, 4] = (right_x - left_x) * (right_y - left_y)
            end
        end
    end

    return mortar_weights
end

# Subcell limiting currently only implemented for certain mesh types
@inline function volume_integral_kernel!(du, u, element,
                                         MeshT::Type{<:Union{TreeMesh{3}, P4estMesh{3}}},
                                         nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralSubcellLimiting,
                                         dg::DGSEM, cache)
    @unpack inverse_weights = dg.basis # Plays role of DG subcell sizes
    @unpack volume_flux_dg, volume_flux_fv, limiter = volume_integral

    # high-order DG fluxes
    @unpack fhat1_L_threaded, fhat1_R_threaded, fhat2_L_threaded, fhat2_R_threaded, fhat3_L_threaded, fhat3_R_threaded = cache

    fhat1_L = fhat1_L_threaded[Threads.threadid()]
    fhat1_R = fhat1_R_threaded[Threads.threadid()]
    fhat2_L = fhat2_L_threaded[Threads.threadid()]
    fhat2_R = fhat2_R_threaded[Threads.threadid()]
    fhat3_L = fhat3_L_threaded[Threads.threadid()]
    fhat3_R = fhat3_R_threaded[Threads.threadid()]
    calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                   u, MeshT, nonconservative_terms, equations, volume_flux_dg,
                   dg, element, cache)

    # low-order FV fluxes
    @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded, fstar3_L_threaded, fstar3_R_threaded = cache

    fstar1_L = fstar1_L_threaded[Threads.threadid()]
    fstar1_R = fstar1_R_threaded[Threads.threadid()]
    fstar2_L = fstar2_L_threaded[Threads.threadid()]
    fstar2_R = fstar2_R_threaded[Threads.threadid()]
    fstar3_L = fstar3_L_threaded[Threads.threadid()]
    fstar3_R = fstar3_R_threaded[Threads.threadid()]
    calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                 u, MeshT, nonconservative_terms, equations, volume_flux_fv,
                 dg, element, cache)

    # antidiffusive flux
    calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                            fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                            u, MeshT, nonconservative_terms, equations, limiter,
                            dg, element, cache)

    # Calculate volume integral contribution of low-order FV flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            du[v, i, j, k, element] += inverse_weights[i] *
                                       (fstar1_L[v, i + 1, j, k] - fstar1_R[v, i, j, k]) +
                                       inverse_weights[j] *
                                       (fstar2_L[v, i, j + 1, k] - fstar2_R[v, i, j, k]) +
                                       inverse_weights[k] *
                                       (fstar3_L[v, i, j, k + 1] - fstar3_R[v, i, j, k])
        end
    end

    return nothing
end

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**without non-conservative terms**).
#
# See also `flux_differencing_kernel!`.
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                                u, ::Type{<:TreeMesh{3}},
                                have_nonconservative_terms::False, equations,
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

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            flux1 = volume_flux(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
                                       equations, dg, ii, j, k)
        end
    end

    # FV-form flux `fhat` in x direction
    for k in eachnode(dg), j in eachnode(dg), i in 1:(nnodes(dg) - 1)
        for v in eachvariable(equations)
            fhat1_L[v, i + 1, j, k] = fhat1_L[v, i, j, k] +
                                      weights[i] * flux_temp[v, i, j, k]
            fhat1_R[v, i + 1, j, k] = fhat1_L[v, i + 1, j, k]
        end
    end

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            flux2 = volume_flux(u_node, u_node_jj, 2, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2,
                                       equations, dg, i, jj, k)
        end
    end

    # FV-form flux `fhat` in y direction
    for k in eachnode(dg), j in 1:(nnodes(dg) - 1), i in eachnode(dg)
        for v in eachvariable(equations)
            fhat2_L[v, i, j + 1, k] = fhat2_L[v, i, j, k] +
                                      weights[j] * flux_temp[v, i, j, k]
            fhat2_R[v, i, j + 1, k] = fhat2_L[v, i, j + 1, k]
        end
    end

    # Split form volume flux in orientation 3: z direction
    flux_temp .= zero(eltype(flux_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            flux3 = volume_flux(u_node, u_node_kk, 3, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], flux3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], flux3,
                                       equations, dg, i, j, kk)
        end
    end

    # FV-form flux `fhat` in z direction
    for k in 1:(nnodes(dg) - 1), j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            fhat3_L[v, i, j, k + 1] = fhat3_L[v, i, j, k] +
                                      weights[k] * flux_temp[v, i, j, k]
            fhat3_R[v, i, j, k + 1] = fhat3_L[v, i, j, k + 1]
        end
    end

    return nothing
end

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**with non-conservative terms in "local * symmetric" form**).
#
# See also `flux_differencing_kernel!`.
#
# The calculation of the non-conservative staggered "fluxes" requires non-conservative
# terms that can be written as a product of local and a symmetric contributions. See, e.g.,
#
# - Rueda-Ramírez, Gassner (2023). A Flux-Differencing Formula for Split-Form Summation By Parts
#   Discretizations of Non-Conservative Systems. https://arxiv.org/pdf/2211.14009.pdf.
#
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                                u, mesh::TreeMesh{3},
                                have_nonconservative_terms::True, equations,
                                volume_flux::Tuple{F_CONS, F_NONCONS}, dg::DGSEM,
                                element,
                                cache) where {
                                              F_CONS <: Function,
                                              F_NONCONS <:
                                              FluxNonConservative{NonConservativeSymmetric()}
                                              }
    @unpack weights, derivative_split = dg.basis
    @unpack flux_temp_threaded, flux_nonconservative_temp_threaded = cache
    @unpack fhat_temp_threaded, fhat_nonconservative_temp_threaded, phi_threaded = cache

    volume_flux_cons, volume_flux_noncons = volume_flux

    flux_temp = flux_temp_threaded[Threads.threadid()]
    flux_noncons_temp = flux_nonconservative_temp_threaded[Threads.threadid()]

    fhat_temp = fhat_temp_threaded[Threads.threadid()]
    fhat_noncons_temp = fhat_nonconservative_temp_threaded[Threads.threadid()]
    phi = phi_threaded[Threads.threadid()]

    # The FV-form fluxes are calculated in a recursive manner, i.e.:
    # fhat_(0,1)   = w_0 * FVol_0,
    # fhat_(j,j+1) = fhat_(j-1,j) + w_j * FVol_j,   for j=1,...,N-1,
    # with the split form volume fluxes FVol_j = -2 * sum_i=0^N D_ji f*_(j,i).

    # To use the symmetry of the `volume_flux`, the split form volume flux is precalculated
    # like in `calc_volume_integral!` for the `VolumeIntegralFluxDifferencing`
    # and saved in in `flux_temp`.

    # Split form volume flux in orientation 1: x direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            flux1 = volume_flux_cons(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
                                       equations, dg, ii, j, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, 1, equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[i, ii],
                                           flux1_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[ii, i],
                                           flux1_noncons,
                                           equations, dg, noncons, ii, j, k)
            end
        end
    end

    # FV-form flux `fhat` in x direction
    fhat_temp[:, 1, :, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, 1, :, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 1, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in eachnode(dg), j in eachnode(dg), i in 1:(nnodes(dg) - 1)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[i] * flux_temp[v, i, j, k]
            fhat_temp[v, i + 1, j, k] = value
            fhat1_L[v, i + 1, j, k] = value
            fhat1_R[v, i + 1, j, k] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[i] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i + 1, j, k] = value

            fhat1_L[v, i + 1, j, k] = fhat1_L[v, i + 1, j, k] +
                                      phi[v, noncons, i, j, k] * value
            fhat1_R[v, i + 1, j, k] = fhat1_R[v, i + 1, j, k] +
                                      phi[v, noncons, i + 1, j, k] * value
        end
    end

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            flux2 = volume_flux_cons(u_node, u_node_jj, 2, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2,
                                       equations, dg, i, jj, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, 2, equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[jj, j],
                                           flux2_noncons,
                                           equations, dg, noncons, i, jj, k)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat_temp[:, :, 1, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, 1, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 2, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in eachnode(dg), j in 1:(nnodes(dg) - 1), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[j] * flux_temp[v, i, j, k]
            fhat_temp[v, i, j + 1, k] = value
            fhat2_L[v, i, j + 1, k] = value
            fhat2_R[v, i, j + 1, k] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[j] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i, j + 1, k] = value

            fhat2_L[v, i, j + 1, k] = fhat2_L[v, i, j + 1, k] +
                                      phi[v, noncons, i, j, k] * value
            fhat2_R[v, i, j + 1, k] = fhat2_R[v, i, j + 1, k] +
                                      phi[v, noncons, i, j + 1, k] * value
        end
    end

    # Split form volume flux in orientation 3: z direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            flux3 = volume_flux_cons(u_node, u_node_kk, 3, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], flux3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], flux3,
                                       equations, dg, i, j, kk)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux3_noncons = volume_flux_noncons(u_node, u_node_kk, 3, equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[k, kk],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[kk, k],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, kk)
            end
        end
    end

    # FV-form flux `fhat` in z direction
    fhat_temp[:, :, :, 1] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, :, 1] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 3, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in 1:(nnodes(dg) - 1), j in eachnode(dg), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[k] * flux_temp[v, i, j, k]
            fhat_temp[v, i, j, k + 1] = value
            fhat3_L[v, i, j, k + 1] = value
            fhat3_R[v, i, j, k + 1] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[k] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i, j, k + 1] = value

            fhat3_L[v, i, j, k + 1] = fhat3_L[v, i, j, k + 1] +
                                      phi[v, noncons, i, j, k] * value
            fhat3_R[v, i, j, k + 1] = fhat3_R[v, i, j, k + 1] +
                                      phi[v, noncons, i, j, k + 1] * value
        end
    end

    return nothing
end

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**with non-conservative terms in "local * jump" form**).
#
# See also `flux_differencing_kernel!`.
#
# The calculation of the non-conservative staggered "fluxes" requires non-conservative
# terms that can be written as a product of local and jump contributions.
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                                u, mesh::TreeMesh{3},
                                nonconservative_terms::True, equations,
                                volume_flux::Tuple{F_CONS, F_NONCONS}, dg::DGSEM,
                                element,
                                cache) where {
                                              F_CONS <: Function,
                                              F_NONCONS <:
                                              FluxNonConservative{NonConservativeJump()}
                                              }
    @unpack weights, derivative_split = dg.basis
    @unpack flux_temp_threaded, flux_nonconservative_temp_threaded = cache
    @unpack fhat_temp_threaded, fhat_nonconservative_temp_threaded, phi_threaded = cache

    volume_flux_cons, volume_flux_noncons = volume_flux

    flux_temp = flux_temp_threaded[Threads.threadid()]
    flux_noncons_temp = flux_nonconservative_temp_threaded[Threads.threadid()]

    fhat_temp = fhat_temp_threaded[Threads.threadid()]
    fhat_noncons_temp = fhat_nonconservative_temp_threaded[Threads.threadid()]
    phi = phi_threaded[Threads.threadid()]

    # The FV-form fluxes are calculated in a recursive manner, i.e.:
    # fhat_(0,1)   = w_0 * FVol_0,
    # fhat_(j,j+1) = fhat_(j-1,j) + w_j * FVol_j,   for j=1,...,N-1,
    # with the split form volume fluxes FVol_j = -2 * sum_i=0^N D_ji f*_(j,i).

    # To use the symmetry of the `volume_flux`, the split form volume flux is precalculated
    # like in `calc_volume_integral!` for the `VolumeIntegralFluxDifferencing`
    # and saved in in `flux_temp`.

    # Split form volume flux in orientation 1: x direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and skew-symmetry of `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            flux1 = volume_flux_cons(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
                                       equations, dg, ii, j, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, 1, equations,
                                                    NonConservativeJump(),
                                                    noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[i, ii],
                                           flux1_noncons,
                                           equations, dg, noncons, i, j, k)
                # Note the sign flip due to skew-symmetry when argument order is swapped
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5f0 * derivative_split[ii, i],
                                           flux1_noncons,
                                           equations, dg, noncons, ii, j, k)
            end
        end
    end

    # FV-form flux `fhat` in x direction
    fhat_temp[:, 1, :, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, 1, :, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 1, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in eachnode(dg), j in eachnode(dg), i in 1:(nnodes(dg) - 1)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[i] * flux_temp[v, i, j, k]
            fhat_temp[v, i + 1, j, k] = value
            fhat1_L[v, i + 1, j, k] = value
            fhat1_R[v, i + 1, j, k] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[i] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i + 1, j, k] = value

            fhat1_L[v, i + 1, j, k] = fhat1_L[v, i + 1, j, k] +
                                      phi[v, noncons, i, j, k] * value
            fhat1_R[v, i + 1, j, k] = fhat1_R[v, i + 1, j, k] +
                                      phi[v, noncons, i + 1, j, k] * value
        end
    end

    # Apply correction term to the flux-differencing formula for nonconservative local * jump fluxes.
    for k in eachnode(dg), j in eachnode(dg)
        u_0 = get_node_vars(u, equations, dg, 1, j, k, element)
        for i in 2:(nnodes(dg) - 1)
            u_i = get_node_vars(u, equations, dg, i, j, k, element)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_i, 1, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat1_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat1_L[v, i + 1, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, nnodes(dg), j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, 1, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing because Trixi multiplies all the non-cons terms with 0.5
                fhat1_R[v, nnodes(dg), j, k] -= phi[v, noncons, nnodes(dg), j, k] *
                                                phi_jump[v]
            end
        end
    end

    ########

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            flux2 = volume_flux_cons(u_node, u_node_jj, 2, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2,
                                       equations, dg, i, jj, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, 2, equations,
                                                    NonConservativeJump(),
                                                    noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j, k)
                # Note the sign flip due to skew-symmetry when argument order is swapped
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5 * derivative_split[jj, j],
                                           flux2_noncons,
                                           equations, dg, noncons, i, jj, k)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat_temp[:, :, 1, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, 1, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 2, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in eachnode(dg), j in 1:(nnodes(dg) - 1), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[j] * flux_temp[v, i, j, k]
            fhat_temp[v, i, j + 1, k] = value
            fhat2_L[v, i, j + 1, k] = value
            fhat2_R[v, i, j + 1, k] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[j] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i, j + 1, k] = value

            fhat2_L[v, i, j + 1, k] = fhat2_L[v, i, j + 1, k] +
                                      phi[v, noncons, i, j, k] * value
            fhat2_R[v, i, j + 1, k] = fhat2_R[v, i, j + 1, k] +
                                      phi[v, noncons, i, j + 1, k] * value
        end
    end

    # Apply correction term to the flux-differencing formula for nonconservative local * jump fluxes.
    for k in eachnode(dg), i in eachnode(dg)
        u_0 = get_node_vars(u, equations, dg, i, 1, k, element)
        for j in 2:(nnodes(dg) - 1)
            u_j = get_node_vars(u, equations, dg, i, j, k, element)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_j, 2, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat2_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat2_L[v, i, j + 1, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, i, nnodes(dg), k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, 2, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
                fhat2_R[v, i, nnodes(dg), k] -= phi[v, noncons, i, nnodes(dg), k] *
                                                phi_jump[v]
            end
        end
    end

    ########

    # Split form volume flux in orientation 3: z direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            flux3 = volume_flux_cons(u_node, u_node_kk, 3, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], flux3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], flux3,
                                       equations, dg, i, j, kk)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux3_noncons = volume_flux_noncons(u_node, u_node_kk, 3, equations,
                                                    NonConservativeJump(),
                                                    noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[k, kk],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, k)
                # Note the sign flip due to skew-symmetry when argument order is swapped
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5 * derivative_split[kk, k],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, kk)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat_temp[:, :, :, 1] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, :, 1] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 3, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in 1:(nnodes(dg) - 1), j in eachnode(dg), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[k] * flux_temp[v, i, j, k]
            fhat_temp[v, i, j, k + 1] = value
            fhat3_L[v, i, j, k + 1] = value
            fhat3_R[v, i, j, k + 1] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[k] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i, j, k + 1] = value

            fhat3_L[v, i, j, k + 1] = fhat3_L[v, i, j, k + 1] +
                                      phi[v, noncons, i, j, k] * value
            fhat3_R[v, i, j, k + 1] = fhat3_R[v, i, j, k + 1] +
                                      phi[v, noncons, i, j, k + 1] * value
        end
    end

    # Apply correction term to the flux-differencing formula for nonconservative local * jump fluxes.
    for j in eachnode(dg), i in eachnode(dg)
        u_0 = get_node_vars(u, equations, dg, i, j, 1, element)
        for k in 2:(nnodes(dg) - 1)
            u_k = get_node_vars(u, equations, dg, i, j, k, element)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_k, 2, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat3_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat3_L[v, i, j, k + 1] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, i, j, nnodes(dg), element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, 3, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
                fhat3_R[v, i, j, nnodes(dg)] -= phi[v, noncons, i, j, nnodes(dg)] *
                                                phi_jump[v]
            end
        end
    end
    return nothing
end

# Calculate the antidiffusive flux `antidiffusive_flux` as the subtraction between `fhat` and `fstar` for conservative systems.
@inline function calcflux_antidiffusive!(fhat1_L, fhat1_R,
                                         fhat2_L, fhat2_R,
                                         fhat3_L, fhat3_R,
                                         fstar1_L, fstar1_R,
                                         fstar2_L, fstar2_R,
                                         fstar3_L, fstar3_R,
                                         u, ::Type{<:Union{TreeMesh{3}, P4estMesh{3}}},
                                         nonconservative_terms::False, equations,
                                         limiter::SubcellLimiterIDP, dg, element, cache)
    @unpack antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R = cache.antidiffusive_fluxes

    # Due to the use of LGL nodes, the DG staggered fluxes `fhat` and FV fluxes `fstar` are equal
    # on the element interfaces. So, they are not computed in the volume integral and set to zero
    # in their respective computation.
    # The antidiffusive fluxes are therefore zero on the element interfaces and don't need to be
    # computed either. They are set to zero directly after resizing the container.
    # This applies to the indices `i=1` and `i=nnodes(dg)+1` for `antidiffusive_flux1_L` and
    # `antidiffusive_flux1_R` and analogously for the other two directions.

    for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)
        for v in eachvariable(equations)
            antidiffusive_flux1_L[v, i, j, k, element] = fhat1_L[v, i, j, k] -
                                                         fstar1_L[v, i, j, k]
            antidiffusive_flux1_R[v, i, j, k, element] = antidiffusive_flux1_L[v,
                                                                               i, j, k,
                                                                               element]
        end
    end
    for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux2_L[v, i, j, k, element] = fhat2_L[v, i, j, k] -
                                                         fstar2_L[v, i, j, k]
            antidiffusive_flux2_R[v, i, j, k, element] = antidiffusive_flux2_L[v,
                                                                               i, j, k,
                                                                               element]
        end
    end
    for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux3_L[v, i, j, k, element] = fhat3_L[v, i, j, k] -
                                                         fstar3_L[v, i, j, k]
            antidiffusive_flux3_R[v, i, j, k, element] = antidiffusive_flux3_L[v,
                                                                               i, j, k,
                                                                               element]
        end
    end

    return nothing
end

# Calculate the antidiffusive flux `antidiffusive_flux` as the subtraction between `fhat` and `fstar` for conservative systems.
@inline function calcflux_antidiffusive!(fhat1_L, fhat1_R,
                                         fhat2_L, fhat2_R,
                                         fhat3_L, fhat3_R,
                                         fstar1_L, fstar1_R,
                                         fstar2_L, fstar2_R,
                                         fstar3_L, fstar3_R,
                                         u, ::Type{<:Union{TreeMesh{3}, P4estMesh{3}}},
                                         nonconservative_terms::True, equations,
                                         limiter::SubcellLimiterIDP, dg, element, cache)
    @unpack antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R = cache.antidiffusive_fluxes

    # Due to the use of LGL nodes, the DG staggered fluxes `fhat` and FV fluxes `fstar` are equal
    # on the element interfaces. So, they are not computed in the volume integral and set to zero
    # in their respective computation.
    # The antidiffusive fluxes are therefore zero on the element interfaces and don't need to be
    # computed either. They are set to zero directly after resizing the container.
    # This applies to the indices `i=1` and `i=nnodes(dg)+1` for `antidiffusive_flux1_L` and
    # `antidiffusive_flux1_R` and analogously for the other two directions.

    for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)
        for v in eachvariable(equations)
            antidiffusive_flux1_L[v, i, j, k, element] = fhat1_L[v, i, j, k] -
                                                         fstar1_L[v, i, j, k]
            antidiffusive_flux1_R[v, i, j, k, element] = fhat1_R[v, i, j, k] -
                                                         fstar1_R[v, i, j, k]
        end
    end
    for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux2_L[v, i, j, k, element] = fhat2_L[v, i, j, k] -
                                                         fstar2_L[v, i, j, k]
            antidiffusive_flux2_R[v, i, j, k, element] = fhat2_R[v, i, j, k] -
                                                         fstar2_R[v, i, j, k]
        end
    end
    for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux3_L[v, i, j, k, element] = fhat3_L[v, i, j, k] -
                                                         fstar3_L[v, i, j, k]
            antidiffusive_flux3_R[v, i, j, k, element] = fhat3_R[v, i, j, k] -
                                                         fstar3_R[v, i, j, k]
        end
    end

    return nothing
end

function prolong2mortars!(cache, u, mesh::TreeMesh{3}, equations,
                          mortar_idp::LobattoLegendreMortarIDP, dg::DGSEM)
    prolong2mortars!(cache, u, mesh, equations, mortar_idp.mortar_l2, dg)

    # The data of both small elements were already copied to the mortar cache
    @threaded for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[5, mortar]

        # Copy solutions
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if cache.mortars.orientations[mortar] == 1
                # IDP mortars in x-direction
                for k in eachnode(dg), j in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, j, k, mortar] = u[v, nnodes(dg), j, k,
                                                                   large_element]
                    end
                end
            elseif cache.mortars.orientations[mortar] == 2
                # IDP mortars in y-direction
                for k in eachnode(dg), i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, i, k, mortar] = u[v, i, nnodes(dg), k,
                                                                   large_element]
                    end
                end
            else
                # IDP mortars in z-direction
                for j in eachnode(dg), i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, i, j, mortar] = u[v, i, j, nnodes(dg),
                                                                   large_element]
                    end
                end
            end
        else # large_sides[mortar] == 2 -> small elements on left side
            if cache.mortars.orientations[mortar] == 1
                # IDP mortars in x-direction
                for k in eachnode(dg), j in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, j, k, mortar] = u[v, 1, j, k,
                                                                   large_element]
                    end
                end
            elseif cache.mortars.orientations[mortar] == 2
                # IDP mortars in y-direction
                for k in eachnode(dg), i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, i, k, mortar] = u[v, i, 1, k,
                                                                   large_element]
                    end
                end
            else
                # IDP mortars in z-direction
                for j in eachnode(dg), i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, i, j, mortar] = u[v, i, j, 1,
                                                                   large_element]
                    end
                end
            end
        end
    end

    return nothing
end

function calc_mortar_flux_low_order!(surface_flux_values,
                                     mesh::TreeMesh{3},
                                     nonconservative_terms::False, equations,
                                     mortar_idp::LobattoLegendreMortarIDP,
                                     surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u_lower_left, u_lower_right, u_upper_left, u_upper_right, u_large, orientations = cache.mortars
    (; mortar_weights, mortar_weights_sums) = mortar_idp

    @threaded for mortar in eachmortar(dg, cache)
        lower_left_element = cache.mortars.neighbor_ids[1, mortar]
        lower_right_element = cache.mortars.neighbor_ids[2, mortar]
        upper_left_element = cache.mortars.neighbor_ids[3, mortar]
        upper_right_element = cache.mortars.neighbor_ids[4, mortar]
        large_element = cache.mortars.neighbor_ids[5, mortar]

        # Calculate fluxes
        orientation = orientations[mortar]

        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientation == 1
                # L2 mortars in x-direction
                direction_small = 1
                direction_large = 2
            elseif orientation == 2
                # L2 mortars in y-direction
                direction_small = 3
                direction_large = 4
            else
                # L2 mortars in z-direction
                direction_small = 5
                direction_large = 6
            end
            small_side = 2
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientation == 1
                # L2 mortars in x-direction
                direction_small = 2
                direction_large = 1
            elseif orientation == 2
                # L2 mortars in y-direction
                direction_small = 4
                direction_large = 3
            else
                # L2 mortars in z-direction
                direction_small = 6
                direction_large = 5
            end
            small_side = 1
        end

        surface_flux_values[:, :, :, direction_small, lower_left_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, :, direction_small, lower_right_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, :, direction_small, upper_left_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, :, direction_small, upper_right_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, :, direction_large, large_element] .= zero(eltype(surface_flux_values))
        # Lower left element
        for j in eachnode(dg), i in eachnode(dg)
            u_lower_left_local = get_surface_node_vars(u_lower_left, equations, dg,
                                                       i, j, mortar)[small_side]
            for l in eachnode(dg), k in eachnode(dg)
                factor = mortar_weights[k, l, i, j, 1]
                if isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, l, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_lower_left_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_lower_left_local, u_large_local, orientation,
                                        equations)
                end

                # Lower left element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[i, j, 1],
                                           flux, equations, dg,
                                           i, j, direction_small, lower_left_element)
                # Large element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[k, l, 2],
                                           flux, equations, dg,
                                           k, l, direction_large, large_element)
            end
        end
        # Lower right element
        for j in eachnode(dg), i in eachnode(dg)
            u_lower_right_local = get_surface_node_vars(u_lower_right, equations, dg,
                                                        i, j, mortar)[small_side]
            for l in eachnode(dg), k in eachnode(dg)
                factor = mortar_weights[k, l, i, j, 2]
                if isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, l, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_lower_right_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_lower_right_local, u_large_local, orientation,
                                        equations)
                end

                # Lower right element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[i, j, 1],
                                           flux, equations, dg,
                                           i, j, direction_small, lower_right_element)
                # Large element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[k, l, 2],
                                           flux, equations, dg,
                                           k, l, direction_large, large_element)
            end
        end
        # Upper left element
        for j in eachnode(dg), i in eachnode(dg)
            u_upper_left_local = get_surface_node_vars(u_upper_left, equations, dg,
                                                       i, j, mortar)[small_side]
            for l in eachnode(dg), k in eachnode(dg)
                factor = mortar_weights[k, l, i, j, 3]
                if isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, l, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_upper_left_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_upper_left_local, u_large_local, orientation,
                                        equations)
                end

                # Upper left element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[i, j, 1],
                                           flux, equations, dg,
                                           i, j, direction_small, upper_left_element)
                # Large element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[k, l, 2],
                                           flux, equations, dg,
                                           k, l, direction_large, large_element)
            end
        end
        # Upper right element
        for j in eachnode(dg), i in eachnode(dg)
            u_upper_right_local = get_surface_node_vars(u_upper_right, equations, dg,
                                                        i, j, mortar)[small_side]
            for l in eachnode(dg), k in eachnode(dg)
                factor = mortar_weights[k, l, i, j, 4]
                if isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, l, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_upper_right_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_upper_right_local, u_large_local, orientation,
                                        equations)
                end

                # Upper right element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[i, j, 1],
                                           flux, equations, dg,
                                           i, j, direction_small, upper_right_element)
                # Large element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[k, l, 2],
                                           flux, equations, dg,
                                           k, l, direction_large, large_element)
            end
        end
    end

    return nothing
end

@inline function calc_lambdas_bar_states!(u, t, mesh::TreeMesh{3},
                                          have_nonconservative_terms, equations,
                                          limiter, dg, cache, boundary_conditions;
                                          calc_bar_states = true)
    if limiter.bar_states == false
        return nothing
    end

    (; lambda1, lambda2, lambda3, bar_states1, bar_states2, bar_states3) = limiter.cache.container_bar_states

    @threaded for element in eachelement(dg, cache)
        # It is sufficient to reset the lambdas and bar states at the interfaces since only the mortar computation adds terms up.
        lambda1[1, :, :, element] .= zero(eltype(lambda1))
        lambda1[end, :, :, element] .= zero(eltype(lambda1))
        lambda2[:, 1, :, element] .= zero(eltype(lambda2))
        lambda2[:, end, :, element] .= zero(eltype(lambda2))
        lambda3[:, :, 1, element] .= zero(eltype(lambda3))
        lambda3[:, :, end, element] .= zero(eltype(lambda3))
        if calc_bar_states
            bar_states1[:, 1, :, :, element] .= zero(eltype(bar_states1))
            bar_states1[:, end, :, :, element] .= zero(eltype(bar_states1))
            bar_states2[:, :, 1, :, element] .= zero(eltype(bar_states2))
            bar_states2[:, :, end, :, element] .= zero(eltype(bar_states2))
            bar_states3[:, :, :, 1, element] .= zero(eltype(bar_states3))
            bar_states3[:, :, :, end, element] .= zero(eltype(bar_states3))
        end

        for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            u_node_im1 = get_node_vars(u, equations, dg, i - 1, j, k, element)
            lambda1[i, j, k, element] = max_abs_speed_naive(u_node_im1, u_node, 1,
                                                            equations)

            calc_bar_states || continue

            flux1 = flux(u_node, 1, equations)
            flux1_im1 = flux(u_node_im1, 1, equations)
            for v in eachvariable(equations)
                bar_states1[v, i, j, k, element] = 0.5 * (u_node[v] + u_node_im1[v]) -
                                                   0.5 * (flux1[v] - flux1_im1[v]) /
                                                   lambda1[i, j, k, element]
            end
        end

        for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            u_node_jm1 = get_node_vars(u, equations, dg, i, j - 1, k, element)
            lambda2[i, j, k, element] = max_abs_speed_naive(u_node_jm1, u_node, 2,
                                                            equations)

            calc_bar_states || continue

            flux2 = flux(u_node, 2, equations)
            flux2_jm1 = flux(u_node_jm1, 2, equations)
            for v in eachvariable(equations)
                bar_states2[v, i, j, k, element] = 0.5 * (u_node[v] + u_node_jm1[v]) -
                                                   0.5 * (flux2[v] - flux2_jm1[v]) /
                                                   lambda2[i, j, k, element]
            end
        end

        for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            u_node_km1 = get_node_vars(u, equations, dg, i, j, k - 1, element)
            lambda3[i, j, k, element] = max_abs_speed_naive(u_node_km1, u_node, 3,
                                                            equations)

            calc_bar_states || continue

            flux3 = flux(u_node, 3, equations)
            flux3_km1 = flux(u_node_km1, 3, equations)
            for v in eachvariable(equations)
                bar_states3[v, i, j, k, element] = 0.5 * (u_node[v] + u_node_km1[v]) -
                                                   0.5 * (flux3[v] - flux3_km1[v]) /
                                                   lambda3[i, j, k, element]
            end
        end
    end

    # Calc lambdas and bar states at element interfaces and periodic boundaries
    calc_lambdas_bar_states_interface!(u, t, limiter, boundary_conditions, mesh,
                                       equations, dg, cache;
                                       calc_bar_states = calc_bar_states)

    # Calc lambdas and bar states at mortar interfaces
    calc_lambdas_bar_states_mortar!(u, t, limiter, boundary_conditions, mesh, equations,
                                    dg, cache; calc_bar_states = calc_bar_states)

    # Calc lambdas and bar states at physical boundaries
    calc_lambdas_bar_states_boundary!(u, t, limiter, boundary_conditions, mesh,
                                      equations, dg, cache;
                                      calc_bar_states = calc_bar_states)

    return nothing
end

@inline function calc_lambdas_bar_states_interface!(u, t, limiter, boundary_conditions,
                                                    mesh::TreeMesh{3}, equations, dg,
                                                    cache; calc_bar_states = true)
    (; lambda1, lambda2, lambda3, bar_states1, bar_states2, bar_states3) = limiter.cache.container_bar_states

    @threaded for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_element = cache.interfaces.neighbor_ids[1, interface]
        right_element = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]

        if orientation == 1
            for k in eachnode(dg), j in eachnode(dg)
                u_left = get_node_vars(u, equations, dg, nnodes(dg), j, k,
                                       left_element)
                u_right = get_node_vars(u, equations, dg, 1, j, k, right_element)
                lambda = max_abs_speed_naive(u_left, u_right, orientation, equations)

                lambda1[nnodes(dg) + 1, j, k, left_element] = lambda
                lambda1[1, j, k, right_element] = lambda

                calc_bar_states || continue

                flux_left = flux(u_left, orientation, equations)
                flux_right = flux(u_right, orientation, equations)
                bar_state = 0.5 * (u_left + u_right) -
                            0.5 * (flux_right - flux_left) / lambda
                for v in eachvariable(equations)
                    bar_states1[v, nnodes(dg) + 1, j, k, left_element] = bar_state[v]
                    bar_states1[v, 1, j, k, right_element] = bar_state[v]
                end
            end
        elseif orientation == 2
            for k in eachnode(dg), i in eachnode(dg)
                u_left = get_node_vars(u, equations, dg, i, nnodes(dg), k,
                                       left_element)
                u_right = get_node_vars(u, equations, dg, i, 1, k, right_element)
                lambda = max_abs_speed_naive(u_left, u_right, orientation, equations)

                lambda2[i, nnodes(dg) + 1, k, left_element] = lambda
                lambda2[i, 1, k, right_element] = lambda

                calc_bar_states || continue

                flux_left = flux(u_left, orientation, equations)
                flux_right = flux(u_right, orientation, equations)
                bar_state = 0.5 * (u_left + u_right) -
                            0.5 * (flux_right - flux_left) / lambda
                for v in eachvariable(equations)
                    bar_states2[v, i, nnodes(dg) + 1, k, left_element] = bar_state[v]
                    bar_states2[v, i, 1, k, right_element] = bar_state[v]
                end
            end
        else # orientation == 3
            for j in eachnode(dg), i in eachnode(dg)
                u_left = get_node_vars(u, equations, dg, i, j, nnodes(dg),
                                       left_element)
                u_right = get_node_vars(u, equations, dg, i, j, 1, right_element)
                lambda = max_abs_speed_naive(u_left, u_right, orientation, equations)

                lambda3[i, j, nnodes(dg) + 1, left_element] = lambda
                lambda3[i, j, 1, right_element] = lambda

                calc_bar_states || continue

                flux_left = flux(u_left, orientation, equations)
                flux_right = flux(u_right, orientation, equations)
                bar_state = 0.5 * (u_left + u_right) -
                            0.5 * (flux_right - flux_left) / lambda
                for v in eachvariable(equations)
                    bar_states3[v, i, j, nnodes(dg) + 1, left_element] = bar_state[v]
                    bar_states3[v, i, j, 1, right_element] = bar_state[v]
                end
            end
        end
    end

    return nothing
end

@inline function calc_lambdas_bar_states_mortar!(u, t, limiter, boundary_conditions,
                                                 mesh::TreeMesh{3}, equations,
                                                 dg, cache; calc_bar_states = true)
    (; lambda1, lambda2, lambda3, bar_states1, bar_states2, bar_states3) = limiter.cache.container_bar_states
    if nmortars(dg, cache) == 0
        return nothing
    end
    (; mortar_weights, mortar_weights_sums) = dg.mortar

    @threaded for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[5, mortar]

        orientation = cache.mortars.orientations[mortar]
        small_elements_on_right = cache.mortars.large_sides[mortar] == 1

        for j_large in eachnode(dg), i_large in eachnode(dg)
            if small_elements_on_right
                if orientation == 1
                    indices_large = (nnodes(dg), i_large, j_large)
                    lambda_indices_large = (nnodes(dg) + 1, i_large, j_large)
                elseif orientation == 2
                    indices_large = (i_large, nnodes(dg), j_large)
                    lambda_indices_large = (i_large, nnodes(dg) + 1, j_large)
                else # orientation == 3
                    indices_large = (i_large, j_large, nnodes(dg))
                    lambda_indices_large = (i_large, j_large, nnodes(dg) + 1)
                end
            else # small elements on left side
                if orientation == 1
                    indices_large = (1, i_large, j_large)
                    lambda_indices_large = (1, i_large, j_large)
                elseif orientation == 2
                    indices_large = (i_large, 1, j_large)
                    lambda_indices_large = (i_large, 1, j_large)
                else # orientation == 3
                    indices_large = (i_large, j_large, 1)
                    lambda_indices_large = (i_large, j_large, 1)
                end
            end
            u_large = get_node_vars(u, equations, dg, indices_large..., large_element)
            flux_large = flux(u_large, orientation, equations)

            for small_element_index in 1:4
                small_element = cache.mortars.neighbor_ids[small_element_index, mortar]
                for j_small in eachnode(dg), i_small in eachnode(dg)
                    weight = mortar_weights[i_large, j_large, i_small, j_small,
                                            small_element_index]
                    if iszero(weight)
                        continue
                    end

                    if small_elements_on_right
                        if orientation == 1
                            indices_small = (1, i_small, j_small)
                            lambda_indices_small = (1, i_small, j_small)
                        elseif orientation == 2
                            indices_small = (i_small, 1, j_small)
                            lambda_indices_small = (i_small, 1, j_small)
                        else # orientation == 3
                            indices_small = (i_small, j_small, 1)
                            lambda_indices_small = (i_small, j_small, 1)
                        end
                    else # small elements on left side
                        if orientation == 1
                            indices_small = (nnodes(dg), i_small, j_small)
                            lambda_indices_small = (nnodes(dg) + 1, i_small,
                                                    j_small)
                        elseif orientation == 2
                            indices_small = (i_small, nnodes(dg), j_small)
                            lambda_indices_small = (i_small, nnodes(dg) + 1,
                                                    j_small)
                        else # orientation == 3
                            indices_small = (i_small, j_small, nnodes(dg))
                            lambda_indices_small = (i_small, j_small,
                                                    nnodes(dg) + 1)
                        end
                    end
                    u_small = get_node_vars(u, equations, dg, indices_small...,
                                            small_element)

                    if small_elements_on_right
                        lambda = max_abs_speed_naive(u_large, u_small, orientation,
                                                     equations)
                    else
                        lambda = max_abs_speed_naive(u_small, u_large, orientation,
                                                     equations)
                    end

                    if orientation == 1
                        lambda1[lambda_indices_large..., large_element] += weight *
                                                                           lambda /
                                                                           mortar_weights_sums[i_large,
                                                                                               j_large,
                                                                                               2]
                        lambda1[lambda_indices_small..., small_element] += weight *
                                                                           lambda /
                                                                           mortar_weights_sums[i_small,
                                                                                               j_small,
                                                                                               1]
                    elseif orientation == 2
                        lambda2[lambda_indices_large..., large_element] += weight *
                                                                           lambda /
                                                                           mortar_weights_sums[i_large,
                                                                                               j_large,
                                                                                               2]
                        lambda2[lambda_indices_small..., small_element] += weight *
                                                                           lambda /
                                                                           mortar_weights_sums[i_small,
                                                                                               j_small,
                                                                                               1]
                    else # orientation == 3
                        lambda3[lambda_indices_large..., large_element] += weight *
                                                                           lambda /
                                                                           mortar_weights_sums[i_large,
                                                                                               j_large,
                                                                                               2]
                        lambda3[lambda_indices_small..., small_element] += weight *
                                                                           lambda /
                                                                           mortar_weights_sums[i_small,
                                                                                               j_small,
                                                                                               1]
                    end

                    calc_bar_states || continue

                    flux_small = flux(u_small, orientation, equations)
                    if small_elements_on_right
                        flux_diff = flux_small - flux_large
                    else
                        flux_diff = flux_large - flux_small
                    end
                    bar_state = 0.5 * (u_small + u_large - flux_diff / lambda)

                    if orientation == 1
                        for v in eachvariable(equations)
                            bar_states1[v, lambda_indices_large..., large_element] += weight *
                                                                                      bar_state[v] /
                                                                                      mortar_weights_sums[i_large,
                                                                                                          j_large,
                                                                                                          2]
                            bar_states1[v, lambda_indices_small..., small_element] += weight *
                                                                                      bar_state[v] /
                                                                                      mortar_weights_sums[i_small,
                                                                                                          j_small,
                                                                                                          1]
                        end
                    elseif orientation == 2
                        for v in eachvariable(equations)
                            bar_states2[v, lambda_indices_large..., large_element] += weight *
                                                                                      bar_state[v] /
                                                                                      mortar_weights_sums[i_large,
                                                                                                          j_large,
                                                                                                          2]
                            bar_states2[v, lambda_indices_small..., small_element] += weight *
                                                                                      bar_state[v] /
                                                                                      mortar_weights_sums[i_small,
                                                                                                          j_small,
                                                                                                          1]
                        end
                    else # orientation == 3
                        for v in eachvariable(equations)
                            bar_states3[v, lambda_indices_large..., large_element] += weight *
                                                                                      bar_state[v] /
                                                                                      mortar_weights_sums[i_large,
                                                                                                          j_large,
                                                                                                          2]
                            bar_states3[v, lambda_indices_small..., small_element] += weight *
                                                                                      bar_state[v] /
                                                                                      mortar_weights_sums[i_small,
                                                                                                          j_small,
                                                                                                          1]
                        end
                    end
                end
            end
        end
    end

    return nothing
end

@inline function calc_lambdas_bar_states_boundary!(u, t, limiter, boundary_conditions,
                                                   mesh::TreeMesh{3}, equations, dg,
                                                   cache; calc_bar_states = true)
    (; lambda1, lambda2, lambda3, bar_states1, bar_states2, bar_states3) = limiter.cache.container_bar_states

    @threaded for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]

        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        if orientation == 1
            if neighbor_side == 2 # Element is on the right, boundary on the left
                for k in eachnode(dg), j in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, 1, j, k, element)
                    u_outer = get_boundary_outer_state(u_inner, t,
                                                       boundary_conditions[1],
                                                       orientation, 1,
                                                       mesh, equations, dg, cache,
                                                       1, j, k, element)
                    lambda1[1, j, k, element] = max_abs_speed_naive(u_inner, u_outer,
                                                                    orientation,
                                                                    equations)

                    calc_bar_states || continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_inner - flux_outer) /
                                lambda1[1, j, k, element]
                    for v in eachvariable(equations)
                        bar_states1[v, 1, j, k, element] = bar_state[v]
                    end
                end
            else # Element is on the left, boundary on the right
                for k in eachnode(dg), j in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, nnodes(dg), j, k,
                                            element)
                    u_outer = get_boundary_outer_state(u_inner, t,
                                                       boundary_conditions[2],
                                                       orientation, 2,
                                                       mesh, equations, dg, cache,
                                                       nnodes(dg), j, k, element)
                    lambda1[nnodes(dg) + 1, j, k, element] = max_abs_speed_naive(u_inner,
                                                                                 u_outer,
                                                                                 orientation,
                                                                                 equations)

                    calc_bar_states || continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_outer - flux_inner) /
                                lambda1[nnodes(dg) + 1, j, k, element]
                    for v in eachvariable(equations)
                        bar_states1[v, nnodes(dg) + 1, j, k, element] = bar_state[v]
                    end
                end
            end
        elseif orientation == 2
            if neighbor_side == 2 # Element is on the right, boundary on the left
                for k in eachnode(dg), i in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, i, 1, k, element)
                    u_outer = get_boundary_outer_state(u_inner, t,
                                                       boundary_conditions[3],
                                                       orientation, 3,
                                                       mesh, equations, dg, cache,
                                                       i, 1, k, element)
                    lambda2[i, 1, k, element] = max_abs_speed_naive(u_inner, u_outer,
                                                                    orientation,
                                                                    equations)

                    calc_bar_states || continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_inner - flux_outer) /
                                lambda2[i, 1, k, element]
                    for v in eachvariable(equations)
                        bar_states2[v, i, 1, k, element] = bar_state[v]
                    end
                end
            else # Element is on the left, boundary on the right
                for k in eachnode(dg), i in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, i, nnodes(dg), k,
                                            element)
                    u_outer = get_boundary_outer_state(u_inner, t,
                                                       boundary_conditions[4],
                                                       orientation, 4,
                                                       mesh, equations, dg, cache,
                                                       i, nnodes(dg), k, element)
                    lambda2[i, nnodes(dg) + 1, k, element] = max_abs_speed_naive(u_inner,
                                                                                 u_outer,
                                                                                 orientation,
                                                                                 equations)

                    calc_bar_states || continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_outer - flux_inner) /
                                lambda2[i, nnodes(dg) + 1, k, element]
                    for v in eachvariable(equations)
                        bar_states2[v, i, nnodes(dg) + 1, k, element] = bar_state[v]
                    end
                end
            end
        else # orientation == 3
            if neighbor_side == 2 # Element is on the right, boundary on the left
                for j in eachnode(dg), i in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, i, j, 1, element)
                    u_outer = get_boundary_outer_state(u_inner, t,
                                                       boundary_conditions[5],
                                                       orientation, 5,
                                                       mesh, equations, dg, cache,
                                                       i, j, 1, element)
                    lambda3[i, j, 1, element] = max_abs_speed_naive(u_inner, u_outer,
                                                                    orientation,
                                                                    equations)

                    calc_bar_states || continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_inner - flux_outer) /
                                lambda3[i, j, 1, element]
                    for v in eachvariable(equations)
                        bar_states3[v, i, j, 1, element] = bar_state[v]
                    end
                end
            else # Element is on the left, boundary on the right
                for j in eachnode(dg), i in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, i, j, nnodes(dg),
                                            element)
                    u_outer = get_boundary_outer_state(u_inner, t,
                                                       boundary_conditions[6],
                                                       orientation, 6,
                                                       mesh, equations, dg, cache,
                                                       i, j, nnodes(dg), element)
                    lambda3[i, j, nnodes(dg) + 1, element] = max_abs_speed_naive(u_inner,
                                                                                 u_outer,
                                                                                 orientation,
                                                                                 equations)

                    calc_bar_states || continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_outer - flux_inner) /
                                lambda3[i, j, nnodes(dg) + 1, element]
                    for v in eachvariable(equations)
                        bar_states3[v, i, j, nnodes(dg) + 1, element] = bar_state[v]
                    end
                end
            end
        end
    end

    return nothing
end

@inline function calc_variable_bounds!(u, mesh::AbstractMesh{3}, nonconservative_terms,
                                       equations, limiter::SubcellLimiterIDP, dg, cache)
    if limiter.bar_states == false
        return nothing
    end

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    (; bar_states1, bar_states2, bar_states3) = limiter.cache.container_bar_states

    (; small_stencil) = limiter
    @assert small_stencil "small_stencil == false is not yet implemented for 3D subcell limiting"

    # Local two-sided limiting for conservative variables
    if limiter.local_twosided
        for v in limiter.local_twosided_variables_cons
            v_string = string(v)
            var_min = variable_bounds[Symbol(v_string, "_min")]
            var_max = variable_bounds[Symbol(v_string, "_max")]
            @threaded for element in eachelement(dg, cache)
                for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                    var_min[i, j, k, element] = typemax(eltype(var_min))
                    var_max[i, j, k, element] = typemin(eltype(var_max))
                end

                for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                    var_min[i, j, k, element] = min(var_min[i, j, k, element],
                                                    u[v, i, j, k, element],
                                                    bar_states1[v, i, j, k,
                                                                element],
                                                    bar_states1[v, i + 1, j, k,
                                                                element],
                                                    bar_states2[v, i, j, k,
                                                                element],
                                                    bar_states2[v, i, j + 1, k,
                                                                element],
                                                    bar_states3[v, i, j, k,
                                                                element],
                                                    bar_states3[v, i, j, k + 1,
                                                                element])
                    var_max[i, j, k, element] = max(var_max[i, j, k, element],
                                                    u[v, i, j, k, element],
                                                    bar_states1[v, i, j, k,
                                                                element],
                                                    bar_states1[v, i + 1, j, k,
                                                                element],
                                                    bar_states2[v, i, j, k,
                                                                element],
                                                    bar_states2[v, i, j + 1, k,
                                                                element],
                                                    bar_states3[v, i, j, k,
                                                                element],
                                                    bar_states3[v, i, j, k + 1,
                                                                element])
                end
            end
        end
    end

    # Local one-sided limiting for nonlinear variables
    if limiter.local_onesided
        for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
            var_minmax = variable_bounds[Symbol(string(variable), "_",
                                                string(min_or_max))]
            @threaded for element in eachelement(dg, cache)
                for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                    var = variable(get_node_vars(u, equations, dg, i, j, k, element),
                                   equations)
                    var_minmax[i, j, k, element] = var
                end

                for k in eachnode(dg), j in eachnode(dg),
                    i in 1:(nnodes(dg) + 1)

                    var = variable(get_node_vars(bar_states1, equations, dg, i,
                                                 j, k, element), equations)
                    if i < nnodes(dg) + 1
                        var_minmax[i, j, k, element] = min_or_max(var_minmax[i,
                                                                             j,
                                                                             k,
                                                                             element],
                                                                  var)
                    end
                    if i > 1
                        var_minmax[i - 1, j, k, element] = min_or_max(var_minmax[i - 1,
                                                                                 j,
                                                                                 k,
                                                                                 element],
                                                                      var)
                    end
                end

                for k in eachnode(dg), j in 1:(nnodes(dg) + 1),
                    i in eachnode(dg)

                    var = variable(get_node_vars(bar_states2, equations, dg, i,
                                                 j, k, element), equations)
                    if j < nnodes(dg) + 1
                        var_minmax[i, j, k, element] = min_or_max(var_minmax[i,
                                                                             j,
                                                                             k,
                                                                             element],
                                                                  var)
                    end
                    if j > 1
                        var_minmax[i, j - 1, k, element] = min_or_max(var_minmax[i,
                                                                                 j - 1,
                                                                                 k,
                                                                                 element],
                                                                      var)
                    end
                end

                for k in 1:(nnodes(dg) + 1), j in eachnode(dg),
                    i in eachnode(dg)

                    var = variable(get_node_vars(bar_states3, equations, dg, i,
                                                 j, k, element), equations)
                    if k < nnodes(dg) + 1
                        var_minmax[i, j, k, element] = min_or_max(var_minmax[i,
                                                                             j,
                                                                             k,
                                                                             element],
                                                                  var)
                    end
                    if k > 1
                        var_minmax[i, j, k - 1, element] = min_or_max(var_minmax[i,
                                                                                 j,
                                                                                 k - 1,
                                                                                 element],
                                                                      var)
                    end
                end
            end
        end
    end

    return nothing
end
end # @muladd
