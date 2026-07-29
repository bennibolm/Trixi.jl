# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**without non-conservative terms**).
#
# See also `flux_differencing_kernel!`.
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                                u, ::Type{<:P4estMesh{3}},
                                nonconservative_terms::False, equations,
                                volume_flux, dg::DGSEM, element, cache)
    (; contravariant_vectors) = cache.elements
    (; weights, derivative_split) = dg.basis
    (; flux_temp_threaded) = cache

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

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors,
                                            i, j, k, element) # x direction

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   ii, j, k, element)
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)

            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], fluxtilde1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], fluxtilde1,
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

        # pull the contravariant vectors in each coordinate direction
        Ja2_node = get_contravariant_vector(2, contravariant_vectors,
                                            i, j, k, element)

        # y direction
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, jj, k, element)
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], fluxtilde2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], fluxtilde2,
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

        # pull the contravariant vectors in each coordinate direction
        Ja3_node = get_contravariant_vector(3, contravariant_vectors,
                                            i, j, k, element)

        # z direction
        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            # pull the contravariant vectors and compute the average
            Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                   i, j, kk, element)
            Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde3 = volume_flux(u_node, u_node_kk, Ja3_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], fluxtilde3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], fluxtilde3,
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
                                u, ::Type{<:P4estMesh{3}},
                                nonconservative_terms::True, equations,
                                volume_flux::Tuple{F_CONS, F_NONCONS}, dg::DGSEM,
                                element,
                                cache) where {
                                              F_CONS <: Function,
                                              F_NONCONS <:
                                              FluxNonConservative{NonConservativeSymmetric()}
                                              }
    (; contravariant_vectors) = cache.elements
    (; weights, derivative_split) = dg.basis
    (; flux_temp_threaded, flux_nonconservative_temp_threaded) = cache
    (; fhat_temp_threaded, fhat_nonconservative_temp_threaded, phi_threaded) = cache

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

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors,
                                            i, j, k, element) # x direction

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   ii, j, k, element)
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)

            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde1 = volume_flux_cons(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], fluxtilde1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], fluxtilde1,
                                       equations, dg, ii, j, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, Ja1_avg,
                                                    equations,
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
        # pull the local contravariant vector
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja1_node, equations,
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

        # pull the contravariant vectors in each coordinate direction
        Ja2_node = get_contravariant_vector(2, contravariant_vectors,
                                            i, j, k, element)

        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, jj, k, element)
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde2 = volume_flux_cons(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], fluxtilde2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], fluxtilde2,
                                       equations, dg, i, jj, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, Ja2_avg,
                                                    equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[jj, j],
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
        # pull the local contravariant vector
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja2_node, equations,
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

        # pull the contravariant vectors in each coordinate direction
        Ja3_node = get_contravariant_vector(3, contravariant_vectors,
                                            i, j, k, element)

        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            # pull the contravariant vectors and compute the average
            Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                   i, j, kk, element)
            Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde3 = volume_flux_cons(u_node, u_node_kk, Ja3_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], fluxtilde3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], fluxtilde3,
                                       equations, dg, i, j, kk)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux3_noncons = volume_flux_noncons(u_node, u_node_kk, Ja3_avg,
                                                    equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[k, kk],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[kk, k],
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
        # pull the local contravariant vector
        Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja3_node, equations,
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
                                u, mesh::P4estMesh{3},
                                nonconservative_terms::True, equations,
                                volume_flux::Tuple{F_CONS, F_NONCONS}, dg::DGSEM,
                                element,
                                cache) where {
                                              F_CONS <: Function,
                                              F_NONCONS <:
                                              FluxNonConservative{NonConservativeJump()}
                                              }
    (; contravariant_vectors) = cache.elements
    (; weights, derivative_split) = dg.basis
    (; flux_temp_threaded, flux_nonconservative_temp_threaded) = cache
    (; fhat_temp_threaded, fhat_nonconservative_temp_threaded, phi_threaded) = cache

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

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors,
                                            i, j, k, element) # x direction

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   ii, j, k, element)
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)

            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde1 = volume_flux_cons(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], fluxtilde1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], fluxtilde1,
                                       equations, dg, ii, j, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, Ja1_avg,
                                                    equations,
                                                    NonConservativeJump(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[i, ii],
                                           flux1_noncons,
                                           equations, dg, noncons, i, j, k)
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
        # pull the local contravariant vector
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja1_node, equations,
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
        Ja1_node_0 = get_contravariant_vector(1, contravariant_vectors,
                                              1, j, k, element)

        for i in 2:(nnodes(dg) - 1)
            u_i = get_node_vars(u, equations, dg, i, j, k, element)
            Ja1_node_i = get_contravariant_vector(1, contravariant_vectors,
                                                  i, j, k, element)
            Ja1_avg = 0.5f0 * (Ja1_node_0 + Ja1_node_i)

            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_i, Ja1_avg, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat1_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat1_L[v, i + 1, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, nnodes(dg), j, k, element)
        Ja1_node_N = get_contravariant_vector(1, contravariant_vectors,
                                              nnodes(dg), j, k, element)
        Ja1_avg = 0.5f0 * (Ja1_node_0 + Ja1_node_N)

        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, Ja1_avg, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing because Trixi multiplies all the non-cons terms with 0.5
                fhat1_R[v, nnodes(dg), j, k] -= phi[v, noncons, nnodes(dg), j, k] *
                                                phi_jump[v]
            end
        end
    end

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja2_node = get_contravariant_vector(2, contravariant_vectors,
                                            i, j, k, element)

        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, jj, k, element)
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde2 = volume_flux_cons(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], fluxtilde2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], fluxtilde2,
                                       equations, dg, i, jj, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, Ja2_avg,
                                                    equations,
                                                    NonConservativeJump(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5f0 * derivative_split[jj, j],
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
        # pull the local contravariant vector
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja2_node, equations,
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
        Ja2_node_0 = get_contravariant_vector(2, contravariant_vectors,
                                              i, 1, k, element)

        for j in 2:(nnodes(dg) - 1)
            u_j = get_node_vars(u, equations, dg, i, j, k, element)
            Ja2_node_j = get_contravariant_vector(2, contravariant_vectors,
                                                  i, j, k, element)
            Ja2_avg = 0.5f0 * (Ja2_node_0 + Ja2_node_j)

            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_j, Ja2_avg, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat2_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat2_L[v, i, j + 1, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, i, nnodes(dg), k, element)
        Ja2_node_N = get_contravariant_vector(2, contravariant_vectors,
                                              i, nnodes(dg), k, element)
        Ja2_avg = 0.5f0 * (Ja2_node_0 + Ja2_node_N)

        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, Ja2_avg, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
                fhat2_R[v, i, nnodes(dg), k] -= phi[v, noncons, i, nnodes(dg), k] *
                                                phi_jump[v]
            end
        end
    end

    # Split form volume flux in orientation 3: z direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja3_node = get_contravariant_vector(3, contravariant_vectors,
                                            i, j, k, element)

        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            # pull the contravariant vectors and compute the average
            Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                   i, j, kk, element)
            Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde3 = volume_flux_cons(u_node, u_node_kk, Ja3_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], fluxtilde3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], fluxtilde3,
                                       equations, dg, i, j, kk)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux3_noncons = volume_flux_noncons(u_node, u_node_kk, Ja3_avg,
                                                    equations,
                                                    NonConservativeJump(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[k, kk],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5f0 * derivative_split[kk, k],
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
        # pull the local contravariant vector
        Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja3_node, equations,
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
        Ja3_node_0 = get_contravariant_vector(3, contravariant_vectors,
                                              i, j, 1, element)

        for k in 2:(nnodes(dg) - 1)
            u_k = get_node_vars(u, equations, dg, i, j, k, element)
            Ja3_node_k = get_contravariant_vector(3, contravariant_vectors,
                                                  i, j, k, element)
            Ja3_avg = 0.5f0 * (Ja3_node_0 + Ja3_node_k)

            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_k, Ja3_avg, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat3_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat3_L[v, i, j, k + 1] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, i, j, nnodes(dg), element)
        Ja3_node_N = get_contravariant_vector(3, contravariant_vectors,
                                              i, j, nnodes(dg), element)
        Ja3_avg = 0.5f0 * (Ja3_node_0 + Ja3_node_N)

        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, Ja3_avg, equations,
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

function prolong2mortars!(cache, u, mesh::P4estMesh{3}, equations,
                          mortar_idp::LobattoLegendreMortarIDP, dg::DGSEM)
    prolong2mortars!(cache, u, mesh, equations, mortar_idp.mortar_l2, dg)

    (; neighbor_ids, node_indices, u_large) = cache.mortars
    index_range = eachnode(dg)

    # The data of all four small elements were already copied to the mortar cache
    @threaded for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[5, mortar]

        # Copy solutions data from large element using "delayed indexing" with
        # a start value and two step sizes to get the correct face and orientation.
        large_indices = node_indices[2, mortar]

        i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_indices[1],
                                                                               index_range)
        j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_indices[2],
                                                                               index_range)
        k_large_start, k_large_step_i, k_large_step_j = index_to_start_step_3d(large_indices[3],
                                                                               index_range)

        i_large = i_large_start
        j_large = j_large_start
        k_large = k_large_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    u_large[v, i, j, mortar] = u[v, i_large, j_large, k_large,
                                                 large_element]
                end
                i_large += i_large_step_i
                j_large += j_large_step_i
                k_large += k_large_step_i
            end
            i_large += i_large_step_j
            j_large += j_large_step_j
            k_large += k_large_step_j
        end
    end

    return nothing
end

@inline function calc_lambdas_bar_states!(u, t, mesh::P4estMesh{3},
                                          have_nonconservative_terms, equations,
                                          limiter, dg, cache, boundary_conditions;
                                          calc_bar_states = true)
    if limiter.bar_states == false
        return nothing
    end

    (; lambda1, lambda2, lambda3, bar_states1, bar_states2, bar_states3) = limiter.cache.container_bar_states
    (; normal_vectors_1, normal_vectors_2, normal_vectors_3) = cache.normal_vectors

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
            # Fetch precomputed freestream-preserving normal vector
            # We access i - 1 here since the normal vector for i = 1 is not used and stored
            normal_direction = get_normal_vector(normal_vectors_1, i - 1, j, k, element)
            lambda1[i, j, k, element] = max_abs_speed_naive(u_node_im1, u_node,
                                                            normal_direction,
                                                            equations)

            calc_bar_states || continue

            flux1 = flux(u_node, normal_direction, equations)
            flux1_im1 = flux(u_node_im1, normal_direction, equations)
            for v in eachvariable(equations)
                bar_states1[v, i, j, k, element] = 0.5 * (u_node[v] + u_node_im1[v]) -
                                                   0.5 * (flux1[v] - flux1_im1[v]) /
                                                   lambda1[i, j, k, element]
            end
        end

        for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            u_node_jm1 = get_node_vars(u, equations, dg, i, j - 1, k, element)
            # Fetch precomputed freestream-preserving normal vector
            # We access j - 1 here since the normal vector for j = 1 is not used and stored
            normal_direction = get_normal_vector(normal_vectors_2, i, j - 1, k, element)
            lambda2[i, j, k, element] = max_abs_speed_naive(u_node_jm1, u_node,
                                                            normal_direction,
                                                            equations)

            calc_bar_states || continue

            flux2 = flux(u_node, normal_direction, equations)
            flux2_jm1 = flux(u_node_jm1, normal_direction, equations)
            for v in eachvariable(equations)
                bar_states2[v, i, j, k, element] = 0.5 * (u_node[v] + u_node_jm1[v]) -
                                                   0.5 * (flux2[v] - flux2_jm1[v]) /
                                                   lambda2[i, j, k, element]
            end
        end

        for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            u_node_km1 = get_node_vars(u, equations, dg, i, j, k - 1, element)
            # Fetch precomputed freestream-preserving normal vector
            # We access k - 1 here since the normal vector for k = 1 is not used and stored
            normal_direction = get_normal_vector(normal_vectors_3, i, j, k - 1, element)
            lambda3[i, j, k, element] = max_abs_speed_naive(u_node_km1, u_node,
                                                            normal_direction,
                                                            equations)

            calc_bar_states || continue

            flux3 = flux(u_node, normal_direction, equations)
            flux3_km1 = flux(u_node_km1, normal_direction, equations)
            for v in eachvariable(equations)
                bar_states3[v, i, j, k, element] = 0.5 * (u_node[v] + u_node_km1[v]) -
                                                   0.5 * (flux3[v] - flux3_km1[v]) /
                                                   lambda3[i, j, k, element]
            end
        end
    end

    calc_lambdas_bar_states_interface!(u, t, limiter, boundary_conditions, mesh,
                                       equations, dg, cache;
                                       calc_bar_states = calc_bar_states)
    calc_lambdas_bar_states_mortar!(u, t, limiter, boundary_conditions, mesh, equations,
                                    dg, cache; calc_bar_states = calc_bar_states)
    calc_lambdas_bar_states_boundary!(u, t, limiter, boundary_conditions, mesh,
                                      equations, dg, cache;
                                      calc_bar_states = calc_bar_states)

    return nothing
end

@inline function calc_lambdas_bar_states_interface!(u, t, limiter, boundary_conditions,
                                                    mesh::P4estMesh{3}, equations, dg,
                                                    cache; calc_bar_states = true)
    (; lambda1, lambda2, lambda3, bar_states1, bar_states2, bar_states3) = limiter.cache.container_bar_states
    (; contravariant_vectors) = cache.elements
    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)

    @threaded for interface in eachinterface(dg, cache)
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]
        primary_direction = indices2direction(primary_indices)

        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]

        i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                     index_range)
        j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                     index_range)
        k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                     index_range)

        i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_indices[1],
                                                                                           index_range)
        j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_indices[2],
                                                                                           index_range)
        k_secondary_start, k_secondary_step_i, k_secondary_step_j = index_to_start_step_3d(secondary_indices[3],
                                                                                           index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        k_primary = k_primary_start
        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        k_secondary = k_secondary_start

        for j in eachnode(dg)
            for i in eachnode(dg)
                normal_direction = get_normal_direction(primary_direction,
                                                        contravariant_vectors,
                                                        i_primary, j_primary,
                                                        k_primary, primary_element)
                u_primary = get_node_vars(u, equations, dg, i_primary, j_primary,
                                          k_primary, primary_element)
                u_secondary = get_node_vars(u, equations, dg, i_secondary,
                                            j_secondary, k_secondary,
                                            secondary_element)
                lambda = max_abs_speed_naive(u_primary, u_secondary, normal_direction,
                                             equations)

                if primary_direction == 1
                    lambda1[i_primary, j_primary, k_primary, primary_element] = lambda
                    lambda1[i_secondary + 1, j_secondary, k_secondary,
                    secondary_element] = lambda
                elseif primary_direction == 2
                    lambda1[i_primary + 1, j_primary, k_primary, primary_element] = lambda
                    lambda1[i_secondary, j_secondary, k_secondary,
                    secondary_element] = lambda
                elseif primary_direction == 3
                    lambda2[i_primary, j_primary, k_primary, primary_element] = lambda
                    lambda2[i_secondary, j_secondary + 1, k_secondary,
                    secondary_element] = lambda
                elseif primary_direction == 4
                    lambda2[i_primary, j_primary + 1, k_primary, primary_element] = lambda
                    lambda2[i_secondary, j_secondary, k_secondary,
                    secondary_element] = lambda
                elseif primary_direction == 5
                    lambda3[i_primary, j_primary, k_primary, primary_element] = lambda
                    lambda3[i_secondary, j_secondary, k_secondary + 1,
                    secondary_element] = lambda
                else # primary_direction == 6
                    lambda3[i_primary, j_primary, k_primary + 1, primary_element] = lambda
                    lambda3[i_secondary, j_secondary, k_secondary,
                    secondary_element] = lambda
                end

                if calc_bar_states
                    flux_primary = flux(u_primary, normal_direction, equations)
                    flux_secondary = flux(u_secondary, normal_direction, equations)
                    bar_state = 0.5 * (u_primary + u_secondary) -
                                0.5 * (flux_secondary - flux_primary) / lambda

                    if primary_direction == 1
                        set_node_vars!(bar_states1, bar_state, equations, dg,
                                       i_primary, j_primary, k_primary,
                                       primary_element)
                        set_node_vars!(bar_states1, bar_state, equations, dg,
                                       i_secondary + 1, j_secondary, k_secondary,
                                       secondary_element)
                    elseif primary_direction == 2
                        set_node_vars!(bar_states1, bar_state, equations, dg,
                                       i_primary + 1, j_primary, k_primary,
                                       primary_element)
                        set_node_vars!(bar_states1, bar_state, equations, dg,
                                       i_secondary, j_secondary, k_secondary,
                                       secondary_element)
                    elseif primary_direction == 3
                        set_node_vars!(bar_states2, bar_state, equations, dg,
                                       i_primary, j_primary, k_primary,
                                       primary_element)
                        set_node_vars!(bar_states2, bar_state, equations, dg,
                                       i_secondary, j_secondary + 1, k_secondary,
                                       secondary_element)
                    elseif primary_direction == 4
                        set_node_vars!(bar_states2, bar_state, equations, dg,
                                       i_primary, j_primary + 1, k_primary,
                                       primary_element)
                        set_node_vars!(bar_states2, bar_state, equations, dg,
                                       i_secondary, j_secondary, k_secondary,
                                       secondary_element)
                    elseif primary_direction == 5
                        set_node_vars!(bar_states3, bar_state, equations, dg,
                                       i_primary, j_primary, k_primary,
                                       primary_element)
                        set_node_vars!(bar_states3, bar_state, equations, dg,
                                       i_secondary, j_secondary, k_secondary + 1,
                                       secondary_element)
                    else # primary_direction == 6
                        set_node_vars!(bar_states3, bar_state, equations, dg,
                                       i_primary, j_primary, k_primary + 1,
                                       primary_element)
                        set_node_vars!(bar_states3, bar_state, equations, dg,
                                       i_secondary, j_secondary, k_secondary,
                                       secondary_element)
                    end
                end

                i_primary += i_primary_step_i
                j_primary += j_primary_step_i
                k_primary += k_primary_step_i
                i_secondary += i_secondary_step_i
                j_secondary += j_secondary_step_i
                k_secondary += k_secondary_step_i
            end
            i_primary += i_primary_step_j
            j_primary += j_primary_step_j
            k_primary += k_primary_step_j
            i_secondary += i_secondary_step_j
            j_secondary += j_secondary_step_j
            k_secondary += k_secondary_step_j
        end
    end

    return nothing
end

@inline function calc_lambdas_bar_states_mortar!(u, t, limiter, boundary_conditions,
                                                 mesh::P4estMesh{3}, equations,
                                                 dg, cache; calc_bar_states = true)
    (; lambda1, lambda2, lambda3, bar_states1, bar_states2, bar_states3) = limiter.cache.container_bar_states
    if nmortars(dg, cache) == 0
        return nothing
    end

    (; mortar_weights, mortar_weights_sums) = dg.mortar
    (; neighbor_ids, node_indices) = cache.mortars
    (; contravariant_vectors) = cache.elements
    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)
        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        large_element = neighbor_ids[5, mortar]
        large_indices = node_indices[2, mortar]
        i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_indices[1],
                                                                               index_range)
        j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_indices[2],
                                                                               index_range)
        k_large_start, k_large_step_i, k_large_step_j = index_to_start_step_3d(large_indices[3],
                                                                               index_range)

        for small_element_index in 1:4
            small_element = neighbor_ids[small_element_index, mortar]

            i_small = i_small_start
            j_small = j_small_start
            k_small = k_small_start
            for j_small_node in eachnode(dg)
                for i_small_node in eachnode(dg)
                    i_mortar_s, j_mortar_s = get_mortar_index(small_indices,
                                                              i_small, j_small, k_small)

                    u_small = get_node_vars(u, equations, dg, i_small, j_small,
                                            k_small, small_element)
                    normal_direction = get_normal_direction(small_direction,
                                                            contravariant_vectors,
                                                            i_small, j_small, k_small,
                                                            small_element)

                    i_large = i_large_start
                    j_large = j_large_start
                    k_large = k_large_start
                    for j_large_node in eachnode(dg)
                        for i_large_node in eachnode(dg)
                            i_mortar_l, j_mortar_l = get_mortar_index(large_indices,
                                                                      i_large,
                                                                      j_large,
                                                                      k_large)

                            weight = mortar_weights[i_mortar_l, j_mortar_l,
                                                    i_mortar_s, j_mortar_s,
                                                    small_element_index]
                            if iszero(weight)
                                i_large += i_large_step_i
                                j_large += j_large_step_i
                                k_large += k_large_step_i
                                continue
                            end

                            u_large = get_node_vars(u, equations, dg, i_large,
                                                    j_large, k_large, large_element)
                            lambda = max_abs_speed_naive(u_small, u_large,
                                                         normal_direction, equations)

                            lambda_small_factor = weight /
                                                  mortar_weights_sums[i_mortar_s,
                                                                      j_mortar_s, 1]
                            lambda_large_factor = weight /
                                                  mortar_weights_sums[i_mortar_l,
                                                                      j_mortar_l, 2]

                            if small_direction == 1
                                lambda1[i_small, j_small, k_small, small_element] += lambda_small_factor *
                                                                                     lambda
                                lambda1[i_large + 1, j_large, k_large, large_element] += lambda_large_factor *
                                                                                         lambda
                            elseif small_direction == 2
                                lambda1[i_small + 1, j_small, k_small, small_element] += lambda_small_factor *
                                                                                         lambda
                                lambda1[i_large, j_large, k_large, large_element] += lambda_large_factor *
                                                                                     lambda
                            elseif small_direction == 3
                                lambda2[i_small, j_small, k_small, small_element] += lambda_small_factor *
                                                                                     lambda
                                lambda2[i_large, j_large + 1, k_large, large_element] += lambda_large_factor *
                                                                                         lambda
                            elseif small_direction == 4
                                lambda2[i_small, j_small + 1, k_small, small_element] += lambda_small_factor *
                                                                                         lambda
                                lambda2[i_large, j_large, k_large, large_element] += lambda_large_factor *
                                                                                     lambda
                            elseif small_direction == 5
                                lambda3[i_small, j_small, k_small, small_element] += lambda_small_factor *
                                                                                     lambda
                                lambda3[i_large, j_large, k_large + 1, large_element] += lambda_large_factor *
                                                                                         lambda
                            else # small_direction == 6
                                lambda3[i_small, j_small, k_small + 1, small_element] += lambda_small_factor *
                                                                                         lambda
                                lambda3[i_large, j_large, k_large, large_element] += lambda_large_factor *
                                                                                     lambda
                            end

                            if calc_bar_states
                                flux_small = flux(u_small, normal_direction, equations)
                                flux_large = flux(u_large, normal_direction, equations)
                                bar_state = 0.5 * (u_small + u_large) -
                                            0.5 * (flux_large - flux_small) / lambda

                                if small_direction == 1
                                    for v in eachvariable(equations)
                                        bar_states1[v, i_small, j_small, k_small, small_element] += lambda_small_factor *
                                                                                                    bar_state[v]
                                        bar_states1[v, i_large + 1, j_large, k_large, large_element] += lambda_large_factor *
                                                                                                        bar_state[v]
                                    end
                                elseif small_direction == 2
                                    for v in eachvariable(equations)
                                        bar_states1[v, i_small + 1, j_small, k_small, small_element] += lambda_small_factor *
                                                                                                        bar_state[v]
                                        bar_states1[v, i_large, j_large, k_large, large_element] += lambda_large_factor *
                                                                                                    bar_state[v]
                                    end
                                elseif small_direction == 3
                                    for v in eachvariable(equations)
                                        bar_states2[v, i_small, j_small, k_small, small_element] += lambda_small_factor *
                                                                                                    bar_state[v]
                                        bar_states2[v, i_large, j_large + 1, k_large, large_element] += lambda_large_factor *
                                                                                                        bar_state[v]
                                    end
                                elseif small_direction == 4
                                    for v in eachvariable(equations)
                                        bar_states2[v, i_small, j_small + 1, k_small, small_element] += lambda_small_factor *
                                                                                                        bar_state[v]
                                        bar_states2[v, i_large, j_large, k_large, large_element] += lambda_large_factor *
                                                                                                    bar_state[v]
                                    end
                                elseif small_direction == 5
                                    for v in eachvariable(equations)
                                        bar_states3[v, i_small, j_small, k_small, small_element] += lambda_small_factor *
                                                                                                    bar_state[v]
                                        bar_states3[v, i_large, j_large, k_large + 1, large_element] += lambda_large_factor *
                                                                                                        bar_state[v]
                                    end
                                else # small_direction == 6
                                    for v in eachvariable(equations)
                                        bar_states3[v, i_small, j_small, k_small + 1, small_element] += lambda_small_factor *
                                                                                                        bar_state[v]
                                        bar_states3[v, i_large, j_large, k_large, large_element] += lambda_large_factor *
                                                                                                    bar_state[v]
                                    end
                                end
                            end

                            i_large += i_large_step_i
                            j_large += j_large_step_i
                            k_large += k_large_step_i
                        end
                        i_large += i_large_step_j
                        j_large += j_large_step_j
                        k_large += k_large_step_j
                    end

                    i_small += i_small_step_i
                    j_small += j_small_step_i
                    k_small += k_small_step_i
                end
                i_small += i_small_step_j
                j_small += j_small_step_j
                k_small += k_small_step_j
            end
        end
    end

    return nothing
end

@inline function calc_lambdas_bar_states_boundary!(u, t, limiter,
                                                   boundary_conditions::BoundaryConditionPeriodic,
                                                   mesh::P4estMesh{3}, equations, dg,
                                                   cache; calc_bar_states = true)
    return nothing
end

@inline function calc_lambdas_bar_states_boundary!(u, t, limiter, boundary_conditions,
                                                   mesh::P4estMesh{3}, equations, dg,
                                                   cache; calc_bar_states = true)
    (; lambda1, lambda2, lambda3, bar_states1, bar_states2, bar_states3) = limiter.cache.container_bar_states
    (; boundary_condition_types, boundary_indices) = boundary_conditions
    (; contravariant_vectors) = cache.elements
    (; boundaries) = cache
    index_range = eachnode(dg)

    foreach_enumerate(boundary_condition_types) do (i, boundary_condition)
        for boundary in boundary_indices[i]
            element = boundaries.neighbor_ids[boundary]
            node_indices = boundaries.node_indices[boundary]
            direction = indices2direction(node_indices)

            i_node_start, i_node_step_i, i_node_step_j = index_to_start_step_3d(node_indices[1],
                                                                                index_range)
            j_node_start, j_node_step_i, j_node_step_j = index_to_start_step_3d(node_indices[2],
                                                                                index_range)
            k_node_start, k_node_step_i, k_node_step_j = index_to_start_step_3d(node_indices[3],
                                                                                index_range)

            i_node = i_node_start
            j_node = j_node_start
            k_node = k_node_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    normal_direction = get_normal_direction(direction,
                                                            contravariant_vectors,
                                                            i_node, j_node, k_node,
                                                            element)
                    u_inner = get_node_vars(u, equations, dg, i_node, j_node, k_node,
                                            element)
                    u_outer = get_boundary_outer_state(u_inner, t, boundary_condition,
                                                       normal_direction,
                                                       mesh, equations, dg, cache,
                                                       i_node, j_node, k_node, element)
                    lambda = max_abs_speed_naive(u_inner, u_outer, normal_direction,
                                                 equations)

                    if direction == 1
                        lambda1[i_node, j_node, k_node, element] = lambda
                    elseif direction == 2
                        lambda1[i_node + 1, j_node, k_node, element] = lambda
                    elseif direction == 3
                        lambda2[i_node, j_node, k_node, element] = lambda
                    elseif direction == 4
                        lambda2[i_node, j_node + 1, k_node, element] = lambda
                    elseif direction == 5
                        lambda3[i_node, j_node, k_node, element] = lambda
                    else # direction == 6
                        lambda3[i_node, j_node, k_node + 1, element] = lambda
                    end

                    if calc_bar_states
                        flux_inner = flux(u_inner, normal_direction, equations)
                        flux_outer = flux(u_outer, normal_direction, equations)
                        bar_state = 0.5 * (u_inner + u_outer) -
                                    0.5 * (flux_outer - flux_inner) / lambda

                        if direction == 1
                            set_node_vars!(bar_states1, bar_state, equations, dg,
                                           i_node, j_node, k_node, element)
                        elseif direction == 2
                            set_node_vars!(bar_states1, bar_state, equations, dg,
                                           i_node + 1, j_node, k_node, element)
                        elseif direction == 3
                            set_node_vars!(bar_states2, bar_state, equations, dg,
                                           i_node, j_node, k_node, element)
                        elseif direction == 4
                            set_node_vars!(bar_states2, bar_state, equations, dg,
                                           i_node, j_node + 1, k_node, element)
                        elseif direction == 5
                            set_node_vars!(bar_states3, bar_state, equations, dg,
                                           i_node, j_node, k_node, element)
                        else # direction == 6
                            set_node_vars!(bar_states3, bar_state, equations, dg,
                                           i_node, j_node, k_node + 1, element)
                        end
                    end

                    i_node += i_node_step_i
                    j_node += j_node_step_i
                    k_node += k_node_step_i
                end
                i_node += i_node_step_j
                j_node += j_node_step_j
                k_node += k_node_step_j
            end
        end
    end

    return nothing
end

function calc_mortar_flux_low_order!(surface_flux_values,
                                     mesh::P4estMesh{3},
                                     nonconservative_terms::False, equations,
                                     mortar_idp::LobattoLegendreMortarIDP,
                                     surface_integral, dg::DG, cache)
    (; surface_flux) = surface_integral
    (; elements, mortars) = cache
    (; neighbor_ids, node_indices, u_large) = mortars
    (; contravariant_vectors) = elements
    (; mortar_weights, mortar_weights_sums) = mortar_idp
    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)
        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        large_indices = node_indices[2, mortar]
        large_direction = indices2direction(large_indices)
        i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_indices[1],
                                                                               index_range)
        j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_indices[2],
                                                                               index_range)
        k_large_start, k_large_step_i, k_large_step_j = index_to_start_step_3d(large_indices[3],
                                                                               index_range)

        for small_element_index in 1:4
            small_element = neighbor_ids[small_element_index, mortar]
            surface_flux_values[:, :, :, small_direction, small_element] .= zero(eltype(surface_flux_values))
        end
        large_element = neighbor_ids[5, mortar]
        surface_flux_values[:, :, :, large_direction, large_element] .= zero(eltype(surface_flux_values))

        i_small = i_small_start
        j_small = j_small_start
        k_small = k_small_start
        for j_small_node in eachnode(dg)
            for i_small_node in eachnode(dg)
                i_mortar_s, j_mortar_s = get_mortar_index(small_indices,
                                                          i_small, j_small, k_small)

                for small_element_index in 1:4
                    small_element = neighbor_ids[small_element_index, mortar]

                    u_small_local, _ = get_surface_node_vars(mortars.u, equations, dg,
                                                             small_element_index,
                                                             i_small_node,
                                                             j_small_node, mortar)

                    # Get the normal direction on the small element.
                    # Note, contravariant vectors at interfaces in negative coordinate direction
                    # are pointing inwards. This is handled by `get_normal_direction`.
                    normal_direction_small = get_normal_direction(small_direction,
                                                                  contravariant_vectors,
                                                                  i_small, j_small,
                                                                  k_small,
                                                                  small_element)

                    i_large = i_large_start
                    j_large = j_large_start
                    k_large = k_large_start
                    for j_large_node in eachnode(dg)
                        for i_large_node in eachnode(dg)
                            i_mortar_l, j_mortar_l = get_mortar_index(large_indices,
                                                                      i_large,
                                                                      j_large,
                                                                      k_large)

                            factor = mortar_weights[i_mortar_l, j_mortar_l,
                                                    i_mortar_s, j_mortar_s,
                                                    small_element_index]
                            if !isapprox(factor, zero(typeof(factor)))
                                u_large_local = get_node_vars(u_large, equations, dg,
                                                              i_large_node,
                                                              j_large_node, mortar)
                                # TODO: Use normal vector of large element for actual curved elements
                                # normal_direction_large = get_normal_direction(large_direction,
                                #                                               contravariant_vectors,
                                #                                               i_large,
                                #                                               j_large,
                                #                                               k_large,
                                #                                               large_element)

                                flux = surface_flux(u_small_local, u_large_local,
                                                    normal_direction_small, equations)

                                # Add flux to small element
                                multiply_add_to_node_vars!(surface_flux_values,
                                                           factor /
                                                           mortar_weights_sums[i_mortar_s,
                                                                               j_mortar_s,
                                                                               1],
                                                           flux, equations, dg,
                                                           i_small_node, j_small_node,
                                                           small_direction,
                                                           small_element)

                                # Add flux to large element
                                # The flux is calculated in the outward direction of the small elements,
                                # so the sign must be switched to get the flux in outward direction
                                # of the large element.
                                # The contravariant vectors of the large element (and therefore the normal
                                # vectors of the large element as well) are four times as large as the
                                # contravariant vectors of the small elements. Therefore, the flux needs
                                # to be scaled by a factor of 4 to obtain the flux of the large element.
                                multiply_add_to_node_vars!(surface_flux_values,
                                                           -4 * factor /
                                                           mortar_weights_sums[i_mortar_l,
                                                                               j_mortar_l,
                                                                               2],
                                                           flux, equations, dg,
                                                           i_large_node, j_large_node,
                                                           large_direction,
                                                           large_element)
                            end

                            i_large += i_large_step_i
                            j_large += j_large_step_i
                            k_large += k_large_step_i
                        end
                        i_large += i_large_step_j
                        j_large += j_large_step_j
                        k_large += k_large_step_j
                    end
                end
                i_small += i_small_step_i
                j_small += j_small_step_i
                k_small += k_small_step_i
            end
            i_small += i_small_step_j
            j_small += j_small_step_j
            k_small += k_small_step_j
        end
    end

    return nothing
end
end # @muladd
