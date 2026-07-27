# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function perform_idp_correction!(u, dt,
                                 mesh::Union{TreeMesh{3}, P4estMesh{3}},
                                 equations, dg, limiter::SubcellLimiterIDP, cache)
    @unpack inverse_weights = dg.basis # Plays role of inverse DG-subcell sizes
    @unpack antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R = cache.antidiffusive_fluxes
    @unpack alpha = limiter.cache.subcell_limiter_coefficients

    # The following code implements the IDP correction in flux-differencing form:
    # u[v, i, j, k, element] += dt * -inverse_jacobian[i, j, k, element] *
    #    (inverse_weights[i] *
    #       ((1 - alpha_1_ip1) * antidiffusive_flux1_ip1[v] - (1 - alpha_1) * antidiffusive_flux1[v]) +
    #     inverse_weights[j] *
    #       ((1 - alpha_2_jp1) * antidiffusive_flux2_jp1[v] - (1 - alpha_2) * antidiffusive_flux2[v]) +
    #     inverse_weights[k] *
    #       ((1 - alpha_3_kp1) * antidiffusive_flux3_kp1[v] - (1 - alpha_3) * antidiffusive_flux3[v]))
    # with
    # alpha_1 = max(alpha[i - 1, j, k, element], alpha[i, j, k, element]),
    # alpha_1_ip1 = max(alpha[i, j, k, element], alpha[i + 1, j, k, element])
    # and equivalently for alpha_2, alpha_2_jp1, alpha_3, alpha_3_kp1.

    # For LGL nodes, the high-order and low-order fluxes at element interfaces are equal
    # and therefore, the antidiffusive fluxes are zero there.
    # To avoid adding zeros and speed up the simulation, we directly loop over the subcell
    # interfaces.

    @threaded for element in eachelement(dg, cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        # Perform correction in 1st/x-direction
        for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)
            # Subcell interface between nodes (i - 1, j, k) and (i, j, k)
            alpha1 = max(alpha[i - 1, j, k, element], alpha[i, j, k, element])

            # Apply to right node (i, j, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, k, element)
            flux1 = get_node_vars(antidiffusive_flux1_R, equations, dg,
                                  i, j, k, element)
            dg_factor = -dt * inverse_jacobian * inverse_weights[i] * (1 - alpha1)
            multiply_add_to_node_vars!(u, dg_factor, flux1,
                                       equations, dg, i, j, k, element)

            # Apply to left node (i - 1, j, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i - 1, j, k, element)
            flux1_ip1 = get_node_vars(antidiffusive_flux1_L, equations, dg,
                                      i, j, k, element)
            dg_factor = dt * inverse_jacobian * inverse_weights[i - 1] * (1 - alpha1)
            multiply_add_to_node_vars!(u, dg_factor, flux1_ip1,
                                       equations, dg, i - 1, j, k, element)
        end

        # Perform correction in 2nd/y-direction
        for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
            # Subcell interface between nodes (i, j - 1, k) and (i, j, k)
            alpha2 = max(alpha[i, j - 1, k, element], alpha[i, j, k, element])

            # Apply to right node (i, j, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, k, element)
            flux2 = get_node_vars(antidiffusive_flux2_R, equations, dg,
                                  i, j, k, element)
            dg_factor = -dt * inverse_jacobian * inverse_weights[j] * (1 - alpha2)
            multiply_add_to_node_vars!(u, dg_factor, flux2,
                                       equations, dg, i, j, k, element)

            # Apply to left node (i, j - 1, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j - 1, k, element)
            flux2_jp1 = get_node_vars(antidiffusive_flux2_L, equations, dg,
                                      i, j, k, element)
            dg_factor = dt * inverse_jacobian * inverse_weights[j - 1] * (1 - alpha2)
            multiply_add_to_node_vars!(u, dg_factor, flux2_jp1,
                                       equations, dg, i, j - 1, k, element)
        end

        # Perform correction in 3rd/z-direction
        for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
            # Subcell interface between nodes (i, j, k - 1) and (i, j, k)
            alpha3 = max(alpha[i, j, k - 1, element], alpha[i, j, k, element])

            # Apply to right node (i, j, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, k, element)
            flux3 = get_node_vars(antidiffusive_flux3_R, equations, dg,
                                  i, j, k, element)
            dg_factor = -dt * inverse_jacobian * inverse_weights[k] * (1 - alpha3)
            multiply_add_to_node_vars!(u, dg_factor, flux3,
                                       equations, dg, i, j, k, element)

            # Apply to left node (i, j, k - 1)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, k - 1, element)
            flux3_kp1 = get_node_vars(antidiffusive_flux3_L, equations, dg,
                                      i, j, k, element)
            dg_factor = dt * inverse_jacobian * inverse_weights[k - 1] * (1 - alpha3)
            multiply_add_to_node_vars!(u, dg_factor, flux3_kp1,
                                       equations, dg, i, j, k - 1, element)
        end
    end

    return nothing
end

function perform_idp_mortar_correction(u, dt, mesh::TreeMesh{3}, equations, dg, cache)
    (; orientations, large_sides, limiting_factor, neighbor_ids) = cache.mortars

    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    for mortar in eachmortar(dg, cache)
        if isapprox(limiting_factor[mortar], one(eltype(limiting_factor)))
            continue
        end
        large_element = neighbor_ids[5, mortar]

        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            direction_small = 2 * orientation - 1
            direction_large = 2 * orientation
            node_small = 1
            node_large = nnodes(dg)

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            direction_small = 2 * orientation
            direction_large = 2 * orientation - 1
            node_small = nnodes(dg)
            node_large = 1

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        for j in eachnode(dg), i in eachnode(dg)
            if orientation == 1
                # L2 mortars in x-direction
                indices_small = (node_small, i, j)
                indices_large = (node_large, i, j)
            elseif orientation == 2
                # L2 mortars in y-direction
                indices_small = (i, node_small, j)
                indices_large = (i, node_large, j)
            else # orientation == 3
                # L2 mortars in z-direction
                indices_small = (i, j, node_small)
                indices_large = (i, j, node_large)
            end

            # large element
            inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_large...,
                                                          large_element)

            flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg, i, j, direction_large,
                                                  large_element)
            flux_large_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, j, direction_large, large_element)
            flux_difference_large = factor_large *
                                    (flux_large_high_order .- flux_large_low_order)

            multiply_add_to_node_vars!(u,
                                       dt * inverse_jacobian_large *
                                       (1 - limiting_factor[mortar]),
                                       flux_difference_large, equations, dg,
                                       indices_large..., large_element)

            # small elements
            for small_element_index in 1:4
                small_element = neighbor_ids[small_element_index, mortar]
                inverse_jacobian_small = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                              mesh, indices_small...,
                                                              small_element)

                flux_small_high_order = get_node_vars(surface_flux_values_high_order,
                                                      equations, dg,
                                                      i, j, direction_small,
                                                      small_element)
                flux_small_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                     i, j, direction_small,
                                                     small_element)
                flux_difference_small = factor_small *
                                        (flux_small_high_order .- flux_small_low_order)

                multiply_add_to_node_vars!(u,
                                           dt * inverse_jacobian_small *
                                           (1 - limiting_factor[mortar]),
                                           flux_difference_small, equations, dg,
                                           indices_small..., small_element)
            end
        end
    end

    return nothing
end

function perform_idp_mortar_correction(u, dt, mesh::P4estMesh{3}, equations, dg, cache)
    (; neighbor_ids, node_indices, limiting_factor) = cache.mortars

    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis
    index_range = eachnode(dg)

    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    for mortar in eachmortar(dg, cache)
        if isapprox(limiting_factor[mortar], one(eltype(limiting_factor)))
            continue
        end
        large_element = neighbor_ids[5, mortar]

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

        i_small = i_small_start
        j_small = j_small_start
        k_small = k_small_start
        i_large = i_large_start
        j_large = j_large_start
        k_large = k_large_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                # large element
                inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                              mesh,
                                                              i_large, j_large, k_large,
                                                              large_element)

                flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                      equations, dg,
                                                      i, j, large_direction,
                                                      large_element)
                flux_large_low_order = get_node_vars(surface_flux_values, equations,
                                                     dg,
                                                     i, j, large_direction,
                                                     large_element)
                flux_difference_large = factor *
                                        (flux_large_high_order .- flux_large_low_order)

                multiply_add_to_node_vars!(u,
                                           dt * inverse_jacobian_large *
                                           (1 - limiting_factor[mortar]),
                                           flux_difference_large, equations, dg,
                                           i_large, j_large, k_large,
                                           large_element)

                # small elements
                for small_element_index in 1:4
                    small_element = neighbor_ids[small_element_index, mortar]
                    inverse_jacobian_small = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                                  mesh, i_small,
                                                                  j_small, k_small,
                                                                  small_element)

                    flux_small_high_order = get_node_vars(surface_flux_values_high_order,
                                                          equations, dg,
                                                          i, j, small_direction,
                                                          small_element)
                    flux_small_low_order = get_node_vars(surface_flux_values,
                                                         equations, dg,
                                                         i, j, small_direction,
                                                         small_element)
                    flux_difference_small = factor *
                                            (flux_small_high_order .-
                                             flux_small_low_order)

                    multiply_add_to_node_vars!(u,
                                               dt * inverse_jacobian_small *
                                               (1 - limiting_factor[mortar]),
                                               flux_difference_small,
                                               equations, dg,
                                               i_small, j_small, k_small,
                                               small_element)
                end

                i_small += i_small_step_i
                j_small += j_small_step_i
                k_small += k_small_step_i
                i_large += i_large_step_i
                j_large += j_large_step_i
                k_large += k_large_step_i
            end

            i_small += i_small_step_j
            j_small += j_small_step_j
            k_small += k_small_step_j
            i_large += i_large_step_j
            j_large += j_large_step_j
            k_large += k_large_step_j
        end
    end

    return nothing
end
end # @muladd
