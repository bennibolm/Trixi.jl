# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function get_large_surface_index(indices, i, j, k)
    # Return the two face-tangential element indices in element-axis order, matching the
    # layout of `surface_flux_values` (cf. `surface_indices` and `mortar_fluxes_to_elements!`).
    if indices[1] === :begin || indices[1] === :end
        return j, k
    elseif indices[2] === :begin || indices[2] === :end
        return i, k
    else # indices[3] === :begin || indices[3] === :end
        return i, j
    end
end

function calc_bounds_twosided_interface!(var_min, var_max, variable, u,
                                         semi, mesh::P4estMesh{3}, equations)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)

    for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]

        # Create the local i,j,k indexing
        i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                     index_range)
        j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                     index_range)
        k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                     index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        k_primary = k_primary_start

        i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_indices[1],
                                                                                           index_range)
        j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_indices[2],
                                                                                           index_range)
        k_secondary_start, k_secondary_step_i, k_secondary_step_j = index_to_start_step_3d(secondary_indices[3],
                                                                                           index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        k_secondary = k_secondary_start

        for j in eachnode(dg)
            for i in eachnode(dg)
                var_primary = u[variable, i_primary, j_primary, k_primary,
                                primary_element]
                var_secondary = u[variable, i_secondary, j_secondary, k_secondary,
                                  secondary_element]

                var_min[i_primary, j_primary, k_primary, primary_element] = min(var_min[i_primary,
                                                                                        j_primary,
                                                                                        k_primary,
                                                                                        primary_element],
                                                                                var_secondary)
                var_max[i_primary, j_primary, k_primary, primary_element] = max(var_max[i_primary,
                                                                                        j_primary,
                                                                                        k_primary,
                                                                                        primary_element],
                                                                                var_secondary)

                var_min[i_secondary, j_secondary, k_secondary, secondary_element] = min(var_min[i_secondary,
                                                                                                j_secondary,
                                                                                                k_secondary,
                                                                                                secondary_element],
                                                                                        var_primary)
                var_max[i_secondary, j_secondary, k_secondary, secondary_element] = max(var_max[i_secondary,
                                                                                                j_secondary,
                                                                                                k_secondary,
                                                                                                secondary_element],
                                                                                        var_primary)

                # Increment the primary element indices
                i_primary += i_primary_step_i
                j_primary += j_primary_step_i
                k_primary += k_primary_step_i
                # Increment the secondary element surface indices
                i_secondary += i_secondary_step_i
                j_secondary += j_secondary_step_i
                k_secondary += k_secondary_step_i
            end
            # Increment the primary element indices
            i_primary += i_primary_step_j
            j_primary += j_primary_step_j
            k_primary += k_primary_step_j
            # Increment the secondary element surface indices
            i_secondary += i_secondary_step_j
            j_secondary += j_secondary_step_j
            k_secondary += k_secondary_step_j
        end
    end

    return nothing
end

@inline function calc_bounds_twosided_mortar!(var_min, var_max, variable, u,
                                              semi, mesh::P4estMesh{3})
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    mortar_weights = get_mortar_weights(dg.mortar)
    index_range = eachnode(dg)

    # `mortar_weights` is defined in mortar reference coordinates, so it has to be
    # indexed with the loop counters (i and j). Using the element-local face indices instead
    # would pair mirror-image subcells whenever the large side is traversed backwards,
    # i.e., for `:i_backward in large_indices`.

    # See comment above TreeMesh version
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[5, mortar]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        # Get index information on the large element
        large_indices = node_indices[2, mortar]
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
                var_small = (u[variable, i_small, j_small, k_small,
                               neighbor_ids[1, mortar]],
                             u[variable, i_small, j_small, k_small,
                               neighbor_ids[2, mortar]],
                             u[variable, i_small, j_small, k_small,
                               neighbor_ids[3, mortar]],
                             u[variable, i_small, j_small, k_small,
                               neighbor_ids[4, mortar]])
                var_large = u[variable, i_large, j_large, k_large, large_element]

                i_small_inner = i_small_start
                j_small_inner = j_small_start
                k_small_inner = k_small_start
                i_large_inner = i_large_start
                j_large_inner = j_large_start
                k_large_inner = k_large_start
                for l in eachnode(dg)
                    for k in eachnode(dg)
                        for small_element_index in 1:4
                            small_element = neighbor_ids[small_element_index,
                                                         mortar]
                            # from large to small element
                            if l2_mortars ||
                               mortar_weights[i, j, k, l, small_element_index] > 0
                                var_min[i_small_inner, j_small_inner, k_small_inner,
                                small_element] = min(var_min[i_small_inner,
                                                             j_small_inner,
                                                             k_small_inner,
                                                             small_element],
                                                     var_large)
                                var_max[i_small_inner, j_small_inner, k_small_inner,
                                small_element] = max(var_max[i_small_inner,
                                                             j_small_inner,
                                                             k_small_inner,
                                                             small_element],
                                                     var_large)
                            end
                            # from small to large element
                            if l2_mortars ||
                               mortar_weights[k, l, i, j, small_element_index] > 0
                                var_min[i_large_inner, j_large_inner, k_large_inner,
                                large_element] = min(var_min[i_large_inner,
                                                             j_large_inner,
                                                             k_large_inner,
                                                             large_element],
                                                     var_small[small_element_index])
                                var_max[i_large_inner, j_large_inner, k_large_inner,
                                large_element] = max(var_max[i_large_inner,
                                                             j_large_inner,
                                                             k_large_inner,
                                                             large_element],
                                                     var_small[small_element_index])
                            end
                        end

                        i_small_inner += i_small_step_i
                        j_small_inner += j_small_step_i
                        k_small_inner += k_small_step_i
                        i_large_inner += i_large_step_i
                        j_large_inner += j_large_step_i
                        k_large_inner += k_large_step_i
                    end
                    i_small_inner += i_small_step_j
                    j_small_inner += j_small_step_j
                    k_small_inner += k_small_step_j
                    i_large_inner += i_large_step_j
                    j_large_inner += j_large_step_j
                    k_large_inner += k_large_step_j
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

@inline function calc_bounds_twosided_boundary!(var_min, var_max, variable, u, t,
                                                boundary_conditions::BoundaryConditionPeriodic,
                                                mesh::P4estMesh{3},
                                                equations, dg, cache)
    return nothing
end

@inline function calc_bounds_twosided_boundary!(var_min, var_max, variable, u, t,
                                                boundary_conditions,
                                                mesh::P4estMesh{3},
                                                equations, dg, cache)
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
                    var_outer = u_outer[variable]

                    var_min[i_node, j_node, k_node, element] = min(var_min[i_node,
                                                                           j_node,
                                                                           k_node,
                                                                           element],
                                                                   var_outer)
                    var_max[i_node, j_node, k_node, element] = max(var_max[i_node,
                                                                           j_node,
                                                                           k_node,
                                                                           element],
                                                                   var_outer)

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

function calc_bounds_onesided_interface!(var_minmax, minmax, variable, u,
                                         semi, mesh::P4estMesh{3})
    _, _, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_values) = subcell_limiter_coefficients(dg.volume_integral)

    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)

    for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]

        # Create the local i,j,k indexing
        i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                     index_range)
        j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                     index_range)
        k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                     index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        k_primary = k_primary_start

        i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_indices[1],
                                                                                           index_range)
        j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_indices[2],
                                                                                           index_range)
        k_secondary_start, k_secondary_step_i, k_secondary_step_j = index_to_start_step_3d(secondary_indices[3],
                                                                                           index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        k_secondary = k_secondary_start

        for j in eachnode(dg)
            for i in eachnode(dg)
                var_primary = variable_values[i_primary, j_primary, k_primary,
                                              primary_element]
                var_secondary = variable_values[i_secondary, j_secondary, k_secondary,
                                                secondary_element]

                var_minmax[i_primary, j_primary, k_primary, primary_element] = minmax(var_minmax[i_primary,
                                                                                                 j_primary,
                                                                                                 k_primary,
                                                                                                 primary_element],
                                                                                      var_secondary)
                var_minmax[i_secondary, j_secondary, k_secondary, secondary_element] = minmax(var_minmax[i_secondary,
                                                                                                         j_secondary,
                                                                                                         k_secondary,
                                                                                                         secondary_element],
                                                                                              var_primary)

                # Increment the primary element indices
                i_primary += i_primary_step_i
                j_primary += j_primary_step_i
                k_primary += k_primary_step_i
                # Increment the secondary element surface indices
                i_secondary += i_secondary_step_i
                j_secondary += j_secondary_step_i
                k_secondary += k_secondary_step_i
            end
            # Increment the primary element indices
            i_primary += i_primary_step_j
            j_primary += j_primary_step_j
            k_primary += k_primary_step_j
            # Increment the secondary element surface indices
            i_secondary += i_secondary_step_j
            j_secondary += j_secondary_step_j
            k_secondary += k_secondary_step_j
        end
    end

    return nothing
end

@inline function calc_bounds_onesided_mortar!(var_minmax, minmax, variable, u,
                                              semi, mesh::P4estMesh{3})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    mortar_weights = get_mortar_weights(dg.mortar)
    index_range = eachnode(dg)

    # `mortar_weights` is defined in mortar reference coordinates, so it has to be
    # indexed with the loop counters (i and j). Using the element-local face indices instead
    # would pair mirror-image subcells whenever the large side is traversed backwards,
    # i.e., for `:i_backward in large_indices`.

    # See comment above TreeMesh version
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[5, mortar]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        # Get index information on the large element
        large_indices = node_indices[2, mortar]
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
                u_small = (get_node_vars(u, equations, dg, i_small, j_small, k_small,
                                         neighbor_ids[1, mortar]),
                           get_node_vars(u, equations, dg, i_small, j_small, k_small,
                                         neighbor_ids[2, mortar]),
                           get_node_vars(u, equations, dg, i_small, j_small, k_small,
                                         neighbor_ids[3, mortar]),
                           get_node_vars(u, equations, dg, i_small, j_small, k_small,
                                         neighbor_ids[4, mortar]))
                u_large = get_node_vars(u, equations, dg, i_large, j_large, k_large,
                                        large_element)
                var_small = (variable(u_small[1], equations),
                             variable(u_small[2], equations),
                             variable(u_small[3], equations),
                             variable(u_small[4], equations))
                var_large = variable(u_large, equations)

                i_small_inner = i_small_start
                j_small_inner = j_small_start
                k_small_inner = k_small_start
                i_large_inner = i_large_start
                j_large_inner = j_large_start
                k_large_inner = k_large_start
                for l in eachnode(dg)
                    for k in eachnode(dg)
                        for small_element_index in 1:4
                            small_element = neighbor_ids[small_element_index, mortar]
                            # values of large element to small elements
                            if l2_mortars ||
                               mortar_weights[i, j, k, l, small_element_index] > 0
                                var_minmax[i_small_inner, j_small_inner, k_small_inner,
                                small_element] = minmax(var_minmax[i_small_inner,
                                                                   j_small_inner,
                                                                   k_small_inner,
                                                                   small_element],
                                                        var_large)
                            end
                            # values of small elements to large element
                            if l2_mortars ||
                               mortar_weights[k, l, i, j, small_element_index] > 0
                                var_minmax[i_large_inner, j_large_inner, k_large_inner,
                                large_element] = minmax(var_minmax[i_large_inner,
                                                                   j_large_inner,
                                                                   k_large_inner,
                                                                   large_element],
                                                        var_small[small_element_index])
                            end
                        end

                        i_small_inner += i_small_step_i
                        j_small_inner += j_small_step_i
                        k_small_inner += k_small_step_i
                        i_large_inner += i_large_step_i
                        j_large_inner += j_large_step_i
                        k_large_inner += k_large_step_i
                    end
                    i_small_inner += i_small_step_j
                    j_small_inner += j_small_step_j
                    k_small_inner += k_small_step_j
                    i_large_inner += i_large_step_j
                    j_large_inner += j_large_step_j
                    k_large_inner += k_large_step_j
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

@inline function calc_bounds_onesided_boundary!(var_minmax, minmax, variable, u, t,
                                                boundary_conditions::BoundaryConditionPeriodic,
                                                mesh::P4estMesh{3},
                                                equations, dg, cache)
    return nothing
end

@inline function calc_bounds_onesided_boundary!(var_minmax, minmax, variable, u, t,
                                                boundary_conditions,
                                                mesh::P4estMesh{3},
                                                equations, dg, cache)
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
                    var_outer = variable(u_outer, equations)

                    var_minmax[i_node, j_node, k_node, element] = minmax(var_minmax[i_node,
                                                                                    j_node,
                                                                                    k_node,
                                                                                    element],
                                                                         var_outer)

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

###############################################################################
# IDP mortar limiting
###############################################################################

@inline function precompute_n_mortars_per_nodes!(volume_integral::VolumeIntegralSubcellLimiting,
                                                 dg, cache,
                                                 mesh::P4estMesh{3})
    if !(dg.mortar isa LobattoLegendreMortarIDP)
        return nothing
    end

    (; n_mortars_per_node) = subcell_limiter_coefficients(volume_integral)
    (; neighbor_ids, node_indices) = cache.mortars
    index_range = eachnode(dg)

    n_mortars_per_node .= zero(eltype(n_mortars_per_node))

    for mortar in eachmortar(dg, cache)
        small_element_1 = neighbor_ids[1, mortar]
        small_element_2 = neighbor_ids[2, mortar]
        small_element_3 = neighbor_ids[3, mortar]
        small_element_4 = neighbor_ids[4, mortar]
        large_element = neighbor_ids[5, mortar]

        # Get index information on the elements
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        large_indices = node_indices[2, mortar]
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
                # Increment the number of mortars per node for each element
                n_mortars_per_node[i_small, j_small, k_small, small_element_1] += 1
                n_mortars_per_node[i_small, j_small, k_small, small_element_2] += 1
                n_mortars_per_node[i_small, j_small, k_small, small_element_3] += 1
                n_mortars_per_node[i_small, j_small, k_small, small_element_4] += 1
                n_mortars_per_node[i_large, j_large, k_large, large_element] += 1

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

###############################################################################
# Local two-sided limiting of conservative variables
@inline function idp_mortar_local_twosided!(limiting_factor, u, dt, semi,
                                            mesh::P4estMesh{3}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; inverse_weights) = dg.basis

    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; variable_bounds, n_mortars_per_node) = subcell_limiter_coefficients(dg.volume_integral)
    variable_string = string(var_index)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        large_element = neighbor_ids[5, mortar]

        # Get index information on the elements
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
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
            for i in eachnode(dg)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                # Large element
                # Map the mortar node to the large-element face since its orientation may be flipped.
                # The small-element face needs no mapping because it is always traversed forward.
                large_node_i, large_node_j = get_large_surface_index(large_indices,
                                                                     i_large, j_large,
                                                                     k_large)
                Q = zalesak_limiting_twosided(u, var_index, i_large, j_large, k_large,
                                              large_element,
                                              large_node_i, large_node_j,
                                              large_direction, factor, dt,
                                              var_min, var_max, n_mortars_per_node,
                                              mesh, cache)

                # Small elements
                for small_element_index in 1:4
                    iszero(Q) && break # Skip if Q is zero, i.e., the limiting factor will be 1

                    small_element = neighbor_ids[small_element_index, mortar]
                    Q = min(Q,
                            zalesak_limiting_twosided(u, var_index, i_small, j_small,
                                                      k_small,
                                                      small_element, i, j,
                                                      small_direction,
                                                      factor, dt,
                                                      var_min, var_max,
                                                      n_mortars_per_node,
                                                      mesh, cache))
                end

                limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Q)

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

##############################################################################
# Local one-sided limiting of nonlinear variables
@inline function idp_mortar_local_onesided!(limiting_factor, u, dt, semi,
                                            mesh::P4estMesh{3}, variable,
                                            min_or_max)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars

    (; inverse_weights) = dg.basis
    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

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
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
            for i in eachnode(dg)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                # Large element
                # Map the mortar node to the large-element face since its orientation may be flipped.
                # The small-element face needs no mapping because it is always traversed forward.
                large_node_i, large_node_j = get_large_surface_index(large_indices,
                                                                     i_large, j_large,
                                                                     k_large)
                newton_loop_mortar!(limiting_factor, mortar, u,
                                    i_large, j_large, k_large, large_element,
                                    large_node_i, large_node_j, large_direction, factor,
                                    dt,
                                    var_minmax, variable, min_or_max,
                                    initial_check_local_onesided_newton_idp,
                                    final_check_local_onesided_newton_idp,
                                    mesh, equations, dg, cache)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                # Small elements
                for small_element_index in 1:4
                    small_element = neighbor_ids[small_element_index, mortar]

                    newton_loop_mortar!(limiting_factor, mortar, u,
                                        i_small, j_small, k_small, small_element,
                                        i, j, small_direction, factor, dt,
                                        var_minmax, variable, min_or_max,
                                        initial_check_local_onesided_newton_idp,
                                        final_check_local_onesided_newton_idp,
                                        mesh, equations, dg, cache)
                    isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
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

###############################################################################
# Global positivity limiting of conservative variables
@inline function idp_mortar_positivity_conservative!(limiting_factor, u, dt, semi,
                                                     mesh::P4estMesh{3}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; inverse_weights) = dg.basis

    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    (; n_mortars_per_node, variable_bounds) = subcell_limiter_coefficients(dg.volume_integral)

    # Check whether the local limiting already computed a bound for this variable in this stage.
    was_limited_locally = limiter.local_twosided &&
                          (var_index in limiter.local_twosided_variables_cons)
    # Without a smoothness indicator, both limiters are enforced completely and `var_min`
    # holds the more restrictive of the two bounds. With a smoothness indicator, local bounds are
    # only enforced fractionally, while positivity limiting is enforced completely.
    # In that case, the local bound is stored in `var_min`, while the positivity bound is stored in
    # `var_min_positivity`.
    enabled_indicator = !isnothing(limiter.indicator)

    # Array the positivity bound was written to. Only with a smoothness indicator it is stored
    # separately; otherwise the more restrictive of the two bounds is kept in `var_min`.
    if was_limited_locally && !enabled_indicator
        # Positivity bound was merged into var_min and therefore already enforced during local limiting.
        # Skip positivity limiting for this variable.
        return nothing
    elseif was_limited_locally && enabled_indicator
        var_min = variable_bounds[Symbol(string(var_index), "_min_positivity")]
    else
        var_min = variable_bounds[Symbol(string(var_index), "_min")]
    end

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

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
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
            for i in eachnode(dg)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                # Large element
                # Map the mortar node to the large-element face since its orientation may be
                # flipped or transposed. The small-element face needs no mapping because it is
                # always traversed forward.
                large_node_i, large_node_j = get_large_surface_index(large_indices,
                                                                     i_large, j_large,
                                                                     k_large)
                Q = zalesak_limiting_onesided(u, var_index, i_large, j_large, k_large,
                                              large_element,
                                              large_node_i, large_node_j,
                                              large_direction, factor, dt,
                                              var_min, n_mortars_per_node,
                                              mesh, cache)

                # Small elements
                for small_element_index in 1:4
                    iszero(Q) && break # Skip if Q is zero, i.e., the limiting factor will be 1

                    small_element = neighbor_ids[small_element_index, mortar]
                    Q = min(Q,
                            zalesak_limiting_onesided(u, var_index, i_small, j_small,
                                                      k_small,
                                                      small_element,
                                                      i, j, small_direction, factor, dt,
                                                      var_min, n_mortars_per_node,
                                                      mesh, cache))
                end

                # Calculate limiting factor
                limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Q)

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

##############################################################################
# Global positivity limiting of nonlinear variables
@inline function idp_mortar_positivity_nonlinear!(limiting_factor, u, dt, semi,
                                                  mesh::P4estMesh{3}, variable)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars

    (; inverse_weights) = dg.basis
    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    # The nonlinear positivity limiting is the only limiter writing this bound: the nonlinear
    # local limiting is only used for entropies, the nonlinear positivity limiting only for
    # the pressure. Therefore, `var_min` holds the positivity bound and can be reused here.
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

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
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
            for i in eachnode(dg)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                # Large element
                # Map the mortar node to the large-element face since its orientation may be
                # flipped or transposed. The small-element face needs no mapping because it is
                # always traversed forward.
                large_node_i, large_node_j = get_large_surface_index(large_indices,
                                                                     i_large, j_large,
                                                                     k_large)
                newton_loop_mortar!(limiting_factor, mortar, u,
                                    i_large, j_large, k_large, large_element,
                                    large_node_i, large_node_j, large_direction, factor,
                                    dt,
                                    var_min, variable, min,
                                    initial_check_nonnegative_newton_idp,
                                    final_check_nonnegative_newton_idp,
                                    mesh, equations, dg, cache)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                # Small elements
                for small_element_index in 1:4
                    small_element = neighbor_ids[small_element_index, mortar]

                    newton_loop_mortar!(limiting_factor, mortar, u,
                                        i_small, j_small, k_small, small_element,
                                        i, j, small_direction, factor, dt,
                                        var_min, variable, min,
                                        initial_check_nonnegative_newton_idp,
                                        final_check_nonnegative_newton_idp,
                                        mesh, equations, dg, cache)
                    isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
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
