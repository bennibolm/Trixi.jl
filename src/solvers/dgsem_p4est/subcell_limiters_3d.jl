# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# TODO: Check if this works for actual 3d problems with crazy meshes
@inline function get_mortar_index(indices, i, j, k)
    if indices[1] == :i_forward || indices[1] == :i_backward
        index_i = i
    elseif indices[2] == :i_forward || indices[2] == :i_backward
        index_i = j
    else # indices[3] == :i_forward || indices[3] == :i_backward
        index_i = k
    end
    if indices[1] == :j_forward || indices[1] == :j_backward
        index_j = i
    elseif indices[2] == :j_forward || indices[2] == :j_backward
        index_j = j
    else # indices[3] == :j_forward || indices[3] == :j_backward
        index_j = k
    end
    return index_i, index_j
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
    index_range = eachnode(dg)

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
                i_mortar_s, j_mortar_s = get_mortar_index(small_indices,
                                                          i_small, j_small, k_small)
                i_mortar_l, j_mortar_l = get_mortar_index(large_indices,
                                                          i_large, j_large, k_large)

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
                        i_mortar_s_inner, j_mortar_s_inner = get_mortar_index(small_indices,
                                                                              i_small_inner,
                                                                              j_small_inner,
                                                                              k_small_inner)
                        i_mortar_l_inner, j_mortar_l_inner = get_mortar_index(large_indices,
                                                                              i_large_inner,
                                                                              j_large_inner,
                                                                              k_large_inner)

                        for small_element_index in 1:4
                            small_element = neighbor_ids[small_element_index,
                                                         mortar]
                            # from large to small element
                            if l2_mortars ||
                               dg.mortar.mortar_weights[i_mortar_l, j_mortar_l,
                                                        i_mortar_s_inner,
                                                        j_mortar_s_inner,
                                                        small_element_index] > 0
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
                               dg.mortar.mortar_weights[i_mortar_l_inner,
                                                        j_mortar_l_inner,
                                                        i_mortar_s, j_mortar_s,
                                                        small_element_index] > 0
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
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

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
                var_primary = variable(get_node_vars(u, equations, dg, i_primary,
                                                     j_primary, k_primary,
                                                     primary_element), equations)
                var_secondary = variable(get_node_vars(u, equations, dg, i_secondary,
                                                       j_secondary, k_secondary,
                                                       secondary_element),
                                         equations)

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
    index_range = eachnode(dg)

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
                i_mortar_s, j_mortar_s = get_mortar_index(small_indices,
                                                          i_small, j_small, k_small)
                i_mortar_l, j_mortar_l = get_mortar_index(large_indices,
                                                          i_large, j_large, k_large)

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
                        i_mortar_s_inner, j_mortar_s_inner = get_mortar_index(small_indices,
                                                                              i_small_inner,
                                                                              j_small_inner,
                                                                              k_small_inner)
                        i_mortar_l_inner, j_mortar_l_inner = get_mortar_index(large_indices,
                                                                              i_large_inner,
                                                                              j_large_inner,
                                                                              k_large_inner)

                        for small_element_index in 1:4
                            small_element = neighbor_ids[small_element_index, mortar]
                            # values of large element to small elements
                            if l2_mortars ||
                               dg.mortar.mortar_weights[i_mortar_l, j_mortar_l,
                                                        i_mortar_s_inner,
                                                        j_mortar_s_inner,
                                                        small_element_index] > 0
                                var_minmax[i_small_inner, j_small_inner, k_small_inner,
                                small_element] = minmax(var_minmax[i_small_inner,
                                                                   j_small_inner,
                                                                   k_small_inner,
                                                                   small_element],
                                                        var_large)
                            end
                            # values of small elements to large element
                            if l2_mortars ||
                               dg.mortar.mortar_weights[i_mortar_l_inner,
                                                        j_mortar_l_inner,
                                                        i_mortar_s, j_mortar_s,
                                                        small_element_index] > 0
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

    (; n_mortars_per_node) = volume_integral.limiter.cache.subcell_limiter_coefficients
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
@inline function limiting_local_conservative!(limiting_factor, u, dt, semi,
                                              mesh::P4estMesh{3}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis

    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; variable_bounds, n_mortars_per_node) = dg.volume_integral.limiter.cache.subcell_limiter_coefficients
    variable_string = string(var_index)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        small_element_1 = neighbor_ids[1, mortar]
        small_element_2 = neighbor_ids[2, mortar]
        small_element_3 = neighbor_ids[3, mortar]
        small_element_4 = neighbor_ids[4, mortar]
        large_element = neighbor_ids[5, mortar]
        if perform_subcell_limiting(dg.volume_integral, large_element) ||
           perform_subcell_limiting(dg.volume_integral, small_element_1) ||
           perform_subcell_limiting(dg.volume_integral, small_element_2) ||
           perform_subcell_limiting(dg.volume_integral, small_element_3) ||
           perform_subcell_limiting(dg.volume_integral, small_element_4)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this mortar
        else
            # Subcell limiting is not necessary for all elements => Skip this mortar
            continue
        end

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
                var_large = u[var_index, i_large, j_large, k_large, large_element]
                if var_large < 0
                    error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
                end

                # Two-sided local bounds
                var_min_large = var_min[i_large, j_large, k_large, large_element]
                var_max_large = var_max[i_large, j_large, k_large, large_element]

                # Real one-sided Zalesak-type limiter
                # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
                # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
                # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
                #       for each mortar, not each node
                Qp_large = max(0, (var_max_large - var_large) / dt)
                Qm_large = min(0, (var_min_large - var_large) / dt)

                # Compute flux differences
                flux_large_high_order = surface_flux_values_high_order[var_index,
                                                                       i, j,
                                                                       large_direction,
                                                                       large_element]
                if !isfinite(flux_large_high_order)
                    limiting_factor[mortar] = 1
                    break
                end
                flux_large_low_order = surface_flux_values[var_index, i, j,
                                                           large_direction,
                                                           large_element]
                flux_difference_large = factor *
                                        (flux_large_high_order - flux_large_low_order)

                Pp_large = max(0, flux_difference_large)
                Pm_large = min(0, flux_difference_large)


                inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                              mesh,
                                                              i_large, j_large, k_large,
                                                              large_element)
                Pp_large = inverse_jacobian_large * Pp_large
                Pm_large = inverse_jacobian_large * Pm_large

                # A node can be on multiple mortars. Scale the antidiffusive flux contribution
                # to account for this. Similar to scaling with `gamma_constant_newton`.
                n_mortars_large = n_mortars_per_node[i_large, j_large, k_large,
                                                     large_element]
                Pp_large = n_mortars_large * Pp_large
                Pm_large = n_mortars_large * Pm_large

                eps_ = eps(typeof(Qp_large)) * 100 * abs(var_max_large)
                Qp_large = abs(Qp_large) / (abs(Pp_large) + eps_)
                Qm_large = abs(Qm_large) / (abs(Pm_large) + eps_)

                Q = min(1, Qp_large, Qm_large)

                # small elements
                for small_element_index in 1:4
                    isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                    small_element = neighbor_ids[small_element_index, mortar]
                    var_small = u[var_index, i_small, j_small, k_small, small_element]
                    if var_small < 0
                        error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
                    end

                    # Two-sided local bounds
                    var_min_small = var_min[i_small, j_small, k_small, small_element]
                    var_max_small = var_max[i_small, j_small, k_small, small_element]

                    Qp_small = max(0, (var_max_small - var_small) / dt)
                    Qm_small = min(0, (var_min_small - var_small) / dt)

                    # Compute flux differences
                    flux_small_high_order = surface_flux_values_high_order[var_index,
                                                                           i, j,
                                                                           small_direction,
                                                                           small_element]
                    if !isfinite(flux_small_high_order)
                        limiting_factor[mortar] = 1
                        break
                    end
                    flux_small_low_order = surface_flux_values[var_index, i, j,
                                                               small_direction,
                                                               small_element]
                    flux_difference_small = factor *
                                            (flux_small_high_order -
                                             flux_small_low_order)

                    Pp_small = max(0, flux_difference_small)
                    Pm_small = min(0, flux_difference_small)

                    inverse_jacobian_small = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                                  mesh, i_small,
                                                                  j_small,
                                                                  k_small,
                                                                  small_element)
                    Pp_small = inverse_jacobian_small * Pp_small
                    Pm_small = inverse_jacobian_small * Pm_small

                    # A node can be on multiple mortars. Scale the antidiffusive flux contribution
                    # to account for this. Similar to scaling with `gamma_constant_newton`.
                    n_mortars_small = n_mortars_per_node[i_small, j_small, k_small,
                                                         small_element]
                    Pp_small = n_mortars_small * Pp_small
                    Pm_small = n_mortars_small * Pm_small

                    eps_ = eps(typeof(Qp_small)) * 100 * abs(var_max_small)
                    Qp_small = abs(Qp_small) / (abs(Pp_small) + eps_)
                    Qm_small = abs(Qm_small) / (abs(Pm_small) + eps_)

                    Q = min(Q, Qp_small, Qm_small)
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
@inline function limiting_local_nonlinear!(limiting_factor, u, dt, semi,
                                           mesh::P4estMesh{3}, variable,
                                           min_or_max)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]

    (; gamma_constant_newton) = limiter

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        small_element_1 = neighbor_ids[1, mortar]
        small_element_2 = neighbor_ids[2, mortar]
        small_element_3 = neighbor_ids[3, mortar]
        small_element_4 = neighbor_ids[4, mortar]
        large_element = neighbor_ids[5, mortar]
        if perform_subcell_limiting(dg.volume_integral, large_element) ||
           perform_subcell_limiting(dg.volume_integral, small_element_1) ||
           perform_subcell_limiting(dg.volume_integral, small_element_2) ||
           perform_subcell_limiting(dg.volume_integral, small_element_3) ||
           perform_subcell_limiting(dg.volume_integral, small_element_4)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this mortar
        else
            # Subcell limiting is not necessary for all elements => Skip this mortar
            continue
        end

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
                u_large = get_node_vars(u, equations, dg,
                                        i_large, j_large, k_large,
                                        large_element)
                bound_large = var_minmax[i_large, j_large, k_large, large_element]

                flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                      equations, dg,
                                                      i, j, large_direction,
                                                      large_element)
                if !all(isfinite, flux_large_high_order)
                    limiting_factor[mortar] = 1
                    break
                end
                flux_large_low_order = get_node_vars(surface_flux_values,
                                                     equations, dg,
                                                     i, j, large_direction,
                                                     large_element)

                inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                              mesh, i_large, j_large,
                                                              k_large,
                                                              large_element)
                antidiffusive_flux_large = gamma_constant_newton * factor *
                                           inverse_jacobian_large *
                                           (flux_large_high_order .-
                                            flux_large_low_order)

                newton_loop!(limiting_factor, bound_large, u_large, (mortar,),
                             variable, min_or_max,
                             initial_check_local_onesided_newton_idp,
                             final_check_local_onesided_newton_idp,
                             equations, dt, limiter, antidiffusive_flux_large)

                # Small elements
                for small_element_index in 1:4
                    isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                    small_element = neighbor_ids[small_element_index, mortar]

                    u_small = get_node_vars(u, equations, dg,
                                            i_small, j_small, k_small,
                                            small_element)
                    bound_small = var_minmax[i_small, j_small, k_small, small_element]

                    flux_small_high_order = get_node_vars(surface_flux_values_high_order,
                                                          equations, dg,
                                                          i, j, small_direction,
                                                          small_element)
                    if !all(isfinite, flux_small_high_order)
                        limiting_factor[mortar] = 1
                        break
                    end
                    flux_small_low_order = get_node_vars(surface_flux_values,
                                                         equations, dg,
                                                         i, j, small_direction,
                                                         small_element)

                    inverse_jacobian_small = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                                  mesh, i_small,
                                                                  j_small,
                                                                  k_small,
                                                                  small_element)
                    antidiffusive_flux_small = gamma_constant_newton * factor *
                                               inverse_jacobian_small *
                                               (flux_small_high_order .-
                                                flux_small_low_order)

                    newton_loop!(limiting_factor, bound_small, u_small, (mortar,),
                                 variable, min_or_max,
                                 initial_check_local_onesided_newton_idp,
                                 final_check_local_onesided_newton_idp,
                                 equations, dt, limiter, antidiffusive_flux_small)
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
@inline function limiting_positivity_conservative!(limiting_factor, u, dt, semi,
                                                   mesh::P4estMesh{3}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis

    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; variable_bounds, n_mortars_per_node) = dg.volume_integral.limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(var_index), "_min")]

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        small_element_1 = neighbor_ids[1, mortar]
        small_element_2 = neighbor_ids[2, mortar]
        small_element_3 = neighbor_ids[3, mortar]
        small_element_4 = neighbor_ids[4, mortar]
        large_element = neighbor_ids[5, mortar]
        if perform_subcell_limiting(dg.volume_integral, large_element) ||
           perform_subcell_limiting(dg.volume_integral, small_element_1) ||
           perform_subcell_limiting(dg.volume_integral, small_element_2) ||
           perform_subcell_limiting(dg.volume_integral, small_element_3) ||
           perform_subcell_limiting(dg.volume_integral, small_element_4)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this mortar
        else
            # Subcell limiting is not necessary for all elements => Skip this mortar
            continue
        end

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
                var_large = u[var_index, i_large, j_large, k_large, large_element]
                if var_large < 0
                    error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
                end

                # Calculate Pm
                flux_large_high_order = surface_flux_values_high_order[var_index,
                                                                       i, j,
                                                                       large_direction,
                                                                       large_element]
                # Check if high-order fluxes are finite. Otherwise, use pure low-order fluxes.
                if !isfinite(flux_large_high_order)
                    limiting_factor[mortar] = 1
                    break
                end
                flux_large_low_order = surface_flux_values[var_index, i, j,
                                                           large_direction,
                                                           large_element]
                flux_difference_large = factor *
                                        (flux_large_high_order - flux_large_low_order)

                # Minimum bound
                var_min_large = var_min[i_large, j_large, k_large, large_element]

                # Real one-sided Zalesak-type limiter
                # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
                # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
                # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
                #       for each mortar, not each node
                Qm_large = min(0, var_min_large - var_large)
                Pm_large = min(0, flux_difference_large)

                # A node can be on multiple mortars. Scale the antidiffusive flux contribution
                # to account for this. Similar to scaling with `gamma_constant_newton`.
                n_mortars_large = n_mortars_per_node[i_large, j_large, k_large,
                                                     large_element]
                Pm_large = n_mortars_large * Pm_large

                inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                              mesh,
                                                              i_large, j_large, k_large,
                                                              large_element)
                Pm_large = dt * inverse_jacobian_large * Pm_large

                # Compute blending coefficient avoiding division by zero
                # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
                eps_ = eps(typeof(Qm_large)) * 100
                Qm_large = abs(Qm_large) / (abs(Pm_large) + eps_)

                Qm = min(1, Qm_large)

                # Small elements
                for small_element_index in 1:4
                    isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                    small_element = neighbor_ids[small_element_index, mortar]
                    var_small = u[var_index, i_small, j_small, k_small, small_element]
                    if var_small < 0
                        error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
                    end

                    flux_small_high_order = surface_flux_values_high_order[var_index,
                                                                           i, j,
                                                                           small_direction,
                                                                           small_element]
                    if !isfinite(flux_small_high_order)
                        limiting_factor[mortar] = 1
                        break
                    end
                    flux_small_low_order = surface_flux_values[var_index, i, j,
                                                               small_direction,
                                                               small_element]
                    flux_difference_small = factor *
                                            (flux_small_high_order -
                                             flux_small_low_order)

                    var_min_small = var_min[i_small, j_small, k_small, small_element]
                    Qm_small = min(0, var_min_small - var_small)
                    Pm_small = min(0, flux_difference_small)

                    # A node can be on multiple mortars. Scale the antidiffusive flux contribution
                    # to account for this. Similar to scaling with `gamma_constant_newton`.
                    n_mortars_small = n_mortars_per_node[i_small, j_small, k_small,
                                                         small_element]
                    Pm_small = n_mortars_small * Pm_small

                    inverse_jacobian_small = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                                  mesh, i_small,
                                                                  j_small, k_small,
                                                                  small_element)
                    Pm_small = dt * inverse_jacobian_small * Pm_small

                    Qm_small = abs(Qm_small) / (abs(Pm_small) + eps_)
                    Qm = min(Qm, Qm_small)
                end

                # Calculate limiting factor
                limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Qm)

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
@inline function limiting_positivity_nonlinear!(limiting_factor, u, dt, semi,
                                                mesh::P4estMesh{3}, variable)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    (; gamma_constant_newton) = limiter

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        small_element_1 = neighbor_ids[1, mortar]
        small_element_2 = neighbor_ids[2, mortar]
        small_element_3 = neighbor_ids[3, mortar]
        small_element_4 = neighbor_ids[4, mortar]
        large_element = neighbor_ids[5, mortar]
        if perform_subcell_limiting(dg.volume_integral, large_element) ||
           perform_subcell_limiting(dg.volume_integral, small_element_1) ||
           perform_subcell_limiting(dg.volume_integral, small_element_2) ||
           perform_subcell_limiting(dg.volume_integral, small_element_3) ||
           perform_subcell_limiting(dg.volume_integral, small_element_4)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this mortar
        else
            # Subcell limiting is not necessary for all elements => Skip this mortar
            continue
        end

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
                u_large = get_node_vars(u, equations, dg,
                                        i_large, j_large, k_large,
                                        large_element)
                var_min_large = var_min[i_large, j_large, k_large, large_element]

                flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                      equations, dg,
                                                      i, j, large_direction,
                                                      large_element)
                if !all(isfinite, flux_large_high_order)
                    limiting_factor[mortar] = 1
                    break
                end
                flux_large_low_order = get_node_vars(surface_flux_values,
                                                     equations, dg,
                                                     i, j, large_direction,
                                                     large_element)

                inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                              mesh, i_large, j_large,
                                                              k_large,
                                                              large_element)
                antidiffusive_flux_large = gamma_constant_newton * factor *
                                           inverse_jacobian_large *
                                           (flux_large_high_order .-
                                            flux_large_low_order)

                newton_loop!(limiting_factor, var_min_large, u_large, (mortar,),
                             variable, min,
                             initial_check_nonnegative_newton_idp,
                             final_check_nonnegative_newton_idp,
                             equations, dt, limiter, antidiffusive_flux_large)

                # Small elements
                for small_element_index in 1:4
                    isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                    small_element = neighbor_ids[small_element_index, mortar]

                    u_small = get_node_vars(u, equations, dg,
                                            i_small, j_small, k_small,
                                            small_element)
                    var_min_small = var_min[i_small, j_small, k_small, small_element]

                    flux_small_high_order = get_node_vars(surface_flux_values_high_order,
                                                          equations, dg,
                                                          i, j, small_direction,
                                                          small_element)
                    if !all(isfinite, flux_small_high_order)
                        limiting_factor[mortar] = 1
                        break
                    end
                    flux_small_low_order = get_node_vars(surface_flux_values,
                                                         equations, dg,
                                                         i, j, small_direction,
                                                         small_element)

                    inverse_jacobian_small = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                                  mesh, i_small,
                                                                  j_small,
                                                                  k_small,
                                                                  small_element)
                    antidiffusive_flux_small = gamma_constant_newton * factor *
                                               inverse_jacobian_small *
                                               (flux_small_high_order .-
                                                flux_small_low_order)

                    newton_loop!(limiting_factor, var_min_small, u_small, (mortar,),
                                 variable, min,
                                 initial_check_nonnegative_newton_idp,
                                 final_check_nonnegative_newton_idp,
                                 equations, dt, limiter, antidiffusive_flux_small)
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
