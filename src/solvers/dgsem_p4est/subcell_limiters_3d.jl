# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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
                if small_indices[1] === :i_forward || small_indices[1] === :i_backward
                    i_mortar_s = i_small
                elseif small_indices[2] === :i_forward ||
                       small_indices[2] === :i_backward
                    i_mortar_s = j_small
                else
                    i_mortar_s = k_small
                end
                if small_indices[1] === :j_forward || small_indices[1] === :j_backward
                    j_mortar_s = i_small
                elseif small_indices[2] === :j_forward ||
                       small_indices[2] === :j_backward
                    j_mortar_s = j_small
                else
                    j_mortar_s = k_small
                end
                if large_indices[1] === :i_forward || large_indices[1] === :i_backward
                    i_mortar_l = i_large
                elseif large_indices[2] === :i_forward ||
                       large_indices[2] === :i_backward
                    i_mortar_l = j_large
                else
                    i_mortar_l = k_large
                end
                if large_indices[1] === :j_forward || large_indices[1] === :j_backward
                    j_mortar_l = i_large
                elseif large_indices[2] === :j_forward ||
                       large_indices[2] === :j_backward
                    j_mortar_l = j_large
                else
                    j_mortar_l = k_large
                end

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
                        if small_indices[1] === :i_forward ||
                           small_indices[1] === :i_backward
                            i_mortar_s_inner = i_small_inner
                        elseif small_indices[2] === :i_forward ||
                               small_indices[2] === :i_backward
                            i_mortar_s_inner = j_small_inner
                        else
                            i_mortar_s_inner = k_small_inner
                        end
                        if small_indices[1] === :j_forward ||
                           small_indices[1] === :j_backward
                            j_mortar_s_inner = i_small_inner
                        elseif small_indices[2] === :j_forward ||
                               small_indices[2] === :j_backward
                            j_mortar_s_inner = j_small_inner
                        else
                            j_mortar_s_inner = k_small_inner
                        end
                        if large_indices[1] === :i_forward ||
                           large_indices[1] === :i_backward
                            i_mortar_l_inner = i_large_inner
                        elseif large_indices[2] === :i_forward ||
                               large_indices[2] === :i_backward
                            i_mortar_l_inner = j_large_inner
                        else
                            i_mortar_l_inner = k_large_inner
                        end
                        if large_indices[1] === :j_forward ||
                           large_indices[1] === :j_backward
                            j_mortar_l_inner = i_large_inner
                        elseif large_indices[2] === :j_forward ||
                               large_indices[2] === :j_backward
                            j_mortar_l_inner = j_large_inner
                        else
                            j_mortar_l_inner = k_large_inner
                        end

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
                if small_indices[1] === :i_forward || small_indices[1] === :i_backward
                    i_mortar_s = i_small
                elseif small_indices[2] === :i_forward ||
                       small_indices[2] === :i_backward
                    i_mortar_s = j_small
                else
                    i_mortar_s = k_small
                end
                if small_indices[1] === :j_forward || small_indices[1] === :j_backward
                    j_mortar_s = i_small
                elseif small_indices[2] === :j_forward ||
                       small_indices[2] === :j_backward
                    j_mortar_s = j_small
                else
                    j_mortar_s = k_small
                end
                if large_indices[1] === :i_forward || large_indices[1] === :i_backward
                    i_mortar_l = i_large
                elseif large_indices[2] === :i_forward ||
                       large_indices[2] === :i_backward
                    i_mortar_l = j_large
                else
                    i_mortar_l = k_large
                end
                if large_indices[1] === :j_forward || large_indices[1] === :j_backward
                    j_mortar_l = i_large
                elseif large_indices[2] === :j_forward ||
                       large_indices[2] === :j_backward
                    j_mortar_l = j_large
                else
                    j_mortar_l = k_large
                end

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
                        if small_indices[1] === :i_forward ||
                           small_indices[1] === :i_backward
                            i_mortar_s_inner = i_small_inner
                        elseif small_indices[2] === :i_forward ||
                               small_indices[2] === :i_backward
                            i_mortar_s_inner = j_small_inner
                        else
                            i_mortar_s_inner = k_small_inner
                        end
                        if small_indices[1] === :j_forward ||
                           small_indices[1] === :j_backward
                            j_mortar_s_inner = i_small_inner
                        elseif small_indices[2] === :j_forward ||
                               small_indices[2] === :j_backward
                            j_mortar_s_inner = j_small_inner
                        else
                            j_mortar_s_inner = k_small_inner
                        end
                        if large_indices[1] === :i_forward ||
                           large_indices[1] === :i_backward
                            i_mortar_l_inner = i_large_inner
                        elseif large_indices[2] === :i_forward ||
                               large_indices[2] === :i_backward
                            i_mortar_l_inner = j_large_inner
                        else
                            i_mortar_l_inner = k_large_inner
                        end
                        if large_indices[1] === :j_forward ||
                           large_indices[1] === :j_backward
                            j_mortar_l_inner = i_large_inner
                        elseif large_indices[2] === :j_forward ||
                               large_indices[2] === :j_backward
                            j_mortar_l_inner = j_large_inner
                        else
                            j_mortar_l_inner = k_large_inner
                        end

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
end # @muladd
