# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

###############################################################################
# IDP Limiting
###############################################################################

###############################################################################
# Calculation of local bounds using low-order FV solution

@inline function calc_bounds_twosided!(var_min, var_max, variable,
                                       u::AbstractArray{<:Any, 5}, t,
                                       semi, equations)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)

    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        # Calculate bounds at Gauss-Lobatto nodes
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, k, element]
            var_min[i, j, k, element] = var
            var_max[i, j, k, element] = var
        end

        # Apply values in x direction
        for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)
            var = u[variable, i - 1, j, k, element]
            var_min[i, j, k, element] = min(var_min[i, j, k, element], var)
            var_max[i, j, k, element] = max(var_max[i, j, k, element], var)

            var = u[variable, i, j, k, element]
            var_min[i - 1, j, k, element] = min(var_min[i - 1, j, k, element], var)
            var_max[i - 1, j, k, element] = max(var_max[i - 1, j, k, element], var)
        end

        # Apply values in y direction
        for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
            var = u[variable, i, j - 1, k, element]
            var_min[i, j, k, element] = min(var_min[i, j, k, element], var)
            var_max[i, j, k, element] = max(var_max[i, j, k, element], var)

            var = u[variable, i, j, k, element]
            var_min[i, j - 1, k, element] = min(var_min[i, j - 1, k, element], var)
            var_max[i, j - 1, k, element] = max(var_max[i, j - 1, k, element], var)
        end

        # Apply values in z direction
        for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, k - 1, element]
            var_min[i, j, k, element] = min(var_min[i, j, k, element], var)
            var_max[i, j, k, element] = max(var_max[i, j, k, element], var)

            var = u[variable, i, j, k, element]
            var_min[i, j, k - 1, element] = min(var_min[i, j, k - 1, element], var)
            var_max[i, j, k - 1, element] = max(var_max[i, j, k - 1, element], var)
        end
    end

    # Calc bounds at element interfaces and periodic boundaries
    calc_bounds_twosided_interface!(var_min, var_max, variable, u,
                                    semi, mesh, equations)

    # Calc bounds at mortars
    calc_bounds_twosided_mortar!(var_min, var_max, variable, u, semi, mesh)

    # Calc bounds at physical boundaries
    (; boundary_conditions) = semi
    calc_bounds_twosided_boundary!(var_min, var_max, variable, u, t,
                                   boundary_conditions,
                                   mesh, equations, dg, cache)
    return nothing
end

@inline function calc_bounds_twosided_interface!(var_min, var_max, variable, u,
                                                 semi, mesh::TreeMesh3D, equations)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations) = cache.interfaces

    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_element = neighbor_ids[1, interface]
        right_element = neighbor_ids[2, interface]

        if perform_subcell_limiting(dg.volume_integral, left_element) ||
           perform_subcell_limiting(dg.volume_integral, right_element)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this interface
        else
            # Subcell limiting is not necessary for both elements => Skip this interface
            continue
        end

        orientation = cache.interfaces.orientations[interface]

        for j in eachnode(dg), i in eachnode(dg)
            # Define node indices for left and right element based on the interface orientation
            if orientation == 1
                # interface in x-direction
                index_left = (nnodes(dg), i, j)
                index_right = (1, i, j)
            elseif orientation == 2
                # interface in y-direction
                index_left = (i, nnodes(dg), j)
                index_right = (i, 1, j)
            else # if orientation == 3
                # interface in z-direction
                index_left = (i, j, nnodes(dg))
                index_right = (i, j, 1)
            end

            if perform_subcell_limiting(dg.volume_integral, right_element)
                var_left = u[variable, index_left..., left_element]
                var_min[index_right..., right_element] = min(var_min[index_right...,
                                                                     right_element],
                                                             var_left)
                var_max[index_right..., right_element] = max(var_max[index_right...,
                                                                     right_element],
                                                             var_left)
            end

            if perform_subcell_limiting(dg.volume_integral, left_element)
                var_right = u[variable, index_right..., right_element]
                var_min[index_left..., left_element] = min(var_min[index_left...,
                                                                   left_element],
                                                           var_right)
                var_max[index_left..., left_element] = max(var_max[index_left...,
                                                                   left_element],
                                                           var_right)
            end
        end
    end

    return nothing
end

@inline function calc_bounds_twosided_mortar!(var_min, var_max, variable, u,
                                              semi, mesh::TreeMesh3D)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars

    # - For LobattoLegendreMortarIDP: include only values of nodes with nonnegative local weights
    # - For LobattoLegendreMortarL2: include all neighboring values (TODO?)
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[5, mortar]

        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            node_small = 1
            node_large = nnodes(dg)
        else # large_sides[mortar] == 2 -> small elements on left side
            node_small = nnodes(dg)
            node_large = 1
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
            else
                # L2 mortars in z-direction
                indices_small = (i, j, node_small)
                indices_large = (i, j, node_large)
            end

            # Get solution data
            var_small = (u[variable, indices_small...,
                           neighbor_ids[1, mortar]],
                         u[variable, indices_small...,
                           neighbor_ids[2, mortar]],
                         u[variable, indices_small...,
                           neighbor_ids[3, mortar]],
                         u[variable, indices_small...,
                           neighbor_ids[4, mortar]])
            var_large = u[variable, indices_large..., large_element]

            for l in eachnode(dg), k in eachnode(dg)
                if orientation == 1
                    # L2 mortars in x-direction
                    indices_small_inner = (node_small, k, l)
                    indices_large_inner = (node_large, k, l)
                elseif orientation == 2
                    # L2 mortars in y-direction
                    indices_small_inner = (k, node_small, l)
                    indices_large_inner = (k, node_large, l)
                else
                    # L2 mortars in z-direction
                    indices_small_inner = (k, l, node_small)
                    indices_large_inner = (k, l, node_large)
                end

                for small_element_index in 1:4
                    small_element = neighbor_ids[small_element_index, mortar]
                    # from large to small element
                    if l2_mortars ||
                       dg.mortar.mortar_weights[i, j, k, l, small_element_index] > 0
                        var_min[indices_small_inner..., small_element] = min(var_min[indices_small_inner...,
                                                                                     small_element],
                                                                             var_large)
                        var_max[indices_small_inner..., small_element] = max(var_max[indices_small_inner...,
                                                                                     small_element],
                                                                             var_large)
                    end
                    # from small to large element
                    if l2_mortars ||
                       dg.mortar.mortar_weights[k, l, i, j, small_element_index] > 0
                        var_min[indices_large_inner..., large_element] = min(var_min[indices_large_inner...,
                                                                                     large_element],
                                                                             var_small[small_element_index])
                        var_max[indices_large_inner..., large_element] = max(var_max[indices_large_inner...,
                                                                                     large_element],
                                                                             var_small[small_element_index])
                    end
                end
            end
        end
    end

    return nothing
end

@inline function calc_bounds_twosided_boundary!(var_min, var_max, variable,
                                                u, t, boundary_conditions,
                                                mesh::TreeMesh{3}, equations,
                                                dg, cache)
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for j in eachnode(dg), i in eachnode(dg)
            # Define node indices and boundary index based on the orientation and neighbor_side
            if neighbor_side == 2 # Element is on the right, boundary on the left
                if orientation == 1 # boundary in x-direction
                    node_index = (1, i, j)
                    boundary_index = 1
                elseif orientation == 2 # boundary in y-direction
                    node_index = (i, 1, j)
                    boundary_index = 3
                else # orientation == 3 # boundary in z-direction
                    node_index = (i, j, 1)
                    boundary_index = 5
                end
            else # Element is on the left, boundary on the right
                if orientation == 1 # boundary in x-direction
                    node_index = (nnodes(dg), i, j)
                    boundary_index = 2
                elseif orientation == 2 # boundary in y-direction
                    node_index = (i, nnodes(dg), j)
                    boundary_index = 4
                else # orientation == 3 # boundary in z-direction
                    node_index = (i, j, nnodes(dg))
                    boundary_index = 6
                end
            end
            u_inner = get_node_vars(u, equations, dg, node_index..., element)
            u_outer = get_boundary_outer_state(u_inner, t,
                                               boundary_conditions[boundary_index],
                                               orientation, boundary_index,
                                               mesh, equations, dg, cache,
                                               node_index..., element)
            var_outer = u_outer[variable]

            var_min[node_index..., element] = min(var_min[node_index..., element],
                                                  var_outer)
            var_max[node_index..., element] = max(var_max[node_index..., element],
                                                  var_outer)
        end
    end

    return nothing
end

@inline function calc_bounds_onesided!(var_minmax, min_or_max, variable,
                                       u::AbstractArray{<:Any, 5}, t,
                                       semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    # The approach used in `calc_bounds_twosided!` is not used here because it requires more
    # evaluations of the variable and is therefore slower.

    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        # Reset bounds
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            if min_or_max === max
                var_minmax[i, j, k, element] = typemin(eltype(var_minmax))
            else
                var_minmax[i, j, k, element] = typemax(eltype(var_minmax))
            end
        end

        # Calculate bounds at Gauss-Lobatto nodes
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            var = variable(get_node_vars(u, equations, dg, i, j, k, element), equations)
            var_minmax[i, j, k, element] = min_or_max(var_minmax[i, j, k, element], var)

            if i > 1
                var_minmax[i - 1, j, k, element] = min_or_max(var_minmax[i - 1, j, k,
                                                                         element], var)
            end
            if i < nnodes(dg)
                var_minmax[i + 1, j, k, element] = min_or_max(var_minmax[i + 1, j, k,
                                                                         element], var)
            end
            if j > 1
                var_minmax[i, j - 1, k, element] = min_or_max(var_minmax[i, j - 1, k,
                                                                         element], var)
            end
            if j < nnodes(dg)
                var_minmax[i, j + 1, k, element] = min_or_max(var_minmax[i, j + 1, k,
                                                                         element], var)
            end
            if k > 1
                var_minmax[i, j, k - 1, element] = min_or_max(var_minmax[i, j, k - 1,
                                                                         element], var)
            end
            if k < nnodes(dg)
                var_minmax[i, j, k + 1, element] = min_or_max(var_minmax[i, j, k + 1,
                                                                         element], var)
            end
        end
    end

    # Calc bounds at element interfaces and periodic boundaries
    calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u,
                                    semi, mesh)

    # Calc bounds at mortars
    calc_bounds_onesided_mortar!(var_minmax, min_or_max, variable, u, semi, mesh)

    # Calc bounds at physical boundaries
    (; boundary_conditions) = semi
    calc_bounds_onesided_boundary!(var_minmax, min_or_max, variable, u, t,
                                   boundary_conditions,
                                   mesh, equations, dg, cache)

    return nothing
end

@inline function calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u,
                                                 semi, mesh::TreeMesh{3})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations) = cache.interfaces

    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_element = neighbor_ids[1, interface]
        right_element = neighbor_ids[2, interface]

        if perform_subcell_limiting(dg.volume_integral, left_element) ||
           perform_subcell_limiting(dg.volume_integral, right_element)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this interface
        else
            # Subcell limiting is not necessary for both elements => Skip this interface
            continue
        end

        orientation = cache.interfaces.orientations[interface]

        for j in eachnode(dg), i in eachnode(dg)
            # Define node indices for left and right element based on the interface orientation
            if orientation == 1
                # interface in x-direction
                index_left = (nnodes(dg), i, j)
                index_right = (1, i, j)
            elseif orientation == 2
                # interface in y-direction
                index_left = (i, nnodes(dg), j)
                index_right = (i, 1, j)
            else # if orientation == 3
                # interface in z-direction
                index_left = (i, j, nnodes(dg))
                index_right = (i, j, 1)
            end

            if perform_subcell_limiting(dg.volume_integral, right_element)
                u_left = get_node_vars(u, equations, dg, index_left..., left_element)
                var_left = variable(u_left, equations)
                var_minmax[index_right..., right_element] = min_or_max(var_minmax[index_right...,
                                                                                  right_element],
                                                                       var_left)
            end
            if perform_subcell_limiting(dg.volume_integral, left_element)
                u_right = get_node_vars(u, equations, dg, index_right..., right_element)
                var_right = variable(u_right, equations)
                var_minmax[index_left..., left_element] = min_or_max(var_minmax[index_left...,
                                                                                left_element],
                                                                     var_right)
            end
        end
    end

    return nothing
end

@inline function calc_bounds_onesided_mortar!(var_minmax, min_or_max, variable, u,
                                              semi, mesh::TreeMesh{3})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars

    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[5, mortar]

        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            node_small = 1
            node_large = nnodes(dg)
        else # large_sides[mortar] == 2 -> small elements on left side
            node_small = nnodes(dg)
            node_large = 1
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
            else
                # L2 mortars in z-direction
                indices_small = (i, j, node_small)
                indices_large = (i, j, node_large)
            end

            u_small = (get_node_vars(u, equations, dg, indices_small...,
                                     neighbor_ids[1, mortar]),
                       get_node_vars(u, equations, dg, indices_small...,
                                     neighbor_ids[2, mortar]),
                       get_node_vars(u, equations, dg, indices_small...,
                                     neighbor_ids[3, mortar]),
                       get_node_vars(u, equations, dg, indices_small...,
                                     neighbor_ids[4, mortar]))
            u_large = get_node_vars(u, equations, dg, indices_large..., large_element)

            var_small = (variable(u_small[1], equations),
                         variable(u_small[2], equations),
                         variable(u_small[3], equations),
                         variable(u_small[4], equations))
            var_large = variable(u_large, equations)

            for l in eachnode(dg), k in eachnode(dg)
                if orientation == 1
                    # L2 mortars in x-direction
                    indices_small_inner = (node_small, k, l)
                    indices_large_inner = (node_large, k, l)
                elseif orientation == 2
                    # L2 mortars in y-direction
                    indices_small_inner = (k, node_small, l)
                    indices_large_inner = (k, node_large, l)
                else
                    # L2 mortars in z-direction
                    indices_small_inner = (k, l, node_small)
                    indices_large_inner = (k, l, node_large)
                end

                for small_element_index in 1:4
                    small_element = neighbor_ids[small_element_index, mortar]
                    # values of large element to small elements
                    if l2_mortars ||
                       dg.mortar.mortar_weights[i, j, k, l, small_element_index] > 0
                        var_minmax[indices_small_inner..., small_element] = min_or_max(var_minmax[indices_small_inner...,
                                                                                                  small_element],
                                                                                       var_large)
                    end
                    # values of small elements to large element
                    if l2_mortars ||
                       dg.mortar.mortar_weights[k, l, i, j, small_element_index] > 0
                        var_minmax[indices_large_inner..., large_element] = min_or_max(var_minmax[indices_large_inner...,
                                                                                                  large_element],
                                                                                       var_small[small_element_index])
                    end
                end
            end
        end
    end

    return nothing
end

@inline function calc_bounds_onesided_boundary!(var_minmax, min_or_max, variable,
                                                u, t, boundary_conditions,
                                                mesh::TreeMesh{3}, equations,
                                                dg, cache)
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for j in eachnode(dg), i in eachnode(dg)
            # Define node indices and boundary index based on the orientation and neighbor_side
            if neighbor_side == 2 # Element is on the right, boundary on the left
                if orientation == 1 # boundary in x-direction
                    node_index = (1, i, j)
                    boundary_index = 1
                elseif orientation == 2 # boundary in y-direction
                    node_index = (i, 1, j)
                    boundary_index = 3
                else # orientation == 3 # boundary in z-direction
                    node_index = (i, j, 1)
                    boundary_index = 5
                end
            else # Element is on the left, boundary on the right
                if orientation == 1 # boundary in x-direction
                    node_index = (nnodes(dg), i, j)
                    boundary_index = 2
                elseif orientation == 2 # boundary in y-direction
                    node_index = (i, nnodes(dg), j)
                    boundary_index = 4
                else # orientation == 3 # boundary in z-direction
                    node_index = (i, j, nnodes(dg))
                    boundary_index = 6
                end
            end
            u_inner = get_node_vars(u, equations, dg, node_index..., element)
            u_outer = get_boundary_outer_state(u_inner, t,
                                               boundary_conditions[boundary_index],
                                               orientation, boundary_index,
                                               mesh, equations, dg, cache,
                                               node_index..., element)
            var_outer = variable(u_outer, equations)

            var_minmax[node_index..., element] = min_or_max(var_minmax[node_index...,
                                                                       element],
                                                            var_outer)
        end
    end

    return nothing
end

###############################################################################
# Local minimum and maximum limiting of conservative variables

@inline function idp_local_twosided!(alpha, limiter, u::AbstractArray{<:Any, 5},
                                     t, dt, semi, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    variable_string = string(variable)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]
    if limiter.bar_states == false
        calc_bounds_twosided!(var_min, var_max, variable, u, t, semi, equations)
    end

    @threaded for element in eachelement(dg, semi.cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            isone(alpha[i, j, k, element]) && continue # Skip if alpha is already 1

            var = u[variable, i, j, k, element]
            # Real Zalesak type limiter
            #   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            #   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            #   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #         for each interface, not each node

            Qp = max(0, (var_max[i, j, k, element] - var) / dt)
            Qm = min(0, (var_min[i, j, k, element] - var) / dt)

            # Calculate Pp and Pm
            # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
            val_flux1_local = inverse_weights[i] *
                              antidiffusive_flux1_R[variable, i, j, k, element]
            val_flux1_local_ip1 = -inverse_weights[i] *
                                  antidiffusive_flux1_L[variable, i + 1, j, k, element]
            val_flux2_local = inverse_weights[j] *
                              antidiffusive_flux2_R[variable, i, j, k, element]
            val_flux2_local_jp1 = -inverse_weights[j] *
                                  antidiffusive_flux2_L[variable, i, j + 1, k, element]
            val_flux3_local = inverse_weights[k] *
                              antidiffusive_flux3_R[variable, i, j, k, element]
            val_flux3_local_kp1 = -inverse_weights[k] *
                                  antidiffusive_flux3_L[variable, i, j, k + 1, element]

            Pp = max(0, val_flux1_local) + max(0, val_flux1_local_ip1) +
                 max(0, val_flux2_local) + max(0, val_flux2_local_jp1) +
                 max(0, val_flux3_local) + max(0, val_flux3_local_kp1)
            Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
                 min(0, val_flux2_local) + min(0, val_flux2_local_jp1) +
                 min(0, val_flux3_local) + min(0, val_flux3_local_kp1)

            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)
            Pp = inverse_jacobian * Pp
            Pm = inverse_jacobian * Pm

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            eps_ = eps(typeof(Qp)) * 100 * abs(var_max[i, j, k, element])
            Qp = abs(Qp) / (abs(Pp) + eps_)
            Qm = abs(Qm) / (abs(Pm) + eps_)

            # Calculate alpha at nodes
            alpha[i, j, k, element] = max(alpha[i, j, k, element], 1 - min(1, Qp, Qm))
        end
    end

    return nothing
end

##############################################################################
# Local one-sided limiting of nonlinear variables

@inline function idp_local_onesided!(alpha, limiter, u::AbstractArray{<:Real, 5},
                                     t, dt, semi,
                                     variable, min_or_max)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]
    if limiter.bar_states == false
        calc_bounds_onesided!(var_minmax, min_or_max, variable, u, t, semi)
    end

    # Perform Newton's bisection method to find new alpha
    @threaded for element in eachelement(dg, cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            isone(alpha[i, j, k, element]) && continue # Skip if alpha is already 1

            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)
            u_local = get_node_vars(u, equations, dg, i, j, k, element)
            newton_loops_alpha!(alpha, var_minmax[i, j, k, element],
                                u_local, i, j, k, element,
                                variable, min_or_max,
                                initial_check_local_onesided_newton_idp,
                                final_check_local_onesided_newton_idp,
                                inverse_jacobian, dt, equations, dg, cache, limiter)
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of conservative variables

@inline function idp_positivity_conservative!(alpha, limiter,
                                              u::AbstractArray{<:Real, 5}, dt, semi,
                                              variable)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)
    (; antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis # Plays role of DG subcell sizes
    (; positivity_correction_factor) = limiter

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in eachelement(dg, semi.cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, k, element]
            if var < 0
                error("Safe low-order method produces negative value for conservative variable $variable. Try a smaller time step.")
            end

            # Compute bound
            if limiter.local_twosided &&
               (variable in limiter.local_twosided_variables_cons) &&
               (var_min[i, j, k, element] >= positivity_correction_factor * var)
                # Local limiting is more restrictive that positivity limiting
                # => Skip positivity limiting for this node
                continue
            end
            var_min[i, j, k, element] = positivity_correction_factor * var

            isone(alpha[i, j, k, element]) && continue # Skip if alpha is already 1

            # Real one-sided Zalesak-type limiter
            # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #       for each interface, not each node
            Qm = min(0, (var_min[i, j, k, element] - var) / dt)

            # Calculate Pm
            # Note: Boundaries of antidiffusive_flux1/2/3 are constant 0, so they make no difference here.
            val_flux1_local = inverse_weights[i] *
                              antidiffusive_flux1_R[variable, i, j, k, element]
            val_flux1_local_ip1 = -inverse_weights[i] *
                                  antidiffusive_flux1_L[variable, i + 1, j, k, element]
            val_flux2_local = inverse_weights[j] *
                              antidiffusive_flux2_R[variable, i, j, k, element]
            val_flux2_local_jp1 = -inverse_weights[j] *
                                  antidiffusive_flux2_L[variable, i, j + 1, k, element]
            val_flux3_local = inverse_weights[k] *
                              antidiffusive_flux3_R[variable, i, j, k, element]
            val_flux3_local_jp1 = -inverse_weights[k] *
                                  antidiffusive_flux3_L[variable, i, j, k + 1, element]

            Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
                 min(0, val_flux2_local) + min(0, val_flux2_local_jp1) +
                 min(0, val_flux3_local) + min(0, val_flux3_local_jp1)

            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)
            Pm = inverse_jacobian * Pm

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qm = abs(Qm) / (abs(Pm) + eps(typeof(Qm)) * 100)

            # Calculate alpha
            alpha[i, j, k, element] = max(alpha[i, j, k, element], 1 - Qm)
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of nonlinear variables

@inline function idp_positivity_nonlinear!(alpha, limiter,
                                           u::AbstractArray{<:Real, 5}, dt, semi,
                                           variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; positivity_correction_factor) = limiter

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in eachelement(dg, semi.cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)

            # Compute bound
            u_local = get_node_vars(u, equations, dg, i, j, k, element)
            var = variable(u_local, equations)
            if var < 0
                error("Safe low-order method produces negative value for variable $variable. Try a smaller time step.")
            end
            var_min[i, j, k, element] = positivity_correction_factor * var

            # Perform Newton's bisection method to find new alpha
            newton_loops_alpha!(alpha, var_min[i, j, k, element],
                                u_local, i, j, k, element,
                                variable, min,
                                initial_check_nonnegative_newton_idp,
                                final_check_nonnegative_newton_idp,
                                inverse_jacobian, dt, equations, dg, cache, limiter)
        end
    end

    return nothing
end

###############################################################################
# Auxiliary functions for Newton-bisection method

@inline function newton_loops_alpha!(alpha, bound, u, i, j, k, element,
                                     variable, min_or_max,
                                     initial_check, final_check,
                                     inverse_jacobian, dt,
                                     equations::AbstractEquations{3},
                                     dg, cache, limiter)
    (; inverse_weights) = dg.basis # Plays role of inverse DG-subcell sizes
    (; antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R) = cache.antidiffusive_fluxes

    (; gamma_constant_newton) = limiter

    indices = (i, j, k, element)
    isone(alpha[indices...]) && return # Skip if alpha is already 1

    # negative xi direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian *
                         inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive xi direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_L, equations, dg,
                                       i + 1, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # negative eta direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive eta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_L, equations, dg,
                                       i, j + 1, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # negative zeta direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian *
                         inverse_weights[k] *
                         get_node_vars(antidiffusive_flux3_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive zeta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[k] *
                         get_node_vars(antidiffusive_flux3_L, equations, dg,
                                       i, j, k + 1, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    return nothing
end

###############################################################################
# IDP mortar limiting
###############################################################################

@inline function precompute_n_mortars_per_nodes!(volume_integral::VolumeIntegralSubcellLimiting,
                                                 dg, cache, mesh::TreeMesh{3})
    if !(dg.mortar isa LobattoLegendreMortarIDP)
        return nothing
    end

    (; n_mortars_per_node) = volume_integral.limiter.cache.subcell_limiter_coefficients
    (; neighbor_ids, orientations, large_sides) = cache.mortars

    n_mortars_per_node .= zero(eltype(n_mortars_per_node))

    for mortar in eachmortar(dg, cache)
        small_element_1 = neighbor_ids[1, mortar]
        small_element_2 = neighbor_ids[2, mortar]
        small_element_3 = neighbor_ids[3, mortar]
        small_element_4 = neighbor_ids[4, mortar]
        large_element = neighbor_ids[5, mortar]

        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            node_small = 1
            node_large = nnodes(dg)
        else # large_sides[mortar] == 2 -> small elements on left side
            node_small = nnodes(dg)
            node_large = 1
        end

        for j in eachnode(dg), i in eachnode(dg)
            if orientation == 1
                indices_small = (node_small, i, j)
                indices_large = (node_large, i, j)
            elseif orientation == 2
                indices_small = (i, node_small, j)
                indices_large = (i, node_large, j)
            else # orientation == 3
                indices_small = (i, j, node_small)
                indices_large = (i, j, node_large)
            end

            n_mortars_per_node[indices_small..., small_element_1] += 1
            n_mortars_per_node[indices_small..., small_element_2] += 1
            n_mortars_per_node[indices_small..., small_element_3] += 1
            n_mortars_per_node[indices_small..., small_element_4] += 1
            n_mortars_per_node[indices_large..., large_element] += 1
        end
    end

    return nothing
end

###############################################################################
# Local minimum and maximum limiting of conservative variables

@inline function limiting_local_conservative!(limiting_factor, u, dt, semi,
                                              mesh::TreeMesh{3}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; variable_bounds, n_mortars_per_node) = dg.volume_integral.limiter.cache.subcell_limiter_coefficients
    variable_string = string(var_index)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]

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

        # Set up correct direction and factors
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

        # Compute limiting factor
        for j in eachnode(dg), i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
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

            # Large element
            var_large = u[var_index, indices_large..., large_element]
            if var_large < 0
                error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
            end

            # Two-sided local bounds
            var_min_large = var_min[indices_large..., large_element]
            var_max_large = var_max[indices_large..., large_element]

            # Real Zalesak type limiter
            #   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            #   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            #   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #         for each interface, not each node
            Qp_large = max(0, (var_max_large - var_large) / dt)
            Qm_large = min(0, (var_min_large - var_large) / dt)

            # Compute flux differences
            flux_large_high_order = surface_flux_values_high_order[var_index, i, j,
                                                                   direction_large,
                                                                   large_element]
            # Check if high-order flux is finite. Otherwise, use pure low-order fluxes.
            if !isfinite(flux_large_high_order)
                limiting_factor[mortar] = 1
                break
            end
            flux_large_low_order = surface_flux_values[var_index, i, j, direction_large,
                                                       large_element]
            flux_difference_large = factor_large *
                                    (flux_large_high_order - flux_large_low_order)

            inverse_jacobian_large = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_large...,
                                                          large_element)
            Pp_large = max(0, flux_difference_large)
            Pm_large = min(0, flux_difference_large)
            Pp_large = inverse_jacobian_large * Pp_large
            Pm_large = inverse_jacobian_large * Pm_large

            # A node can be on multiple mortars. Scale the antidiffusive flux contribution
            # to account for this. Similar to scaling with `gamma_constant_newton`.
            n_mortars_large = n_mortars_per_node[indices_large..., large_element]
            Pp_large *= n_mortars_large
            Pm_large *= n_mortars_large

            eps_ = eps(typeof(Qp_large)) * 100 * abs(var_max_large)
            Qp_large = abs(Qp_large) / (abs(Pp_large) + eps_)
            Qm_large = abs(Qm_large) / (abs(Pm_large) + eps_)

            # Calculate limiting factor
            Q = min(1, Qp_large, Qm_large)

            # Small elements
            for small_element_index in 1:4
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                small_element = neighbor_ids[small_element_index, mortar]
                var_small = u[var_index, indices_small..., small_element]
                if var_small < 0
                    error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
                end

                var_min_small = var_min[indices_small..., small_element]
                var_max_small = var_max[indices_small..., small_element]

                Qp_small = max(0, (var_max_small - var_small) / dt)
                Qm_small = min(0, (var_min_small - var_small) / dt)

                # Compute flux differences
                flux_small_high_order = surface_flux_values_high_order[var_index, i, j,
                                                                       direction_small,
                                                                       small_element]
                if !isfinite(flux_small_high_order)
                    limiting_factor[mortar] = 1
                    break
                end
                flux_small_low_order = surface_flux_values[var_index, i, j,
                                                           direction_small,
                                                           small_element]
                flux_difference_small = factor_small *
                                        (flux_small_high_order - flux_small_low_order)

                inverse_jacobian_small = get_inverse_jacobian(inverse_jacobian, mesh,
                                                              indices_small...,
                                                              small_element)
                Pp_small = max(0, flux_difference_small)
                Pm_small = min(0, flux_difference_small)
                Pp_small = inverse_jacobian_small * Pp_small
                Pm_small = inverse_jacobian_small * Pm_small

                n_mortars_small = n_mortars_per_node[indices_small..., small_element]
                Pp_small *= n_mortars_small
                Pm_small *= n_mortars_small

                eps_ = eps(typeof(Qp_small)) * 100 * abs(var_max_small)
                Qp_small = abs(Qp_small) / (abs(Pp_small) + eps_)
                Qm_small = abs(Qm_small) / (abs(Pm_small) + eps_)

                Q = min(Q, Qp_small, Qm_small)
            end

            # Calculate limiting factor
            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Q)
        end
    end

    return nothing
end

##############################################################################
# Local minimum or maximum limiting of nonlinear variables

@inline function limiting_local_nonlinear!(limiting_factor, u, dt, semi,
                                           mesh::TreeMesh{3}, variable,
                                           min_or_max)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]

    (; gamma_constant_newton) = limiter

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

            u_large = get_node_vars(u, equations, dg, indices_large..., large_element)

            bound_small = (var_minmax[indices_small..., small_element_1],
                           var_minmax[indices_small..., small_element_2],
                           var_minmax[indices_small..., small_element_3],
                           var_minmax[indices_small..., small_element_4])
            bound_large = var_minmax[indices_large..., large_element]

            # Large element
            flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, j, direction_large, large_element)
            flux_large_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, j, direction_large, large_element)
            if !all(isfinite, flux_large_high_order)
                limiting_factor[mortar] = 1
                break
            end
            inverse_jacobian_large = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_large...,
                                                          large_element)
            antidiffusive_flux_large = gamma_constant_newton * factor_large *
                                       inverse_jacobian_large *
                                       (flux_large_high_order .- flux_large_low_order)

            newton_loop!(limiting_factor, bound_large, u_large, (mortar,), variable,
                         min_or_max, initial_check_local_onesided_newton_idp,
                         final_check_local_onesided_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_large)

            # Small elements
            for small_element_index in 1:4
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                small_element = neighbor_ids[small_element_index, mortar]
                u_small = get_node_vars(u, equations, dg, indices_small...,
                                        small_element)

                flux_small_high_order = get_node_vars(surface_flux_values_high_order,
                                                      equations, dg,
                                                      i, j, direction_small,
                                                      small_element)
                flux_small_low_order = get_node_vars(surface_flux_values, equations,
                                                     dg, i, j, direction_small,
                                                     small_element)
                if !all(isfinite, flux_small_high_order)
                    limiting_factor[mortar] = 1
                    break
                end
                inverse_jacobian_small = get_inverse_jacobian(inverse_jacobian, mesh,
                                                              indices_small...,
                                                              small_element)
                antidiffusive_flux_small = gamma_constant_newton * factor_small *
                                           inverse_jacobian_small *
                                           (flux_small_high_order .-
                                            flux_small_low_order)

                newton_loop!(limiting_factor, bound_small[small_element_index],
                             u_small, (mortar,), variable, min_or_max,
                             initial_check_local_onesided_newton_idp,
                             final_check_local_onesided_newton_idp,
                             equations, dt, limiter, antidiffusive_flux_small)
            end
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of conservative variables
@inline function limiting_positivity_conservative!(limiting_factor, u, dt, semi,
                                                   mesh::TreeMesh{3}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; variable_bounds, n_mortars_per_node) = dg.volume_integral.limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(var_index), "_min")]

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

        # Set up correct direction and factors
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

        # Compute limiting factor
        for j in eachnode(dg), i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

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

            # Large element
            var_large = u[var_index, indices_large..., large_element]
            if var_large < 0
                error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
            end

            # Minimum bound
            var_min_large = var_min[indices_large..., large_element]

            flux_large_high_order = surface_flux_values_high_order[var_index, i, j,
                                                                   direction_large,
                                                                   large_element]
            # Check if high-order flux is finite. Otherwise, use pure low-order fluxes.
            if !isfinite(flux_large_high_order)
                limiting_factor[mortar] = 1
                break
            end
            flux_large_low_order = surface_flux_values[var_index, i, j, direction_large,
                                                       large_element]
            flux_difference_large = factor_large *
                                    (flux_large_high_order - flux_large_low_order)

            # Real one-sided Zalesak-type limiter
            # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #       for each mortar, not each node
            Qm_large = min(0, var_min_large - var_large)
            Pm_large = min(0, flux_difference_large)

            # A node can be on multiple mortars. Scale the antidiffusive flux contribution
            # to account for this. Similar to scaling with `gamma_constant_newton`.
            Pm_large *= n_mortars_per_node[indices_large..., large_element]

            inverse_jacobian_large = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_large...,
                                                          large_element)
            Pm_large = dt * inverse_jacobian_large * Pm_large

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            eps_ = eps(typeof(Qm_large)) * 100
            Qm_large = abs(Qm_large) / (abs(Pm_large) + eps_)
            Qm = min(1, Qm_large)

            # small elements
            for small_element_index in 1:4
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                small_element = neighbor_ids[small_element_index, mortar]
                var_small = u[var_index, indices_small..., small_element]
                if var_small < 0
                    error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
                end

                # Compute flux differences
                flux_small_high_order = surface_flux_values_high_order[var_index, i, j,
                                                                       direction_small,
                                                                       small_element]
                if !isfinite(flux_small_high_order)
                    limiting_factor[mortar] = 1
                    break
                end
                flux_small_low_order = surface_flux_values[var_index, i, j,
                                                           direction_small,
                                                           small_element]
                flux_difference_small = factor_small *
                                        (flux_small_high_order - flux_small_low_order)

                # Minimum bound
                var_min_small = var_min[indices_small..., small_element]
                Qm_small = min(0, var_min_small - var_small)
                Pm_small = min(0, flux_difference_small)

                # A node can be on multiple mortars. Scale the antidiffusive flux contribution
                # to account for this. Similar to scaling with `gamma_constant_newton`.
                Pm_small *= n_mortars_per_node[indices_small..., small_element]

                inverse_jacobian_small = get_inverse_jacobian(inverse_jacobian, mesh,
                                                              indices_small...,
                                                              small_element)
                Pm_small = dt * inverse_jacobian_small * Pm_small

                # Compute blending coefficient avoiding division by zero
                # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
                Qm_small = abs(Qm_small) / (abs(Pm_small) + eps_)
                Qm = min(Qm, Qm_small)
            end

            # Calculate limiting factor
            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Qm)
        end
    end

    return nothing
end

##############################################################################
# Global positivity limiting of nonlinear variables
@inline function limiting_positivity_nonlinear!(limiting_factor, u, dt, semi,
                                                mesh::TreeMesh{3}, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; orientations, large_sides, neighbor_ids) = cache.mortars
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis

    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.volume_integral
    (; variable_bounds) = dg.volume_integral.limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    (; gamma_constant_newton) = limiter

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        small_element_1 = neighbor_ids[1, mortar]
        small_element_2 = neighbor_ids[2, mortar]
        small_element_3 = neighbor_ids[3, mortar]
        small_element_4 = neighbor_ids[4, mortar]
        large_element = neighbor_ids[5, mortar]

        if perform_subcell_limiting(dg.volume_integral, small_element_1) ||
           perform_subcell_limiting(dg.volume_integral, small_element_2) ||
           perform_subcell_limiting(dg.volume_integral, small_element_3) ||
           perform_subcell_limiting(dg.volume_integral, small_element_4) ||
           perform_subcell_limiting(dg.volume_integral, large_element)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this mortar
        else
            # Subcell limiting is not necessary for all elements => Skip this mortar
            continue
        end

        # Set up correct direction and factors
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

        # Compute limiting factor
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

            u_small_1 = get_node_vars(u, equations, dg, indices_small...,
                                      small_element_1)
            u_small_2 = get_node_vars(u, equations, dg, indices_small...,
                                      small_element_2)
            u_small_3 = get_node_vars(u, equations, dg, indices_small...,
                                      small_element_3)
            u_small_4 = get_node_vars(u, equations, dg, indices_small...,
                                      small_element_4)
            u_large = get_node_vars(u, equations, dg, indices_large..., large_element)
            var_small_1 = variable(u_small_1, equations)
            var_small_1 = variable(u_small_1, equations)
            var_small_2 = variable(u_small_2, equations)
            var_small_3 = variable(u_small_3, equations)
            var_small_4 = variable(u_small_4, equations)
            var_large = variable(u_large, equations)
            if var_small_1 < 0 || var_small_2 < 0 ||
               var_small_3 < 0 || var_small_4 < 0 ||
               var_large < 0
                error("Safe low-order method produces negative value for variable $variable. Try a smaller time step.")
            end

            # Minimum bound
            var_min_small_1 = var_min[indices_small..., small_element_1]
            var_min_small_2 = var_min[indices_small..., small_element_2]
            var_min_small_3 = var_min[indices_small..., small_element_3]
            var_min_small_4 = var_min[indices_small..., small_element_4]
            var_min_small = (var_min_small_1, var_min_small_2,
                             var_min_small_3, var_min_small_4)
            var_min_large = var_min[indices_large..., large_element]

            # small elements
            for small_element_index in 1:4
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

                small_element = neighbor_ids[small_element_index, mortar]

                # Compute flux differences
                flux_small_high_order = get_node_vars(surface_flux_values_high_order,
                                                      equations, dg,
                                                      i, j, direction_small,
                                                      small_element)
                flux_small_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                     i, j, direction_small,
                                                     small_element)

                # Use pure low-order fluxes if high-order fluxes are not finite.
                if !(all(isfinite.(flux_small_high_order)))
                    limiting_factor[mortar] = 1
                    break
                end
                flux_difference_small = flux_small_high_order .- flux_small_low_order

                inverse_jacobian_small = get_inverse_jacobian(inverse_jacobian, mesh,
                                                              indices_small...,
                                                              small_element)
                antidiffusive_flux_small = gamma_constant_newton * factor_small *
                                           inverse_jacobian_small *
                                           flux_difference_small

                u_small = get_node_vars(u, equations, dg, indices_small...,
                                        small_element)
                newton_loop!(limiting_factor, var_min_small[small_element_index],
                             u_small, (mortar,), variable, min,
                             initial_check_nonnegative_newton_idp,
                             final_check_nonnegative_newton_idp,
                             equations, dt, limiter, antidiffusive_flux_small)
            end
            if limiting_factor[mortar] == 1
                break
            end

            # large element
            inverse_jacobian_large = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_large...,
                                                          large_element)
            # Compute flux differences
            flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, j, direction_large, large_element)
            flux_large_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, j, direction_large, large_element)
            # Use pure low-order fluxes if high-order fluxes are not finite.
            if !(all(isfinite.(flux_large_high_order)))
                limiting_factor[mortar] = 1
                break
            end
            flux_difference_large = flux_large_high_order .- flux_large_low_order
            antidiffusive_flux_large = gamma_constant_newton * factor_large *
                                       inverse_jacobian_large *
                                       flux_difference_large

            newton_loop!(limiting_factor, var_min_large, u_large, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_large)
        end
    end

    return nothing
end
end # @muladd
