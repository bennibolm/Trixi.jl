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

    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_element = cache.interfaces.neighbor_ids[1, interface]
        right_element = cache.interfaces.neighbor_ids[2, interface]

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
            var_left = u[variable, index_left..., left_element]
            var_right = u[variable, index_right..., right_element]

            var_min[index_right..., right_element] = min(var_min[index_right...,
                                                                 right_element],
                                                         var_left)
            var_max[index_right..., right_element] = max(var_max[index_right...,
                                                                 right_element],
                                                         var_left)

            var_min[index_left..., left_element] = min(var_min[index_left...,
                                                               left_element], var_right)
            var_max[index_left..., left_element] = max(var_max[index_left...,
                                                               left_element], var_right)
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

    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_element = cache.interfaces.neighbor_ids[1, interface]
        right_element = cache.interfaces.neighbor_ids[2, interface]

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
            var_left = variable(get_node_vars(u, equations, dg, index_left...,
                                              left_element),
                                equations)
            var_right = variable(get_node_vars(u, equations, dg, index_right...,
                                               right_element),
                                 equations)

            var_minmax[index_right..., right_element] = min_or_max(var_minmax[index_right...,
                                                                              right_element],
                                                                   var_left)
            var_minmax[index_left..., left_element] = min_or_max(var_minmax[index_left...,
                                                                            left_element],
                                                                 var_right)
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
    calc_bounds_twosided!(var_min, var_max, variable, u, t, semi, equations)

    @threaded for element in eachelement(dg, semi.cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)
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
            val_flux3_local_jp1 = -inverse_weights[k] *
                                  antidiffusive_flux3_L[variable, i, j, k + 1, element]

            Pp = max(0, val_flux1_local) + max(0, val_flux1_local_ip1) +
                 max(0, val_flux2_local) + max(0, val_flux2_local_jp1) +
                 max(0, val_flux3_local) + max(0, val_flux3_local_jp1)
            Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
                 min(0, val_flux2_local) + min(0, val_flux2_local_jp1) +
                 min(0, val_flux3_local) + min(0, val_flux3_local_jp1)

            Pp = inverse_jacobian * Pp
            Pm = inverse_jacobian * Pm

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qp = abs(Qp) /
                 (abs(Pp) + eps(typeof(Qp)) * 100 * abs(var_max[i, j, k, element]))
            Qm = abs(Qm) /
                 (abs(Pm) + eps(typeof(Qm)) * 100 * abs(var_max[i, j, k, element]))

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
    calc_bounds_onesided!(var_minmax, min_or_max, variable, u, t, semi)

    # Perform Newton's bisection method to find new alpha
    @threaded for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
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
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)
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
# Newton-bisection method

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

# Local two-sided limiting of conservative variables
@inline function limiting_positivity_conservative!(limiting_factor, u, dt, semi,
                                                   mesh::TreeMesh{3}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; orientations) = cache.mortars
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; positivity_correction_factor) = dg.volume_integral.limiter

    for mortar in eachmortar(dg, cache)
        small_element_1 = cache.mortars.neighbor_ids[1, mortar]
        small_element_2 = cache.mortars.neighbor_ids[2, mortar]
        small_element_3 = cache.mortars.neighbor_ids[3, mortar]
        small_element_4 = cache.mortars.neighbor_ids[4, mortar]
        large_element = cache.mortars.neighbor_ids[5, mortar]

        # Compute minimal bound
        var_min_small_1 = typemax(eltype(surface_flux_values))
        var_min_small_2 = typemax(eltype(surface_flux_values))
        var_min_small_3 = typemax(eltype(surface_flux_values))
        var_min_small_4 = typemax(eltype(surface_flux_values))
        var_min_large = typemax(eltype(surface_flux_values))
        for j in eachnode(dg), i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i, j)
                    indices_large = (nnodes(dg), i, j)
                elseif orientations[mortar] == 2
                    # L2 mortars in y-direction
                    indices_small = (i, 1, j)
                    indices_large = (i, nnodes(dg), j)
                else # orientations[mortar] == 3
                    # L2 mortars in z-direction
                    indices_small = (i, j, 1)
                    indices_large = (i, j, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i, j)
                    indices_large = (1, i, j)
                elseif orientations[mortar] == 2
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg), j)
                    indices_large = (i, 1, j)
                else # orientations[mortar] == 3
                    # L2 mortars in z-direction
                    indices_small = (i, j, nnodes(dg))
                    indices_large = (i, j, 1)
                end
            end
            var_small_1 = u[var_index, indices_small..., small_element_1]
            var_small_2 = u[var_index, indices_small..., small_element_2]
            var_small_3 = u[var_index, indices_small..., small_element_3]
            var_small_4 = u[var_index, indices_small..., small_element_4]
            var_large = u[var_index, indices_large..., large_element]
            var_min_small_1 = min(var_min_small_1, var_small_1)
            var_min_small_2 = min(var_min_small_2, var_small_2)
            var_min_small_3 = min(var_min_small_3, var_small_3)
            var_min_small_4 = min(var_min_small_4, var_small_4)
            var_min_large = min(var_min_large, var_large)
        end
        var_min_small_1 = positivity_correction_factor * var_min_small_1
        var_min_small_1 = positivity_correction_factor * var_min_small_1
        var_min_small_2 = positivity_correction_factor * var_min_small_2
        var_min_small_3 = positivity_correction_factor * var_min_small_3
        var_min_small_4 = positivity_correction_factor * var_min_small_4
        var_min_large = positivity_correction_factor * var_min_large

        # Set up correct direction and factors
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 1
                direction_large = 2
            elseif orientations[mortar] == 2
                # L2 mortars in y-direction
                direction_small = 3
                direction_large = 4
            elseif orientations[mortar] == 3
                # L2 mortars in z-direction
                direction_small = 5
                direction_large = 6
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 2
                direction_large = 1
            elseif orientations[mortar] == 2
                # L2 mortars in y-direction
                direction_small = 4
                direction_large = 3
            elseif orientations[mortar] == 3
                # L2 mortars in z-direction
                direction_small = 6
                direction_large = 5
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        # Compute limiting factor
        for j in eachnode(dg), i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i, j)
                    indices_large = (nnodes(dg), i, j)
                elseif orientations[mortar] == 2
                    # L2 mortars in y-direction
                    indices_small = (i, 1, j)
                    indices_large = (i, nnodes(dg), j)
                else # orientations[mortar] == 3
                    # L2 mortars in z-direction
                    indices_small = (i, j, 1)
                    indices_large = (i, j, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i, j)
                    indices_large = (1, i, j)
                elseif orientations[mortar] == 2
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg), j)
                    indices_large = (i, 1, j)
                else # orientations[mortar] == 3
                    # L2 mortars in z-direction
                    indices_small = (i, j, nnodes(dg))
                    indices_large = (i, j, 1)
                end
            end
            var_small_1 = u[var_index, indices_small..., small_element_1]
            var_small_2 = u[var_index, indices_small..., small_element_2]
            var_small_3 = u[var_index, indices_small..., small_element_3]
            var_small_4 = u[var_index, indices_small..., small_element_4]
            var_large = u[var_index, indices_large..., large_element]

            if min(var_small_1, var_small_2, var_small_3, var_small_4, var_large) < 0
                error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
            end

            # Compute flux differences
            flux_small_1_high_order = surface_flux_values_high_order[var_index, i, j,
                                                                     direction_small,
                                                                     small_element_1]
            flux_small_1_low_order = surface_flux_values[var_index, i, j,
                                                         direction_small,
                                                         small_element_1]
            flux_difference_small_1 = factor_small *
                                      (flux_small_1_high_order - flux_small_1_low_order)

            flux_small_2_high_order = surface_flux_values_high_order[var_index, i, j,
                                                                     direction_small,
                                                                     small_element_2]
            flux_small_2_low_order = surface_flux_values[var_index, i, j,
                                                         direction_small,
                                                         small_element_2]
            flux_difference_small_2 = factor_small *
                                      (flux_small_2_high_order - flux_small_2_low_order)

            flux_small_3_high_order = surface_flux_values_high_order[var_index, i, j,
                                                                     direction_small,
                                                                     small_element_3]
            flux_small_3_low_order = surface_flux_values[var_index, i, j,
                                                         direction_small,
                                                         small_element_3]
            flux_difference_small_3 = factor_small *
                                      (flux_small_3_high_order - flux_small_3_low_order)

            flux_small_4_high_order = surface_flux_values_high_order[var_index, i, j,
                                                                     direction_small,
                                                                     small_element_4]
            flux_small_4_low_order = surface_flux_values[var_index, i, j,
                                                         direction_small,
                                                         small_element_4]
            flux_difference_small_4 = factor_small *
                                      (flux_small_4_high_order - flux_small_4_low_order)

            flux_large_high_order = surface_flux_values_high_order[var_index, i, j,
                                                                   direction_large,
                                                                   large_element]
            flux_large_low_order = surface_flux_values[var_index, i, j, direction_large,
                                                       large_element]
            flux_difference_large = factor_large *
                                    (flux_large_high_order - flux_large_low_order)

            # Use pure low-order fluxes if high-order fluxes are not finite.
            if !isfinite(flux_small_1_high_order) ||
               !isfinite(flux_small_2_high_order) ||
               !isfinite(flux_small_3_high_order) ||
               !isfinite(flux_small_4_high_order) ||
               !isfinite(flux_large_high_order)
                limiting_factor[mortar] = 1
                break
            end

            inverse_jacobian_small_1 = get_inverse_jacobian(inverse_jacobian, mesh,
                                                            indices_small...,
                                                            small_element_1)
            inverse_jacobian_small_2 = get_inverse_jacobian(inverse_jacobian, mesh,
                                                            indices_small...,
                                                            small_element_2)
            inverse_jacobian_small_3 = get_inverse_jacobian(inverse_jacobian, mesh,
                                                            indices_small...,
                                                            small_element_3)
            inverse_jacobian_small_4 = get_inverse_jacobian(inverse_jacobian, mesh,
                                                            indices_small...,
                                                            small_element_4)
            inverse_jacobian_large = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_large...,
                                                          large_element)

            # Real one-sided Zalesak-type limiter
            # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #       for each mortar, not each node
            Qm_small_1 = min(0, var_min_small_1 - var_small_1)
            Qm_small_2 = min(0, var_min_small_2 - var_small_2)
            Qm_small_3 = min(0, var_min_small_3 - var_small_3)
            Qm_small_4 = min(0, var_min_small_4 - var_small_4)
            Qm_large = min(0, var_min_large - var_large)

            Pm_small_1 = min(0, flux_difference_small_1)
            Pm_small_2 = min(0, flux_difference_small_2)
            Pm_small_3 = min(0, flux_difference_small_3)
            Pm_small_4 = min(0, flux_difference_small_4)
            Pm_large = min(0, flux_difference_large)

            Pm_small_1 = dt * inverse_jacobian_small_1 * Pm_small_1
            Pm_small_2 = dt * inverse_jacobian_small_2 * Pm_small_2
            Pm_small_3 = dt * inverse_jacobian_small_3 * Pm_small_3
            Pm_small_4 = dt * inverse_jacobian_small_4 * Pm_small_4
            Pm_large = dt * inverse_jacobian_large * Pm_large

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qm_small_1 = abs(Qm_small_1) /
                         (abs(Pm_small_1) + eps(typeof(Qm_small_1)) * 100)
            Qm_small_2 = abs(Qm_small_2) /
                         (abs(Pm_small_2) + eps(typeof(Qm_small_2)) * 100)
            Qm_small_3 = abs(Qm_small_3) /
                         (abs(Pm_small_3) + eps(typeof(Qm_small_3)) * 100)
            Qm_small_4 = abs(Qm_small_4) /
                         (abs(Pm_small_4) + eps(typeof(Qm_small_4)) * 100)
            Qm_large = abs(Qm_large) / (abs(Pm_large) + eps(typeof(Qm_large)) * 100)

            # Calculate limiting factor
            limiting_factor[mortar] = max(limiting_factor[mortar],
                                          1 - Qm_small_1, 1 - Qm_small_2,
                                          1 - Qm_small_3, 1 - Qm_small_4,
                                          1 - Qm_large)
        end
    end

    return nothing
end

##############################################################################
# Local one-sided limiting of nonlinear variables
@inline function limiting_positivity_nonlinear!(limiting_factor, u, dt, semi,
                                                mesh::TreeMesh{3}, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; orientations, neighbor_ids) = cache.mortars
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis

    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.volume_integral
    (; positivity_correction_factor) = limiter

    for mortar in eachmortar(dg, cache)
        small_element_1 = neighbor_ids[1, mortar]
        small_element_2 = neighbor_ids[2, mortar]
        small_element_3 = neighbor_ids[3, mortar]
        small_element_4 = neighbor_ids[4, mortar]
        large_element = neighbor_ids[5, mortar]

        # Set up correct direction and factors
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 1
                direction_large = 2
            elseif orientations[mortar] == 2
                # L2 mortars in y-direction
                direction_small = 3
                direction_large = 4
            else # orientations[mortar] == 3
                # L2 mortars in z-direction
                direction_small = 5
                direction_large = 6
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 2
                direction_large = 1
            elseif orientations[mortar] == 2
                # L2 mortars in y-direction
                direction_small = 4
                direction_large = 3
            else # orientations[mortar] == 3
                # L2 mortars in z-direction
                direction_small = 6
                direction_large = 5
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        # Compute limiting factor
        for j in eachnode(dg), i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i, j)
                    indices_large = (nnodes(dg), i, j)
                elseif orientations[mortar] == 2
                    # L2 mortars in y-direction
                    indices_small = (i, 1, j)
                    indices_large = (i, nnodes(dg), j)
                else # orientations[mortar] == 3
                    # L2 mortars in z-direction
                    indices_small = (i, j, 1)
                    indices_large = (i, j, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i, j)
                    indices_large = (1, i, j)
                elseif orientations[mortar] == 2
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg), j)
                    indices_large = (i, 1, j)
                else # orientations[mortar] == 3
                    # L2 mortars in z-direction
                    indices_small = (i, j, nnodes(dg))
                    indices_large = (i, j, 1)
                end
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

            var_min_small_1 = positivity_correction_factor * var_small_1
            var_min_small_2 = positivity_correction_factor * var_small_2
            var_min_small_3 = positivity_correction_factor * var_small_3
            var_min_small_4 = positivity_correction_factor * var_small_4
            var_min_small = (var_min_small_1, var_min_small_2,
                             var_min_small_3, var_min_small_4)
            var_min_large = positivity_correction_factor * var_large

            # small elements
            for small_element_index in 1:4
                small_element = neighbor_ids[small_element_index, mortar]

                inverse_jacobian_small = get_inverse_jacobian(inverse_jacobian, mesh,
                                                              indices_small...,
                                                              small_element)
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
                    break # TODO: Actually, I need a double break here
                end
                flux_difference_small = factor_small *
                                        (flux_small_high_order .- flux_small_low_order)
                antidiffusive_flux_small = inverse_jacobian_small *
                                           flux_difference_small

                u_small = get_node_vars(u, equations, dg, indices_small...,
                                        small_element)
                newton_loop!(limiting_factor, var_min_small[small_element_index],
                             u_small, (mortar,), variable, min,
                             initial_check_nonnegative_newton_idp,
                             final_check_nonnegative_newton_idp,
                             equations, dt, limiter, antidiffusive_flux_small)
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
            flux_difference_large = factor_large *
                                    (flux_large_high_order .- flux_large_low_order)
            antidiffusive_flux_large = inverse_jacobian_large * flux_difference_large

            newton_loop!(limiting_factor, var_min_large, u_large, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_large)
        end
    end

    return nothing
end
end # @muladd
