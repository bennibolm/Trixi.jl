# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function refine_element!(u::AbstractArray{<:Any, 2}, element_id,
                         old_u, old_element_id,
                         adaptor::Nothing, equations::AbstractEquations{3}, solver::FV)
    # @unpack forward_upper, forward_lower = adaptor

    # Store new element ids
    bottom_lower_left_id = element_id
    bottom_lower_right_id = element_id + 1
    bottom_upper_left_id = element_id + 2
    bottom_upper_right_id = element_id + 3
    top_lower_left_id = element_id + 4
    top_lower_right_id = element_id + 5
    top_upper_left_id = element_id + 6
    top_upper_right_id = element_id + 7

    @boundscheck begin
        @assert old_element_id >= 1
        @assert size(old_u, 1) == nvariables(equations)
        @assert size(old_u, 2) >= old_element_id
        @assert element_id >= 1
        @assert size(u, 1) == nvariables(equations)
        @assert size(u, 2) >= element_id + 7
    end

    # # Interpolate to bottom lower left element
    # multiply_dimensionwise!(view(u, :, :, :, :, bottom_lower_left_id), forward_lower,
    #                         forward_lower, forward_lower,
    #                         view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # # Interpolate to bottom lower right element
    # multiply_dimensionwise!(view(u, :, :, :, :, bottom_lower_right_id), forward_upper,
    #                         forward_lower, forward_lower,
    #                         view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # # Interpolate to bottom upper left element
    # multiply_dimensionwise!(view(u, :, :, :, :, bottom_upper_left_id), forward_lower,
    #                         forward_upper, forward_lower,
    #                         view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # # Interpolate to bottom upper right element
    # multiply_dimensionwise!(view(u, :, :, :, :, bottom_upper_right_id), forward_upper,
    #                         forward_upper, forward_lower,
    #                         view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # # Interpolate to top lower left element
    # multiply_dimensionwise!(view(u, :, :, :, :, top_lower_left_id), forward_lower,
    #                         forward_lower, forward_upper,
    #                         view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # # Interpolate to top lower right element
    # multiply_dimensionwise!(view(u, :, :, :, :, top_lower_right_id), forward_upper,
    #                         forward_lower, forward_upper,
    #                         view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # # Interpolate to top upper left element
    # multiply_dimensionwise!(view(u, :, :, :, :, top_upper_left_id), forward_lower,
    #                         forward_upper, forward_upper,
    #                         view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # # Interpolate to top upper right element
    # multiply_dimensionwise!(view(u, :, :, :, :, top_upper_right_id), forward_upper,
    #                         forward_upper, forward_upper,
    #                         view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    u_local = get_node_vars(old_u, equations, solver, old_element_id)
    set_node_vars!(u, u_local, equations, solver, bottom_lower_left_id)
    set_node_vars!(u, u_local, equations, solver, bottom_lower_right_id)
    set_node_vars!(u, u_local, equations, solver, bottom_upper_left_id)
    set_node_vars!(u, u_local, equations, solver, bottom_upper_right_id)
    set_node_vars!(u, u_local, equations, solver, top_lower_left_id)
    set_node_vars!(u, u_local, equations, solver, top_lower_right_id)
    set_node_vars!(u, u_local, equations, solver, top_upper_left_id)
    set_node_vars!(u, u_local, equations, solver, top_upper_right_id)

    return nothing
end

function coarsen_elements!(u::AbstractArray{<:Any, 2}, element_id,
                           old_u, old_element_id,
                           adaptor::Nothing, equations::AbstractEquations{3},
                           solver::FV)
    # @unpack reverse_upper, reverse_lower = adaptor

    # Store old element ids
    bottom_lower_left_id = old_element_id
    bottom_lower_right_id = old_element_id + 1
    bottom_upper_left_id = old_element_id + 2
    bottom_upper_right_id = old_element_id + 3
    top_lower_left_id = old_element_id + 4
    top_lower_right_id = old_element_id + 5
    top_upper_left_id = old_element_id + 6
    top_upper_right_id = old_element_id + 7

    @boundscheck begin
        @assert old_element_id >= 1
        @assert size(old_u, 1) == nvariables(equations)
        @assert size(old_u, 2) >= old_element_id + 3
        @assert element_id >= 1
        @assert size(u, 1) == nvariables(equations)
        @assert size(u, 2) >= element_id
    end

    acc = zero(get_node_vars(u, equations, solver, element_id))

    # Project from bottom lower left element
    acc += get_node_vars(old_u, equations, solver, bottom_lower_left_id) #* reverse_lower[1, 1] * reverse_lower[1, 1]

    # Project from bottom lower right element_variables
    acc += get_node_vars(old_u, equations, solver, bottom_lower_right_id) #* reverse_lower[1, 1] * reverse_lower[1, 1]

    # Project from bottom upper left element
    acc += get_node_vars(old_u, equations, solver, bottom_upper_left_id) #* reverse_lower[1, 1] * reverse_lower[1, 1]

    # Project from bottom upper right element
    acc += get_node_vars(old_u, equations, solver, bottom_upper_right_id) #* reverse_lower[1, 1] * reverse_lower[1, 1]

    # Project from top lower left element
    acc += get_node_vars(old_u, equations, solver, top_lower_left_id) #* reverse_lower[1, 1] * reverse_lower[1, 1]

    # Project from top lower right element
    acc += get_node_vars(old_u, equations, solver, top_lower_right_id) #* reverse_lower[1, 1] * reverse_lower[1, 1]

    # Project from top upper left element
    acc += get_node_vars(old_u, equations, solver, top_upper_left_id) #* reverse_lower[1, 1] * reverse_lower[1, 1]

    # Project from top upper right element
    acc += get_node_vars(old_u, equations, solver, top_upper_right_id) #* reverse_lower[1, 1] * reverse_lower[1, 1]

    # Update value
    set_node_vars!(u, 1 / 8 * acc, equations, solver, element_id)
end
end # @muladd
