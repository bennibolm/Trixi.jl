# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function analyze_coefficient(mesh::TreeMesh3D, equations, dg, cache,
                             limiter::SubcellLimiterIDP)
    @unpack weights = dg.basis
    @unpack alpha = limiter.cache.subcell_limiter_coefficients

    alpha_avg = zero(eltype(alpha))
    total_volume = zero(eltype(alpha))
    for element in eachelement(dg, cache)
        jacobian = inv(cache.elements.inverse_jacobian[element])
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            alpha_avg += jacobian * weights[i] * weights[j] * weights[k] *
                         alpha[i, j, k, element]
            total_volume += jacobian * weights[i] * weights[j] * weights[k]
        end
    end

    return alpha_avg / total_volume
end

function analyze_coefficient(mesh::Union{StructuredMesh{3}, P4estMesh{3}},
                             equations, dg, cache,
                             limiter::SubcellLimiterIDP)
    @unpack weights = dg.basis
    @unpack alpha = limiter.cache.subcell_limiter_coefficients

    alpha_avg = zero(eltype(alpha))
    total_volume = zero(eltype(alpha))
    for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            jacobian = inv(cache.elements.inverse_jacobian[i, j, k, element])
            alpha_avg += jacobian * weights[i] * weights[j] * weights[k] *
                         alpha[i, j, k, element]
            total_volume += jacobian * weights[i] * weights[j] * weights[k]
        end
    end

    return alpha_avg / total_volume
end

@inline function average_mortar_limiting_factor(limiting_factor, mesh::TreeMesh{3},
                                                dg, cache)
    (; neighbor_ids, large_sides, orientations) = cache.mortars
    (; node_coordinates) = cache.elements

    avg_type = promote_type(eltype(limiting_factor), eltype(node_coordinates))
    weighted_sum = zero(avg_type)
    total_weight = zero(avg_type)
    n_nodes = nnodes(dg)

    for mortar in eachindex(limiting_factor)
        large_element = neighbor_ids[5, mortar]

        if large_sides[mortar] == 1 # small elements on right side
            index = n_nodes
        else # large_sides[mortar] == 2, small elements on left side
            index = 1
        end

        if orientations[mortar] == 1
            dy = node_coordinates[2, index, end, end, large_element] -
                 node_coordinates[2, index, 1, 1, large_element]
            dz = node_coordinates[3, index, end, end, large_element] -
                 node_coordinates[3, index, 1, 1, large_element]
            size = abs(dy * dz)
        elseif orientations[mortar] == 2
            dx = node_coordinates[1, end, index, end, large_element] -
                 node_coordinates[1, 1, index, 1, large_element]
            dz = node_coordinates[3, end, index, end, large_element] -
                 node_coordinates[3, 1, index, 1, large_element]
            size = abs(dx * dz)
        else # orientations[mortar] == 3
            dx = node_coordinates[1, end, end, index, large_element] -
                 node_coordinates[1, 1, 1, index, large_element]
            dy = node_coordinates[2, end, end, index, large_element] -
                 node_coordinates[2, 1, 1, index, large_element]
            size = abs(dx * dy)
        end

        weighted_sum += limiting_factor[mortar] * size
        total_weight += size
    end

    return weighted_sum / total_weight
end

@inline function average_mortar_limiting_factor(limiting_factor, mesh::P4estMesh{3},
                                                dg, cache)
    # TODO: Proper weighted average for P4estMesh
    return sum(limiting_factor) / length(limiting_factor)
end
end # @muladd
