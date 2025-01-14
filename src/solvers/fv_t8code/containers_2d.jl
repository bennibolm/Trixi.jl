# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function init_mortar_neighbor_ids!(mortars::T8codeFVMortarContainer{2}, my_face,
                                           other_face, orientation, neighbor_ielements,
                                           mortar_id)
    if orientation == 0
        mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[1] + 1
        mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[2] + 1
    else
        mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[2] + 1
        mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[1] + 1
    end
    return nothing
end

@inline function init_mortar_faces!(mortars::T8codeFVMortarContainer{2}, faces,
                                    orientation, mortar_id)
    dual_faces, iface = faces

    mortars.faces[end, mortar_id] = iface + 1
    if orientation == 0
        mortars.faces[1, mortar_id] = dual_faces[1] + 1
        mortars.faces[2, mortar_id] = dual_faces[2] + 1
    else
        mortars.faces[1, mortar_id] = dual_faces[2] + 1
        mortars.faces[2, mortar_id] = dual_faces[1] + 1
    end
    return nothing
end
end # @muladd
