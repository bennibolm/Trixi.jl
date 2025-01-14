# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function init_mortar_neighbor_ids!(mortars::T8codeFVMortarContainer{3}, my_face,
                                           other_face, orientation, neighbor_ielements,
                                           mortar_id)
    # For more information see init_mortar_neighbor_ids! for DGSEM
    lower = my_face <= other_face
    my_right_handed = my_face in (1, 2, 5)
    other_right_handed = other_face in (1, 2, 5)
    flipped = my_right_handed == other_right_handed
    if !flipped
        if orientation == 0
            # Corner 0 of other side matches corner 0 of my side
            #   2┌──────┐3   2┌──────┐3
            #    │      │     │      │
            #    │      │     │      │
            #   0└──────┘1   0└──────┘1
            #     η            η
            #     ↑            ↑
            #     │            │
            #     └───> ξ      └───> ξ

            mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[1] + 1
            mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[2] + 1
            mortars.neighbor_ids[3, mortar_id] = neighbor_ielements[3] + 1
            mortars.neighbor_ids[4, mortar_id] = neighbor_ielements[4] + 1

        elseif ((lower && orientation == 2) # Corner 0 of my side matches corner 2 of other side
                ||
                (!lower && orientation == 1)) # Corner 0 of other side matches corner 1 of my side
            #   2┌──────┐3   0┌──────┐2
            #    │      │     │      │
            #    │      │     │      │
            #   0└──────┘1   1└──────┘3
            #     η            ┌───> η
            #     ↑            │
            #     │            ↓
            #     └───> ξ      ξ

            mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[2] + 1
            mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[4] + 1
            mortars.neighbor_ids[3, mortar_id] = neighbor_ielements[1] + 1
            mortars.neighbor_ids[4, mortar_id] = neighbor_ielements[3] + 1

        elseif ((lower && orientation == 1) # Corner 0 of my side matches corner 1 of other side
                ||
                (!lower && orientation == 2)) # Corner 0 of other side matches corner 2 of my side
            #   2┌──────┐3   3┌──────┐1
            #    │      │     │      │
            #    │      │     │      │
            #   0└──────┘1   2└──────┘0
            #     η                 ξ
            #     ↑                 ↑
            #     │                 │
            #     └───> ξ     η <───┘

            mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[3] + 1
            mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[1] + 1
            mortars.neighbor_ids[3, mortar_id] = neighbor_ielements[4] + 1
            mortars.neighbor_ids[4, mortar_id] = neighbor_ielements[2] + 1

        else # orientation == 3
            # Corner 0 of my side matches corner 3 of other side and
            # corner 0 of other side matches corner 3 of my side.
            #   2┌──────┐3   1┌──────┐0
            #    │      │     │      │
            #    │      │     │      │
            #   0└──────┘1   3└──────┘2
            #     η           ξ <───┐
            #     ↑                 │
            #     │                 ↓
            #     └───> ξ           η

            mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[4] + 1
            mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[3] + 1
            mortars.neighbor_ids[3, mortar_id] = neighbor_ielements[2] + 1
            mortars.neighbor_ids[4, mortar_id] = neighbor_ielements[1] + 1
        end
    else # flipped
        if orientation == 0
            # Corner 0 of other side matches corner 0 of my side
            #   2┌──────┐3   1┌──────┐3
            #    │      │     │      │
            #    │      │     │      │
            #   0└──────┘1   0└──────┘2
            #     η            ξ
            #     ↑            ↑
            #     │            │
            #     └───> ξ      └───> η

            mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[1] + 1
            mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[3] + 1
            mortars.neighbor_ids[3, mortar_id] = neighbor_ielements[2] + 1
            mortars.neighbor_ids[4, mortar_id] = neighbor_ielements[4] + 1

        elseif orientation == 2
            # Corner 0 of my side matches corner 2 of other side and
            # corner 0 of other side matches corner 2 of my side.
            #   2┌──────┐3   0┌──────┐1
            #    │      │     │      │
            #    │      │     │      │
            #   0└──────┘1   2└──────┘3
            #     η            ┌───> ξ
            #     ↑            │
            #     │            ↓
            #     └───> ξ      η

            mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[3] + 1
            mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[4] + 1
            mortars.neighbor_ids[3, mortar_id] = neighbor_ielements[1] + 1
            mortars.neighbor_ids[4, mortar_id] = neighbor_ielements[2] + 1

        elseif orientation == 1
            # Corner 0 of my side matches corner 1 of other side and
            # corner 0 of other side matches corner 1 of my side.
            #   2┌──────┐3   3┌──────┐2
            #    │      │     │      │
            #    │      │     │      │
            #   0└──────┘1   1└──────┘0
            #     η                 η
            #     ↑                 ↑
            #     │                 │
            #     └───> ξ     ξ <───┘

            mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[2] + 1
            mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[1] + 1
            mortars.neighbor_ids[3, mortar_id] = neighbor_ielements[4] + 1
            mortars.neighbor_ids[4, mortar_id] = neighbor_ielements[3] + 1

        else # orientation == 3
            # Corner 0 of my side matches corner 3 of other side and
            # corner 0 of other side matches corner 3 of my side.
            #   2┌──────┐3   2┌──────┐0
            #    │      │     │      │
            #    │      │     │      │
            #   0└──────┘1   3└──────┘1
            #     η           η <───┐
            #     ↑                 │
            #     │                 ↓
            #     └───> ξ           ξ

            mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[4] + 1
            mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[2] + 1
            mortars.neighbor_ids[3, mortar_id] = neighbor_ielements[3] + 1
            mortars.neighbor_ids[4, mortar_id] = neighbor_ielements[1] + 1
        end
    end
end

@inline function init_mortar_faces!(mortars::T8codeFVMortarContainer{3}, faces,
                                    orientation, mortar_id)
    dual_faces, iface = faces

    # TODO: Test this with more complicated meshes
    # what should happen for orientation = 1, 2 or 3
    mortars.faces[end, mortar_id] = iface + 1
    if orientation == 0
        mortars.faces[1, mortar_id] = dual_faces[1] + 1
        mortars.faces[2, mortar_id] = dual_faces[2] + 1
        mortars.faces[3, mortar_id] = dual_faces[3] + 1
        mortars.faces[4, mortar_id] = dual_faces[4] + 1
    else
        mortars.faces[1, mortar_id] = dual_faces[4] + 1
        mortars.faces[2, mortar_id] = dual_faces[3] + 1
        mortars.faces[3, mortar_id] = dual_faces[2] + 1
        mortars.faces[4, mortar_id] = dual_faces[1] + 1
    end
    return nothing
end
end # @muladd
