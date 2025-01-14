# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# this method is called when an `ControllerThreeLevel` is constructed
function create_cache(::Type{ControllerThreeLevel},
                      mesh::T8codeMesh, equations,
                      solver::FV, cache)
    controller_value = Vector{Int}(undef, nelements(mesh, solver, cache))
    return (; controller_value)
end
end # @muladd
