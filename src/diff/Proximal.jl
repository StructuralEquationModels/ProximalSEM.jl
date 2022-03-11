struct SemDiffProximal{A, B, C, D} <: SemDiff
    algorithm::A
    options::B
    operator_g::C
    operator_h::D
end

SemDiffProximal(;algorithm = ProximalAlgorithms.PANOC(), options = Dict{Symbol, Any}(), operator_g, operator_h = nothing, kwargs...) = 
    SemDiffProximal(algorithm, options, operator_g, operator_h)

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemDiffProximal)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end