module ProximalSEM

    using StructuralEquationModels
    import StructuralEquationModels: print_type_name, print_field_types, sem_fit, start_val
    import ProximalCore, ProximalAlgorithms, ProximalOperators
    import ProximalCore: prox!

    #ProximalCore.prox!(y, f, x, gamma) = ProximalOperators.prox!(y, f, x, gamma)

    include("diff/Proximal.jl")
    include("optimizer/ProximalAlgorithms.jl")

    export SemOptimizerProximal, sem_fit

end