## connect do ProximalAlgorithms.jl as backend
ProximalCore.gradient!(grad, model::AbstractSem, parameters) = objective_gradient!(grad, model::AbstractSem, parameters)

function sem_fit(
    model::AbstractSemSingle{O, I, L, D}; 
    start_val = start_val, 
    kwargs...) where {O, I, L, D <: SemOptimizerProximal}
    
    if !isa(start_val, Vector)
        start_val = start_val(model; kwargs...)
    end

    if isnothing(model.optimizer.operator_h)
        solution, iterations = model.optimizer.algorithm(
            x0 = start_val, 
            f = model, 
            g = model.optimizer.operator_g
        )
    else
        solution, iterations = model.optimizer.algorithm(
            x0=start_val, 
            f=model, 
            g=model.optimizer.operator_g, 
            h=model.optimizer.operator_h
        )
    end

    return SemFit(objective!(model, solution), solution, start_val, model, Dict(:iterations => iterations))

end