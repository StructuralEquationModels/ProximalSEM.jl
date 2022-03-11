## connect do ProximalAlgorithms.jl as backend
ProximalCore.gradient!(grad, model::AbstractSem, parameters) = objective_gradient!(grad, model::AbstractSem, parameters)

#= function SemFit(optimization_result::Optim.MultivariateOptimizationResults, model::AbstractSem, start_val)
    return SemFit(
        optimization_result.minimum,
        optimization_result.minimizer,
        start_val,
        model,
        optimization_result
    )
end =#

function sem_fit(model::Sem{O, I, L, D}; start_val = start_val, kwargs...) where {O, I, L, D <: SemDiffProximal}
    
    if !isa(start_val, Vector)
        start_val = start_val(model; kwargs...)
    end

    if isnothing(model.diff.operator_h)
        solution, iterations = model.diff.algorithm(x0 = start_val, f = model, g = model.diff.operator_g)
    else
        solution, iterations = model.diff.algorithm(x0=start_val, f=model, g=model.diff.operator_g, h=model.diff.operator_h)
    end

    return SemFit(objective!(model, solution), solution, start_val, model, Dict(:iterations => iterations))

end