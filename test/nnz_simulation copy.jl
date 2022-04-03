using StructuralEquationModels, ProximalSEM, ProximalOperators, CSV, DataFrames, StenoGraphs, LinearAlgebra, Distributions, Optim
import StructuralEquationModels as SEM

############################################################################
### true model
############################################################################

S = zeros(18, 18)
for i in 1:15 S[i, i] = 0.5 end
S[16:18, 16:18] .= 0.5
for i in 16:18 S[i, i] = 1.0 end

A = zeros(18, 18)
for j in 1:3
    for i in 1:5
        A[i+(j-1)*5, 15+j] = 0.7
    end
end

F = zeros(15, 18)
F[diagind(F)] .= 1.0

Σ = F*inv(I-A)*S*inv(I-A)'*F'
dist = MvNormal(Σ)

############################################################################
### define model
############################################################################
latent_vars = Symbol.(:f, 1:3)
observed_vars = Symbol.(:x, 1:15)

unknown_loadings = Symbol.(:x, [3:5..., 8:10..., 13:15...])

label_f1 = label.(Symbol.(:a, 1:9))
label_f2 = label.(Symbol.(:b, 1:9))
label_f3 = label.(Symbol.(:c, 1:9))

full_graph = @StenoGraph begin
    f1 → x1 + x2
    f2 → x6 + x7
    f3 → x11 + x12

    f1 → _(label_f1).*_(unknown_loadings)
    f2 → _(label_f2).*_(unknown_loadings)
    f3 → _(label_f3).*_(unknown_loadings)

    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ [fixed(1.0)].*_(latent_vars)

    f1 ↔ f2
    f1 ↔ f3
    f3 ↔ f2
end

partable = ParameterTable(;
    graph = full_graph, 
    observed_vars = observed_vars,
    latent_vars = latent_vars)

ram_mat = RAMMatrices(partable)
############################################################################
### get parameter indices
############################################################################
ind = get_identifier_indices([Symbol.(:a, 1:9)..., Symbol.(:b, 1:9)..., Symbol.(:c, 1:9)...], ram_mat)
ind_known = get_identifier_indices(Symbol.(:θ_, 1:6), ram_mat)
ind_correct = get_identifier_indices([Symbol.(:a, 1:3)..., Symbol.(:b, 4:6)..., Symbol.(:c, 7:9)...], ram_mat)
ind_incorrect = get_identifier_indices([Symbol.(:a, 4:9)..., Symbol.(:b, [1,2,3,7,8,9])..., Symbol.(:c, 1:6)...], ram_mat)

############################################################################
### result storage
############################################################################
α = [0.001:0.002:0.01..., 0.01:0.02:0.1..., 0.2:0.2:3...]
α = [0.0001:0.0002:0.001..., 0.001:0.002:0.01..., 0.02:0.2:1...]


res_ml = DataFrame((-99).*ones(100, 3), [:error_ic, :error_c, :converged])

res_l0 = DataFrame(
    (-99).*ones(100, 8), [
        :error_ic,
        :error_c,

        :α,
        :which_α,
        :bic,

        :converged,

        :nonzero_ic,
        :zero_c
        ])

res_l1 = copy(res_l0)

res_inner_l0 = DataFrame(
    (-99).*ones(length(α), 6), [
        :error_ic,
        :error_c,

        :bic,

        :converged,

        :nonzero_ic,
        :zero_c
        ])

res_inner_l1 = copy(res_inner_l0)

############################################################################
### simulation
############################################################################
using Random
Random.seed!(398578237498)

n_dat = 100
n_obs = 100

for i in 1:n_dat

    # simulate data
    obs_data = permutedims(rand(dist, n_obs))
    obs_data = DataFrame(obs_data, observed_vars)

    # fit ml model
    model_ml = Sem(
        specification = ram_mat,
        data = obs_data,
        loss = SemML
    )
    solution_ml = sem_fit(model_ml)

    # access ml model convergence
    objective!(model_ml, solution_ml.solution)
    Σ = model_ml.imply.Σ
    converged = (
        isposdef(Symmetric(Σ)) & 
        all(abs.(solution_ml.solution) .< 100) & 
        Optim.converged(solution_ml.optimization_result))

    res_ml.converged[i] = converged

    if !converged
        res_ml.error_ic[i] = Inf
        res_ml.error_c[i] = Inf
    else
        res_ml.error_ic[i] = mean(abs.(solution_ml.solution[ind_incorrect] .- 0.0))
        res_ml.error_c[i] = mean(abs.(solution_ml.solution[ind_correct] .- 0.7))
    end

    for (j, α₁) in enumerate(α)

        # l1 regularization ------------------------------------------------------------
        λ = zeros(51)
        λ[ind] .= α₁

        model_l1 = Sem(
            specification = ram_mat,
            data = obs_data,
            loss = SemML,
            diff = SemDiffProximal,
            operator_g = NormL1(λ)
        )

        solution_l1 = sem_fit(model_l1)
        
        objective!(model_l1, solution_l1.solution)
        Σ_l1 = model_l1.imply.Σ
        
        res_inner_l1.converged[j] = my_converged(Σ_l1, solution_l1)
        update_res_inner!(res_inner_l1, j, solution_l1, ind_incorrect, ind_correct, 1e-2)

        # l0 regularization ---------------------------------------------------------
        sepsum = SlicedSeparableSum(Tuple(NormL0(λᵢ) for λᵢ in λ), Tuple(1:51))

        model_l0 = Sem(
            specification = ram_mat,
            data = obs_data,
            loss = SemML,
            diff = SemDiffProximal,
            operator_g = sepsum
        )

        solution_l0 = sem_fit(model_l0)
        
        objective!(model_l0, solution_l0.solution)
        Σ_l0 = model_l0.imply.Σ

        res_inner_l0.converged[j]  = my_converged(Σ_l0, solution_l0)
        update_res_inner!(res_inner_l0, j, solution_l0, ind_incorrect, ind_correct, 0.0)

    end

    # update best alpha values
    update_res!(res_l0, res_inner_l0, i)
    update_res!(res_l1, res_inner_l1, i)
end

# save as CSV
CSV.write("res_l0.csv", res_l0)
CSV.write("res_l1.csv", res_l1)
CSV.write("res_ml.csv", res_ml)

function update_res!(res, res_inner, i)
    k = Int(findmin(res_inner.bic)[2])

    res.which_α[i] = k
    res.α[i] = α[k]
    res.bic[i] = res_inner.bic[k]

    res.converged[i] = res_inner.converged[k]
    # errors
    res.error_ic[i] = res_inner.error_ic[k]
    res.error_c[i] = res_inner.error_c[k]
    # structural errors
    res.nonzero_ic[i] = res_inner.nonzero_ic[k]
    res.zero_c[i] = res_inner.zero_c[k]
end

function update_res_inner!(res_inner, j, solution, ind_incorrect, ind_correct, atol_zero)
    if !Bool(res_inner_l1.converged[j])
        res_inner.error_ic[j] = Inf
        res_inner.error_c[j] = Inf
        res_inner.bic[j] = Inf
        res_inner.nonzero_ic[j] = Inf
        res_inner.zero_c[j] = Inf
    else
        res_inner.error_ic[j] = mean(abs.(solution.solution[ind_incorrect] .- 0.0))
        res_inner.error_c[j] = mean(abs.(solution.solution[ind_correct] .- 0.7))
        res_inner.bic[j] = my_bic(solution, atol_zero)
        res_inner.nonzero_ic[j] = sum(.!iszero.(solution.solution[ind_incorrect]))
        res_inner.zero_c[j] = sum(isapprox.(solution.solution[ind_correct], 0; atol = atol_zero))
    end
end

function my_bic(solution, atol)
    minus2ll(solution) + log(SEM.n_obs(solution))*sum(.!isapprox.(solution.solution, 0; atol = atol))
end

function my_converged(Σ, solution)
    is_converged = 
        (isposdef(Symmetric(Σ)) & 
        all(abs.(solution.solution) .< 100) & 
        solution.optimization_result[:iterations] < 1000)
    return is_converged
end

