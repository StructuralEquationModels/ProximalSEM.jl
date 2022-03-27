using StructuralEquationModels, ProximalSEM, ProximalOperators, CSV, DataFrames, StenoGraphs, LinearAlgebra, Distributions, Optim
import StructuralEquationModels as SEM

# load data
dat = DataFrame(CSV.File("data_dem.csv"))

############################################################################
### define models
############################################################################

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60

    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    y1 ↔ label(:cov_15)*y5
    y2 ↔ label(:cov_24)*y4 + label(:cov_26)*y6
    y3 ↔ label(:cov_37)*y7
    y4 ↔ label(:cov_48)*y8
    y6 ↔ label(:cov_68)*y8

end

partable = ParameterTable(;
    graph = graph, 
    latent_vars = latent_vars, 
    observed_vars = observed_vars)

ram_mat = RAMMatrices(partable)

# use lasso from ProximalSEM
ind = get_identifier_indices([:cov_15, :cov_24, :cov_26, :cov_37, :cov_48, :cov_68], ram_mat)
λ = zeros(31); λ[ind] .= 0.002

model_prox = Sem(
    specification = partable,
    data = dat,
    loss = SemML,
    diff = SemDiffProximal,
    operator_g = NormL1(λ)
)

solution_prox = sem_fit(model_prox)

## L0 Norm

model_l0 = Sem(
    specification = partable,
    data = dat,
    loss = SemML,
    diff = SemDiffProximal,
    operator_g = NormL0(0.002)
)

solution_l0 = sem_fit(model_l0)

update_estimate!(partable, solution_l0)

solution_l0.solution[ind]

sum(iszero.(solution_l0.solution))

# separable sum
λ = zeros(31); λ[ind] .= 0.002
sepsum = SlicedSeparableSum(Tuple(NormL0(λᵢ) for λᵢ in λ), Tuple(1:31))

model_sepsum = Sem(
    specification = partable,
    data = dat,
    loss = SemML,
    diff = SemDiffProximal,
    operator_g = sepsum
)

solution_sepsum = sem_fit(model_sepsum)

sum(iszero.(solution_sepsum.solution))

############################################################################
### Simulate Data from true model
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

# simulate Data

using Distributions

n = 1000
obs_data = permutedims(rand(dist, n))
obs_cov = cov(obs_data)
colnames = Symbol.(:x, 1:15)

############################################################################
### ML test model
############################################################################

latent_vars = Symbol.(:f, 1:3)
observed_vars = Symbol.(:x, 1:15)

true_graph = @StenoGraph begin
    f1 → x1 + x2 + x3 + x4 + x5
    f2 → x6 + x7 + x8 + x9 + x10
    f3 → x11 + x12 + x13 + x14 + x15

    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ [fixed(1.0)].*_(latent_vars)
    f1 ↔ f2
    f1 ↔ f3
    f3 ↔ f2
end

partable = ParameterTable(;
    graph = true_graph, 
    observed_vars = observed_vars,
    latent_vars = latent_vars)

ram_mat = RAMMatrices(partable)

model_ml = Sem(
    specification = ram_mat,
    obs_cov = obs_cov,
    cov_colnames = colnames,
    loss = SemML
)

solution_ml = sem_fit(model_ml)

# variances/covariances
solution_ml.solution[1:15]

# loadings
solution_ml.solution[16:end]

############################################################################
### define models
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

# define operators
α = 0.02
ind = get_identifier_indices([Symbol.(:a, 1:9)..., Symbol.(:b, 1:9)..., Symbol.(:c, 1:9)...], ram_mat)
λ = zeros(51); λ[ind] .= α

model_l1 = Sem(
    specification = ram_mat,
    obs_cov = obs_cov,
    cov_colnames = colnames,
    loss = SemML,
    diff = SemDiffProximal,
    operator_g = NormL1(λ)
)

sepsum = SlicedSeparableSum(Tuple(NormL0(λᵢ) for λᵢ in λ), Tuple(1:51))

model_l0 = Sem(
    specification = ram_mat,
    obs_cov = obs_cov,
    cov_colnames = colnames,
    loss = SemML,
    diff = SemDiffProximal,
    operator_g = sepsum
)

model_ml = Sem(
    specification = ram_mat,
    obs_cov = obs_cov,
    cov_colnames = colnames,
    loss = SemML
)

solution_ml = sem_fit(model_ml)
solution_l0 = sem_fit(model_l0)
solution_l1 = sem_fit(model_l1)

ind_known = get_identifier_indices(Symbol.(:θ_, 1:6), ram_mat)
ind_correct = get_identifier_indices([Symbol.(:a, 1:3)..., Symbol.(:b, 4:6)..., Symbol.(:c, 7:9)...], ram_mat)
ind_incorrect = get_identifier_indices([Symbol.(:a, 4:9)..., Symbol.(:b, [1,2,3,7,8,9])..., Symbol.(:c, 1:6)...], ram_mat)

mean((solution_ml.solution[ind_known] .- 0.7).^2)
mean((solution_ml.solution[ind_correct] .- 0.7).^2)
mean((solution_ml.solution[ind_incorrect] .- 0.0).^2)

mean((solution_l0.solution[ind_known] .- 0.7).^2)
mean((solution_l0.solution[ind_correct] .- 0.7).^2)
mean((solution_l0.solution[ind_incorrect] .- 0.0).^2)
18 - sum(iszero.(solution_l0.solution[ind_incorrect]))

mean((solution_l1.solution[ind_known] .- 0.7).^2)
mean((solution_l1.solution[ind_correct] .- 0.7).^2)
mean((solution_l1.solution[ind_incorrect] .- 0.0).^2)

# regularization path
α = [0.01:0.01:0.1..., 0.2:0.1:1.5...]
n_dat = 100

ind_known = get_identifier_indices(Symbol.(:θ_, 1:6), ram_mat)
ind_correct = get_identifier_indices([Symbol.(:a, 1:3)..., Symbol.(:b, 4:6)..., Symbol.(:c, 7:9)...], ram_mat)
ind_incorrect = get_identifier_indices([Symbol.(:a, 4:9)..., Symbol.(:b, [1,2,3,7,8,9])..., Symbol.(:c, 1:6)...], ram_mat)

# save errors for all alpha values
error_incorrect_ml = zeros(n_dat, length(α))
error_incorrect_l0 = zeros(n_dat, length(α))
error_incorrect_l1 = zeros(n_dat, length(α))

error_correct_ml = zeros(n_dat, length(α))
error_correct_l0 = zeros(n_dat, length(α))
error_correct_l1 = zeros(n_dat, length(α))

# save α values for the best model
α_l0_best = zeros(n_dat)
α_l1_best = zeros(n_dat)

iter_l0 = zeros(n_dat)
iter_l1 = zeros(n_dat)

best_logl_l0 = zeros(n_dat)
best_logl_l1 = zeros(n_dat)

error_best_l0 = zeros(n_dat)
error_best_l1 = zeros(n_dat)

n_obs = 300

using Random
Random.seed!(1234)

for i in 1:n_dat
    obs_data = permutedims(rand(dist, n_obs))
    holdout = obs_data[1:150, :]
    obs_data = obs_data[151:300, :]

    holdout_cov = cov(holdout)
    obs_cov = cov(obs_data)

    logl_l1 = Float64[]
    logl_l0 = Float64[]

    iter_inner_l1 = Float64[]
    iter_inner_l0 = Float64[]

    for (j, α₁) in enumerate(α)
        λ = zeros(51)
        λ[ind] .= α₁

        model_l1 = Sem(
            specification = ram_mat,
            obs_cov = obs_cov,
            cov_colnames = colnames,
            loss = SemML,
            diff = SemDiffProximal,
            operator_g = NormL1(λ)
        )

        sepsum = SlicedSeparableSum(Tuple(NormL0(λᵢ) for λᵢ in λ), Tuple(1:51))

        model_l0 = Sem(
            specification = ram_mat,
            obs_cov = obs_cov,
            cov_colnames = colnames,
            loss = SemML,
            diff = SemDiffProximal,
            operator_g = sepsum
        )

        model_ml = Sem(
            specification = ram_mat,
            obs_cov = obs_cov,
            cov_colnames = colnames,
            loss = SemML
        )

        solution_ml = sem_fit(model_ml)
        solution_l1 = sem_fit(model_l1)
        solution_l0 = sem_fit(model_l0)

        # save errors for all alpha values
        error_incorrect_ml[i, j] = mean(abs.(solution_ml.solution[ind_incorrect] .- 0.0))
        error_incorrect_l0[i, j] = mean(abs.(solution_l0.solution[ind_incorrect] .- 0.0))
        error_incorrect_l1[i, j] = mean(abs.(solution_l1.solution[ind_incorrect] .- 0.0))

        error_correct_ml[i, j] = mean(abs.(solution_ml.solution[ind_correct] .- 0.7))
        error_correct_l0[i, j] = mean(abs.(solution_l0.solution[ind_correct] .- 0.7))
        error_correct_l1[i, j] = mean(abs.(solution_l1.solution[ind_correct] .- 0.7))

        # save logl
        objective!(model_l0, solution_l0.solution)
        Σ_l0 = model_l0.imply.Σ

        objective!(model_l1, solution_l1.solution)
        Σ_l1 = model_l1.imply.Σ

        if det(Σ_l1) > 0 & all(abs.(solution_l1.solution) .< 100)
            push!(logl_l1, logdet(Σ_l1) + tr(inv(Σ_l1)*holdout_cov))
        else
            push!(logl_l1, Inf)
        end

        if det(Σ_l0) > 0 & all(abs.(solution_l0.solution) .< 100)
            push!(logl_l0, logdet(Σ_l0) + tr(inv(Σ_l0)*holdout_cov))
        else 
            push!(logl_l0, Inf)
        end

        push!(iter_inner_l1, solution_l1.optimization_result[:iterations])
        push!(iter_inner_l0, solution_l0.optimization_result[:iterations])

    end

    # update best alpha values
    α_l0_best[i] = findmin(logl_l0)[2]
    α_l1_best[i] = findmin(logl_l1)[2]

    iter_l0[i] = iter_inner_l0[Int(α_l0_best[i])]
    iter_l1[i] = iter_inner_l1[Int(α_l1_best[i])]

    best_logl_l0[i] = logl_l0[Int(α_l0_best[i])]
    best_logl_l1[i] = logl_l1[Int(α_l0_best[i])]

end

# save as CSV
CSV.write("error_incorrect_ml.csv", DataFrame(error_incorrect_ml, :auto))
CSV.write("error_incorrect_l0.csv", DataFrame(error_incorrect_l0, :auto))
CSV.write("error_incorrect_l1.csv", DataFrame(error_incorrect_l1, :auto))

CSV.write("error_correct_ml.csv", DataFrame(error_correct_ml, :auto))
CSV.write("error_correct_l0.csv", DataFrame(error_correct_l0, :auto))
CSV.write("error_correct_l1.csv", DataFrame(error_correct_l1, :auto))

# save as CSV
CSV.write("α_l0_best.csv", DataFrame(reshape(α_l0_best, 100, 1), [:which_alpha]))
CSV.write("α_l1_best.csv", DataFrame(reshape(α_l1_best, 100, 1), [:which_alpha]))

### second simulation
α = [0.01:0.01:0.1..., 0.2:0.1:1.5...]
n_dat = 100

res = zeros(100, 27); res .= -99

res = DataFrame(
    res, [
        :error_ic_ml,
        :error_c_ml,

        :error_ic_l0_cv,
        :error_ic_l1_cv,
        :error_c_l0_cv,
        :error_c_l1_cv,

        :error_ic_l0_bic,
        :error_ic_l1_bic,
        :error_c_l0_bic,
        :error_c_l1_bic,

        :α_l0_cv,
        :α_l1_cv,
        :α_l0_bic,
        :α_l1_bic,

        :logl_l0,
        :logl_l1,
        :bic_l0,
        :bic_l1,

        :converged_l0_cv,
        :converged_l1_cv,
        :converged_l0_bic,
        :converged_l1_bic,

        :converged_ml,

        :nonzero_ic_cv,
        :zero_c_cv,
        :nonzero_ic_bic,
        :zero_c_bic,
        ])

res_inner = zeros(24, 20); res_inner .= -99

res_inner = DataFrame(
    res_inner, [
        :error_ic_l0_bic,
        :error_ic_l1_bic,
        :error_c_l0_bic,
        :error_c_l1_bic,

        :error_ic_l0_cv,
        :error_ic_l1_cv,
        :error_c_l0_cv,
        :error_c_l1_cv,

        :logl_l0,
        :logl_l1,
        :bic_l0,
        :bic_l1,

        :converged_l0_cv,
        :converged_l1_cv,
        :converged_l0_bic,
        :converged_l1_bic,

        :nonzero_ic_cv,
        :zero_c_cv,
        :nonzero_ic_bic,
        :zero_c_bic,
        ])

n_obs = 50

using Random
Random.seed!(398578237498)

for i in 1:n_dat

    # simulate data
    obs_data = permutedims(rand(dist, n_obs))
    obs_data = DataFrame(obs_data, colnames)

    holdout = obs_data[1:25, :]
    obs_data_cv = obs_data[26:50, :]

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

    res.converged_ml[i] = converged

    if !converged
        res.error_ic_ml[i] = Inf
        res.error_c_ml[i] = Inf
    else
        res.error_ic_ml[i] = mean(abs.(solution_ml.solution[ind_incorrect] .- 0.0))
        res.error_c_ml[i] = mean(abs.(solution_ml.solution[ind_correct] .- 0.7))
    end

    for (j, α₁) in enumerate(α)

        # l1 regularization
        λ = zeros(51)
        λ[ind] .= α₁

        model_l1_cv = Sem(
            specification = ram_mat,
            data = obs_data_cv,
            loss = SemML,
            diff = SemDiffProximal,
            operator_g = NormL1(λ)
        )

        solution_l1_cv = sem_fit(model_l1_cv)

        model_l1_bic = Sem(
            specification = ram_mat,
            data = obs_data,
            loss = SemML,
            diff = SemDiffProximal,
            operator_g = NormL1(λ)
        )

        solution_l1_bic = sem_fit(model_l1_bic)

        objective!(model_l1_cv, solution_l1_cv.solution)
        Σ_l1 = model_l1_cv.imply.Σ
        res_inner.converged_l1_cv[j]  = (
            isposdef(Symmetric(Σ_l1)) & 
            all(abs.(solution_l1_cv.solution) .< 100) & 
            (solution_l1_cv.optimization_result[:iterations] < 1000))

        if !Bool(res_inner.converged_l1_cv[j] )
            res_inner.error_ic_l1_cv[j] = Inf
            res_inner.error_c_l1_cv[j] = Inf
            res_inner.logl_l1[j] = -Inf
        else
            res_inner.error_ic_l1_cv[j] = mean(abs.(solution_l1_cv.solution[ind_incorrect] .- 0.0))
            res_inner.error_c_l1_cv[j] = mean(abs.(solution_l1_cv.solution[ind_correct] .- 0.7))
            res_inner.logl_l1[j] = logdet(Σ_l1) + tr(inv(Σ_l1)*cov(Matrix(holdout)))
        end
        
        objective!(model_l1_bic, solution_l1_bic.solution)
        Σ_l1 = model_l1_bic.imply.Σ
        res_inner.converged_l1_bic[j]  = (
            isposdef(Symmetric(Σ_l1)) & 
            all(abs.(solution_l1_bic.solution) .< 100) & 
            solution_l1_bic.optimization_result[:iterations] < 1000)

        if !Bool(res_inner.converged_l1_bic[j] )
            res_inner.error_ic_l1_bic[j] = Inf
            res_inner.error_c_l1_bic[j] = Inf
            res_inner.bic_l1[j] = Inf
        else
            res_inner.error_ic_l1_bic[j] = mean(abs.(solution_l1_bic.solution[ind_incorrect] .- 0.0))
            res_inner.error_c_l1_bic[j] = mean(abs.(solution_l1_bic.solution[ind_correct] .- 0.7))
            res_inner.bic_l1[j] = 
            minus2ll(solution_l1_bic) + log(SEM.n_obs(solution_l1_bic))*sum(solution_l1_bic.solution .> 1e-3)
        end

        # l0 regularization

        sepsum = SlicedSeparableSum(Tuple(NormL0(λᵢ) for λᵢ in λ), Tuple(1:51))

        model_l0_cv = Sem(
            specification = ram_mat,
            data = obs_data_cv,
            loss = SemML,
            diff = SemDiffProximal,
            operator_g = sepsum
        )

        solution_l0_cv = sem_fit(model_l0_cv)

        model_l0_bic = Sem(
            specification = ram_mat,
            data = obs_data,
            loss = SemML,
            diff = SemDiffProximal,
            operator_g = sepsum
        )

        solution_l0_bic = sem_fit(model_l0_bic)

        objective!(model_l0_cv, solution_l0_cv.solution)
        Σ_l0 = model_l0_cv.imply.Σ
        res_inner.converged_l0_cv[j]  = (
            isposdef(Symmetric(Σ_l0)) & 
            all(abs.(solution_l0_cv.solution) .< 100) & 
            (solution_l0_cv.optimization_result[:iterations] < 1000))

        if !Bool(res_inner.converged_l0_cv[j])
            res_inner.error_ic_l0_cv[j] = Inf
            res_inner.error_c_l0_cv[j] = Inf
            res_inner.logl_l0[j] = -Inf
            res_inner.nonzero_ic_cv[j] = Inf
            res_inner.zero_c_cv[j] = Inf
        else
            res_inner.error_ic_l0_cv[j] = mean(abs.(solution_l0_cv.solution[ind_incorrect] .- 0.0))
            res_inner.error_c_l0_cv[j] = mean(abs.(solution_l0_cv.solution[ind_correct] .- 0.7))
            res_inner.logl_l0[j] = logdet(Σ_l0) + tr(inv(Σ_l0)*cov(Matrix(holdout)))
            res_inner.nonzero_ic_cv[j] = sum(.!iszero.(solution_l0_cv.solution[ind_incorrect]))
            res_inner.zero_c_cv[j] = sum(iszero.(solution_l0_cv.solution[ind_correct]))
        end
        
        objective!(model_l0_bic, solution_l0_bic.solution)
        Σ_l0 = model_l0_bic.imply.Σ
        res_inner.converged_l0_bic[j]  = (
            isposdef(Symmetric(Σ_l0)) & 
            all(abs.(solution_l0_bic.solution) .< 100) & 
            solution_l0_bic.optimization_result[:iterations] < 1000)

        if !Bool(res_inner.converged_l0_bic[j]) 
            res_inner.error_ic_l0_bic[j] = Inf
            res_inner.error_c_l0_bic[j] = Inf
            res_inner.bic_l0[j] = Inf
            res_inner.nonzero_ic_bic[j] = Inf
            res_inner.zero_c_bic[j] = Inf
        else
            res_inner.error_ic_l0_bic[j] = mean(abs.(solution_l0_bic.solution[ind_incorrect] .- 0.0))
            res_inner.error_c_l0_bic[j] = mean(abs.(solution_l0_bic.solution[ind_correct] .- 0.7))
            res_inner.bic_l0[j] = 
                minus2ll(solution_l0_bic) + log(SEM.n_obs(solution_l0_bic))*sum(.!iszero.(solution_l0_bic.solution))
            res_inner.nonzero_ic_bic[j] = sum(.!iszero.(solution_l0_bic.solution[ind_incorrect]))
            res_inner.zero_c_bic[j] = sum(iszero.(solution_l0_bic.solution[ind_correct]))
        end

    end

    # update best alpha values
    res.α_l0_cv[i] = findmax(res_inner.logl_l0)[2]
    res.α_l0_bic[i] = findmin(res_inner.bic_l0)[2]
    res.α_l1_cv[i] = findmax(res_inner.logl_l1)[2]
    res.α_l1_bic[i] = findmin(res_inner.bic_l1)[2]

    res.logl_l0[i] = res_inner.logl_l0[Int(res.α_l0_cv[i])]
    res.bic_l0[i] = res_inner.bic_l0[Int(res.α_l0_bic[i])]
    res.logl_l1[i] = res_inner.logl_l1[Int(res.α_l1_cv[i])]
    res.bic_l1[i] = res_inner.bic_l1[Int(res.α_l1_bic[i])]

    res.converged_l0_cv[i] = res_inner.converged_l0_cv[Int(res.α_l0_cv[i])]
    res.converged_l0_bic[i] = res_inner.converged_l0_bic[Int(res.α_l0_bic[i])]
    res.converged_l1_cv[i] = res_inner.converged_l1_cv[Int(res.α_l1_cv[i])]
    res.converged_l1_bic[i] = res_inner.converged_l1_bic[Int(res.α_l1_bic[i])]

    res.error_ic_l0_cv[i] = res_inner.error_ic_l0_cv[Int(res.α_l0_cv[i])]
    res.error_ic_l0_bic[i] = res_inner.error_ic_l0_bic[Int(res.α_l0_bic[i])]
    res.error_ic_l1_cv[i] = res_inner.error_ic_l1_cv[Int(res.α_l1_cv[i])]
    res.error_ic_l1_bic[i] = res_inner.error_ic_l1_bic[Int(res.α_l1_bic[i])]

    res.error_c_l0_cv[i] = res_inner.error_c_l0_cv[Int(res.α_l0_cv[i])]
    res.error_c_l0_bic[i] = res_inner.error_c_l0_bic[Int(res.α_l0_bic[i])]
    res.error_c_l1_cv[i] = res_inner.error_c_l1_cv[Int(res.α_l1_cv[i])]
    res.error_c_l1_bic[i] = res_inner.error_c_l1_bic[Int(res.α_l1_bic[i])]

    res.nonzero_ic_cv[i] = res_inner.nonzero_ic_cv[Int(res.α_l0_cv[i])]
    res.zero_c_cv[i] = res_inner.zero_c_cv[Int(res.α_l0_cv[i])]
    res.nonzero_ic_bic[i] = res_inner.nonzero_ic_bic[Int(res.α_l0_bic[i])]
    res.zero_c_bic[i] = res_inner.zero_c_bic[Int(res.α_l0_bic[i])]
end

# save as CSV
CSV.write("res.csv", res)