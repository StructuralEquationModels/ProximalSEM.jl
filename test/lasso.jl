using StructuralEquationModels, ProximalSEM, ProximalOperators, CSV, DataFrames, StenoGraphs

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
λ = zeros(31); λ[16:20] .= 0.002

model_prox = Sem(
    specification = partable,
    data = dat,
    loss = (SemML,),
    diff = SemDiffProximal,
    operator_g = NormL1(λ)
)

solution_prox = sem_fit(model_prox)

@testset "lasso_converged" begin
    @test solution_prox.solution.optimization_result[:iterations] < 1000
end