using StructuralEquationModels, ProximalSEM, ProximalOperators, CSV, DataFrames

# load data
dat = DataFrame(CSV.File("data_dem.csv"))

############################################################################
### define models
############################################################################

graph_1 = """
    ind60 =∼ 1*x1 + x2 + x3
    dem60 =∼ 1*y1 + y2 + y3 + y4
    dem65 =∼ 1*y5 + y6 + y7 + y8
    dem60 ∼ ind60
    dem65 ∼ dem60
    dem65 ∼ ind60
    ind60 ∼∼ ind60
    dem60 ∼∼ dem60
    dem65 ∼∼ dem65
    x1 ∼∼ x1
    x2 ∼∼ x2
    x3 ∼∼ x3
    y1 ∼∼ y1
    y2 ∼∼ y2
    y3 ∼∼ y3
    y4 ∼∼ y4
    y5 ∼∼ y5
    y6 ∼∼ y6
    y7 ∼∼ y7
    y8 ∼∼ y8
    y1 ∼∼ y5
    y2 ∼∼ y4 + y6
    y3 ∼∼ y7
    y4 ∼∼ y8
    y6 ∼∼ y8
"""

observed_vars = ["x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"]
latent_vars = ["ind60", "dem60", "dem65"]

partable = ParameterTable(latent_vars, observed_vars, graph_1)

# use ridge from StructuralEquationModels
model_ridge = Sem(
    specification = partable,
    data = dat,
    loss = (SemML, SemRidge,),
    α_ridge = .001,
    which_ridge = 16:20
)

solution_ridge = sem_fit(model_ridge)

# use ridge from ProximalSEM
λ = zeros(31); λ[16:20] .= 0.002

model_prox = Sem(
    specification = partable,
    data = dat,
    loss = (SemML,),
    diff = SemDiffProximal,
    operator_g = SqrNormL2(λ)
)

solution_prox = sem_fit(model_prox)

@testset "ridge_solution" begin
    @test isapprox(solution_prox.solution, solution_ridge.solution; rtol = 1e-4)
end