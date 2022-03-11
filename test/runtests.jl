using Test, SafeTestsets

@safetestset "Ridge" begin include("ridge.jl") end
@safetestset "Lasso" begin include("lasso.jl") end