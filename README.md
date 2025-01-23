> [!CAUTION]
> This package is deprecated and all functionality has been moved to the main package `StructuralEquationModels.jl`.
> All information on how to now fit regularized SEM with `StructuralEquationModels.jl` can be found in the [online docs](https://structuralequationmodels.github.io/StructuralEquationModels.jl/).

# ProximalSEM.jl

This is a package for regularized structural equation modeling. It connects [StructuralEquationModels.jl](https://github.com/StructuralEquationModels/StructuralEquationModels.jl) to [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl). As a result, it can be used to fit structural equation models with

- regularization (L0-L2 and beyond)
- penalties
- projections onto sets

i.e. everything [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) implements.

The documentation is part of the documentation of StructuralEquationModels.jl, the relevant chapter can be found [here](https://structuralequationmodels.github.io/StructuralEquationModels.jl/stable/tutorials/regularization/regularization/).

| **Documentation**                                                               | **Build Status**                                                                                | Citation                                                                                        |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://structuralequationmodels.github.io/StructuralEquationModels.jl/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://structuralequationmodels.github.io/StructuralEquationModels.jl/dev/) | [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Github Action CI](https://github.com/StructuralEquationModels/ProximalSEM.jl/workflows/CI/badge.svg)](https://github.com/StructuralEquationModels/ProximalSEM.jl/actions/) [![codecov](https://codecov.io/gh/StructuralEquationModels/ProximalSEM.jl/branch/main/graph/badge.svg?token=WjN6i2koPY)](https://codecov.io/gh/StructuralEquationModels/ProximalSEM.jl)|[![DOI](https://zenodo.org/badge/468414762.svg)](https://zenodo.org/badge/latestdoi/468414762) |
