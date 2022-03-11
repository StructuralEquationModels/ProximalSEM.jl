using ProximalSEM
using Documenter

DocMeta.setdocmeta!(ProximalSEM, :DocTestSetup, :(using ProximalSEM); recursive=true)

makedocs(;
    modules=[ProximalSEM],
    authors="Maximilian S. Ernst, Aaron Peikert",
    repo="https://github.com/StructuralEquationModels/ProximalSEM.jl/blob/{commit}{path}#{line}",
    sitename="ProximalSEM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://StructuralEquationModels.github.io/ProximalSEM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/StructuralEquationModels/ProximalSEM.jl",
    devbranch="main",
)
