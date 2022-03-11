using ProximalSEM
using Documenter

DocMeta.setdocmeta!(ProximalSEM, :DocTestSetup, :(using ProximalSEM); recursive=true)

makedocs(;
    modules=[ProximalSEM],
    authors="Maximilian S. Ernst, Aaron Peikert",
    repo="https://github.com/Maximilian-Stefan-Ernst/ProximalSEM.jl/blob/{commit}{path}#{line}",
    sitename="ProximalSEM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Maximilian-Stefan-Ernst.github.io/ProximalSEM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Maximilian-Stefan-Ernst/ProximalSEM.jl",
    devbranch="main",
)
