#!/usr/bin/env julia

import PyPlot
const plt = PyPlot.plt
using LaTeXStrings

using DelimitedFiles
using HDF5

get_params() = (
    # data_gp = "data/tangle_1024_t110.h5",
    # data_gp = "data/2048/tangle_2048_t42_res4.h5",
    data_gp = "data/2048/tangle_2048_res4_turbulence.h5",
    # data_ns = "data_NS/NS1024/Nu1/t14000.h5",
    # data_ns = "data_NS/NS1024/Nu2/t15000.h5",

    # data_ns = "data_NS/NS512/Re1/NS_512_forced_t70000.h5",
    # NS_dir = "data_NS/NS512/Re1/latu_data",

    data_ns = "data_NS/NS1024_2013/R0_GradientForcing/NS_1024_t74000.h5",
    NS_dir = "data_NS/NS1024_2013/R0_GradientForcing/latu_data",

    # NS_dir = expanduser("~/Work/simulation_data/LatuABC/Runs1024/Nu1/data/NS_r3_1024"),
    # Parameters of ABC flow
    L_ABC = π,
    v_ABC = sqrt(0.2517 * 4 * 2 / 3),
    ℓ_gp = 2π / 35,  # TODO not sure!!
)

const MOMENTS_FROM_HISTOGRAM = Ref(true)

function read_all_pdfs(g::HDF5Group; bin_centres=true)
    if exists(g, "hist_filt")
        H = g["hist_filt"][:, :] :: Matrix{Float64}
        Γ = g["bin_center_filt"][:] :: Vector{Float64}
        # These circulation values need to be unnormalised...
        κ = read(file(g)["/ParamsGP/kappa"]) :: Float64
        @assert bin_centres
        Γ .*= κ
    else
        H = Float64.(g["hist"][:, :] :: Matrix{Int})
        bin_edges = g["bin_edges"][:] :: Vector{Float64}
        Γ = if bin_centres
            (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
        else
            bin_edges
        end
    end
    H ./= sum(H, dims=1)  # normalise histogram
    Γ, H
end

function moment_from_histogram(g::HDF5Group, args...)
    Γ, H = read_all_pdfs(g)
    # n = searchsortedlast(Γ, 0)
    # @show Γ[n]
    moment_from_histogram(H, Γ, args...)
end

function moment_from_histogram(H::AbstractMatrix, bin_centres, ::Val{p}) where {p}
    Nb, Nr = size(H)
    @assert length(bin_centres) == Nb
    # @show sum(H, dims=1)
    M = zeros(Nr)
    @inbounds for i = 1:Nb
        x = bin_centres[i]
        xp = abs(x)^p
        for r = 1:Nr
            M[r] += xp * H[i, r]
        end
    end
    # M ./= vec(sum(H, dims=1))
    M
end

function load_viscosity_NS(dir)
    filename = joinpath(dir, "src", "programmParameter.init")
    ν = nothing
    for line in eachline(filename)
        if startswith(line, "mu ")
            a, b = split(line, limit=2)
            ν = parse(Float64, b)
            break
        end
    end
    ν
end

function load_dissipation_NS(dir, timestep)
    filename = joinpath(dir, "diagnostic", "energyDiss.log")
    ε = nothing
    pat = Regex("^$timestep\\s+\\S+\\s+(\\S+)")
    for line in eachline(filename)
        m = match(pat, line)
        if m !== nothing
            ε = parse(Float64, m[1])
            break
        end
    end
    ε
end

function load_energy_NS(dir, timestep)
    filename = joinpath(dir, "diagnostic", "energy.log")
    E = nothing
    pat = Regex("^$timestep\\s+\\S+\\s+(\\S+)")
    for line in eachline(filename)
        m = match(pat, line)
        if m !== nothing
            E = parse(Float64, m[1])
            break
        end
    end
    E
end

function read_loop_sizes(g)
    ff = file(g)
    dx = load_dx(ff)
    # Note: loop sizes are in units of grid steps
    rs = g["loop_sizes"][:]::Vector{Int} .* dx
    rs
end

function plot_moment!(
        ax, g; order::Val{p}, dx, norm_r=1, norm_Γ=1, plot_kw...) where {p}
    # rs = g["loop_sizes"][:]::Vector{Int} .* dx
    rs = read_loop_sizes(g)
    if MOMENTS_FROM_HISTOGRAM[]
        gg = g_open(g, "Histogram")
        M = moment_from_histogram(gg, order)
    else
        gg = g_open(g, "Moments")
        p_all = gg["p_abs"][:]::Vector{Int}
        n = searchsortedlast(p_all, p)
        @assert p_all[n] == p
        M = gg["M_abs"][n, :]::Vector{Float64} # ./ (norm_Γ^p)
        # FIXME input values are not normalised!
        M ./= gg["total_samples"][n]
    end
    Nr = length(rs) - 2  # skip very large loops (periodicity problems)
    let r = rs[1:Nr], M = M[1:Nr]
        A = r.^2
        # To check small-scale prediction in NS: <Γ^2> = <|ω|^2> A^2 / 3 (in 3D)
        @show filename(g), M[1] / A[1]^p
        M ./= norm_Γ^p
        A ./= norm_r^2
        @show A[1] M[1]
        ax.plot(A, M; plot_kw...)
        # ax.plot(A, M .* A.^(-2/3); plot_kw...)
        # ax.plot(r[2:end], diff(log.(M)) ./ diff(log.(r)); plot_kw...)
    end
    nothing
end

load_dx(ff) = let g = g_open(ff, "ParamsGP")
    g["L"][1] / g["dims"][1]
end :: Float64

function plot_moment!(ax, filename::AbstractString, h5field="Velocity"; etc...)
    h5open(filename, "r") do ff
        dx = load_dx(ff)
        g = g_open(ff, "/Circulation/$h5field")
        plot_moment!(ax, g; dx=dx, etc...)
    end
    ax
end

function parse_timestep_NS(data_file) :: Int
    m = match(r"t(\d+)\.h5$", data_file)
    @assert m !== nothing
    parse(Int, m[1])
end

function plot_moment_NS!(ax, params; etc...)
    data_file = params.data_ns
    dir = params.NS_dir
    step = parse_timestep_NS(data_file)
    ν = load_viscosity_NS(dir)
    ε = load_dissipation_NS(dir, step)
    E = load_energy_NS(dir, step)
    Ω = ε / 2ν        # enstrophy
    λ = sqrt(5E / Ω)  # Taylor scale
    # η = (ν^3 / ε)^(1/4)
    L = λ
    Γ_0 = L^2 * sqrt(2Ω / 3)
    plot_moment!(ax, data_file;
                 norm_r = L,
                 norm_Γ = Γ_0,
                 # norm_Γ = Γ_L,
                 etc...)
end

function plot_moment_GP!(ax, params; etc...)
    data_file = params.data_gp
    ℓ = params.ℓ_gp
    Γ_L = params.L_ABC * params.v_ABC
    κ = h5read(data_file, "/ParamsGP/kappa") :: Float64
    plot_moment!(ax, data_file;
                 norm_r=ℓ,
                 # norm_Γ=Γ_L,
                 norm_Γ=κ,
                 etc...)
end

logrange(a, b, N) = exp.(LinRange(log.((a, b))..., N))

function plot_moment(params; order::Val{p}, etc...) where {p}
    fig, ax = plt.subplots()
    ax.set_xscale(:log)
    ax.set_yscale(:log)
    ax.set_yticks(10.0 .^ (-3:3))
    # ax.grid(true)
    ax.set_xlabel(L"A / ℓ^2")
    # ax.set_ylabel(L"\left< Γ^2 \right>^{1/2} / (σ_v L)")
    let p = replace(string(p), "//" => "/")
        ax.set_ylabel(latexstring("\\left< Γ^{$p} \\right> / κ^{$p}"))
    end
    ax.axvline(1.0, color="tab:gray", alpha=0.6, lw=1, ls="--")
    plot_moment_GP!(
        ax, params;
        order, label="gGP", ls="-", color="tab:blue", etc...,
    )
    plot_moment_NS!(ax, params;
                    order, label="NS", ls=":", color="tab:orange", etc...)
    let xvar = "A", kw = (lw=0.7, color="black", ls="--")
        ℓ = params.ℓ_gp
        κ = h5read(params.data_gp, "/ParamsGP/kappa") :: Float64
        plot_power_law!(ax, (2e-3, 0.5), 1, 2, xvar; kw...)
        λ = isinteger(3000p) ? 2Int(3000p)//(3 * 3000) : 2p/3
        plot_power_law!(ax, (1.2, 320), λ, 2, xvar; kw...)
        plot_power_law!(ax, (1.2e-2, 1.2e-1), p, 0.5, xvar;
                        ha="left", va="top", kw...)
    end
    ax.legend()
    fig
end

function plot_power_law!(ax, (xmin, xmax), n, α=1, xvar="x";
                         ha="right", va="bottom", text_shift=(1, 1), color,
                         kw...)
    x = logrange(xmin, xmax, 3)
    y = α * x.^n
    ax.plot(x, y; color=color, kw...)
    let pow_s = replace(string(n), "//" => "/")
        xy = (x[2], y[2]) .* text_shift
        ax.text(xy..., latexstring(xvar, "^{", pow_s, "}"),
                va=va, ha=ha, color=color)
    end
    ax
end

function plot_prob_zero_NS!(ax, params)
    data_file = params.data_ns
    dir = params.NS_dir
    ν = load_viscosity_NS(dir)
    step = parse_timestep_NS(data_file)
    ε = load_dissipation_NS(dir, step)
    E = load_energy_NS(dir, step)
    Ω = ε / 2ν        # enstrophy
    @show E Ω ν
    η = (ν^3 / ε)^(1/4)
    λ = sqrt(5E / Ω)  # Taylor scale
    dx = 2π / 1024
    @show η / dx
    @show λ / η
    rs, Γ, H = h5open(data_file, "r") do ff
        g = g_open(ff, "/Circulation/Velocity/Histogram")
        rs = read_loop_sizes(parent(g))
        Γ, H = read_all_pdfs(g, bin_centres=false)  # return bin edges
        @assert length(Γ) == size(H, 1) + 1
        Γ ./= ν
        rs ./= λ
        rs, Γ, H
    end
    # Assume bins are centred at 0 -> there is a [-a, a] bin
    n = searchsortedlast(Γ, 0)
    @assert Γ[n] < 0 < Γ[n + 1]
    @assert Γ[n] ≈ -Γ[n + 1]
    @show Γ[n], Γ[n + 1]
    dn = 0
    ind = (n - dn):(n + dn)
    @assert Γ[first(ind)] ≈ -Γ[last(ind) + 1]
    Γ_max = Γ[last(ind) + 1]
    Γ_max_round = round(Γ_max, digits=1)
    prob = @views vec(sum(H[ind, :], dims=1))
    write_prob_zero_NS("prob_zero_NS.txt", rs, prob; Γ_max)
    ax.plot(rs, prob, "o-", color="tab:green",
            label="Classical (\$|Γ| / ν < $Γ_max_round\$)")
    ax
end

function write_prob_zero_NS(filename, r, P; Γ_max)
    open(filename, "w") do ff
        write(ff, "#  (1) r/eta \t (2) Prob(|Gamma|/nu < $Γ_max)\n")
        writedlm(ff, zip(r, P))
    end
    nothing
end

function plot_prob_zero(params)
    fig, ax = plt.subplots(figsize=(4, 3) .* 0.9)
    ax.set_xlabel("\$r / ℓ\$, \$r / λ\$")
    ax.set_ylabel(L"\mathbb{P}(Γ_r = 0)")
    ax.set_xscale(:log)
    ax.set_yscale(:log)
    rs, Γ, H = h5open(params.data_gp, "r") do ff
        κ = read(ff, "/ParamsGP/kappa") :: Float64
        g = g_open(ff, "/Circulation/Velocity/Histogram")
        rs = read_loop_sizes(parent(g))
        rs ./= params.ℓ_gp
        Γ, H = read_all_pdfs(g)
        Γ ./= κ
        rs, Γ, H
    end
    n = searchsortedlast(Γ, 0)
    @assert Γ[n] == 0
    @views ax.plot(rs, H[n, :], "^-", label="Quantum")
    plot_power_law!(ax, (0.6, 700), -4//3, 0.3, "r";
                    text_shift=(0.9, 0.9), lw=1.5,
                    ha="right", va="top", color="tab:orange", zorder=100)
    plot_prob_zero_NS!(ax, params)
    ax.legend(loc="lower left", frameon=false)
    fig
end

function make_plots()
    params = get_params()
    let fig = plot_moment(params, order=Val(2))
        display(fig)
        fig.savefig("gamma_variance")
    end
    let fig = plot_prob_zero(params)
        display(fig)
        # fig.savefig("prob_zero")
    end
    nothing
end

make_plots()
