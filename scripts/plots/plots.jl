#!/usr/bin/env julia

using PyPlot: plt
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

function read_all_pdfs(g::HDF5.Group; bin_centres=true, normalise=true)
    if haskey(g, "hist_filt")
        H = g["hist_filt"][:, :] :: Matrix{Float64}
        Γ = g["bin_center_filt"][:] :: Vector{Float64}
        # These circulation values need to be unnormalised...
        κ = read(HDF5.file(g)["/ParamsGP/kappa"]) :: Float64
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
    if normalise
        dΓ = Γ[2] - Γ[1]  # assume uniform distribution
        Hsum = sum(H, dims=1)
        H ./= Hsum * dΓ
    end
    Γ, H
end

function moment_from_histogram(g::HDF5.Group, args...; kws...)
    Γ, H = read_all_pdfs(g; kws...)
    # n = searchsortedlast(Γ, 0)
    # @show Γ[n]
    moment_from_histogram(H, Γ, args...)
end

function moment_from_histogram(
        f::Function, H::AbstractMatrix, bin_centres,
        ::Val{p}) where {p}
    Nb, Nr = size(H)
    @assert length(bin_centres) == Nb
    # @show sum(H, dims=1)
    M = zeros(Nr)
    dx = bin_centres[2] - bin_centres[1]  # assume uniform distribution
    @inbounds for i = 1:Nb
        x = bin_centres[i]
        xp = f(x)^p
        for r = 1:Nr
            M[r] += xp * H[i, r] * dx
        end
    end
    # M ./= vec(sum(H, dims=1))
    M
end

moment_from_histogram(f::Function, H::AbstractVector, etc...) =
    moment_from_histogram(f, reshape(H, :, 1), etc...)[1]

# By default, apply absolute value to Γ
moment_from_histogram(H::AbstractArray, etc...) =
    moment_from_histogram(abs, H, etc...)

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
    ff = HDF5.file(g)
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
        gg = open_group(g, "Histogram")
        M = moment_from_histogram(gg, order)
    else
        gg = open_group(g, "Moments")
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
        @show HDF5.filename(g), M[1] / A[1]^p
        M ./= norm_Γ^p
        A ./= norm_r^2
        @show A[1] M[1]
        ax.plot(A, M; plot_kw...)
        # ax.plot(A, M .* A.^(-2/3); plot_kw...)
        # ax.plot(r[2:end], diff(log.(M)) ./ diff(log.(r)); plot_kw...)
    end
    nothing
end

load_dx(ff) = let g = open_group(ff, "ParamsGP")
    g["L"][1] / g["dims"][1]
end :: Float64

function plot_moment!(ax, filename::AbstractString, h5field="Velocity"; etc...)
    h5open(filename, "r") do ff
        dx = load_dx(ff)
        g = open_group(ff, "/Circulation/$h5field")
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

function diagnostics_NS(datafile, simdir)
    dir = simdir
    step = parse_timestep_NS(datafile)
    ε = load_dissipation_NS(dir, step)
    E = load_energy_NS(dir, step)
    ν = load_viscosity_NS(dir)
    Ω = ε / 2ν  # enstrophy
    @show E Ω ν
    (
        η = (ν^3 / ε)^(1/4),
        λ = sqrt(5E / Ω),  # Taylor scale
        ν,
    )
end

function plot_prob_zero_NS!(ax, params)
    data_file = params.data_ns
    η, λ, ν = let L = diagnostics_NS(data_file, params.NS_dir)
        L.η, L.λ, L.ν
    end
    let dx = 2π / 1024
        @show η / dx
    end
    @show λ / η
    rs, Γ, H = h5open(data_file, "r") do ff
        g = open_group(ff, "/Circulation/Velocity/Histogram")
        rs = read_loop_sizes(parent(g))
        Γ, H = read_all_pdfs(g, bin_centres=false)  # return bin edges
        @assert length(Γ) == size(H, 1) + 1
        Γ ./= ν
        H .*= ν
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
    dΓ = Γ[2] - Γ[1]
    prob = @views vec(sum(H[ind, :], dims=1)) .* dΓ
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
        g = open_group(ff, "/Circulation/Velocity/Histogram")
        rs = read_loop_sizes(parent(g))
        rs ./= params.ℓ_gp
        Γ, H = read_all_pdfs(g)
        Γ ./= κ
        H .*= κ
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

function plot_pdfs_NS(params)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_yscale(:log)
    NS = diagnostics_NS(params.data_ns, params.NS_dir)
    h5open(params.data_ns, "r") do ff
        gbase = open_group(ff, "Circulation/Velocity")
        plot_pdfs_NS!(
            ax, gbase, NS;
            xnorm = :rms,
        )
    end
    ax.text(
        0.97, 0.98, "(c)"; fontsize = :large,
        ha = :right, va = :top, transform = ax.transAxes,
    )
    fig
end

function plot_pdfs_NS!(ax, gbase, NS; xnorm = nothing, kws...)
    rs = read_loop_sizes(gbase)
    Nr = length(rs)
    # rind = 2:12:round(Int, 0.9 * Nr)
    λ = NS.λ  # Taylor scale
    rλ_wanted = (1, 5, 12)
    rind = map(rλ -> searchsortedlast(rs, rλ * λ), rλ_wanted)
    ghist = open_group(gbase, "Histogram")
    Γin, Hin = read_all_pdfs(ghist)
    @assert all(sum(Hin, dims=1) .* (Γin[2] - Γin[1]) .≈ 1)  # PDFs are normalised
    Γ = similar(Γin)
    P = similar(Hin, size(Hin, 1))
    for n in rind
        r = rs[n]
        rλ = round(r / λ, digits=1)  # normalised by Taylor scale
        copy!(Γ, Γin)
        copy!(P, view(Hin, :, n))
        plot_pdf!(ax, Γ, P, NS; xnorm, label = latexstring(rλ), kws...)
    end
    ax.legend(
        title = L"r / λ_{\mathrm{T}}", fontsize = :small, loc = "upper left")
    if xnorm == :rms
        ax.set_xlim(-16, 16)
        ax.set_ylim(1e-7, 0.9)
        ax.set_xlabel(L"Γ_{\! r} / \left\langle Γ_{\! r}^2 \right\rangle^{1/2}")
        let x = -10:0.1:10
            # Pnormal = @. exp(-x^2 / 2) / sqrt(2π)
            # ax.plot(x, Pnormal; lw=1, ls=:dashed, c="0.4")
        end
    end
    plot_pdf_fit!(ax; xnorm)
    ax
end

function plot_pdf!(ax, Γ, P, NS; xnorm, plot_kws...)
    scale = if xnorm == :ν
        NS.ν
    elseif xnorm == :rms
        # identity: don't apply absolute value to Γ
        mean1 = moment_from_histogram(identity, P, Γ, Val(1))
        mean2 = moment_from_histogram(identity, P, Γ, Val(2))
        # mean3 = moment_from_histogram(identity, P, Γ, Val(3))
        sqrt(mean2 - mean1^2)
    else
        1
    end
    Γ ./= scale
    P .*= scale
    ax.plot(Γ, P; plot_kws...)
    ax
end

function plot_pdf_fit!(ax; xnorm, plot_kws...)
    # Fit as in Iyer+ 2020 (arXiv)
    if xnorm == :rms
        α = 0.8
        b = 1.2
        x = 1:0.1:10
    end
    p = @. α * exp(-b * x) / sqrt(x)
    ax.plot(x, p; ls=:dotted, c="0.3", plot_kws...)
    ax.plot(-x, p; ls=:dotted, c="0.3", plot_kws...)
    ax
end

function fig_exponential_tails()
    fig, ax = plt.subplots(figsize=(3.8, 1.6))
    ax.set_yscale(:log)
    ax.set_xlim(-55, 55)
    ax.set_ylim(1e-7, 2)
    x = 5:0.5:40
    α = 1
    β = 0.8
    p = @. exp(-α * x^β)
    kws = (color = "0.3", ls = :dotted)
    ax.plot(x, p; kws...)
    ax.plot(-x, p; kws...)
    fig
end

function make_plots()
    params = get_params()
    fig_exponential_tails()
    let fig = plot_pdfs_NS(params)
        display(fig)
    end
    return
    let fig = plot_moment(params, order=Val(2))
        display(fig)
        # fig.savefig("gamma_variance")
    end
    let fig = plot_prob_zero(params)
        display(fig)
        # fig.savefig("prob_zero")
    end
    nothing
end

make_plots()
