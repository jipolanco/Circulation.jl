import PyPlot: plt
const mpl = plt.matplotlib
using LaTeXStrings
using TimerOutputs

using DelimitedFiles
using HDF5

# Did we compute histograms with circulation divided by area?
const CIRCULATION_DIVIDED_BY_AREA = Ref(true)

const to = TimerOutput()

@timeit to function load_histograms1D(g, rs, field)
    xs = g["bin_edges"][:] :: Vector{Float64}
    hist = g["hist"][:, :] :: Array{Int64,2}
    samples = g["total_samples"][:] :: Vector{Int}

    # check_histogram(hist, samples, basename(HDF5.name(g)))

    pdfs = similar(hist, Float64)
    for j in axes(pdfs, 2)
        p = @view pdfs[:, j]
        h = @view hist[:, j]
        N = samples[j]
        A = rs[j]^2
        for i in eachindex(p)
            dx = xs[i + 1] - xs[i]
            p[i] = h[i] / (N * dx)
        end
    end
    (;
        field, bin_edges = xs, rs, hist, pdfs, samples,
    )
end

@timeit to function load_stats(g, params)
    rs = g["kernel_size"][:] :: Vector{Float64}
    As = g["kernel_area"][:] :: Vector{Float64}
    @assert As ≈ rs.^2

    let g = open_group(g, "FieldMetadata")
        @assert read(g["CirculationField/divided_by_area"]) ==
            CIRCULATION_DIVIDED_BY_AREA[]
    end

    histograms_Γ = load_histograms1D(g["HistogramCirculation"], rs, :Γ)

    ν = params.ν
    ε_mean = params.DNS_dissipation

    Lbox = params.Ls[1]
    Δx = Lbox / params.Ns[1]
    η = (ν^3 / ε_mean)^(1/4)
    rs_η = rs ./ η
    As_η = As ./ η^2

    taylor_scale = let E = params.DNS_energy
        sqrt(10ν * E / ε_mean)
    end

    @show taylor_scale
    rs_λ = rs ./ taylor_scale

    enstrophy = ε_mean / (2ν)
    Γ_taylor = taylor_scale^2 * sqrt(2enstrophy / 3)

    @show ε_mean, enstrophy
    @show η, Lbox / η, Δx / η, taylor_scale / η

    (; rs, As, rs_η, As_η, rs_λ, L = Lbox, η, ν,
        taylor_scale, enstrophy, Γ_taylor,
        histograms_Γ,
        ε_mean,
    )
end

function load_simulation_params(g, paramsNS)
    (;
        Ns = Tuple(g["dims"][:]) :: NTuple{3,Int},
        Ls = Tuple(g["L"][:]) :: NTuple{3,Float64},
        paramsNS...,
        # η = 0.00245,  # from DNS
    )
end

function load_data(ff::HDF5.File, paramsNS)
    params = load_simulation_params(ff["SimParams"], paramsNS)
    stats = load_stats(ff["Statistics/Velocity"], params)
    (; params, stats)
end

@inline moment(f::Function, edges, pdf, p) = sum(eachindex(pdf)) do i
    dx = edges[i + 1] - edges[i]
    xc = f((edges[i] + edges[i + 1]) / 2)
    xc^p * pdf[i] * dx
end

@timeit to function fractional_moments(hists, stats; ps = 0:0.1:1)
    xs = hists.bin_edges
    pdfs = hists.pdfs :: AbstractMatrix
    rs = collect(hists.rs)

    multiply_by_area = hists.field == :Γ && CIRCULATION_DIVIDED_BY_AREA[]

    As = stats.As
    λ = stats.taylor_scale

    Nx, Nr = size(pdfs)
    @assert length(xs) == Nx + 1
    Np = length(ps)

    @assert length(rs) == Nr
    dlog_r = diff(log.(rs))

    moments = zeros(Nr, Np)
    slopes = zeros(Nr - 1, Np)

    for (j, p) in enumerate(ps)
        ys = @view moments[:, j]
        sl = @view slopes[:, j]
        for (i, r) in enumerate(rs)
            pdf = @view pdfs[:, i]
            ys[i] = moment(abs, xs, pdf, p)
            if multiply_by_area
                ys[i] *= As[i]^p
            end
        end

        dlog_y = diff(log.(ys))
        sl[:] .= dlog_y ./ dlog_r
    end

    rs_slopes = map(eachindex(rs)[2:end]) do i
        exp((log(rs[i - 1]) + log(rs[i])) / 2)
    end

    (; rs, rs_slopes, ps, moments, slopes)
end

# function plot_moment!(ax, rs, moments, p; cmap)
#     ax.plot()
# end

function plot_moments(M, stats)
    fig, axs = plt.subplots(2, 1; figsize = (4.5, 5), dpi = 200, sharex = true)

    Γ_norm = stats.ν
    r_norm = stats.taylor_scale

    inds_p = eachindex(M.ps)

    cmap = let is = inds_p
        cm = plt.cm.Oranges
        di = is[end] - is[begin] + 2
        i -> cm((i - is[begin] + 2) / di)
    end

    let ax = axs[1]
        ax.set_xscale(:log)
        ax.set_yscale(:log)
        ax.set_ylabel(L"\left< |Γ_{\!r}|^{\,p} \right> / ν^{\,p}")
        xs = M.rs ./ r_norm
        for (j, p) in enumerate(M.ps)
            color = cmap(j)
            ys = @view(M.moments[:, j]) ./ Γ_norm^p
            ax.plot(xs, ys; color, label = p)
        end
        ax.axhline(1; lw = 2, color = "0.3", ls = :dotted)
        ax.legend(
            title = L"p", ncol = 2, frameon = true, labelspacing = 0.4,
            # borderpad = 0.1,
            fontsize = :small,
        )
    end

    let ax = axs[2]
        ax.set_xscale(:log)
        ax.set_xlabel(L"r / λ")
        ax.set_ylabel(L"λ_p(r)")
        xs = M.rs_slopes ./ r_norm

        for (j, p) in enumerate(M.ps)
            color = cmap(j)
            ys = @view M.slopes[:, j]
            ax.plot(xs, ys; color, label = p)

            let
                a = searchsortedlast(xs, 0.5)  # r / λ ≈ 0.5
                b = searchsortedlast(xs, 20)
                xs_ir = @view xs[a:b]
                slopes_kolm = fill!(similar(xs_ir), 4p / 3)
                ax.plot(xs_ir, slopes_kolm; color, ls = :dashed, lw = 1.5)
            end
        end

        ax.axhline(0; lw = 2, color = "0.5", ls = :dotted)
        # ax.legend(title = L"p", ncol = 2)
    end

    fig
end

function main()
    reset_timer!(to)

    mpl.rc(:font, size = 12)
    mpl.rc(:lines, linewidth = 1.5)

    params_NS_2048 = (;
        DNS_timestep = 32_000,
        DNS_energy = 3.6588,
        DNS_dissipation = 1.2523,
        ν = 2.8e-4,
    )
    params = (;
        results_file = "../../results/data_NS/NS2048/circulation_cond.h5",
        NS = params_NS_2048,
    )

    data = h5open(io -> load_data(io, params.NS), params.results_file, "r")
    stats = data.stats

    ps = vcat(0, 1e-4, 0.1, 0.2:0.2:1)
    moments = fractional_moments(stats.histograms_Γ, stats; ps)
    plot_moments(moments, stats)

    println(to)

    nothing
end

main()
