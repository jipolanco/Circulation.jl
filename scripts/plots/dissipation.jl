using PyPlot: plt
const mpl = plt.matplotlib
using LaTeXStrings

using DelimitedFiles
using HDF5

params = (;
    results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/circulation_test1.h5",
)

function make_dissipation_intervals(εs; join = 50)
    dε = (εs[2] - εs[1]) * join
    range(first(εs), last(εs); step = dε)
end

# PDFs of Γ_r conditioned on ε_r
function conditional_pdfs_ε(εs_all, hist_orig)
    @assert ndims(hist_orig) == 3
    εs = make_dissipation_intervals(εs_all)
    dims = (size(hist_orig, 1), length(εs))
    hist = zeros(eltype(hist_orig), dims)
    for j ∈ axes(hist_orig, 2)
        ε = (εs_all[j] + εs_all[j + 1]) / 2
        jj = searchsortedlast(εs, ε)
        @assert !iszero(jj)
        for k ∈ axes(hist_orig, 3), i ∈ axes(hist_orig, 1)
            hist[i, jj] += hist_orig[i, j, k]
        end
    end
    hist_ε = partial_sum(hist; dims = 1)
    pdf = similar(hist, Float32)
    for (I, val) in pairs(IndexCartesian(), hist)
        j = Tuple(I)[2]
        pdf[I] = val / hist_ε[j]
    end
    (;
        εs,
        cond_pdf_Γ = pdf,
    )
end

# PDFs of Γ_r conditioned on ε_r * A
# This doesn't give very good results, because the Γ bins are resampled...
function conditional_pdfs_εA(As, εs, ωs, εA_kolmogorov, hist_orig; ν)
    εAs = range(0, 10εA_kolmogorov; length = 101)
    Γs = range(-1, 1; length = 1001) .* 10ν
    Nx = length(Γs) - 1
    Ny = length(εAs) - 1
    hist = zeros(eltype(hist_orig), Nx, Ny)
    @inbounds for k ∈ axes(hist_orig, 3), j ∈ axes(hist_orig, 2)
        A = As[k]
        εA = εs[j] * A
        jj = searchsortedlast(εAs, εA)
        if jj ∈ (0, Ny + 1)
            continue
        end
        for i ∈ axes(hist_orig, 1)
            Γ = A * ωs[i]
            ii = searchsortedlast(Γs, Γ)
            if ii ∉ (0, Nx + 1)
                hist[ii, jj] += hist_orig[i, j, k]
            end
        end
    end
    hist_εA = partial_sum(hist; dims = 1)
    # @show hist_εA  # show number of events for each interval
    pdf = similar(hist, Float32)
    for (I, val) in pairs(IndexCartesian(), hist)
        j = Tuple(I)[2]
        pdf[I] = val / hist_εA[j]
    end
    (;
        εAs,
        Γs,
        cond_pdf_Γ = pdf,
    )
end

function load_histograms(g, As; params)
    hist = g["hist"][:, :, :] :: Array{Int,3}        # (Γ, ε, r)
    minima = g["minimum"][:, :] :: Array{Float64,2}  # (Γ, ε)
    maxima = g["maximum"][:, :] :: Array{Float64,2}

    @show minima[1, :] .* As
    @show maxima[1, :] .* As

    # This is actually Γ / A (→ Γ bins vary with loop size!!)
    bins_Γ = g["bin_edges1"][:] :: Vector{Float64}

    bins_ε = g["bin_edges2"][:] :: Vector{Float64}
    samples = g["total_samples"][:] :: Vector{Int}

    hist_Γ = partial_sum(hist; dims = 2)
    hist_ε = partial_sum(hist; dims = 1)
    totals = partial_sum(hist_Γ; dims = 1)

    # conditional_ε = conditional_pdfs_ε(bins_ε, hist)

    # Check that (almost) all samples are in the PDFs
    @assert isapprox(totals, samples; rtol=1e-4)

    # Estimate mean from smallest loop
    ε_mean = let H = @view hist_ε[:, 1]
        moment(bins_ε, H, Val(1)) / moment(bins_ε, H, Val(0))
    end
    @show ε_mean

    η = params.η
    ν = params.ν
    εA_kolmogorov = ε_mean * η^2  # used to define εA bins
    conditional_εA = conditional_pdfs_εA(
        As, bins_ε, bins_Γ, εA_kolmogorov, hist; ν,
    )

    (; hist, hist_Γ, hist_ε, minima, maxima, bins_Γ, bins_ε, samples,
        # conditional_ε,
        conditional_εA,
        ε_mean,
    )
end

partial_sum(u; dims) = dropdims(sum(u; dims); dims)

function r_indices_to_plot(rs)
    inds = eachindex(rs)
    N = length(inds)
    subinds = Iterators.flatten((inds[1:10], inds[12:4:(N - 8)]))
    collect(subinds)  # we want an AbstractVector...
end

function mappable_colour(cmap, vmin, vmax)
    norm = mpl.colors.Normalize(vmin, vmax)
    let cmap = mpl.cm.ScalarMappable(; norm, cmap)
        v -> cmap.to_rgba(v)
    end
end

function mappable_colour(cmap, inds::AbstractVector)
    r = eachindex(inds)
    mappable_colour(cmap, first(r), last(r))
end

function load_stats(g, params)
    rs = g["kernel_size"][:] :: Vector{Float64}
    As = g["kernel_area"][:] :: Vector{Float64}
    @assert As ≈ rs.^2
    Lbox = params.Ls[1]
    Δx = Lbox / params.Ns[1]
    η = params.η
    @show Lbox / η, Δx / η
    rs_η = rs ./ η
    As_η = As ./ η^2
    let g = open_group(g, "FieldMetadata")
        @assert read(g["CirculationField/divided_by_area"]) == true
        @assert read(g["DissipationField/divided_by_area"]) == true
    end
    histograms = load_histograms(g["Histogram2D"], As; params)
    (; rs, As, rs_η, As_η, L = Lbox, η, histograms)
end

function load_simulation_params(g)
    (
        Ns = Tuple(g["dims"][:]) :: NTuple{3,Int},
        Ls = Tuple(g["L"][:]) :: NTuple{3,Float64},
        η = 0.00245,  # from DNS
        ν = 5e-5,
    )
end

function load_data(ff::HDF5.File)
    params = load_simulation_params(ff["SimParams"])
    stats = load_stats(ff["Statistics/Velocity"], params)
    (; params, stats)
end

function plot_pdf_2D!(ax, stats; r)
    histograms = stats.histograms
    hist = @view histograms.hist[:, :, r]
    A = stats.As[r]
    pdf = Float32.(hist) ./ (histograms.samples[r] * A)
    Γ = histograms.bins_Γ .* A
    ε = histograms.bins_ε
    ax.set_xlabel(L"Γ_r")
    ax.set_ylabel(L"ε_r")
    norm = mpl.colors.LogNorm(1e-8, 1e-3)
    plot = ax.pcolormesh(Γ, ε, pdf'; norm, shading = :flat)
    fig = ax.get_figure()
    fig.colorbar(plot; ax)
    ax
end

moment(edges, pdf, ::Val{p}) where {p} = sum(eachindex(pdf)) do i
    dx = edges[i + 1] - edges[i]
    xc = (edges[i] + edges[i + 1]) / 2
    xc^p * pdf[i] * dx
end

function variance(edges, pdf)
    mean1 = moment(edges, pdf, Val(1))
    mean2 = moment(edges, pdf, Val(2))
    mean2 - mean1^2
end

function normal_pdf(x; μ = 0, σ = 1)
    exp(-0.5 * (x - μ)^2 / σ^2) / (σ * sqrt(2π))
end

function plot_pdf_Γr!(ax, stats)
    histograms = stats.histograms
    hists = histograms.hist_Γ
    As = stats.As
    ax.set_yscale(:log)
    ax.set_xlim(-20, 20)
    ax.set_xlabel(L"Γ_r / \left\langle Γ_r^2 \right\rangle^{1/2}")
    ax.set_ylabel("Probability")
    r_indices = r_indices_to_plot(As)
    cmap = mappable_colour(plt.cm.viridis_r, r_indices)

    for (i, r) in enumerate(r_indices)
        color = cmap(i)
        A = As[r]
        hist = @view hists[:, r]
        xs = histograms.bins_Γ

        # Multiply by area to get actual circulation
        xs .*= A

        pdf = let dx = xs[2] - xs[1]
            Float32.(hist) ./ (histograms.samples[r] * dx)
        end

        xrms = sqrt(variance(xs, pdf))
        pdf .*= xrms
        xs ./= xrms

        xs_centre = @views (xs[1:end-1] .+ xs[2:end]) ./ 2
        r_η = round(stats.rs_η[r], digits=0)
        ax.plot(
            xs_centre, pdf;
            marker = ".", color, label = latexstring("$r_η"),
        )
    end
    ax
end

function plot_condpdf_Γ_εA!(ax, stats)
    histograms = stats.histograms
    conditional = histograms.conditional_εA
    η = stats.η
    ε_mean = histograms.ε_mean
    εA_norm = η^2 * ε_mean

    pdfs = conditional.cond_pdf_Γ
    εAs = conditional.εAs

    ax.set_yscale(:log)
    # ax.set_xlim(-40, 40)
    # ax.set_ylim(1e-6, 1)
    # ax.set_xlabel(L"ω_r / \left\langle ω_r^2 | ε_r \right\rangle^{1/2}")
    # ax.set_ylabel(L"P(ω_r | ε_r)")

    εA_indices = eachindex(εAs)[5:5:end]
    Ncurves = length(εA_indices)
    cmap = mappable_colour(plt.cm.cividis_r, 0, 1)

    for j in εA_indices
        pdf = pdfs[:, j]
        xs = conditional.Γs

        εA_nominal = round.((εAs[j], εAs[j + 1]) ./ εA_norm, digits=4)

        # ε_label = round.((εs[j], εs[j + 1]) ./ histograms.ε_mean, digits=2)
        # ε_label[1] > max_ε && continue
        # color = cmap(ε_label[1])
        color = cmap((j - 1) / (Ncurves - 1))

        # xrms = sqrt(variance(xs, pdf))
        # pdf .*= xrms
        # xs ./= xrms

        xs_centre = @views (xs[1:end-1] .+ xs[2:end]) ./ 2
        ax.plot(
            xs_centre, pdf;
            marker = ".", color,
            label = εA_nominal,
            # label = latexstring("$r_η"),
        )
    end

    ax
end

function plot_pdf_εr!(ax, stats)
    histograms = stats.histograms
    hists = histograms.hist_ε
    As = stats.As
    ax.set_yscale(:log)
    ax.set_xlabel(L"\left( \log ε_r - \left\langle \log ε_r \right\rangle \right) / σ(\log ε_r)")
    r_indices = r_indices_to_plot(As)
    cmap = mappable_colour(plt.cm.viridis_r, r_indices)
    for (i, r) in enumerate(r_indices)
        color = cmap(i)
        A = As[r]
        hist = @view hists[:, r]
        xs = histograms.bins_ε
        dx = xs[2] - xs[1]
        pdf = Float32.(hist) ./ (dx * histograms.samples[r])

        xs_centre = @views (xs[1:end-1] .+ xs[2:end]) ./ 2
        xs_centre_log = log.(xs_centre)

        pdf_log = pdf .* xs_centre

        xs_log = log.(xs)
        if isinf(xs_log[1])
            xs_log[1] = xs_log[2]
        end

        εlog_mean = moment(xs_log, pdf_log, Val(1))
        εlog_std = sqrt(moment(xs_log, pdf_log, Val(2)) - εlog_mean^2)

        # Normalise by mean and std
        xs_centre_log .= (xs_centre_log .- εlog_mean) ./ εlog_std
        pdf_log .*= εlog_std

        r_η = round(stats.rs_η[r], digits=0)
        ax.plot(
            xs_centre_log, pdf_log;
            color,
            marker = ".", label = latexstring("$r_η"),
        )
    end

    let xs = -5:0.1:5
        pdf = normal_pdf.(xs)
        ax.plot(xs, pdf; lw = 1.5, color = "tab:orange", ls = :dashed)
    end

    ax.set_xlim(-5, 4)

    ax
end

data = h5open(load_data, params.results_file, "r");
stats = data.stats;

let
    fig, axes = plt.subplots(1, 3; figsize = (10, 4))
    let ax = axes[1]
        plot_pdf_Γr!(ax, stats)
        # ax.legend()
    end
    let ax = axes[3]
        plot_condpdf_Γ_εA!(ax, stats)
        ax.legend(
            title = L"ε_r r^2 / \langle ε \rangle η^2", fontsize = "x-small",
            framealpha = 0.5,
        )
    end
    let ax = axes[2]
        plot_pdf_εr!(ax, stats)
        ax.legend(title = L"r / η", fontsize = :small)
    end
    fig.savefig("dissipation.png", dpi=300)
end
