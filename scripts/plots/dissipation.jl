import PyPlot: plt
const mpl = plt.matplotlib
using LaTeXStrings
using TimerOutputs

using DelimitedFiles
using HDF5

# Did we compute histograms with circulation divided by area?
const CIRCULATION_DIVIDED_BY_AREA = Ref(true)

# Did we compute histograms with dissipation multiplied by area?
const DISSIPATION_MULTIPLIED_BY_AREA = Ref(false)

const TEX_MEAN_ε = raw"\left< ε \right>"

const to = TimerOutput()

function to_steprange(xs::AbstractVector)
    islinear = all(eachindex(xs)[2:end-1]) do i
        a = xs[i] - xs[i - 1]
        b = xs[i + 1] - xs[i]
        a ≈ b  # same step
    end
    @assert islinear
    step = xs[2] - xs[1]
    range(xs[1], xs[end]; step)
end

function check_histogram(hist, samples, label; tol = 1e-3)
    colons = ntuple(_ -> Colon(), Val(ndims(hist) - 1))
    # Check that (almost) all samples are in the PDFs
    for r in eachindex(samples)
        ratio = @views sum(hist[colons..., r]) / samples[r]
        if 1 - ratio > tol
            @show label, r, ratio
        end
    end
    nothing
end

@timeit to function load_histograms2D(g, rs)
    merged_scales = Bool(read(g["merged_scales"]))
    hist_full = g["hist"][:, :, :] :: Array{Int64,3}  # (Γ, ε, r)

    hist1 = read(g["hist1"]) :: Array{Int64,2}  # (Γ, r)
    hist2 = read(g["hist2"]) :: Array{Int64,2}  # (ε, r)

    bins1 = to_steprange(g["bin_edges1"][:] :: Vector{Float64})
    bins2 = g["bin_edges2"][:] :: Vector{Float64}

    minima = g["minimum"][:, :] :: Array{Float64,2}  # (Γ, ε)
    maxima = g["maximum"][:, :] :: Array{Float64,2}

    samples = g["total_samples"][:] :: Vector{Int64}

    check_histogram(hist_full, samples, basename(HDF5.name(g)))

    conditional = conditional_pdfs(hist_full, hist2, rs, bins1, bins2, :Γ)

    (; hist_full, minima, maxima, samples, bins_Γ = bins1, conditional)
end

@timeit to function load_histograms1D(g, rs, field)
    xs = g["bin_edges"][:] :: Vector{Float64}
    hist = g["hist"][:, :] :: Array{Int64,2}
    samples = g["total_samples"][:] :: Vector{Int}

    check_histogram(hist, samples, basename(HDF5.name(g)))

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

@timeit to function conditional_pdfs(
        hist2D, hist_y_all, rs, xs, ys_all, field::Symbol,
    )
    @assert size(hist2D, 2) == size(hist_y_all, 1) == length(ys_all) - 1

    # Merge the first 4 bins, then the next 4 bins, then the next 8 bins, then
    # the rest...
    mergebins = [4, 4, 8]
    append!(mergebins, length(ys_all) - sum(mergebins) - 1)
    mergetot = cumsum(mergebins)
    @assert sum(mergebins) == mergetot[end] == length(ys_all) - 1

    ys = similar(ys_all, length(mergebins) + 1)
    ys[begin] = ys_all[begin]
    for i in eachindex(mergetot)
        ys[1 + i] = ys_all[1 + mergetot[i]]
    end
    @assert ys[end] == ys_all[end]

    Nx = length(xs) - 1
    Ny = length(ys) - 1
    Nr = size(hist2D, 3)
    hist = zeros(eltype(hist2D), Nx, Ny, Nr)
    hist_y = zeros(eltype(hist2D), Ny, Nr)
    for k ∈ axes(hist2D, 3), j ∈ axes(hist2D, 2)
        jj = searchsortedfirst(mergetot, j)
        hist_y[jj, k] += hist_y_all[j, k]
        for i in axes(hist2D, 1)
            hist[i, jj, k] += hist2D[i, j, k]
        end
    end

    pdfs = zeros(Float64, size(hist))
    for k in axes(pdfs, 3)
        for j in axes(pdfs, 2)
            h = @view hist[:, j, k]
            s = sum(h)
            s_expected = hist_y[j, k]
            # if 1 - s / s_expected > 1e-4
            if s_expected > 0 && !isapprox(s / s_expected, 1; atol = 1e-6)
                @warn(
                    "Some samples are not included in the 2D histogram." *
                    " The range of circulation bins should be enlarged!" *
                    " ($s ≠ $s_expected)"
                )
            end
            s == 0 && continue  # this conditional PDF has no events!
            for i in axes(pdfs, 1)
                dx = xs[i + 1] - xs[i]
                pdfs[i, j, k] = h[i] / (s * dx)
            end
        end
    end
    εs = ys
    @show εs
    (; field,
        bin_edges = xs,
        εs, rs, pdfs,
    )
end

partial_sum(u; dims) = dropdims(sum(u; dims); dims)

function r_indices_to_plot(rs)
    inds = eachindex(rs)
    N = length(inds)
    # subinds = Iterators.flatten((inds[2:2:10], inds[12:1:(N - 8)]))
    # collect(subinds)  # we want an AbstractVector...
    if N > 30
        # inds[12:2:(N - 6)]  # case of 2048^3 data
        inds[14:1:(N - 10)]
    else
        inds[4:1:N-6]
    end
end

function ε_indices_to_plot(εs)
    inds = eachindex(εs)
    N = length(inds) - 1
    n = min(8, N)
    inds[1:1:n]
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

@timeit to function load_stats(g, params)
    rs = g["kernel_size"][:] :: Vector{Float64}
    As = g["kernel_area"][:] :: Vector{Float64}
    @assert As ≈ rs.^2

    using_enstrophy = haskey(g, "FieldMetadata/EnstrophyField")

    let g = open_group(g, "FieldMetadata")
        @assert read(g["CirculationField/divided_by_area"]) ==
            CIRCULATION_DIVIDED_BY_AREA[]

        gname = using_enstrophy ? "EnstrophyField" : "DissipationField"

        @assert read(g["$gname/divided_by_area"]) ==
            !DISSIPATION_MULTIPLIED_BY_AREA[]

        let name = "$gname/inplane (2D)"
            inplane = haskey(g, name) ? Bool(read(g[name])) : false
            println("In-plane dissipation: ", inplane)
        end
    end

    histograms2D = load_histograms2D(g["Histogram2D"], rs)
    histograms_Γ = load_histograms1D(g["HistogramCirculation"], rs, :Γ)
    histograms_ε = load_histograms1D(g["HistogramDissipation"], rs, :ε)

    ν = params.ν

    # Estimate ⟨ε⦒ from smallest loop size.
    ε_mean_est = let H = histograms_ε, j = 1
        mean = @views moment(H.bin_edges, H.pdfs[:, j, 1], Val(1))
        if DISSIPATION_MULTIPLIED_BY_AREA[]
            mean /= As[j]
        end
        mean
    end

    ε_mean = params.DNS_dissipation

    @assert isapprox(ε_mean, ε_mean_est; rtol = 1e-3)

    Lbox = params.Ls[1]
    Δx = Lbox / params.Ns[1]
    η = (ν^3 / ε_mean)^(1/4)
    rs_η = rs ./ η
    As_η = As ./ η^2

    taylor_scale = let E = params.DNS_energy
        sqrt(10ν * E / ε_mean)
    end

    rs_λ = rs ./ taylor_scale

    enstrophy = ε_mean / (2ν)
    Γ_taylor = taylor_scale^2 * sqrt(2enstrophy / 3)

    @show ε_mean, ε_mean_est, enstrophy
    @show η, Lbox / η, Δx / η, taylor_scale / η

    (; rs, As, rs_η, As_η, rs_λ, L = Lbox, η, ν,
        using_enstrophy,
        taylor_scale, enstrophy, Γ_taylor,
        histograms2D,
        histograms_Γ, histograms_ε,
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

moment(f::Function, edges, pdf, ::Val{p}) where {p} = sum(eachindex(pdf)) do i
    dx = edges[i + 1] - edges[i]
    xc = f((edges[i] + edges[i + 1]) / 2)
    xc^p * pdf[i] * dx
end

moment(args...) = moment(identity, args...)

function variance(edges, pdf)
    mean1 = moment(edges, pdf, Val(1))
    mean2 = moment(edges, pdf, Val(2))
    mean2 - mean1^2
end

function normal_pdf(x; μ = 0, σ = 1)
    exp(-0.5 * (x - μ)^2 / σ^2) / (σ * sqrt(2π))
end

function plot_normal_pdf!(ax, xs; kws...)
    ys = normal_pdf.(xs)
    ax.plot(xs, ys; color = "tab:red", ls = :dashed, lw = 1.5, kws...)
end

function write_distances(fname, rs; varname = nothing)
    open(fname, "w") do ff
        if varname !== nothing
            println(ff, "# (1) i  (2) $varname  (3) ($varname)^2")
        end
        writedlm(ff, zip(eachindex(rs), rs, rs.^2))
    end
end

function plot_pdf_Γr!(ax, stats; cmap = plt.cm.viridis_r, write_data = false)
    histograms = stats.histograms_Γ

    hists = histograms.hist
    Γs = histograms.bin_edges

    As = stats.As
    λ = stats.taylor_scale

    rs_λ = stats.rs ./ λ

    ax.set_yscale(:log)
    ax.set_xlim(-20, 20)
    ax.set_xlabel(L"Γ_r / \left< Γ_r^2 \right>^{1/2}")
    ax.set_ylabel("Probability")
    r_indices = r_indices_to_plot(As)
    cmap_norm = mappable_colour(cmap, r_indices)

    if write_data
        write_distances(
            "pdf.rs_lambda.dat", @view(rs_λ[r_indices]); varname = "r / lambda",
        )
    end

    for (i, r) in enumerate(r_indices)
        color = cmap_norm(i)
        A = As[r]
        hist = @view hists[:, r]
        xs = collect(Γs)

        if CIRCULATION_DIVIDED_BY_AREA[]
            # Multiply by area to get actual circulation
            xs .*= A
        end

        pdf = let dx = xs[2] - xs[1]
            hist ./ (histograms.samples[r] * dx)
        end

        xrms = sqrt(variance(xs, pdf))
        pdf .*= xrms
        xs ./= xrms

        xs_centre = @views (xs[1:end-1] .+ xs[2:end]) ./ 2
        r_λ = round(rs_λ[r], digits = 2)

        if write_data
            writedlm("pdf.r$i.dat", zip(xs_centre, pdf))
        end

        ax.plot(
            xs_centre, pdf;
            # marker = ".",
            color, label = latexstring("$r_λ"),
        )
    end

    plot_normal_pdf!(ax, -6:0.1:6)

    ax
end

function plot_condpdfs!(
        ax, stats;
        rind = nothing, εind = nothing, cmap,
        write_data = false,
    )
    histograms = stats.histograms2D
    conditional = histograms.conditional
    using_enstrophy = stats.using_enstrophy

    @assert sum(isnothing, (rind, εind)) == 1 "exactly one must be nothing"

    rs_η = stats.rs_η
    rs_λ = stats.rs_λ
    rs = stats.rs
    Γs = conditional.bin_edges
    εs = conditional.εs
    pdfs = conditional.pdfs

    @assert length(rs) == size(pdfs, 3)

    ε_mean = stats.ε_mean
    η = stats.η
    λ = stats.taylor_scale
    ν = stats.ν
    τ_η = sqrt(ν / ε_mean)

    # PDFs are conditioned by ε, not A * ε.
    @assert !DISSIPATION_MULTIPLIED_BY_AREA[]

    ax.set_yscale(:log)
    # ax.set_xlim(-6, 6)
    # ax.set_ylim(1e-6, 1)

    if using_enstrophy
        ax.set_xlabel(L"Γ_{\!r} / {\left< Γ_{\!r}^2 \, | \, Ω_r \right>}^{1/2}")
        ax.set_ylabel(L"P(Γ_{\!r} \, | \, Ω_r, r)")
        ε_norm = 1 / τ_η^2
    else
        ax.set_xlabel(L"Γ_{\!r} / {\left< Γ_{\!r}^2 \, | \, ε_r \right>}^{1/2}")
        ax.set_ylabel(L"P(Γ_{\!r} \, | \, ε_r, r)")
        ε_norm = ε_mean
    end

    if !isnothing(rind)
        pdf_indices = ε_indices_to_plot(εs)
        text = let r = round(rs_λ[rind], digits = 2)
            latexstring("r / λ = $r")
        end
    elseif !isnothing(εind)
        pdf_indices = r_indices_to_plot(rs)
        ε_range = (εs[εind], εs[εind + 1])
        text = let ε = round.(ε_range ./ ε_norm, digits = 2)
            latexstring("ε_r / \\left< ε \\right> ∈ $ε")
        end
    end

    if write_data
        write_distances(
            "condpdf.rs_lambda.dat", @view(rs_λ[pdf_indices]);
            varname = "r / lambda",
        )
        if !isnothing(εind)
            writedlm("condpdf.eps_range.dat", ε_range ./ ε_norm)
        end
    end

    ax.text(
        0.02, 0.98, text;
        ha = :left, va = :top, transform = ax.transAxes,
    )

    Ncurves = length(pdf_indices)
    cmap_norm = mappable_colour(cmap, 0, 1)

    for (n, j) in enumerate(pdf_indices)
        if !isnothing(rind)
            ri = rind
            εi = j
            pdf = pdfs[:, j, rind]
            label = ntuple(2) do δ
                round(εs[j - 1 + δ] / ε_norm, digits = 2)
            end
        elseif !isnothing(εind)
            ri = j
            εi = εind
            pdf = pdfs[:, εind, j]
            label = round(rs_λ[j], digits = 2)
        end

        xs = collect(Γs)

        if CIRCULATION_DIVIDED_BY_AREA[]
            A = rs[ri]^2
            xs .*= A
            pdf ./= A
        end

        color = cmap_norm((n - 1) / (Ncurves - 1))

        xrms = sqrt(variance(xs, pdf))
        pdf .*= xrms
        xs ./= xrms

        xs_centre = @views (xs[1:end-1] .+ xs[2:end]) ./ 2

        if write_data
            writedlm("condpdf.r$n.dat", zip(xs_centre, pdf))
        end

        ax.plot(
            xs_centre, pdf;
            # marker = ".",
            color, label,
        )
    end

    plot_normal_pdf!(ax, -6:0.1:6)

    ax
end

function plot_dissipation_K62!(ax, stats)
    histograms = stats.histograms_ε
    using_enstrophy = stats.using_enstrophy

    pdfs = histograms.pdfs
    As = stats.As
    εs = histograms.bin_edges
    ε_mean = stats.ε_mean

    εr_log_mean = map(axes(pdfs, 2)) do j
        pdf = @view pdfs[:, j]
        y = sum(axes(pdfs, 1)) do i
            xc = (εs[i] + εs[i + 1]) / 2
            dx = εs[i + 1] - εs[i]
            log(xc) * pdf[i] * dx
        end
        if DISSIPATION_MULTIPLIED_BY_AREA[]
            # In this case, we computed ⟨ log(A * ε_r) ⟩ = log(A) + ⟨ log(ε_r) ⟩.
            y -= log(As[j])
        end
        y -= log(ε_mean)
    end

    λ = stats.taylor_scale
    rs = stats.rs ./ λ

    ax.plot(rs, εr_log_mean; marker = :o)

    # Linear regression fit: y = A + μ * x
    if using_enstrophy
        inds_fit = 0.1 .≤ rs .≤ 1.0
    else
        inds_fit = 0.2 .≤ rs .≤ 2.0
    end
    rs_fit = rs[inds_fit]
    A, μ = let
        y = εr_log_mean[inds_fit]
        x = hcat(ones(length(y)), log.(rs_fit))
        b = (x'x) \ (x'y)
        b[1], b[2]
    end
    yfit = A .+ μ * log.(rs)
    @show A, μ

    ax.axvspan(rs_fit[1], rs_fit[end]; color = "tab:green", alpha = 0.2)

    let inds = 1:length(rs)-6
        @views ax.plot(rs[inds], yfit[inds]; ls = :dashed, c = :black)
    end

    μ_round = round(μ, digits = 3)
    ax.text(
        0.04, 0.96,
        """
        Fit: \$A + μ \\, \\log r\$
        with \$μ = $μ_round\$
        """
        ;
        transform = ax.transAxes,
        va = "top", ha = "left",
        fontsize = :large,
        bbox = Dict(:facecolor => :white, :alpha => 1),
    )

    ax.set_xscale(:log)
    ax.set_xlabel(L"r / λ")
    ax.set_ylabel(L"\left< \log (ε_r / ε_0) \right>")

    ax
end

function plot_pdf_εr!(ax, stats; logpdf = false, cmap = plt.cm.viridis_r)
    histograms = stats.histograms_ε

    hists = histograms.hist
    εs = histograms.bin_edges

    As = stats.As
    λ = stats.taylor_scale
    ε_mean = stats.ε_mean

    if logpdf
        ax.set_xlabel(L"\left( \log ε_r - \left\langle \log ε_r \right\rangle \right) / σ(\log ε_r)")
    end

    ax.set_yscale(:log)
    r_indices = r_indices_to_plot(As)
    cmap_norm = mappable_colour(cmap, r_indices)

    for (i, r) in enumerate(r_indices)
        color = cmap_norm(i)
        A = As[r]
        hist = @view hists[:, r]

        xs = collect(εs)

        if DISSIPATION_MULTIPLIED_BY_AREA[]
            # Divide by area to get actual dissipation
            xs ./= A
        end

        pdf = map(eachindex(hist)) do i
            dx = xs[i + 1] - xs[i]
            hist[i] / (histograms.samples[r] * dx)
        end

        r_λ = round(stats.rs[r] / λ, digits = 2)

        xs_centre = @views (xs[1:end-1] .+ xs[2:end]) ./ 2

        if !logpdf
            xs_centre ./= ε_mean
            pdf .*= ε_mean
            ax.plot(
                xs_centre, pdf;
                color,
                # marker = ".",
                label = latexstring("$r_λ"),
            )
            ax.set_xlim(0, 5)
            continue
        end

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

        ax.plot(
            xs_centre_log, pdf_log;
            color,
            # marker = ".",
            label = latexstring("$r_λ"),
        )
    end

    if logpdf
        plot_normal_pdf!(ax, -5:0.1:5)
        ax.set_xlim(-5, 5)
    end

    ax
end

function plot_pdfs(stats; εind = 2, rind = 8, write_data = false)
    fig, axes = plt.subplots(2, 2; figsize = 0.9 .* (8, 6), sharey = true)
    using_enstrophy = stats.using_enstrophy
    let ax = axes[1, 1]
        plot_pdf_Γr!(ax, stats; cmap = plt.cm.viridis_r, write_data)
        ax.set_xlim(-10, 10)
        ax.set_ylim(1e-6, 1)
        ax.legend(
            fontsize = "x-small", title = L"r / λ",
            frameon = false, loc = "upper right",
        )
    end
    let ax = axes[2, 2]
        plot_pdf_εr!(ax, stats; logpdf = true, cmap = plt.cm.viridis_r)
        ax.set_ylim(1e-6, 1)
        ax.legend(
            title = L"r / λ", fontsize = "x-small", frameon = false,
            loc = "lower center",
        )
    end
    let ax = axes[2, 1]
        plot_condpdfs!(ax, stats; rind, cmap = plt.cm.cividis_r)
        ax.set_xlim(-12, 12)
        ax.set_ylim(1e-6, 1)
        legtitle = if using_enstrophy
            L"Ω_r τ_η^2"
        else
            L"ε_r / \langle ε \rangle"
        end
        ax.legend(
            fontsize = "small", frameon = false,
            loc = "upper right",
            title = legtitle,
        )
    end
    let ax = axes[1, 2]
        plot_condpdfs!(ax, stats; εind, cmap = plt.cm.viridis_r, write_data)
        ax.set_xlim(-10, 10)
        ax.set_ylim(1e-6, 1)
        ax.set_ylabel("")
        ax.legend(
            frameon = false, loc = "upper right", title = L"r / λ",
            fontsize = "x-small",
        )
    end
    fig
end

dissipation_bin(xs, i, xmean) = (xs[i], xs[i + 1]) ./ xmean

function dissipation_bin_text(args...)
    x_bin = dissipation_bin(args...)
    x_range = round.(x_bin, digits = 2)
    string(x_range)
end

dissipation_bin_text(::Nothing, etc...) = "(0, ∞)"

cmap_dissipation_exponents(::Nothing, etc...) = nothing
cmap_dissipation_exponents(εs, imax) =
    mappable_colour(plt.cm.cividis_r, εs[1], εs[imax])

function plot_moments!(
        axs, stats, moments;
        ess = false, ε_ind = 1,
    )
    rs = moments.rs
    rs_norm = rs ./ moments.rnorm
    rlabel = moments.rlabel
    rlabel_norm = moments.rlabel_norm
    Γ_norm = stats.Γ_taylor
    # Γ_norm = stats.ν

    ps = moments.ps
    ys = moments.moments
    Nr, Np, Nε = size(ys)
    ε_mean = stats.ε_mean

    εs = moments.εs
    εind = min(Nε, ε_ind)
    ε_text = dissipation_bin_text(εs, εind, ε_mean)
    ε_text_full = latexstring("ε ∈ $ε_text")
    ε_ind_max = min(4, Nε)

    cmap_ε = cmap_dissipation_exponents(εs, ε_ind_max)

    slopes = moments.slopes
    slopes_fitted = moments.slopes_fitted

    cmap = mappable_colour(plt.cm.viridis_r, ps)

    let ax = axs[1]
        if ess
            ax.set_xlabel(L"\left< Γ_r^2 \right> / Γ_{\! \mathrm{T}}^2")
        else
            ax.set_xlabel(rlabel_norm)
        end
        ax.set_ylabel(
            latexstring(
                "\\langle Γ_{\\!r}^p \\, | \\, $rlabel \\rangle " *
                "/ Γ_{\\! \\mathrm{T}}^p"
            )
        )
        ax.set_xscale(:log)
        ax.set_yscale(:log)
        ax.set_xlim(5e-2, 75)
        ax.set_ylim(1e-13, 1e17)
        ax.axvspan(moments.fit_region_λ...; alpha = 0.2, color = "tab:blue")
    end
    let ax = axs[2]
        ax.set_xlabel(rlabel_norm)
        dlog = raw"\mathrm{d} \, \log"
        ax.set_ylabel("Local slopes")
        ax.set_xscale(:log)
        ax.set_xlim(5e-2, 75)
        ax.set_ylim(-0.2, 14.2)
        ax.axvspan(moments.fit_region_λ...; alpha = 0.2, color = "tab:blue")
    end

    for (j, p) in enumerate(ps)
        color = cmap(p)
        let ax = axs[1]
            k = εind
            y = @view(ys[:, j, k]) ./ Γ_norm^p
            x = if ess
                @assert ps[2] == 2
                @view(ys[:, 2, k]) ./ Γ_norm^2
            else
                rs_norm
            end
            ax.plot(x, y; color, marker = :., label = p)
            ax.text(
                0.5, 0.98, ε_text_full;
                ha = :center, va = :top, transform = ax.transAxes,
            )
        end
        let ax = axs[2]
            x = @views (rs_norm[1:end-1] .+ rs_norm[2:end]) ./ 2
            k = εind
            if ess
                @assert ps[2] == 2
                y = @views slopes[:, j, k] ./ slopes[:, 2, k] .* 4 * ps[2] / 3
            else
                y = @view slopes[:, j, k]
            end
            ax.text(
                0.98, 0.98, ε_text_full;
                ha = :right, va = :top, transform = ax.transAxes,
            )
            ax.axhline(4p/3; color, ls = :dotted)
            ax.plot(x, y; color, marker = :.)
            # ax.axhline(slopes_fitted[j]; color, ls = :solid)
        end
    end

    let ax = axs[3]
        slopes_k41 = 4/3 .* ps
        ax.plot(ps, slopes_k41; color = :black, ls = :dotted)
        ax.set_ylabel(L"λ_p")
        for k in 1:ε_ind_max
            if isnothing(εs)
                color = :black
            else
                color = cmap_ε(εs[k])
            end
            ax.plot(
                ps, @view(slopes_fitted[:, k]);
                marker = :o, color,
                label = latexstring(dissipation_bin_text(εs, k, ε_mean)),
            )
        end
        ax.set_xlabel(L"p")
        ax.legend(frameon = false, title = latexstring("ε / $TEX_MEAN_ε"))
        ax.set_xlim((extrema(ps) .+ (-1, 1))...)
        ax.set_ylim(0, 14)
    end

    let ax = axs[1]
        ax.legend(title = L"p", loc = "upper left", frameon = false, ncol = 2)
    end

    nothing
end

function plot_circulation_scaling_exponents!(
        ax, stats, moments;
        plot_selfsimilar = false,
        compensate = false,
        write_data = false,
    )
    ps = moments.ps

    slopes_fitted = moments.slopes_fitted
    εs = moments.εs
    ε_mean = stats.ε_mean

    conditioned = εs !== nothing

    Np, Nε = size(slopes_fitted)
    ε_ind_max = min(10, Nε)

    if conditioned
        open("exponents.eps_ranges.dat", "w") do ff
            for k in 1:ε_ind_max
                a, b = round.(dissipation_bin(εs, k, ε_mean), digits = 2)
                println(ff, "$a\t$b")
            end
        end
    end

    ax.set_xlabel(L"p")
    ax.set_xlim((extrema(ps) .+ (-1, 1))...)

    if compensate
        ax.set_ylabel(L"λ_p^{\mathrm{K41}} - λ_p")
    else
        ax.set_ylim(0, 14)
        ax.set_ylabel(L"λ_p")
    end

    cmap_ε = cmap_dissipation_exponents(εs, ε_ind_max)

    for k in 1:ε_ind_max
        if conditioned
            color = cmap_ε(εs[k])
            marker = :o
            fname = "exponents_cond_eps$k.dat"
        else
            color = :black
            marker = :x
            fname = "exponents_base.dat"
        end
        y = slopes_fitted[:, k]
        if write_data
            open(fname, "w") do ff
                println(ff, "# (1) p  (2) lambda_p")
                writedlm(ff, zip(ps, y))
            end
        end
        if compensate
            @. y = 4/3 * ps - y
        end
        ax.plot(
            ps, y;
            marker, color, markersize = 4,
            label = latexstring(dissipation_bin_text(εs, k, ε_mean)),
        )
    end

    if plot_selfsimilar
        let style = (color = "tab:red", lw = 1, ls = :dashed)
            if compensate
                ax.axhline(0; style...)
            else
                slopes_k41 = 4/3 .* ps
                ax.plot(ps, slopes_k41; style...)
            end
        end
    end

    ax
end

function fit_indices(rs::AbstractVector, λ, r_λ_estimates::NTuple{2})
    r_indices_fit = map(a -> searchsortedlast(rs, a * λ), r_λ_estimates)
    if r_indices_fit[1] == 0
        r_indices_fit = r_indices_fit .+ 1
    end
    r_indices_fit
end

@timeit to function compute_moments(hists, stats; ps = 1:10)
    using_enstrophy = stats.using_enstrophy
    xs = hists.bin_edges

    # 3 dimensions: [Γ_r, ε_r, r]
    # If there's no conditioning by ε, then the second dimension has size 1.
    pdfs = let p = hists.pdfs
        ndims(p) == 2 ? reshape(p, size(p, 1), 1, size(p, 2)) : p
    end

    rs = collect(hists.rs)

    multiply_by_area = hists.field == :Γ && CIRCULATION_DIVIDED_BY_AREA[]

    As = stats.As
    λ = stats.taylor_scale

    # Estimate of r/λ region where slopes are fitted
    fit_region_λ_est = (1, 7)
    rlabel = "r"
    rlabel_norm = L"r / λ"

    Nx, Nε, Nr = size(pdfs)
    @assert length(xs) == Nx + 1
    Np = length(ps)

    @assert length(rs) == Nr
    dlog_r = diff(log.(rs))

    moments = zeros(Nr, Np, Nε)
    slopes = zeros(Nr - 1, Np, Nε)
    slopes_fitted = similar(slopes, Np, Nε)

    r_indices_fit = fit_indices(rs, λ, fit_region_λ_est)
    fit_region_λ = getindex.(Ref(rs), r_indices_fit) ./ λ  # actual fit region
    @show fit_region_λ

    for l in axes(moments, 3)  # for every ε
        for (j, p) in enumerate(ps)
            ys = @view moments[:, j, l]
            sl = @view slopes[:, j, l]
            slopes_fitted[j, l] = _compute_moments!(
                Val(p), ys, sl, @view(pdfs[:, l, :]), xs, As;
                multiply_by_area, dlog_r, r_indices_fit,
            )
        end
    end

    εs = get(hists, :εs, nothing)
    @assert isnothing(εs) == (Nε == 1)

    (;
        field = hists.field,
        ps, rs, εs,
        rnorm = λ,
        rlabel,
        rlabel_norm,
        moments,
        slopes,
        slopes_fitted,
        fit_region_λ,
    )
end

@timeit to function _compute_all_moments!(
        moments, slopes, pdf, pmax,
    )
    # TODO
end

@timeit to function _compute_moments!(
        ::Val{p}, ys::AbstractVector, slopes::AbstractVector,
        pdfs::AbstractMatrix, xs, As;
        multiply_by_area, dlog_r, r_indices_fit,
    ) where {p}
    for i ∈ eachindex(ys)
        pdf = @view pdfs[:, i]
        ys[i] = moment(abs, xs, pdf, Val(p))
        if multiply_by_area
            ys[i] *= As[i]^p
        end
    end

    dlog_y = diff(log.(ys))
    slopes[:] .= dlog_y ./ dlog_r

    ra, rb = r_indices_fit
    slope_fit = sum(n -> slopes[n], ra:rb) / (rb - ra + 1)

    slope_fit
end

function plot_circulation_flatness!(ax, stats, moments)
    εs = moments.εs
    ε_mean = stats.ε_mean
    ys = moments.moments
    ps = moments.ps
    rs = moments.rs
    rs_norm = rs ./ moments.rnorm

    @assert ps[2] == 2 && ps[4] == 4

    Nr, Np, Nε = size(ys)
    εs = moments.εs
    ε_ind_max = min(4, Nε)

    if !isnothing(εs)
        cmap_ε = mappable_colour(plt.cm.cividis_r, εs[1], εs[ε_ind_max])
    end

    # xs = @views (rs_norm[1:end-1] .+ rs_norm[2:end]) ./ 2
    xs = rs_norm

    for k = 1:ε_ind_max
        if isnothing(εs)
            color = :black
        else
            color = cmap_ε(εs[k])
        end
        y = @views ys[:, 4, k] ./ ys[:, 2, k].^2
        ax.plot(
            xs, y;
            color,
            label = latexstring(dissipation_bin_text(εs, k, ε_mean)),
        )
    end

    ax.set_xscale(:log)
    ax.set_xlabel(L"r / λ")
    ax.set_ylabel("Flatness")

    ax
end

function plot_circulation_moments(stats, moments; kws...)
    fig, axes = plt.subplots(1, 3; figsize = (10, 4))
    plot_moments!(axes, stats, moments; kws...)
    # fig.savefig("moments.svg", dpi=300)
    nothing
end

function plot_circulation_scaling_exponents(
        stats, moments_orig, moments_cond;
        compensate = false, write_data = false,
    )
    fig, ax = plt.subplots(figsize = (4, 3) .* 0.8)
    plot_circulation_scaling_exponents!(
        ax, stats, moments_cond;
        plot_selfsimilar = false, compensate, write_data,
    )
    plot_circulation_scaling_exponents!(
        ax, stats, moments_orig;
        plot_selfsimilar = true, compensate, write_data,
    )
    ax.legend(frameon = false, title = latexstring("ε_r / $TEX_MEAN_ε"), fontsize = :small)
    # fig.savefig("exponents.png"; dpi = 300)
    fig
end

function plot_circulation_flatness(stats, moments_orig, moments_cond)
    fig, ax = plt.subplots()
    plot_circulation_flatness!(ax, stats, moments_orig)
    plot_circulation_flatness!(ax, stats, moments_cond)
    ax.axhline(3; color = "tab:green", ls = :dashed)
    ax.axhline(249 / 80; color = "tab:red", ls = :dashed)
    ax.legend(frameon = false, title = latexstring("ε_r / $TEX_MEAN_ε"))
    # fig.savefig("flatness.png"; dpi = 300)
    fig
end

function plot_conditional_moments(stats, moments_cond; kws...)
    hists = stats.histograms2D
    fig, axes = plt.subplots(1, 3; figsize = (10, 4))
    plot_moments!(axes, stats, moments_cond; kws...)
    # fig.savefig("moments_conditioned.svg", dpi=300)
    nothing
end

function plot_dissipation_stats(stats)
    fig, axes = plt.subplots(1, 3; figsize = (10, 4))
    let ax = axes[1]
        plot_pdf_εr!(ax, stats; logpdf = true)
        ax.set_ylim(1e-6, 1)
        ax.legend(
            title = L"r / λ", fontsize = :small, frameon = false,
            loc = "lower center",
        )
    end
    let ax = axes[2]
        plot_dissipation_K62!(ax, stats)
    end
    nothing
end

function analyse()
    plt.close(:all)
    reset_timer!(to)

    params_NS_1024 = (;
        DNS_timestep = 70_000,
        DNS_energy = 0.0521297,  # from DNS, timestep 70k
        DNS_dissipation = 0.00376189,
        ν = 5e-5,
    )

    params_NS_2048 = (;
        DNS_timestep = 32_000,
        DNS_energy = 3.6588,
        DNS_dissipation = 1.2523,
        ν = 2.8e-4,
    )

    params = (;
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/circulation_dissA.logbins.v2.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/circulation_diss2D.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/circulation_enstrophy2D.v2.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/test.v3-dissipation3d.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/v3-enstrophy2D.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/v3-dissipation2d.h5",

        # NS = params_NS_1024,
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/v3-dissipation3d.h5",

        results_file = "../../results/data_NS/NS2048/circulation_cond.h5",
        NS = params_NS_2048,
    )

    data = h5open(io -> load_data(io, params.NS), params.results_file, "r")
    stats = data.stats

    moments = compute_moments(stats.histograms_Γ, stats)
    moments_cond = compute_moments(stats.histograms2D.conditional, stats)

    println(to)

    stats, moments, moments_cond
end

function make_plots(stats, moments, moments_cond)
    ess = false
    write_data = false

    fig = plot_pdfs(stats; rind = 12, εind = 2, write_data)
    # fig.savefig("pdfs.png")

    fig = plot_circulation_scaling_exponents(
        stats, moments, moments_cond; write_data)
    # fig.savefig("exponents.png")

    return
    # plot_dissipation_stats(stats)
    # plot_circulation_moments(stats, moments; ess)
    # plot_conditional_moments(stats, moments_cond; ess, ε_ind = 2)
    # plot_conditional_moments(stats, moments_cond; ess, ε_ind = 2)
    plot_conditional_moments(stats, moments_cond; ess, ε_ind = 3)
    # plot_conditional_moments(stats, moments_cond; ess, ε_ind = 9)
    # plot_circulation_flatness(stats, moments, moments_cond)

    nothing
end

data = analyse();

make_plots(data...);
