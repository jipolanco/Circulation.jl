import PyPlot: plt
const mpl = plt.matplotlib
using LaTeXStrings

using DelimitedFiles
using HDF5

# Did we compute histograms with circulation divided by area?
const CIRCULATION_DIVIDED_BY_AREA = Ref(true)

# Did we compute histograms with dissipation multiplied by area?
const DISSIPATION_MULTIPLIED_BY_AREA = Ref(false)

const TEX_MEAN_ε = raw"\left< ε \right>"

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

function load_histograms2D(g, rs)
    merged_scales = Bool(read(g["merged_scales"]))
    hist_full = g["hist"][:, :, :] :: Array{Int64,3}  # (Γ, εA, r)

    bins1 = to_steprange(g["bin_edges1"][:] :: Vector{Float64})
    bins2 = g["bin_edges2"][:] :: Vector{Float64}

    minima = g["minimum"][:, :] :: Array{Float64,2}  # (Γ, ε)
    maxima = g["maximum"][:, :] :: Array{Float64,2}

    samples = g["total_samples"][:] :: Vector{Int64}

    # Check that (almost) all samples are in the PDFs
    # @show sum(hist_full) / samples
    # @assert isapprox(sum(hist_full), samples; rtol=1e-3)

    conditional = conditional_pdfs(hist_full, rs, bins1, bins2, :Γ)

    (; hist_full, minima, maxima, samples, bins_Γ = bins1, conditional)
end

function load_histograms1D(g, rs, field)
    xs = g["bin_edges"][:] :: Vector{Float64}
    hist = g["hist"][:, :] :: Array{Int64,2}
    samples = g["total_samples"][:] :: Vector{Int}
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

function conditional_pdfs(hist2D, rs, xs, ys_all, field::Symbol)
    mergebins = 2
    @assert (length(ys_all) - 1) % mergebins == 0
    ys = [ys_all[i] for i = 1:mergebins:lastindex(ys_all)]
    @assert ys[end] == ys_all[end]
    # ys = ys_all
    Nx = length(xs) - 1
    Ny = length(ys) - 1
    Nr = size(hist2D, 3)
    hist = zeros(eltype(hist2D), Nx, Ny, Nr)
    for k ∈ axes(hist2D, 3), j ∈ axes(hist2D, 2)
        jj = div(j, mergebins, RoundUp)
        for i in axes(hist2D, 1)
            hist[i, jj, k] += hist2D[i, j, k]
        end
    end
    pdfs = similar(hist, Float64)
    for k in axes(pdfs, 3)
        for j in axes(pdfs, 2)
            h = @view hist[:, j, k]
            s = sum(h)
            for i in axes(pdfs, 1)
                dx = xs[i + 1] - xs[i]
                pdfs[i, j, k] = h[i] / (s * dx)
            end
        end
    end
    εs = ys
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
    inds[4:1:N-6]
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
    # ε_mean = let H = histograms_ε, j = 1
    #     mean = @views moment(H.bin_edges, H.pdfs[:, j, 1], Val(1))
    #     if DISSIPATION_MULTIPLIED_BY_AREA[]
    #         mean /= As[j]
    #     end
    #     mean
    # end

    ε_mean = params.DNS_dissipation

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

    @show ε_mean, enstrophy
    @show η, Lbox / η, Δx / η, taylor_scale / η

    (; rs, As, rs_η, As_η, rs_λ, L = Lbox, η, ν,
        using_enstrophy,
        taylor_scale, enstrophy, Γ_taylor,
        histograms2D,
        histograms_Γ, histograms_ε,
        ε_mean,
    )
end

function load_simulation_params(g)
    (
        Ns = Tuple(g["dims"][:]) :: NTuple{3,Int},
        Ls = Tuple(g["L"][:]) :: NTuple{3,Float64},
        # η = 0.00245,  # from DNS
        DNS_timestep = 70_000,
        DNS_energy = 0.0521297,  # from DNS, timestep 70k
        DNS_dissipation = 0.00376189,
        ν = 5e-5,
    )
end

function load_data(ff::HDF5.File)
    params = load_simulation_params(ff["SimParams"])
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

function plot_pdf_Γr!(ax, stats)
    histograms = stats.histograms_Γ

    hists = histograms.hist
    Γs = histograms.bin_edges

    As = stats.As
    λ = stats.taylor_scale

    ax.set_yscale(:log)
    ax.set_xlim(-20, 20)
    ax.set_xlabel(L"Γ_r / \left< Γ_r^2 \right>^{1/2}")
    ax.set_ylabel("Probability")
    r_indices = r_indices_to_plot(As)
    cmap = mappable_colour(plt.cm.viridis_r, r_indices)

    for (i, r) in enumerate(r_indices)
        color = cmap(i)
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
        # r_η = round(stats.rs_η[r], digits=0)
        r_λ = round(stats.rs[r] / λ, digits = 2)
        ax.plot(
            xs_centre, pdf;
            marker = ".", color, label = latexstring("$r_λ"),
        )
    end
    ax
end

function plot_condpdfs!(ax, stats; rind = nothing, εind = nothing)
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
        pdf_indices = eachindex(εs)[1:end-5]
        text = let r = round(rs_λ[rind], digits = 2)
            latexstring("r / λ = $r")
        end
    elseif !isnothing(εind)
        pdf_indices = r_indices_to_plot(rs)
        ε_range = (εs[εind], εs[εind + 1])
        text = let ε = round.(ε_range ./ ε_norm, digits = 2)
            latexstring("ε / \\left< ε \\right> ∈ $ε")
        end
    end

    ax.text(
        0.98, 0.98, text;
        ha = :right, va = :top, transform = ax.transAxes,
    )

    Ncurves = length(pdf_indices)
    cmap = mappable_colour(plt.cm.cividis_r, 0, 1)

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

        color = cmap((n - 1) / (Ncurves - 1))

        xrms = sqrt(variance(xs, pdf))
        pdf .*= xrms
        xs ./= xrms

        xs_centre = @views (xs[1:end-1] .+ xs[2:end]) ./ 2
        ax.plot(
            xs_centre, pdf;
            marker = ".", color, label,
        )
    end

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

function plot_pdf_εr!(ax, stats; logpdf = false)
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
    cmap = mappable_colour(plt.cm.viridis_r, r_indices)

    for (i, r) in enumerate(r_indices)
        color = cmap(i)
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
                marker = ".", label = latexstring("$r_λ"),
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
            marker = ".", label = latexstring("$r_λ"),
        )
    end

    if logpdf
        let xs = -5:0.1:5
            pdf = normal_pdf.(xs)
            ax.plot(xs, pdf; lw = 1.5, color = "tab:orange", ls = :dashed)
        end

        ax.set_xlim(-5, 5)
    end

    ax
end

function plot_pdfs(stats)
    fig, axes = plt.subplots(2, 2; figsize = (8, 6))
    using_enstrophy = stats.using_enstrophy
    let ax = axes[1, 1]
        plot_pdf_Γr!(ax, stats)
        ax.set_xlim(-12, 12)
        ax.set_ylim(1e-6, 1)
        ax.legend(
            fontsize = :small, title = L"r / λ",
            frameon = false, loc = "upper right",
        )
    end
    let ax = axes[1, 2]
        plot_pdf_εr!(ax, stats; logpdf = true)
        ax.set_ylim(1e-6, 1)
        ax.legend(
            title = L"r / λ", fontsize = :small, frameon = false,
            loc = "lower center",
        )
    end
    let ax = axes[2, 1]
        plot_condpdfs!(ax, stats; rind = 8)
        ax.set_xlim(-12, 12)
        ax.set_ylim(1e-6, 1)
        legtitle = if using_enstrophy
            L"Ω_r τ_η^2"
        else
            L"ε_r / \langle ε \rangle"
        end
        ax.legend(
            fontsize = "small", frameon = false,
            loc = "upper left",
            title = legtitle,
        )
    end
    let ax = axes[2, 2]
        plot_condpdfs!(ax, stats; εind = 3)
        ax.set_xlim(-12, 12)
        ax.set_ylim(1e-6, 1)
        ax.legend(
            frameon = false, loc = "upper left", title = L"r / λ",
            fontsize = :small,
        )
    end
    # fig.savefig("pdfs.svg", dpi=300)
end

function dissipation_bin_text(xs, i, xmean)
    x_range = round.((xs[i], xs[i + 1]) ./ xmean, digits = 1)
    string(x_range)
end

dissipation_bin_text(::Nothing, etc...) = "(0, ∞)"

function plot_moments!(axs, stats, moments; ess = false)
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
    εind = min(Nε, 2)
    ε_text = dissipation_bin_text(εs, εind, ε_mean)
    ε_text_full = latexstring("ε ∈ $ε_text")
    ε_ind_max = min(4, Nε)

    if !isnothing(εs)
        cmap_ε = mappable_colour(plt.cm.plasma, εs[1], εs[ε_ind_max])
    end

    slopes = moments.slopes
    slopes_fitted = moments.slopes_fitted

    cmap = mappable_colour(plt.cm.viridis_r, ps)

    let ax = axs[1]
        ax.set_xlabel(rlabel_norm)
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
            ax.plot(rs_norm, y; color, marker = :., label = p)
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
    )
    ps = moments.ps

    slopes_fitted = moments.slopes_fitted
    εs = moments.εs
    ε_mean = stats.ε_mean

    Np, Nε = size(slopes_fitted)
    ε_ind_max = min(4, Nε)

    ax.set_xlabel(L"p")
    ax.set_ylabel(L"λ_p")
    ax.set_xlim((extrema(ps) .+ (-1, 1))...)
    ax.set_ylim(0, 14)

    if !isnothing(εs)
        cmap_ε = mappable_colour(plt.cm.plasma, εs[1], εs[ε_ind_max])
    end

    if plot_selfsimilar
        slopes_k41 = 4/3 .* ps
        ax.plot(ps, slopes_k41; color = :black, ls = :dotted)
    end

    for k in 1:ε_ind_max
        if isnothing(εs)
            color = :black
        else
            color = cmap_ε(εs[k])
        end
        ax.plot(
            ps, @view(slopes_fitted[:, k]);
            marker = :o, color, markersize = 3,
            label = latexstring(dissipation_bin_text(εs, k, ε_mean)),
        )
    end

    ax
end

function compute_moments(hists, stats; ps = 1:10)
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

    r_indices_fit = map(a -> searchsortedlast(rs, a * λ), fit_region_λ_est)
    if r_indices_fit[1] == 0
        r_indices_fit = r_indices_fit .+ 1
    end

    fit_region_λ = getindex.(Ref(rs), r_indices_fit) ./ λ  # actual fit region
    @show fit_region_λ

    for l in axes(moments, 3)  # for every ε
        for (j, p) in enumerate(ps)
            ys = @view moments[:, j, l]

            for i ∈ 1:Nr
                pdf = @view pdfs[:, l, i]
                ys[i] = moment(abs, xs, pdf, Val(p))
                if multiply_by_area
                    ys[i] *= As[i]^p
                end
            end

            dlog_y = diff(log.(ys))
            sl = @view slopes[:, j, l]
            sl[:] .= dlog_y ./ dlog_r

            ra, rb = r_indices_fit
            slopes_fitted[j, l] = sum(n -> sl[n], ra:rb) / (rb - ra + 1)
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
        cmap_ε = mappable_colour(plt.cm.plasma, εs[1], εs[ε_ind_max])
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

function plot_circulation_moments(stats; kws...)
    hists = stats.histograms_Γ
    moments = compute_moments(hists, stats)
    fig, axes = plt.subplots(1, 3; figsize = (10, 4))
    plot_moments!(axes, stats, moments; kws...)
    # fig.savefig("moments.svg", dpi=300)
    nothing
end

function plot_circulation_scaling_exponents(stats)
    moments_orig = compute_moments(stats.histograms_Γ, stats)
    moments_cond = compute_moments(stats.histograms2D.conditional, stats)
    fig, ax = plt.subplots()
    plot_circulation_scaling_exponents!(
        ax, stats, moments_orig; plot_selfsimilar = true,
    )
    plot_circulation_scaling_exponents!(
        ax, stats, moments_cond; plot_selfsimilar = false,
    )
    ax.legend(frameon = false, title = latexstring("ε_r / $TEX_MEAN_ε"))
    # fig.savefig("exponents.png"; dpi = 300)
    fig
end

function plot_circulation_flatness(stats)
    moments_orig = compute_moments(stats.histograms_Γ, stats)
    moments_cond = compute_moments(stats.histograms2D.conditional, stats)
    fig, ax = plt.subplots()
    plot_circulation_flatness!(ax, stats, moments_orig)
    plot_circulation_flatness!(ax, stats, moments_cond)
    ax.axhline(3; color = "tab:green", ls = :dashed)
    ax.axhline(249 / 80; color = "tab:red", ls = :dashed)
    ax.legend(frameon = false, title = latexstring("ε_r / $TEX_MEAN_ε"))
    # fig.savefig("flatness.png"; dpi = 300)
    fig
end

function plot_conditional_moments(stats; kws...)
    hists = stats.histograms2D
    moments = compute_moments(hists.conditional, stats)
    fig, axes = plt.subplots(1, 3; figsize = (10, 4))
    plot_moments!(axes, stats, moments; kws...)
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

function main()
    params = (;
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/circulation_dissA.logbins.v2.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/circulation_diss2D.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/circulation_enstrophy2D.v2.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/test.v3-dissipation3d.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/v3-enstrophy2D.h5",
        results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/v3-dissipation3d.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/v3-dissipation2d.h5",
    )
    ess = false
    data = h5open(load_data, params.results_file, "r")
    stats = data.stats
    # plot_pdfs(stats)
    # plot_dissipation_stats(stats)
    # plot_circulation_moments(stats; ess)
    # plot_conditional_moments(stats; ess)
    plot_circulation_scaling_exponents(stats)
    plot_circulation_flatness(stats)
end

main()
