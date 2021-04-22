using PyPlot: plt
const mpl = plt.matplotlib
using LaTeXStrings

using DelimitedFiles
using HDF5

# Did we compute histograms with circulation divided by area?
const CIRCULATION_DIVIDED_BY_AREA = Ref(false)

# Did we compute histograms with dissipation multiplied by area?
const DISSIPATION_MULTIPLIED_BY_AREA = Ref(true)

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

function load_histograms2D(g)
    merged_scales = Bool(read(g["merged_scales"]))
    hist_base = g["hist"][:, :, :] :: Array{Int64,3}  # (Γ, εA, r)

    hist_full = if merged_scales
        dropdims(hist_base; dims = 3)
    else
        partial_sum(hist_base; dims = 3)
    end

    bins_Γ = to_steprange(g["bin_edges1"][:] :: Vector{Float64})
    # bins_εA = to_steprange(g["bin_edges2"][:] :: Vector{Float64})
    bins_εA = g["bin_edges2"][:] :: Vector{Float64}

    minima = g["minimum"][:, :] :: Array{Float64,2}  # (Γ, ε)
    maxima = g["maximum"][:, :] :: Array{Float64,2}

    samples = sum(g["total_samples"][:] :: Vector{Int64})

    # Check that (almost) all samples are in the PDFs
    @show sum(hist_full) / samples
    # @assert isapprox(sum(hist_full), samples; rtol=1e-3)

    conditional = conditional_pdfs(hist_full, bins_Γ, bins_εA, :Γ_cond)

    (; hist_full, minima, maxima, samples, bins_Γ, conditional)
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
        for i in eachindex(p)
            dx = xs[i + 1] - xs[i]
            p[i] = h[i] / (N * dx)
        end
    end
    (;
        field, rname = :r, xs, rs, hist, pdfs, samples,
    )
end

function conditional_pdfs(hist2D, xs, ys_all, field::Symbol)
    mergebins = 2
    @assert (length(ys_all) - 1) % mergebins == 0
    ys = [ys_all[i] for i = 1:mergebins:lastindex(ys_all)]
    @assert ys[end] == ys_all[end]
    Nx = length(xs) - 1
    Ny = length(ys) - 1
    hist = zeros(eltype(hist2D), Nx, Ny)
    for j ∈ axes(hist2D, 2)
        jj = div(j, mergebins, RoundUp)
        for i in axes(hist2D, 1)
            hist[i, jj] += hist2D[i, j]
        end
    end
    pdfs = similar(hist, Float64)
    for j in axes(pdfs, 2)
        h = @view hist[:, j]
        s = sum(h)
        for i in axes(pdfs, 1)
            dx = xs[i + 1] - xs[i]
            pdfs[i, j] = h[i] / (s * dx)
        end
    end
    rs = @views (ys[1:end-1] .+ ys[2:end]) ./ 2
    (; field,
        rname = :εA,  # name of "independent" variable
        xs, rs, pdfs,
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

    let g = open_group(g, "FieldMetadata")
        @assert read(g["CirculationField/divided_by_area"]) ==
            CIRCULATION_DIVIDED_BY_AREA[]
        @assert read(g["DissipationField/divided_by_area"]) ==
            !DISSIPATION_MULTIPLIED_BY_AREA[]
    end

    histograms2D = load_histograms2D(g["Histogram2D"])
    histograms_Γ = load_histograms1D(g["HistogramCirculation"], rs, :Γ)
    histograms_ε = load_histograms1D(g["HistogramDissipation"], rs, :ε)

    # Estimate ⟨ε⦒ from smallest loop size.
    ε_mean = let H = histograms_ε, j = 1
        mean = @views moment(H.xs, H.pdfs[:, j], Val(1))
        if DISSIPATION_MULTIPLIED_BY_AREA[]
            mean /= As[j]
        end
        mean
    end

    Lbox = params.Ls[1]
    Δx = Lbox / params.Ns[1]
    ν = params.ν
    η = (ν^3 / ε_mean)^(1/4)
    rs_η = rs ./ η
    As_η = As ./ η^2

    taylor_scale = let E = params.DNS_energy
        sqrt(10ν * E / ε_mean)
    end

    enstrophy = ε_mean / (2ν)
    Γ_taylor = taylor_scale^2 * sqrt(2enstrophy / 3)

    @show η, Lbox / η, Δx / η, taylor_scale / η

    (; rs, As, rs_η, As_η, L = Lbox, η, ν,
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
    Γs = histograms.xs

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

function plot_condpdfs!(ax, stats)
    histograms = stats.histograms2D
    conditional = histograms.conditional

    Γs = conditional.xs
    εAs = conditional.rs
    pdfs = conditional.pdfs

    ε_mean = stats.ε_mean
    η = stats.η
    λ = stats.taylor_scale

    εA_norm = ε_mean * λ^2
    @show εA_norm

    ax.set_yscale(:log)
    # ax.set_xlim(-6, 6)
    # ax.set_ylim(1e-6, 1)
    ax.set_xlabel(L"Γ_{\!r} / {\left< Γ_{\!r}^2 \, | \, r^2 ε_r \right>}^{1/2}")
    ax.set_ylabel(L"P(Γ_{\!r} \, | \, r^2 ε_r)")

    εA_indices = eachindex(εAs)[4:1:end-6]
    Ncurves = length(εA_indices)
    cmap = mappable_colour(plt.cm.cividis_r, 0, 1)

    for (n, j) in enumerate(εA_indices)
        pdf = pdfs[:, j]
        xs = collect(Γs)

        # εA_nominal = round.((εAs[j], εAs[j + 1]) ./ εA_norm, digits=4)
        r_norm = ntuple(2) do δ
            round(sqrt(εAs[j - 1 + δ] / εA_norm), digits = 2)
        end

        color = cmap((n - 1) / (Ncurves - 1))

        xrms = sqrt(variance(xs, pdf))
        pdf .*= xrms
        xs ./= xrms

        xs_centre = @views (xs[1:end-1] .+ xs[2:end]) ./ 2
        ax.plot(
            xs_centre, pdf;
            marker = ".", color,
            label = r_norm,
            # label = latexstring("$r_η"),
        )
    end

    # let xs = -5:0.1:5
    #     pdf = normal_pdf.(xs)
    #     ax.plot(xs, pdf; lw = 1.5, color = "tab:orange", ls = :dashed)
    # end

    ax
end

function plot_pdf_εr!(ax, stats; logpdf = false)
    histograms = stats.histograms_ε

    hists = histograms.hist
    εs = histograms.xs

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
    fig, axes = plt.subplots(1, 3; figsize = (10, 4))
    let ax = axes[1]
        plot_pdf_Γr!(ax, stats)
        ax.set_xlim(-12, 12)
        ax.set_ylim(1e-6, 1)
        ax.legend(
            fontsize = :small, title = L"r / λ",
            frameon = false, loc = "upper right",
        )
    end
    let ax = axes[2]
        plot_pdf_εr!(ax, stats; logpdf = true)
        ax.set_ylim(1e-6, 1)
        ax.legend(
            title = L"r / λ", fontsize = :small, frameon = false,
            loc = "lower center",
        )
    end
    let ax = axes[3]
        plot_condpdfs!(ax, stats)
        ax.set_xlim(-12, 12)
        ax.set_ylim(1e-6, 1)
        ax.legend(
            fontsize = "x-small", frameon = false,
            loc = "upper left",
            title = L"(ε_r / \langle ε \rangle)^{1/2} \, r / λ",
        )
    end
    fig.savefig("pdfs.svg", dpi=300)
end

function plot_moments!(axes, stats, moments)
    rname = moments.rname
    rs = moments.rs
    rs_norm = rs ./ moments.rnorm
    rlabel = moments.rlabel
    rlabel_norm = moments.rlabel_norm
    Γ_norm = stats.Γ_taylor
    # Γ_norm = stats.ν

    ps = moments.ps
    ys = moments.moments
    slopes = moments.slopes
    slopes_fitted = moments.slopes_fitted

    cmap = mappable_colour(plt.cm.viridis_r, ps)

    let ax = axes[1]
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
    let ax = axes[2]
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
        let ax = axes[1]
            y = @view(ys[:, j]) ./ Γ_norm^p
            ax.plot(rs_norm, y; color, marker = :., label = p)
        end
        let ax = axes[2]
            x = @views (rs_norm[1:end-1] .+ rs_norm[2:end]) ./ 2
            y = @view slopes[:, j]
            ax.plot(x, y; color, marker = :.)
            ax.axhline(4p/3; color, ls = :dotted)
            # ax.axhline(slopes_fitted[j]; color, ls = :solid)
        end
    end

    let ax = axes[3]
        slopes_k41 = 4/3 .* ps
        ax.plot(ps, slopes_k41; color = :black, ls = :dotted)
        ax.set_ylabel(L"λ_p")
        ax.plot(ps, slopes_fitted; marker = :o)
        ax.set_xlabel(L"p")
    end

    let ax = axes[1]
        ax.legend(title = L"p", loc = "upper left", frameon = false, ncol = 2)
    end

    nothing
end

function compute_moments(hists, stats; ps = 1:10)
    xs = hists.xs
    pdfs = hists.pdfs
    rs = collect(hists.rs)
    rname = hists.rname  # name of "independent" variable

    if rname == :r
        As = stats.As
        divide_by_area = hists.field == :ε && DISSIPATION_MULTIPLIED_BY_AREA[]
        λ = stats.taylor_scale
        # Estimate of r/λ region where slopes are fitted
        fit_region_λ_est = (1, 7)
        rlabel = "r"
        rlabel_norm = L"r / λ"
    elseif rname == :εA
        divide_by_area = false
        λ = stats.taylor_scale * sqrt(stats.ε_mean)
        rs .= sqrt.(rs)
        fit_region_λ_est = (1, 4)
        rlabel = "ε_r r^2"
        rlabel_norm = L"(ε_r / \langle ε \rangle)^{1/2} r / λ"
    end

    Nr = size(pdfs, 2)
    Np = length(ps)

    @assert length(rs) == Nr
    dlog_r = diff(log.(rs))

    moments = zeros(Nr, Np)
    slopes = zeros(Nr - 1, Np)
    slopes_fitted = similar(slopes, Np)

    r_indices_fit = map(a -> searchsortedlast(rs, a * λ), fit_region_λ_est)
    fit_region_λ = getindex.(Ref(rs), r_indices_fit) ./ λ  # actual fit region
    @show fit_region_λ

    for (j, p) in enumerate(ps)
        ys = @view moments[:, j]

        for i ∈ 1:Nr
            pdf = @view pdfs[:, i]
            ys[i] = moment(abs, xs, pdf, Val(p))
            if divide_by_area
                ys[i] /= As[i]^p
            end
        end

        dlog_y = diff(log.(ys))
        sl = @view slopes[:, j]
        sl[:] .= dlog_y ./ dlog_r

        ra, rb = r_indices_fit
        slopes_fitted[j] = sum(n -> sl[n], ra:rb) / (rb - ra + 1)
    end

    (;
        field = hists.field,
        ps,
        rs,
        rname,
        rnorm = λ,
        rlabel,
        rlabel_norm,
        moments,
        slopes,
        slopes_fitted,
        fit_region_λ,
    )
end

function plot_circulation_moments(stats)
    hists = stats.histograms_Γ
    moments = compute_moments(hists, stats)
    fig, axes = plt.subplots(1, 3; figsize = (10, 4))
    plot_moments!(axes, stats, moments)
    fig.savefig("moments.svg", dpi=300)
    nothing
end

function plot_conditional_moments(stats)
    hists = stats.histograms2D
    moments = compute_moments(hists.conditional, stats)
    fig, axes = plt.subplots(1, 3; figsize = (10, 4))
    plot_moments!(axes, stats, moments)
    fig.savefig("moments_conditioned.svg", dpi=300)
    nothing
end

function main()
    params = (;
        results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/circulation_dissA.logbins.v2.h5",
        # results_file = "../../results/data_NS/NS1024_2013/R0_GradientForcing/Dissipation/circulation_dissA.h5",
    )
    data = h5open(load_data, params.results_file, "r")
    stats = data.stats
    plot_pdfs(stats)
    plot_circulation_moments(stats)
    plot_conditional_moments(stats)
end

main()
