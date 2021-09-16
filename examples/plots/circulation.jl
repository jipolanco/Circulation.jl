using CairoMakie
using LaTeXStrings
using HDF5

filename = if isempty(ARGS)
    "../../circulation_NS.h5"
else
    ARGS[1]
end

@info "Plotting from $filename"
if !isfile(filename)
    error("file not found: $filename")
end

make_plots(filename) = h5open(make_plots, filename, "r")

function make_plots(ff::HDF5.File)
    g_base = open_group(ff, "/Circulation/Velocity")
    g_moments = open_group(g_base, "Moments")
    g_hist = open_group(g_base, "Histogram")

    # Load loop sizes
    rs = read(g_base["kernel_size"]) :: Vector{Float64}

    # Load second-order moments (circulation variance)
    Γ_var = g_moments["M_abs"][2, :] :: Vector{Float64}

    fig = Figure(resolution = (1000, 500))

    # 1. Plot circulation variance as a function of loop size
    let gl = fig[1, 1] = GridLayout()
        ax = Axis(gl[1, 1]; xscale = log10, yscale = log10)
        ax.xlabel = "Loop size r"
        ax.ylabel = "Circulation variance"

        # This is just to make sure that p[2] == 2 (i.e. we're indeed plotting
        # the second-order moment)
        @assert g_moments["p_abs"][2] == 2

        @views scatterlines!(ax, rs[begin:end-1], Γ_var[begin:end-1]; label = "Variance")

        # Plot K41 prediction for reference
        var_K41 = @. 5 * rs^(8/3)
        @views lines!(
            ax, rs[begin+1:end-4], var_K41[begin+1:end-4];
            label = latexstring(raw"$r^{8/3}$ (K41)"),
        )

        axislegend(ax; position = :lt)
    end

    # 2. Plot circulation PDF for different loop sizes
    let gl = fig[1, 2] = GridLayout()
        ax = Axis(gl[1, 1]; yscale = log10, xlabel = L"Γ / ⟨ Γ^2 ⟩^{1/2}")
        ax.ylabel = "Probability"

        bin_edges = read(g_hist["bin_edges"]) :: Vector{Float64}  # [Nb + 1]
        hists = read(g_hist["hist"]) :: Matrix{Int}  # [Nb, Nr]
        nsamples = read(g_hist["total_samples"]) :: Vector{Int}  # [Nr]

        Nb = size(hists, 1)  # number of bins
        @assert length(bin_edges) == Nb + 1

        # Plot for every `r`
        pdf = similar(hists, Float64, Nb)
        bin_centres = @views (bin_edges[begin:end-1] .+ bin_edges[begin+1:end]) ./ 2
        xs = similar(bin_centres)
        colourmap = cgrad(:viridis)
        yshift = 1.0  # shift PDFs for better visualisation

        for j ∈ eachindex(rs)
            # Normalise bins by the standard deviation
            Γ_std = sqrt(Γ_var[j])
            xs .= bin_centres ./ Γ_std
            n = nsamples[j]
            
            for i ∈ eachindex(pdf)
                dx = (bin_edges[i + 1] - bin_edges[i]) / Γ_std
                pdf[i] = hists[i, j] / (n * dx) * yshift
            end

            yshift *= 0.5

            # Replace zeros by NaN's to avoid error when plotting zeros in log
            # scale.
            replace!(y -> iszero(y) ? oftype(y, NaN) : y, pdf)

            label = string(round(rs[j], digits = 1))
            alpha = 0.8
            color = (colourmap[j / length(rs)], alpha)

            lines!(ax, xs, pdf; label, color)
        end

        Legend(gl[1, 2], ax, "Loop size")
    end

    outfile = splitext(HDF5.filename(ff))[1] * ".svg"
    @info "Saving figure to $outfile"
    save(outfile, fig)

    nothing
end

make_plots(filename)
