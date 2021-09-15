# Generate synthetic incompressible 3D velocity field with energy spectrum E ~ k^{-α}.

using LinearAlgebra
using Printf: @sprintf
using Random
using StaticArrays: SVector

using FFTW
using WriteVTK

import UnicodePlots
const UP = UnicodePlots

sampling_freq(N, L) = 2π * N / L

# Generate synthetic vector field in Fourier space.
# The resulting energy spectrum has k^{-α} scaling.
# Note that this works in arbitrary dimensions!
function generate_synthetic_field_fourier(
        dims::Dims{N}, α;
        incompressible = true,
        L = ntuple(i -> 2π, length(dims)),
    ) where {N}
    # Wave numbers
    ks = (
        rfftfreq(dims[1], sampling_freq(dims[1], L[1])),
        ntuple(i -> (j = i + 1; fftfreq(dims[j], sampling_freq(dims[j], L[j]))),
               length(dims) - 1)...,
    )

    # Dimensions in Fourier space
    dims_f = length.(ks)
    vf = ntuple(d -> zeros(ComplexF64, dims_f...), length(dims))  # vector field
    pow = -(N - 1 + α) / 4

    k2max = ks[1][end]^2

    rng = MersenneTwister(42)

    for I in CartesianIndices(vf[1])
        kvec = SVector{N}(getindex.(ks, Tuple(I)))
        k2 = sum(abs2, kvec)
        k2 == 0 && continue
        k2 > k2max && continue
        γ = k2^pow
        w = randn(rng, SVector{N,ComplexF64}) * γ
        if incompressible
            # Remove compressible part.
            c = conj((w ⋅ kvec) / k2)
            w -= c * kvec
            @assert abs(kvec ⋅ w) < eps(k2)
        end
        for (v, w) in zip(vf, w)
            v[I] = w
        end
    end

    for v in vf
        # Set zero mode to zero
        v[1] = 0

        # Normalise by the energy (roughly)
        v .*= 1.0 / sqrt(sum(abs2, v))
    end

    ks, vf
end

function spectrum(ks, vf)
    k = ks[1]
    N = length(k)
    E = zeros(N)
    # kmax = k[end] + k[2]
    for I in CartesianIndices(vf[1])
        kvec = getindex.(ks, Tuple(I))
        knorm = sqrt(sum(abs2, kvec))
        # knorm > kmax && continue
        n = searchsortedlast(k, knorm)
        n == 0 && continue
        factor = kvec[1] == 0 ? 1 : 2
        E[n] += factor * sum(abs2, getindex.(vf, Ref(I)))
    end
    k, E
end

function plot_spectrum(kin, Ein, α)
    k = @view kin[2:end-1]
    E = @view Ein[2:end-1]
    fig = UP.lineplot(
        k, E;
        xscale = log10, yscale = log10,
        title = "Energy spectrum",
        xlabel = "k", ylabel = "E(k)",
        name = "Synthetic field",
    )
    α_str = replace(string(α), "//" => "/")
    UP.lineplot!(fig, k, 5 .* k.^(-α); name = "k^{-$α_str}")
    display(fig)
    fig
end

function save_vtk(basename, v, Ls)
    dims = size(v[1])
    coords = map(dims, Ls) do N, L
        LinRange(0, L, N + 1)[1:N]
    end
    files = vtk_grid(basename, coords..., compress=false) do vtk
        vtk["v"] = v
    end
    @info "Saved $files"
    nothing
end

# Note: incompressible velocity files are named like "VIx_d.042.dat".
function save_binary(
        vs::Tuple, timestep=0;
        directory = ".",
        filename_fmt = "VI{component}_d.{timestep}.dat",
    )
    @assert length(vs) == 3
    fmt = replace(filename_fmt, "{timestep}" => @sprintf("%03d", timestep))
    for (v, c) in zip(vs, "xyz")
        fname = joinpath(directory, replace(fmt, "{component}" => c))
        write(fname, v)
        @info "Saved $fname"
    end
    nothing
end

function main()
    N = 64
    dims = (N, N, N)
    L = (2π, 2π, 2π)
    α = 5 // 3
    # α = 2

    ks, vf = generate_synthetic_field_fourier(
        dims, α; L = L, incompressible = true,
    )

    # Compute and verify spectrum
    spec = spectrum(ks, vf)
    plot_spectrum(spec..., α)

    # Save velocity field in physical space
    v = brfft.(vf, N)
    save_binary(v)

    # For visualisation
    save_vtk("synthetic", v, L)

    nothing
end

main()
