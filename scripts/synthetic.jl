#!/usr/bin/env julia

# Generate synthetic incompressible 3D velocity field with energy spectrum E ~ k^{-α}.

using LinearAlgebra
using Printf: @sprintf
using Random
using Statistics: mean
using StaticArrays

using FFTW
using WriteVTK

import PyPlot
const plt = PyPlot
using LaTeXStrings

sampling_freq(N, L) = 2π * N / L

# Generate synthetic vector field in Fourier space.
# The resulting energy spectrum has k^{-α} scaling.
# Note that this works in arbitrary dimensions!
function generate_synthetic_field_fourier(
        dims::Dims{N}, α;
        incompressible = true,
        L = ntuple(i -> 2π, length(dims))) where {N}
    # Wave numbers
    ks = (
        rfftfreq(dims[1], sampling_freq(dims[1], L[1])),
        ntuple(i -> (j = i + 1; fftfreq(dims[j], sampling_freq(dims[j], L[j]))),
               length(dims) - 1)...,
    )

    # Dimensions in Fourier space
    dims_f = length.(ks)
    vf = ntuple(d -> zeros(ComplexF64, dims_f...), length(dims))  # vector field
    pow = -(2 + α) / 4

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
    fig, ax = plt.subplots()
    ax.set_xscale(:log)
    ax.set_yscale(:log)
    α_str = replace(string(α), "//" => "/")
    ax.plot(k, E, ".-", label=L"E")
    ax.plot(k, 2 .* k.^(-α), color="black", ls=":",
            label=latexstring("k^{-$α_str}"))
    ax.set_title("Energy spectrum (synthetic field)")
    ax.legend()
    fig
end

# Compute longitudinal structure function of order p.
function structure_function(vs, rs; p = 2)
    S = zeros(length(rs))
    samples = zeros(Int, length(rs))
    dims = size(vs[1])
    for I in CartesianIndices(vs[1])
        for (n, v) in enumerate(vs)  # for each velocity component
            N = dims[n]  # number of points along direction n
            v1 = v[I]
            i = I[n]
            for (m, r) in enumerate(rs)  # for each increment
                j = mod1(i + r, N)
                # Replace index in the n-th dimension.
                J = CartesianIndex(ntuple(d -> (d == n ? j : I[d]), length(I)))
                v2 = v[J]
                S[m] += (v2 - v1)^p
                samples[m] += 1
            end
        end
    end
    S ./= samples
    S
end

function plot_S2(rs, S2, α; compensate=true)
    fig, ax = plt.subplots()
    ax.set_xscale(:log)
    ax.set_yscale(:log)
    n = α - 1
    nstr = replace(string(n), "//" => "/")
    ax.set_xlabel(L"r")
    ax.set_title("Second-order structure function (synthetic field)")
    if compensate
        ax.plot(rs, S2 ./ rs.^n, ".-")
        ax.set_ylabel(latexstring("S_2(r) / r^{$nstr}"))
    else
        ax.plot(rs, S2, ".-", label=L"S_2")
        ax.plot(rs, 1.2 * rs.^n, ":", color="black",
                label=latexstring("r^{$nstr}"))
        ax.set_ylabel(latexstring("S_2(r)"))
        ax.legend()
    end
    fig
end

function save_vtk(basename, v, Ls)
    dims = size(v[1])
    coords = map(dims, Ls) do N, L
        LinRange(0, L, N + 1)[1:N]
    end
    vtk_grid(basename, coords..., compress=false) do vtk
        vtk["v"] = v
    end
    nothing
end

# Note: incompressible velocity files are named like "VIx_d.042.dat".
function save_binary(vs::Tuple, timestep=0;
                     filename_fmt="VI{component}_d.{timestep}.dat")
    @assert length(vs) == 3
    fmt = replace(filename_fmt, "{timestep}" => @sprintf("%03d", timestep))
    for (v, c) in zip(vs, "xyz")
        fname = replace(fmt, "{component}" => c)
        write(fname, v)
    end
    nothing
end

function main()
    N = 128
    dims = (N, N, N)
    L = (2π, 2π, 2π)
    α = 5 // 3
    # α = 2
    @time ks, vf = generate_synthetic_field_fourier(
        dims, α; L=L, incompressible=true)
    spec = spectrum(ks, vf)
    plot_spectrum(spec..., α)
    v = brfft.(vf, N)
    save_vtk("synthetic", v, L)
    save_binary(v)

    let rs = unique([min(N ÷ 2, round(Int, 1.1^n)) for n = 0:100])
        @time S2 = structure_function(v, rs, p=2)
        Δx = L[1] / dims[1]
        for comp in (false, true)
            plot_S2(rs .* Δx, S2, α, compensate=comp)
        end
    end

    nothing
end

main()
