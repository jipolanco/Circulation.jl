#!/usr/bin/env julia

# Generate synthetic 3D velocity field with energy spectrum E ~ k^{-α}.

using FFTW
using LinearAlgebra
# using PyPlot
using Random
using UnicodePlots
using WriteVTK

sampling_freq(N, L) = 2π * N / L

# Generate synthetic vector field in Fourier space.
# The resulting energy spectrum has k^{-α} scaling.
# Note that this works in arbitrary dimensions!
# Also note that the generated field is compressible.
function generate_synthetic_field_fourier(dims, α;
                                          L = ntuple(i -> 2π, length(dims)))
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
        kvec = getindex.(ks, Tuple(I))
        k2 = sum(abs2, kvec)
        k2 > k2max && continue
        γ = k2^pow
        for v in vf  # for each velocity component
            v[I] = randn(rng) * γ
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
    plt = lineplot(log10.(k), log10.(E), name="E")
    lineplot!(plt, log10.(k), log10.(2 .* k.^(-α)), name="k^-$α")
    println(plt)
end

# Compute longitudinal structure function of order p.
function structure_function(vs, rs; p = 2)
    S = zeros(length(rs))
    samples = similar(S, Int)
    dims = size(vs[1])
    for I in CartesianIndices(vs[1])
        for (n, v) in enumerate(vs)  # for each velocity component
            N = dims[n]  # number of points along direction n
            v1 = v[I]
            i = I[n]
            for (m, r) in enumerate(rs)  # for each increment
                j = mod1(i + r, N)
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

function main()
    N = 64
    dims = (N, N, N)
    L = (2π, 2π, 2π)
    α = 5 // 3
    # α = 2
    ks, vf = generate_synthetic_field_fourier(dims, α; L=L)
    spec = spectrum(ks, vf)
    plot_spectrum(spec..., α)
    v = brfft.(vf, N) ./ sqrt(prod(dims))
    let rs = 1:(N ÷ 2)
        S2 = structure_function(v, rs, p=2)
        plt = lineplot(log10.(rs), log10.(S2),
                       title="Structure function (p = 2)")
        println(plt)
    end
    save_vtk("synthetic", v, L)
end

main()
