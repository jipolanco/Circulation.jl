#!/usr/bin/env julia

using GPFields
using Circulation

using FFTW
using TimerOutputs
using HDF5

import Base.Threads

dims = (1024, 1024)
gp_input = ParamsGP(dims; L = (2π, 2π), c = 1, nxi = 1)
resampling_factor = 2
gp = ParamsGP(gp_input, dims = resampling_factor .* dims)

ψ_input = load_psi(
    gp_input,
    expanduser("~/Dropbox/circulation/data/1024/2D/*2D_1024_slice6_t100.bin"),
)

ψ = similar(ψ_input, resampling_factor .* size(ψ_input))
GPFields.resample_field_fourier!(ψ, fft!(ψ_input), gp_input)
ifft!(ψ)

ρ = density(ψ)
p = momentum(ψ, gp)
v = map(p -> p ./ ρ, p)
vreg = map(p -> p ./ sqrt.(ρ), p)

function curl_phys(v, gp::ParamsGP{2}; plan = plan_rfft(v[1]))
    vF = map(v -> plan * v, v)
    ωF = similar(vF[1])
    curlF!(ωF, vF, gp)
    plan \ ωF
end

@time vreg_curl = curl_phys(vreg, gp)

