module GPFields

export ParamsGP
export get_coordinates

using FFTW
using Printf: @sprintf

include("params.jl")

# Type definitions
const ComplexArray{T,N} = AbstractArray{Complex{T},N} where {T<:Real,N}
const RealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

function check_size(psi, io_r, io_c)
    size_r = stat(io_r).size
    size_i = stat(io_c).size
    size_r == size_i || error("files have different sizes")
    if sizeof(psi) != size_r + size_i
        error("dimensions of `psi` are inconsistent with file sizes")
    end
    nothing
end

function load_psi!(psi::ComplexArray{T}, io_r::IO, io_c::IO) where {T}
    check_size(psi, io_r, io_c)
    N = length(psi)  # number of points

    for n = 1:N
        x = read(io_r, T)
        y = read(io_c, T)
        psi[n] = Complex(x, y)
    end

    psi
end

"""
    load_psi!(psi, datadir, timestep)

Load complex ψ(x) field from files for `ψ_r` and `ψ_c`.

Writes data to preallocated output `psi`.
"""
function load_psi!(psi::ComplexArray, datadir::AbstractString,
                   timestep::Integer)
    ts = @sprintf "%03d" timestep  # e.g. "007" if timestep = 7

    fname_r = joinpath(datadir, "ReaPsi.$ts.dat")
    fname_i = joinpath(datadir, "ImaPsi.$ts.dat")

    for fname in (fname_r, fname_i)
        isfile(fname) || error("file not found: $fname")
    end

    open(fname_r, "r") do io_r
        open(fname_i, "r") do io_i
            load_psi!(psi, io_r, io_i)
        end
    end

    psi
end

"""
    load_psi(gp::ParamsGP, datadir, timestep)

Load complex ψ(x) field from files for `ψ_r` and `ψ_c`.

Allocates output `psi`.
"""
function load_psi(gp::ParamsGP, args...)
    psi = Array{ComplexF64}(undef, gp.dims...)
    load_psi!(psi, args...) :: ComplexArray
end

"""
    compute_momentum!(p::NTuple, ψ::ComplexArray, gp::ParamsGP)

Compute momentum from complex array ψ.
"""
function compute_momentum!(p::NTuple{D,<:RealArray},
                           ψ::ComplexArray{T,D},
                           gp::ParamsGP{D}) where {T,D}
    @assert all(size(pj) === size(ψ) for pj in p)

    dψ = similar(ψ)  # ∇ψ component

    ks = get_wavenumbers(gp)  # (kx, ky, ...)
    @assert length.(ks) === size(ψ)

    α = 2 * gp.c * gp.ξ / sqrt(2)

    # Loop over momentum components.
    for (n, pj) in enumerate(p)
        # 1. Compute dψ/dx[n].
        kn = ks[n]
        plan = plan_fft!(ψ, n)  # in-place FFT along n-th dimension
        copy!(dψ, ψ)
        plan * dψ  # apply in-place FFT
        @inbounds for I in CartesianIndices(dψ)
            kloc = kn[I[n]]
            dψ[I] *= im * kloc
        end
        plan \ dψ  # apply in-place backward FFT

        # 2. Evaluate momentum p[n].
        @inbounds for i in eachindex(ψ)
            pj[i] = α * imag(conj(ψ[i]) * dψ[i])
        end
    end

    p
end

"""
    compute_momentum(ψ::AbstractArray, gp::ParamsGP)

Allocate and compute momentum from complex array ψ.
"""
function compute_momentum(ψ::ComplexArray{T,D}, gp::ParamsGP{D}) where {T,D}
    p = ntuple(d -> similar(ψ, T), Val(D))  # allocate arrays
    compute_momentum!(p, ψ, gp) :: NTuple
end

"""
    compute_density!(ρ::AbstractArray, ψ::AbstractArray)

Compute density from ψ.
"""
function compute_density!(ρ::AbstractArray{<:Real,N},
                          ψ::AbstractArray{<:Complex,N}) where {N}
    size(ρ) === size(ψ) || throw(ArgumentError("incompatible array dimensions"))
    @inbounds for n in eachindex(ρ)
        ρ[n] = abs2(ψ[n])
    end
    ρ
end

"""
    compute_density(ψ::AbstractArray)

Allocate and compute density from ψ.
"""
function compute_density(ψ::ComplexArray{T}) where {T}
    ρ = similar(ψ, T)
    compute_density!(ρ, ψ) :: RealArray
end

end
