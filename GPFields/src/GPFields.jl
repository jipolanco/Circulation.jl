module GPFields

export ParamsGP
export get_coordinates

using FFTW
using Printf: @sprintf
import Mmap

# Type definitions
const ComplexArray{T,N} = AbstractArray{Complex{T},N} where {T<:Real,N}
const RealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

# Defines a slice in N dimensions.
const Slice{N} = Tuple{Vararg{Union{Int,Colon}, N}} where {N}

include("slices.jl")
include("params.jl")

function check_size(::Type{T}, dims, io_r, io_c) where {T <: Complex}
    size_r = stat(io_r).size
    size_i = stat(io_c).size
    size_r == size_i || error("files have different sizes")
    N = prod(dims)
    if sizeof(T) * N != size_r + size_i
        sr = size_r ÷ sizeof(T)
        error(
            """
            given GP dimensions are inconsistent with file sizes
                given dimensions:    $N  $dims
                expected from files: $sr
            """
        )
    end
    nothing
end

# Read the full data
function load_psi!(psi::ComplexArray{T}, vr::RealArray{T}, vi::RealArray{T},
                   slice::Nothing) where {T}
    @assert length(psi) == length(vr) == length(vi)
    for n in eachindex(psi)
        psi[n] = Complex{T}(vr[n], vi[n])
    end
    psi
end

# Read a data slice
function load_psi!(psi::ComplexArray{T}, vr::RealArray{T}, vi::RealArray{T},
                   slice::Slice) where {T}
    inds = view(CartesianIndices(vr), slice...)
    @assert size(vr) == size(vi)
    if size(psi) != size(inds)
        throw(DimensionMismatch(
            "output array has different dimensions from slice: " *
            "$(size(psi)) ≠ $(size(inds))"
        ))
    end
    for (n, I) in enumerate(inds)
        psi[n] = Complex{T}(vr[I], vi[I])
    end
    psi
end

"""
    load_psi!(psi, gp::ParamsGP, datadir, field_index; slice=nothing)

Load complex ψ(x) field from files for `ψ_r` and `ψ_c`.

Writes data to preallocated output `psi`.

The optional `slice` parameter may designate a slice of the domain,
such as `(:, 42, :)`.
"""
function load_psi!(psi::ComplexArray{T}, gp::ParamsGP{N},
                   datadir::AbstractString, field_index::Integer;
                   slice::Union{Nothing,Slice{N}} = nothing) where {T,N}
    ts = @sprintf "%03d" field_index  # e.g. "007" if field_index = 7

    fname_r = joinpath(datadir, "ReaPsi.$ts.dat")
    fname_i = joinpath(datadir, "ImaPsi.$ts.dat")

    for fname in (fname_r, fname_i)
        isfile(fname) || error("file not found: $fname")
    end

    check_size(Complex{T}, gp.dims, fname_r, fname_i)

    # Memory-map data from file.
    # That is, data is not loaded into memory until needed.
    vr = Mmap.mmap(fname_r, Array{T,N}, gp.dims)
    vi = Mmap.mmap(fname_i, Array{T,N}, gp.dims)

    load_psi!(psi, vr, vi, slice)

    psi
end

"""
    load_psi(gp::ParamsGP, datadir, field_index)

Load full complex ψ(x) field from files for `ψ_r` and `ψ_c`.

Allocates output `psi`.
"""
function load_psi(gp::ParamsGP, args...)
    psi = Array{ComplexF64}(undef, gp.dims...)
    load_psi!(psi, gp, args...) :: ComplexArray
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
