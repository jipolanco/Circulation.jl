module GPFields

export ParamsGP
export get_coordinates

using FFTW
using HDF5
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
    create_fft_plans_1d!(ψ::ComplexArray{T,N}) -> (plans_1, plans_2, ...)

Create in-place complex-to-complex FFT plans.

Returns `N` pairs of forward/backward plans along each dimension.
"""
function create_fft_plans_1d!(ψ::ComplexArray{T,D}) where {T,D}
    ntuple(Val(D)) do d
        p = plan_fft!(ψ, d)
        (fw=p, bw=inv(p))
    end
end

"""
    compute_momentum!(p::NTuple, ψ::ComplexArray, gp::ParamsGP;
                      buf=similar(ψ),
                      fft_plans = create_fft_plans_1d!(ψ),
                      )

Compute momentum from complex array ψ.

Optionally, to avoid memory allocations, a buffer array may be passed.
The array must have the same type and dimensions as ψ.

Precomputed FFT plans may be passed via the `fft_plans` argument.
These should be generated using `create_fft_plans_1d!`.
This is not only good for performance, but it also avoids problems when using
threads.
"""
function compute_momentum!(
        p::NTuple{D,<:RealArray},
        ψ::ComplexArray{T,D},
        gp::ParamsGP{D};
        buf::ComplexArray{T,D} = similar(ψ),
        fft_plans = create_fft_plans_1d!(ψ),
    ) where {T,D}
    @assert all(size(pj) === size(ψ) for pj in p)
    if size(buf) !== size(ψ)
        throw(DimensionMismatch(
            "inconsistent dimensions between ψ and buffer array"
        ))
    end

    dψ = buf  # ∇ψ component

    ks = get_wavenumbers(gp)  # (kx, ky, ...)
    @assert length.(ks) === size(ψ)

    α = 2 * gp.c * gp.ξ / sqrt(2)

    # Loop over momentum components.
    for (n, pj) in enumerate(p)
        plans = fft_plans[n]

        # 1. Compute dψ/dx[n].
        kn = ks[n]

        copy!(dψ, ψ)
        plans.fw * dψ  # apply in-place FFT

        @inbounds for I in CartesianIndices(dψ)
            kloc = kn[I[n]]
            dψ[I] *= im * kloc
        end

        plans.bw * dψ  # apply in-place backward FFT

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
    size(ρ) === size(ψ) || throw(DimensionMismatch())
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

"""
    resample_field_fourier!(
        dest::AbstractArray, src::AbstractArray,
        params_src::ParamsGP; destroy_input = true,
    )

Resample complex field by zero-padding in Fourier space.

The resampling factor is determined from the dimensions of the two arrays.
It must be the same along all dimensions.

For now, the resampling factor must also be a non-negative power of two.

Resampling is performed in Fourier space.
No transforms are performed in this function, meaning that the input and output
are also in Fourier space.
"""
function resample_field_fourier!(dst::ComplexArray{T,N}, src::ComplexArray{T,N},
                                 p_src::ParamsGP{N}) where {T,N}
    if size(src) === size(dst)
        if src !== dst
            copy!(dst, src)
        end
        return dst
    end

    p_dst = ParamsGP(p_src, dims=size(dst))

    ksrc = get_wavenumbers(p_src)
    kdst = get_wavenumbers(p_dst)

    kmap = _wavenumber_map.(ksrc, kdst)

    # 1. Set everything to zero.
    fill!(dst, 0)

    # 2. Copy all modes in src.
    for I in CartesianIndices(src)
        is = Tuple(I)
        js = getindex.(kmap, is)
        dst[js...] = src[I]
    end

    dst
end

# Maps ki index to ko index, such that ki[n] = ko[kmap[n]].
function _wavenumber_map(ki::Frequencies, ko::Frequencies)
    Base.require_one_based_indexing.((ki, ko))
    Ni = length(ki)
    No = length(ko)
    if Ni > No
        error("downscaling (Fourier truncation) is not allowed")
    end
    if any(isodd.((Ni, No)))
        error("data length must be even (got $((Ni, No)))")
    end
    if ki[Ni] > 0 || ko[No] > 0
        error("negative wave numbers must be included")
    end
    h = Ni >> 1
    kmap = similar(ki, Int)
    for n = 1:h
        kmap[n] = n
        kmap[Ni - n + 1] = No - n + 1
    end
    # Verification
    for n in eachindex(ki)
        @assert ki[n] ≈ ko[kmap[n]]
    end
    kmap
end

end
