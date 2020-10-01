module GPFields

export ParamsGP
export get_coordinates

using FFTW
using HDF5
using Printf: @sprintf
import Mmap
using Base.Threads

# Type definitions
const ComplexArray{T,N} = AbstractArray{Complex{T},N} where {T<:Real,N}
const RealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}
const RealVector{T,N} = NTuple{N, RealArray{T,N}} where {T<:Real,N}

# Defines a slice in N dimensions.
const Slice{N} = Tuple{Vararg{Union{Int,Colon}, N}} where {N}

include("slices.jl")
include("params.jl")
include("resampling.jl")
include("io.jl")

"""
    create_fft_plans_1d!(ψ::ComplexArray{T,N}) -> (plans_1, plans_2, ...)

Create in-place complex-to-complex FFT plans.

Returns `N` pairs of forward/backward plans along each dimension.
"""
function create_fft_plans_1d!(ψ::ComplexArray{T,D}) where {T,D}
    FFTW.set_num_threads(nthreads())
    ntuple(Val(D)) do d
        p = plan_fft!(ψ, d, flags=FFTW.MEASURE)
        (fw=p, bw=inv(p))
    end
end

"""
    compute_momentum!(
        p::NTuple, ψ::ComplexArray, gp::ParamsGP;
        buf=similar(ψ), fft_plans = create_fft_plans_1d!(ψ),
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

        @threads for i in eachindex(ψ)
            @inbounds dψ[i] = ψ[i]
        end

        plans.fw * dψ  # apply in-place FFT

        @inbounds @threads for I in CartesianIndices(dψ)
            kloc = kn[I[n]]
            dψ[I] *= im * kloc
        end

        plans.bw * dψ  # apply in-place backward FFT

        # 2. Evaluate momentum p[n].
        @threads for i in eachindex(ψ)
            @inbounds pj[i] = α * imag(conj(ψ[i]) * dψ[i])
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
    size(ρ) === size(ψ) || throw(DimensionMismatch(
        "ρ and ψ must have the same dimensions: $(size(ρ)) ≠ $(size(ψ))"
    ))
    @threads for n in eachindex(ρ)
        @inbounds ρ[n] = abs2(ψ[n])
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
