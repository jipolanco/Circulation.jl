"""
    momentum!(
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
function momentum!(
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
    momentum(ψ::AbstractArray, gp::ParamsGP)

Allocate and compute momentum from complex array ψ.
"""
function momentum(ψ::ComplexArray{T,D}, gp::ParamsGP{D}) where {T,D}
    p = ntuple(d -> similar(ψ, T), Val(D))  # allocate arrays
    momentum!(p, ψ, gp) :: NTuple
end

"""
    density!(ρ::AbstractArray, ψ::AbstractArray)

Compute density from ψ.
"""
function density!(ρ::AbstractArray{<:Real,N},
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
    density(ψ::AbstractArray)

Allocate and compute density from ψ.
"""
function density(ψ::ComplexArray{T}) where {T}
    ρ = similar(ψ, T)
    density!(ρ, ψ) :: RealArray
end
