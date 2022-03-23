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

    α = let p = gp.phys
        @assert p !== nothing
        2 * p.c * p.ξ / sqrt(2)
    end

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

"""
    velocity!([f::Function,] vs::NTuple, ps::NTuple, ρ; eps = 0)

Compute velocity field from momentum and density fields.

The velocity and momentum fields may be aliased.

The optional function may be a modifier for the density `ρ`. That is, it
replaces each value of `ρ` by the output of `f(ρ)`. By default `f` is the
`identity` function, i.e. it doesn't modify `ρ`. This may be used to compute the
regularised velocity (if `f = sqrt`). It may also be used to add a small
increment to the density, to avoid division by zero (e.g. `f = ρ -> ρ + 1e-6`).
"""
function velocity!(f::Function,
                   vs::NTuple{D,<:RealArray},
                   ps::NTuple{D,<:RealArray},
                   ρ::RealArray{T,D}) where {T,D}
    @inbounds @threads for n in eachindex(ρ)
        one_over_rho = inv(f(ρ[n]))
        for (v, p) in zip(vs, ps)  # for each velocity component
            v[n] = one_over_rho * p[n]
        end
    end
    vs
end

velocity!(vs::NTuple, etc...) = velocity!(identity, vs, etc...)
