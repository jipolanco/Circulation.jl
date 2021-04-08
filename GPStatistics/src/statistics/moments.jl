export ParamsMoments, Moments

struct ParamsMoments{
        T <: AbstractFloat,
        Fields <: Tuple{AbstractScalarField},
        F <: Union{Nothing, Int},
    } <: BaseStatsParams

    fields     :: Fields
    integer    :: Int  # number of integer moments to compute (should be even)
    fractional :: F    # number of fractional moments to compute

    function ParamsMoments(::Type{T}, field; integer, fractional = nothing) where {T}
        fields = (field,)
        new{T, typeof(fields), typeof(fractional)}(fields, integer, fractional)
    end
end

ParamsMoments(field; kws...) = ParamsMoments(Float64, field; kws...)

init_statistics(p::ParamsMoments, etc...) = Moments(p, etc...)

"""
    Moments{T}

Integer moments of a scalar quantity (circulation, velocity increments, ...).

---

    Moments(params::ParamsMoments, Nr::Integer)
    Moments(N::Integer, Nr::Integer, ::Type{T} = Float64; Nfrac = nothing)

Construct object for computation of moments.
Moments are computed up to order `N`, for `Nr` different loop sizes (or spatial
increments, for structure functions).

Optionally, fractional exponents of the absolute value can also be computed.
For this, set the value of `Nfrac` to the number of exponents to consider in the
range `0 < p ≤ 1`.
For example, if `Nfrac = 10`, the exponents in `0.1:0.1:1` will be considered.
"""
struct Moments{
        T, FracMatrix <: Union{Matrix{T},Nothing},
        Params <: ParamsMoments,
    } <: AbstractBaseStats

    params :: Params
    finalised :: Base.RefValue{Bool}
    Nr     :: Int  # number of "columns" of data (e.g. one per loop size)
    Nm     :: Int  # number of moments to compute (assumed to be even)
    Nm_odd :: Int  # number of odd moments to compute (= N / 2)

    # Number of fractional moments to compute.
    # Fractional moments are computed for 0 < p < 1.
    # The separation between fractional moment exponents to be computed is
    # Δp = 1 / (Nm_frac + 1).
    # For instance, if Nm_frac = 9, then Δp = 0.1, and moments in 0.1:0.1:0.9
    # are computed.
    Nm_frac :: Int

    Nsamples :: Vector{Int}   # number of samples per column [Nr]

    Mabs :: Matrix{T}    # moments of |Γ|   [Nm, Nr]
    Modd :: Matrix{T}    # odd moments of Γ [Nm_odd, Nr]
    Mfrac :: FracMatrix   # fractional moments of |Γ| [Nm_frac, Nr]

    function Moments(p::ParamsMoments{T}, Nr::Integer) where {T}
        N = p.integer
        Nfrac = p.fractional
        iseven(N) || throw(ArgumentError("`N` should be even!"))
        Nodd = N >> 1
        @assert 2Nodd == N
        Nsamples = zeros(Int, Nr)
        Mabs = zeros(T, N, Nr)
        Modd = zeros(T, Nodd, Nr)

        if Nfrac === nothing
            # Don't compute fractional moments.
            Nm_frac = 0
            Mfrac = nothing
        else
            Nfrac >= 1 || throw(ArgumentError("`Nfrac` should be >= 1"))
            Nm_frac = Nfrac - 1
            Mfrac = zeros(T, Nm_frac, Nr)
        end

        FracMatrix = typeof(Mfrac)

        new{T, FracMatrix, typeof(p)}(
            p, Ref(false), Nr, N, Nodd, Nm_frac, Nsamples, Mabs, Modd, Mfrac,
        )
    end
end

# Check whether fractional exponents are being computed.
has_fractional(s::Moments) = s.Mfrac !== nothing

# Get computed moment exponents.
function exponents(s::Moments)
    (
        abs = range(1, length=s.Nm, step=1),
        odd = range(1, length=s.Nm_odd, step=2),
        exponents(Val(has_fractional(s)), s)...,
    )
end

# Fractional exponents
exponents(frac::Val{false}, s::Moments) = NamedTuple()
exponents(frac::Val{true}, s::Moments) =
    (; frac = (1:s.Nm_frac) ./ (s.Nm_frac + 1))

Base.eltype(::Type{<:Moments{T}}) where {T} = T
Base.zero(s::Moments) = Moments(s.params, s.Nr)

function update!(::NoConditioning, s::Moments, Γ, r)
    @assert 1 <= r <= s.Nr

    s.Nsamples[r] += length(Γ)

    @assert iseven(s.Nm)
    Nhalf = s.Nm_odd

    with_frac = has_fractional(s)

    if with_frac
        Nm_frac = s.Nm_frac
        frac_base = inv(Nm_frac + 1)  # first fractional exponent
    end

    # Loop over values of circulation, then over moment orders.
    @inbounds for v in Γ
        u = one(v)  # initialise u = 1 (with the same type of v)
        n = 0

        for m = 1:Nhalf
            # 1. Odd moment
            n += 1
            u *= v  # = v^(2m - 1)
            s.Mabs[n, r] += abs(u)
            s.Modd[m, r] += u

            # 2. Even moment
            n += 1
            u *= v  # = v^(2m)
            s.Mabs[n, r] += u  # u should already be positive
        end

        with_frac || continue

        b = abs(v)^frac_base  # e.g. |v|^(1/10), if Nm_frac = 9
        u = one(b)

        for m = 1:Nm_frac
            u *= b  # b^m = |v|^(m/10)
            s.Mfrac[m, r] += u
        end
    end

    s
end

function update!(cond::ConditionOnDissipation, s::Moments, fields, r)
    Γ = fields.Γ
    ε = fields.ε
    # TODO continue...
end

function reduce!(s::Moments, v)
    for src in v
        @assert s.Nr == src.Nr
        @assert s.Nm == src.Nm
        @assert s.Nm_frac == src.Nm_frac
        s.Nsamples .+= src.Nsamples
        s.Mabs .+= src.Mabs
        s.Modd .+= src.Modd
        if has_fractional(s)
            s.Mfrac .+= src.Mfrac
        end
    end
    s
end

function finalise!(s::Moments)
    @assert !was_finalised(s)
    for r = 1:s.Nr
        # Divide by number of samples, to get ⟨Γ^n⟩ and ⟨|Γ|^n⟩.
        Ns = s.Nsamples[r]
        s.Mabs[:, r] ./= Ns
        s.Modd[:, r] ./= Ns
        if has_fractional(s)
            s.Mfrac[:, r] ./= Ns
        end
    end
    s.finalised[] = true
    s
end

function reset!(s::Moments)
    s.finalised[] = false
    fill!.((s.Nsamples, s.Mabs, s.Modd), 0)
    if has_fractional(s)
        fill!(s.Mfrac, 0)
    end
    s
end

function Base.write(g, s::Moments)
    @assert was_finalised(s)
    g["total_samples"] = s.Nsamples
    g["M_odd"] = s.Modd
    g["M_abs"] = s.Mabs

    if has_fractional(s)
        g["M_frac"] = s.Mfrac
    end

    let p = exponents(s)
        g["p_abs"] = collect(p.abs)
        g["p_odd"] = collect(p.odd)
        if has_fractional(s)
            g["p_frac"] = collect(p.frac)
        end
    end

    g
end
