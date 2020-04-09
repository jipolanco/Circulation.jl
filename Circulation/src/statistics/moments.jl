"""
    Moments{T}

Integer moments of a scalar quantity.

---

    Moments(N::Integer, Nr::Integer, ::Type{T} = Float64; Nfrac = nothing)

Construct object for computation of moments.
Moments are computed up to order `N`, for `Nr` different loop sizes.

Optionally, fractional exponents of the absolute value can also be computed.
For this, set the value of `Nfrac` to the number of exponents to consider such that
0 <= p < 1. Negative exponents will also be computed.
For example, if `Nfrac = 10`, the exponents in `-0.9:0.1:0.9` will be
considered.

"""
struct Moments{T, FracMatrix}
    Nr     :: Int  # number of "columns" of data (e.g. one per loop size)
    Nm     :: Int  # number of moments to compute (assumed to be even)
    Nm_odd :: Int  # number of odd moments to compute (= N / 2)

    # Number of positive fractional moments to compute.
    # Fractional moments are computed for -1 < p < 1.
    # The separation between fractional moment exponents to be computed is
    # Δp = 1 / (Nm_frac + 1).
    # For instance, if Nm_frac = 9, then Δp = 0.1, and moments in -0.9:0.1:0.9
    # are computed.
    Nm_frac :: Int

    Nsamples :: Vector{Int}   # number of samples per column [Nr]

    # Number of samples for negative fractional exponents ([Nr] or [0])
    # Can be less than Nsamples since zero values are skipped (which may or may
    # not make sense, depending on the value of the exponent)
    Nsamples_neg :: Vector{Int}

    Mabs :: Matrix{T}    # moments of |Γ|   [Nm, Nr]
    Modd :: Matrix{T}    # odd moments of Γ [Nm_odd, Nr]
    Mfrac :: FracMatrix  # fractional moments of |Γ| [-Nm_frac:Nm_frac, Nr]

    function Moments(N::Integer, Nr::Integer, ::Type{T} = Float64;
                     Nfrac::Union{Int,Nothing} = nothing) where {T}
        iseven(N) || throw(ArgumentError("`N` should be even!"))
        Nodd = N >> 1
        @assert 2Nodd == N
        Nsamples = zeros(Int, Nr)
        Mabs = zeros(T, N, Nr)
        Modd = zeros(T, Nodd, Nr)

        Nsamples_neg = copy(Nsamples)

        if Nfrac === nothing
            # Don't compute fractional moments.
            Nm_frac = 0
            Mfrac = nothing
            resize!(Nsamples_neg, 0)  # I don't need this array
        else
            Nfrac >= 1 || throw(ArgumentError("`Nfrac` should be >= 1"))
            Nm_frac = Nfrac - 1
            # Array that can be accessed with negative indices.
            Mfrac = OffsetArray(zeros(T, 2Nm_frac + 1, Nr),
                                -Nm_frac:Nm_frac, 1:Nr)
        end

        FracMatrix = typeof(Mfrac)

        new{T, FracMatrix}(Nr, N, Nodd, Nm_frac, Nsamples, Nsamples_neg,
                           Mabs, Modd, Mfrac)
    end
end

# Check whether fractional exponents are being computed.
has_fractional(s::Moments) = s.Mfrac !== nothing

# Reconstruct `Nfrac` keyword argument to be passed to constructor.
_get_Nfrac_kwarg(s::Moments) = _get_Nfrac_kwarg(Val(has_fractional(s)), s)
_get_Nfrac_kwarg(frac::Val{true}, s::Moments) = s.Nm_frac + 1
_get_Nfrac_kwarg(frac::Val{false}, s::Moments) = nothing

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
    (; frac = (-s.Nm_frac:s.Nm_frac) ./ (s.Nm_frac + 1))

Base.eltype(::Type{<:Moments{T}}) where {T} = T
Base.zero(s::Moments) = Moments(s.Nm, s.Nr, eltype(s),
                                Nfrac=_get_Nfrac_kwarg(s))

function update!(s::Moments, Γ, r)
    @assert 1 <= r <= s.Nr

    s.Nsamples[r] += length(Γ)

    @assert iseven(s.Nm)
    Nhalf = s.Nm_odd

    with_frac = has_fractional(s)

    if with_frac
        Nm_frac = s.Nm_frac
        frac_base = inv(Nm_frac + 1)  # first positive fractional exponent
        s.Mfrac[0, r] += length(Γ)  # zero moment
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

        b⁺ = abs(v)^frac_base  # e.g. |v|^( 1/10), if Nm_frac = 9
        b⁻ = inv(b⁺)           # e.g. |v|^(-1/10)
        u⁺ = one(b⁺)
        u⁻ = one(b⁻)

        # Skip negative exponents if b⁻ = ∞ (i.e. if v = 0).
        skip_neg = isinf(b⁻)

        if !skip_neg
            s.Nsamples_neg[r] += 1
        end

        for m = 1:Nm_frac
            u⁺ *= b⁺  # b^m    = |v|^( m/10)
            u⁻ *= b⁻  # b^(-m) = |v|^(-m/10)
            s.Mfrac[m, r] += u⁺
            if !skip_neg
                s.Mfrac[-m, r] += u⁻
            end
        end
    end

    s
end

function reduce!(s::Moments, v::AbstractVector{<:Moments})
    for src in v
        @assert s.Nr == src.Nr
        @assert s.Nm == src.Nm
        @assert s.Nm_frac == src.Nm_frac
        s.Nsamples .+= src.Nsamples
        s.Nsamples_neg .+= src.Nsamples_neg
        s.Mabs .+= src.Mabs
        s.Modd .+= src.Modd
        if has_fractional(s)
            s.Mfrac .+= src.Mfrac
        end
    end
    s
end

function finalise!(s::Moments)
    for r in s.Nr
        # Divide by number of samples, to get ⟨Γ^n⟩ and ⟨|Γ|^n⟩.
        Ns = s.Nsamples[r]
        s.Mabs[:, r] ./= Ns
        s.Modd[:, r] ./= Ns
        if has_fractional(s)
            Ns_neg = s.Nsamples_neg[r]
            N = last(axes(s.Mfrac, 1))
            @assert axes(s.Mfrac, 1) == -N:N
            s.Mfrac[0:N, r] ./= Ns
            s.Mfrac[-N:-1, r] ./= Ns_neg  # negative exponents
        end
    end
    s
end

function reset!(s::Moments)
    fill!.((s.Nsamples, s.Nsamples_neg, s.Mabs, s.Modd), 0)
    if has_fractional(s)
        fill!(s.Mfrac, 0)
    end
    s
end

function Base.write(g, s::Moments)
    g["total_samples"] = s.Nsamples
    g["M_odd"] = s.Modd
    g["M_abs"] = s.Mabs

    if has_fractional(s)
        g["total_samples_neg"] = s.Nsamples_neg  # negative exponents
        g["M_frac"] = parent(s.Mfrac)  # I can't directly write an OffsetArray...
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
