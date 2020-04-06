"""
    Moments{T}

Integer moments of a scalar quantity.
"""
struct Moments{T}
    Nr     :: Int  # number of "columns" of data (e.g. one per loop size)
    Nm     :: Int  # number of moments to compute (assumed to be even)
    Nm_odd :: Int  # number of odd moments to compute (= N / 2)

    Nsamples :: Vector{Int}  # number of samples per column (Nr)

    Mabs :: Matrix{T}  # moments of |Γ|   (Nm, Nr)
    Modd :: Matrix{T}  # odd moments of Γ (Nm_odd, Nr)

    function Moments(N::Integer, Nr::Integer, ::Type{T} = Float64) where {T}
        iseven(N) || throw(ArgumentError("`N` should be even!"))
        Nodd = N >> 1
        @assert 2Nodd == N
        Nsamples = zeros(Int, Nr)
        Mabs = zeros(T, N, Nr)
        Modd = zeros(T, Nodd, Nr)
        new{T}(Nr, N, Nodd, Nsamples, Mabs, Modd)
    end
end

Base.eltype(::Type{<:Moments{T}}) where {T} = T
Base.zero(s::Moments) = Moments(s.Nm, s.Nr, eltype(s))

function update!(s::Moments, Γ, r)
    @assert 1 <= r <= s.Nr

    s.Nsamples[r] += length(Γ)

    @assert iseven(s.Nm)
    Nhalf = s.Nm_odd

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
    end

    s
end

function reduce!(s::Moments, v::AbstractVector{<:Moments})
    for src in v
        @assert s.Nr == src.Nr
        @assert s.Nm == src.Nm
        s.Nsamples .+= src.Nsamples
        s.Mabs .+= src.Mabs
        s.Modd .+= src.Modd
    end
    s
end

function finalise!(s::Moments)
    for r in s.Nr
        # Divide by number of samples, to get ⟨Γ^n⟩ and ⟨|Γ|^n⟩.
        Ns = s.Nsamples[r]
        s.Mabs[:, r] ./= Ns
        s.Modd[:, r] ./= Ns
        # TODO subtract mean?
    end
    s
end

function reset!(s::Moments)
    fill!.((s.Nsamples, s.Mabs, s.Modd), 0)
    s
end

function Base.write(g, s::Moments)
    g["total_samples"] = s.Nsamples
    g["M_odd"] = s.Modd
    g["M_abs"] = s.Mabs
    g
end
