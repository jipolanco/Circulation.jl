export ParamsHistogram, Histogram

Base.@kwdef struct ParamsHistogram{Edges <: AbstractVector} <: BaseStatsParams
    bin_edges :: Edges
end

init_statistics(p::ParamsHistogram, etc...) = Histogram(p, etc...)

"""
    Histogram{T}

Histogram of a scalar quantity.
"""
struct Histogram{T, Tb, BinType<:AbstractVector{Tb}} <: AbstractBaseStats
    finalised :: Base.RefValue{Bool}
    Nr    :: Int          # number of "columns" of data (e.g. one per loop size)
    Nbins :: Int          # number of bins
    bin_edges :: BinType  # sorted list of bin edges [Nbins + 1]
    H :: Matrix{T}        # histogram [Nbins, Nr]

    vmin :: Vector{Tb}    # minimum sampled value [Nr]
    vmax :: Vector{Tb}    # maximum sampled value [Nr]

    # Number of samples per column (Nr).
    # This includes outliers, i.e. events falling outside of the histogram.
    Nsamples :: Vector{Int}

    function Histogram(bin_edges::AbstractVector, Nr::Integer,
                       ::Type{T} = Int) where {T}
        Nbins = length(bin_edges) - 1
        H = zeros(T, Nbins, Nr)
        Nsamples = zeros(Int, Nr)
        BinType = typeof(bin_edges)
        Tb = eltype(bin_edges)
        vmin = zeros(Tb, Nr)
        vmax = zeros(Tb, Nr)
        new{T, Tb, BinType}(
            Ref(false), Nr, Nbins, bin_edges, H, vmin, vmax, Nsamples,
        )
    end
end

Histogram(p::ParamsHistogram, etc...) = Histogram(p.bin_edges, etc...)

Base.eltype(::Type{<:Histogram{T}}) where {T} = T
Base.zero(s::Histogram) = Histogram(s.bin_edges, s.Nr, eltype(s))

function update!(::NoConditioning, s::Histogram, Γ, r)
    @assert 1 <= r <= s.Nr

    s.Nsamples[r] += length(Γ)
    Ne = length(s.bin_edges)

    vmin = s.vmin[r]
    vmax = s.vmax[r]

    @inbounds for v in Γ
        # The "oftype" is just to make sure that `vmin` doesn't change type
        # (e.g. from Float64 to Float32, in case `v` is the latter), which would
        # be very bad for performance.
        vmin = min(oftype(vmin, v), vmin)
        vmax = max(oftype(vmax, v), vmax)
        i = searchsortedlast(s.bin_edges, v)
        if i ∉ (0, Ne)
            s.H[i, r] += 1
        end
    end

    s.vmin[r] = vmin
    s.vmax[r] = vmax

    s
end

function update!(
        cond::ConditionOnDissipation, s::Histogram,
        fields, r,
    )
    Γ = fields.Γ
    ε = fields.ε
    # TODO continue...
end

function reduce!(s::Histogram, v)
    for src in v
        @assert s.Nr == src.Nr
        @assert s.Nbins == src.Nbins
        @assert s.bin_edges == src.bin_edges
        s.H .+= src.H
        s.Nsamples .+= src.Nsamples
        s.vmin .= min.(s.vmin, src.vmin)
        s.vmax .= max.(s.vmax, src.vmax)
    end
    s
end

function finalise!(s::Histogram)
    @assert !was_finalised(s)
    s.finalised[] = true
    s  # nothing to do
end

function reset!(s::Histogram)
    s.finalised[] = false
    fill!(s.H, 0)
    fill!(s.Nsamples, 0)
    fill!(s.vmin, 0)
    fill!(s.vmax, 0)
    s
end

function Base.write(g, s::Histogram)
    @assert was_finalised(s)
    g["bin_edges"] = collect(s.bin_edges)
    g["total_samples"] = s.Nsamples

    g["minimum"] = s.vmin
    g["maximum"] = s.vmax

    # Write compressed histogram (compression ratio can be huge!)
    let Nbins = size(s.H, 1)
        chunks = (Nbins, 1)  # 1 chunk = 1 histogram (single loop size)
        g["hist", chunk=chunks, compress=6] = s.H
    end

    g
end
