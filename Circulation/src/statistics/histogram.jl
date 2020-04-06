"""
    Histogram{T}

Histogram of a scalar quantity.
"""
struct Histogram{T, BinType<:AbstractVector}
    Nr    :: Int          # number of "columns" of data (e.g. one per loop size)
    Nbins :: Int          # number of bins
    bin_edges :: BinType  # sorted list of bin edges (length Nbins + 1)
    H :: Matrix{T}        # histogram (Nbins, Nr)

    # Number of samples per column (Nr).
    # This includes outliers, i.e. events falling outside of the histogram.
    Nsamples :: Vector{Int}

    function Histogram(bin_edges::AbstractVector, Nr::Integer,
                       ::Type{T} = Int) where {T}
        Nbins = length(bin_edges) - 1
        H = zeros(T, Nbins, Nr)
        BinType = typeof(bin_edges)
        Nsamples = zeros(Int, Nr)
        new{T, BinType}(Nr, Nbins, bin_edges, H, Nsamples)
    end
end

Base.eltype(::Type{<:Histogram{T}}) where {T} = T
Base.zero(s::Histogram) = Histogram(s.bin_edges, s.Nr, eltype(s))

function update!(s::Histogram, Γ, r)
    @assert 1 <= r <= s.Nr

    s.Nsamples[r] += length(Γ)
    Ne = length(s.bin_edges)

    @inbounds for v in Γ
        i = searchsortedlast(s.bin_edges, v)
        if i ∉ (0, Ne)
            s.H[i, r] += 1
        end
    end

    s
end

function reduce!(s::Histogram, v::AbstractVector{<:Histogram})
    for src in v
        @assert s.Nr == src.Nr
        @assert s.Nbins == src.Nbins
        @assert s.bin_edges == src.bin_edges
        s.H .+= src.H
        s.Nsamples .+= src.Nsamples
    end
    s
end

finalise!(s::Histogram) = s  # nothing to do

function reset!(s::Histogram)
    fill!(s.H, 0)
    fill!(s.Nsamples, 0)
    s
end

function Base.write(g, s::Histogram)
    g["bin_edges"] = collect(s.bin_edges)
    g["hist"] = s.H
    g["total_samples"] = s.Nsamples
    g
end
