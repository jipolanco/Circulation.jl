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

function update!(s::Histogram, Γ, r)
    @assert 1 <= r <= s.Nr

    s.Nsamples[r] += length(Γ)
    Ne = length(s.bin_edges)

    # TODO this may be faster if we sort Γ first.
    # In that case we can avoid using searchsortedlast.
    @inbounds for v in Γ
        i = searchsortedlast(s.bin_edges, v)
        if i ∉ (0, Ne)
            s.H[i] += 1
        end
    end

    s
end

finalise!(s::Histogram) = s  # nothing to do
