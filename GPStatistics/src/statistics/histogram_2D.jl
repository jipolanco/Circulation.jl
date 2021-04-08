export ParamsHistogram2D, Histogram2D

Base.@kwdef struct ParamsHistogram2D{Edges1, Edges2} <: BaseStatsParams
    bin_edges1 :: Edges1
    bin_edges2 :: Edges2
end

init_statistics(p::ParamsHistogram2D, etc...) = Histogram2D(p, etc...)

struct Histogram2D{
        T, Tb,
        BinType <: Tuple{Vararg{AbstractVector,2}},
    } <: AbstractBaseStats
    finalised :: Base.RefValue{Bool}
    Nr    :: Int          # number of "columns" of data (e.g. one per loop size)
    Nbins :: Dims{2}      # number of bins (Nx, Ny)
    bin_edges :: BinType  # sorted lists of bin edges [Nbins + 1]
    H :: Array{T,3}       # histogram [Nx, Ny, Nr]

    vmin :: Vector{NTuple{2,Tb}}  # minimum sampled value for each variable [Nr]
    vmax :: Vector{NTuple{2,Tb}}  # maximum sampled value for each variable [Nr]

    # Number of samples per column (Nr).
    # This includes outliers, i.e. events falling outside of the histogram.
    Nsamples :: Vector{Int}

    function Histogram2D(p::ParamsHistogram2D, Nr::Integer, ::Type{T} = Int) where {T}
        edges = (p.bin_edges1, p.bin_edges2)
        Nbins = length.(edges) .- 1
        Nsamples = zeros(Int, Nr)
        BinType = typeof(edges)
        H = zeros(T, Nbins..., Nr)
        Tb = promote_type(eltype.(edges)...)
        vmin = zeros(Tb, Nr)
        vmax = zeros(Tb, Nr)
        new{T, Tb, BinType}(
            Ref(false), Nr, Nbins, edges, H, vmin, vmax, Nsamples,
        )
    end
end
