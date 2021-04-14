export ParamsHistogram2D, Histogram2D

struct ParamsHistogram2D{
        T,
        Fields <: Tuple{Vararg{AbstractScalarField,2}},
        Edges  <: Tuple{Vararg{AbstractVector,2}},
    } <: BaseStatsParams

    fields    :: Fields
    bin_edges :: Edges

    ParamsHistogram2D(::Type{T}, fields; bin_edges) where {T} =
        new{T, typeof(fields), typeof(bin_edges)}(fields, bin_edges)
end

ParamsHistogram2D(fields; kws...) = ParamsHistogram2D(Int64, fields; kws...)

init_statistics(p::ParamsHistogram2D, etc...) = Histogram2D(p, etc...)

struct Histogram2D{
        T, Tb,
        BinType <: Tuple{Vararg{AbstractVector,2}},
        Params <: ParamsHistogram2D,
    } <: AbstractBaseStats
    params :: Params
    finalised :: Base.RefValue{Bool}
    Nr    :: Int          # number of "columns" of data (e.g. one per loop size)
    Nbins :: Dims{2}      # number of bins (Nx, Ny)
    bin_edges :: BinType  # sorted lists of bin edges [Nbins + 1]
    H :: Array{T,3}       # histogram [Nx, Ny, Nr]

    vmin :: Matrix{Tb}  # minimum sampled value for each variable [2, Nr]
    vmax :: Matrix{Tb}  # maximum sampled value for each variable [2, Nr]

    # Number of samples per column (Nr).
    # This includes outliers, i.e. events falling outside of the histogram.
    Nsamples :: Vector{Int}

    function Histogram2D(p::ParamsHistogram2D{T}, Nr::Integer) where {T}
        edges = p.bin_edges
        Nbins = length.(edges) .- 1
        Nsamples = zeros(Int, Nr)
        BinType = typeof(edges)
        H = zeros(T, Nbins..., Nr)
        Tb = promote_type(eltype.(edges)...)
        vmin = zeros(Tb, length(edges), Nr)
        vmax = zeros(Tb, length(edges), Nr)
        new{T, Tb, BinType, typeof(p)}(
            p, Ref(false), Nr, Nbins, edges, H, vmin, vmax, Nsamples,
        )
    end
end

Base.eltype(::Type{<:Histogram2D{T}}) where {T} = T
Base.zero(s::Histogram2D) = Histogram2D(s.params, s.Nr)

function update!(s::Histogram2D, fields::NamedTuple, r)
    u, v = getfields(s, fields)
    s
end

function reduce!(s::Histogram2D, v)
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

function finalise!(s::Histogram2D)
    @assert !was_finalised(s)
    s.finalised[] = true
    s  # nothing to do
end

function reset!(s::Histogram2D)
    s.finalised[] = false
    fill!(s.H, 0)
    fill!(s.Nsamples, 0)
    fill!(s.vmin, 0)
    fill!(s.vmax, 0)
    s
end

function Base.write(g, s::Histogram2D)
    @assert was_finalised(s)
    # TODO
    g
end
