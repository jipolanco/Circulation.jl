export ParamsHistogram2D, Histogram2D

struct ParamsHistogram2D{
        T,
        Fields <: Tuple{Vararg{AbstractScalarField,2}},
        Edges  <: Tuple{Vararg{AbstractVector,2}},
    } <: BaseStatsParams

    label     :: String
    fields    :: Fields
    bin_edges :: Edges
    merge_scales :: Bool  # merge scale-wise statistics?

    function ParamsHistogram2D(
            ::Type{T}, fields;
            bin_edges, merge_scales = false, label = "Histogram2D",
        ) where {T}
        new{T, typeof(fields), typeof(bin_edges)}(
            label, fields, bin_edges, merge_scales,
        )
    end
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
        if p.merge_scales
            Nr = one(Nr)
        end
        edges = p.bin_edges
        Nfields = length(edges)
        @assert Nfields == 2
        Nbins = length.(edges) .- 1
        Nsamples = zeros(Int, Nr)
        BinType = typeof(edges)
        H = zeros(T, Nbins..., Nr)
        Tb = promote_type(eltype.(edges)...)
        vmin = zeros(Tb, Nfields, Nr)
        vmax = zeros(Tb, Nfields, Nr)
        new{T, Tb, BinType, typeof(p)}(
            p, Ref(false), Nr, Nbins, edges, H, vmin, vmax, Nsamples,
        )
    end
end

Base.eltype(::Type{<:Histogram2D{T}}) where {T} = T
Base.zero(s::Histogram2D) = Histogram2D(s.params, s.Nr)

function update!(s::Histogram2D, fields::NamedTuple, r)
    if s.params.merge_scales
        @assert s.Nr == 1
        r = one(r)
    end

    fs = getfields(s, fields)  # for instance Γ and ε
    us, vs = fs
    @assert 1 <= r <= s.Nr
    @assert length(us) == length(vs)
    Nes = map(length, s.bin_edges)

    s.Nsamples[r] += length(us)

    vmins = @view s.vmin[:, r]
    vmaxs = @view s.vmax[:, r]

    @inbounds for n in eachindex(us)
        uv = getindex.(fs, n)
        for l ∈ eachindex(uv)
            vmins[l] = min(vmins[l], uv[l])
            vmaxs[l] = max(vmins[l], uv[l])
        end
        is = map(searchsortedlast, s.bin_edges, uv)
        within_bounds = all(map((i, Ne) -> i ∉ (0, Ne), is, Nes))
        if within_bounds
            s.H[is..., r] += 1
        end
    end

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
    merge_scales = s.params.merge_scales

    for i ∈ eachindex(s.bin_edges)
        g["bin_edges$i"] = collect(s.bin_edges[i])
    end

    g["total_samples"] = s.Nsamples
    g["merged_scales"] = merge_scales

    g["minimum"] = s.vmin
    g["maximum"] = s.vmax

    Nfields = length(s.bin_edges)

    # Write compressed histogram (compression ratio can be huge!)
    hist = s.H
    _write_histogram_chunked(g, hist, "hist")

    g
end

function _write_histogram_chunked(g, hist, name)
    s = size(hist)
    N = ndims(hist)
    chunk = ntuple(i -> i == N ? 1 : s[i], N)  # chunk along last dimension
    g[name, chunk = chunk, compress = 6] = hist
end
