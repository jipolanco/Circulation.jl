"""
    ParamsGP{D}

Parameters of D-dimensional GP data.
"""
struct ParamsGP{D}  # D: dimension
    dims :: NTuple{D,Int}      # (Nx, Ny, Nz)
    L    :: NTuple{D,Float64}  # (Lx, Ly, Lz)
    c    :: Float64            # speed of sound
    nξ   :: Float64
    ξ    :: Float64            # healing length
    κ    :: Float64            # quantum of circulation
end

Base.size(p::ParamsGP) = p.dims
Base.ndims(::ParamsGP{D}) where {D} = D

"""
    ParamsGP(dims::Dims{D}; L, c, nxi)

Construct GP data parameters.

The domain length should be given as a tuple of length `D`.
For instance, for a cubic box of size `2π`, `L = (2pi, 2pi, 2pi)`.
"""
function ParamsGP(dims::Dims{D}; L::NTuple{D}, c, nxi) where {D}
    @assert D >= 1

    Lx = L[1]
    Nx = dims[1]
    ξ = Lx * nxi / Nx
    κ = Lx * sqrt(2) * c * ξ

    ParamsGP{D}(dims, L, c, nxi, ξ, κ)
end

"""
    ParamsGP(p::ParamsGP, slice; dims=p.dims, L=p.L)

Construct parameters associated to slice of GP field.
"""
function ParamsGP(p::ParamsGP{D}, slice::Slice{D};
                  dims=p.dims, L=p.L) where {D}
    idims = slice_dimensions(slice)
    Ns = getindex.(Ref(dims), idims)
    Ls = getindex.(Ref(L), idims)
    ParamsGP(Ns; L=Ls, c=p.c, nxi=p.nξ)
end

"""
    ParamsGP(p::ParamsGP; dims=p.dims, L=p.L, c=p.c, nxi=p.nξ)

Copy `ParamsGP`, optionally modifying some parameters.
"""
ParamsGP(p::ParamsGP; dims=p.dims, L=p.L, c=p.c, nxi=p.nξ) =
    ParamsGP(dims, L=L, c=c, nxi=nxi)

function get_coordinates(g::ParamsGP, i::Integer)
    N = g.dims[i]
    L = g.L[i]
    LinRange(0, L, N + 1)[1:N]
end

get_coordinates(g::ParamsGP) = ntuple(d -> get_coordinates(g, d), Val(ndims(g)))

function get_wavenumbers(g::ParamsGP, i::Integer)
    N = g.dims[i]
    L = g.L[i]
    sampling_freq = 2pi * N / L  # = 2π / Δx
    fftfreq(N, sampling_freq)
end

get_wavenumbers(g::ParamsGP) = ntuple(d -> get_wavenumbers(g, d), Val(ndims(g)))

"""
    write(g::Union{HDF5File,HDF5Group}, p::ParamsGP)

Write GP parameters to HDF5 file.
"""
function Base.write(g::Union{HDF5File,HDF5Group}, p::ParamsGP)
    g["dims"] = collect(p.dims)
    g["L"] = collect(p.L)
    g["c"] = p.c
    g["nxi"] = p.nξ
    g["xi"] = p.ξ
    g["kappa"] = p.κ
    g
end
