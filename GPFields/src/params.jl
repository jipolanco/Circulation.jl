"""
    ParamsGP{D}

Parameters of D-dimensional GP data.
"""
struct ParamsGP{D}  # D: dimension (2 or 3)
    dims :: NTuple{D,Int}      # (Nx, Ny, Nz)
    L    :: NTuple{D,Float64}  # (Lx, Ly, Lz)
    c    :: Float64            # speed of sound
    nξ   :: Float64
    ξ    :: Float64            # healing length
    κ    :: Float64            # quantum of circulation
end

Base.ndims(::ParamsGP{D}) where {D} = D

"""
    ParamsGP(dims::Dims{D}; L = (2π, ...), c = 1.0, nxi = 1.0)

Construct GP data parameters.
"""
function ParamsGP(
        dims::Dims{D};
        L::NTuple{D}=ntuple(d -> 2π, D),
        c=1.0,
        nxi=1.0,
    ) where {D}
    @assert D in (2, 3)

    Lx = L[1]
    Nx = dims[1]
    ξ = Lx * nxi / Nx
    κ = Lx * sqrt(2) * c * ξ

    ParamsGP{D}(dims, L, c, nxi, ξ, κ)
end

"""
    ParamsGP(p::ParamsGP, slice)

Construct parameters associated to slice of GP field.
"""
function ParamsGP(p::ParamsGP{D}, slice::Slice{D}) where {D}
    idims = slice_dimensions(slice)
    Ns = getindex.(Ref(p.dims), idims)
    Ls = getindex.(Ref(p.L), idims)
    ParamsGP(Ns; L=Ls, c=p.c, nxi=p.nξ)
end

get_coordinates(g::ParamsGP) = map(g.dims, g.L) do N, L
    LinRange(0, L, N + 1)[1:N]
end

get_wavenumbers(g::ParamsGP) = map(g.dims, g.L) do N, L
    sampling_freq = 2pi * N / L  # = 2π / Δx
    fftfreq(N, sampling_freq)
end
