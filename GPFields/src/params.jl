"""
    ParamsGP{D}

Parameters of D-dimensional GP data.

---

    ParamsGP(dims::Dims{D}; L, c, nxi = nothing, ξ = nothing)

Construct GP data parameters.

The domain length should be given as a tuple of length `D`.
For instance, for a cubic box of size `2π`, `L = (2pi, 2pi, 2pi)`.

Either `nxi` or `ξ` must be given as a keyword argument.
"""
struct ParamsGP{D}  # D: dimension
    dims :: NTuple{D,Int}      # (Nx, Ny, Nz)
    L    :: NTuple{D,Float64}  # (Lx, Ly, Lz)
    dx   :: NTuple{D,Float64}  # (dx, dy, dz)
    c    :: Float64            # speed of sound
    nξ   :: Float64
    ξ    :: Float64            # healing length
    κ    :: Float64            # quantum of circulation

    function ParamsGP(
            dims::Dims{D}; L::NTuple{D}, c, nxi = nothing, ξ = nothing,
        ) where {D}
        @assert D >= 1

        Lx = L[1]
        Nx = dims[1]

        if ξ === nothing && nxi === nothing
            throw(ArgumentError("either `ξ` or `nxi` must be given"))
        elseif ξ !== nothing && nxi !== nothing
            throw(ArgumentError("cannot pass both `ξ` and `nxi`"))
        elseif nxi !== nothing
            nxi_out = nxi
            ξ_out = Lx * nxi / Nx
        elseif ξ !== nothing
            ξ_out = ξ
            nxi_out = ξ_out * Nx / Lx
        end

        dx = L ./ dims
        κ = Lx * sqrt(2) * c * ξ_out

        new{D}(dims, L, dx, c, nxi_out, ξ_out, κ)
    end
end

Base.size(p::ParamsGP) = p.dims
Base.ndims(::ParamsGP{D}) where {D} = D

function Base.show(io::IO, p::ParamsGP)
    print(io,
        """
        Gross–Pitaevskii parameters
          - Field resolution:  $(p.dims)
          - Domain dimensions: $(p.L)
          - c = $(p.c)
          - ξ = $(p.ξ)
          - κ = $(p.κ)""")
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
    ParamsGP(Ns; L=Ls, c=p.c, ξ=p.ξ)
end

"""
    ParamsGP(p::ParamsGP; dims=p.dims, L=p.L)

Copy `ParamsGP`, optionally modifying some parameters.

Physical parameters such as `ξ` and `c` stay the same.
"""
ParamsGP(p::ParamsGP; dims=p.dims, L=p.L) = ParamsGP(dims, L=L, c=p.c, ξ=p.ξ)

function get_coordinates(g::ParamsGP, i::Integer)
    N = g.dims[i]
    L = g.L[i]
    LinRange(0, L, N + 1)[1:N]
end

get_coordinates(g::ParamsGP) = ntuple(d -> get_coordinates(g, d), Val(ndims(g)))

function get_wavenumbers(f::Function, g::ParamsGP, i::Integer)
    N = g.dims[i]
    L = g.L[i]
    sampling_freq = 2pi * N / L  # = 2π / Δx
    f(N, sampling_freq)
end

# Wave numbers for complex-to-complex transform.
get_wavenumbers(g::ParamsGP, ::Val{:c2c}) =
    ntuple(d -> get_wavenumbers(fftfreq, g, d), Val(ndims(g)))

# Wave numbers for real-to-complex transform.
get_wavenumbers(g::ParamsGP, ::Val{:r2c}) =
    (
        get_wavenumbers(rfftfreq, g, 1),
        ntuple(d -> get_wavenumbers(fftfreq, g, d + 1), Val(ndims(g) - 1))...,
    )

get_wavenumbers(g::ParamsGP, i::Integer) = get_wavenumbers(fftfreq, g, i)
get_wavenumbers(g::ParamsGP) = get_wavenumbers(g, Val(:c2c))

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
