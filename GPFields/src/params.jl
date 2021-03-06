"""
    PhysicalParamsGP

Contains parameters specific to the GP model, such as the speed of sound ``c``
or the quantum of circulation ``κ``.
"""
struct PhysicalParamsGP
    c  :: Float64            # speed of sound
    nξ :: Float64
    ξ  :: Float64            # healing length
    κ  :: Float64            # quantum of circulation
end

function PhysicalParamsGP(Lx::Real, Nx::Integer; c::Real, nxi = nothing, ξ = nothing)
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
    κ = Lx * sqrt(2) * c * ξ_out
    PhysicalParamsGP(c, nxi_out, ξ_out, κ)
end

function print_params(io::IO, p::PhysicalParamsGP)
    print(io,
        """
          - c = $(p.c)
          - ξ = $(p.ξ)
          - κ = $(p.κ)"""
    )
end

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
struct ParamsGP{D, Phys <: Union{Nothing, PhysicalParamsGP}}  # D: dimension
    dims :: NTuple{D,Int}      # (Nx, Ny, Nz)
    L    :: NTuple{D,Float64}  # (Lx, Ly, Lz)
    dx   :: NTuple{D,Float64}  # (dx, dy, dz)
    phys :: Phys
end

function ParamsGP(dims::Dims{D}; L::NTuple{D}, kwargs...) where {D}
    @assert D >= 1
    phys = if isempty(kwargs)
        nothing
    else
        PhysicalParamsGP(L[1], dims[1]; kwargs...)
    end
    dx = L ./ dims
    ParamsGP(dims, L, dx, phys)
end

Base.size(p::ParamsGP) = p.dims
Base.size(p::ParamsGP, i) = size(p)[i]
Base.ndims(::ParamsGP{D}) where {D} = D

box_size(p::ParamsGP) = p.L

function Base.show(io::IO, p::ParamsGP)
    print(io,
        """
        Field parameters
          - Field resolution:  $(p.dims)
          - Domain dimensions: $(p.L)"""
    )
    if p.phys !== nothing
        print_params(io, p.phys)
    end
    nothing
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
    dx = getindex.(Ref(p.dx), idims)
    ParamsGP(Ns, Ls, dx, p.phys)
end

"""
    ParamsGP(p::ParamsGP; dims=p.dims, L=p.L)

Copy `ParamsGP`, optionally modifying some parameters.

Physical parameters such as `ξ` and `c` stay the same.
"""
function ParamsGP(p::ParamsGP; dims=p.dims, L=p.L)
    q = p.phys
    if q === nothing
        ParamsGP(dims; L)
    else
        ParamsGP(dims; L, c = q.c, ξ = q.ξ)
    end
end

"""
    coordinates(p::ParamsGP, [dim])

Get physical coordinates `(x, y, ...)` of domain.

Note that the endpoint of the periodic domain is also included.
"""
function coordinates(g::ParamsGP, i::Integer)
    N = g.dims[i]
    L = g.L[i]
    range(0, L, length = N + 1)
end

coordinates(g::ParamsGP) = ntuple(d -> coordinates(g, d), Val(ndims(g)))

@deprecate get_coordinates coordinates

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
    write(g::Union{HDF5.File,HDF5.Group}, p::ParamsGP)

Write GP parameters to HDF5 file.
"""
function Base.write(g::Union{HDF5.File,HDF5.Group}, p::ParamsGP)
    g["dims"] = collect(p.dims)
    g["L"] = collect(p.L)
    let p = p.phys
        if p !== nothing
            g["c"] = p.c
            g["nxi"] = p.nξ
            g["xi"] = p.ξ
            g["kappa"] = p.κ
        end
    end
    g
end
