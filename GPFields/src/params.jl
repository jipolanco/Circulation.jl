struct ParamsGP{D}  # D: dimension (2 or 3)
    dims :: NTuple{D,Int}      # (Nx, Ny, Nz)
    L    :: NTuple{D,Float64}  # (Lx, Ly, Lz)
    c    :: Float64            # speed of sound
    ξ    :: Float64            # healing length
    κ    :: Float64            # quantum of circulation
end

Base.ndims(::ParamsGP{D}) where {D} = D

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

    ParamsGP{D}(dims, L, c, ξ, κ)
end

get_coordinates(g::ParamsGP) = map(g.dims, g.L) do N, L
    LinRange(0, L, N + 1)[1:N]
end

get_wavenumbers(g::ParamsGP) = map(g.dims, g.L) do N, L
    # TODO verify that this works for L != 2pi
    sampling_freq = 2pi * N / L  # = 2π / Δx
    fftfreq(N, sampling_freq)
end
