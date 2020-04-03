include("moments.jl")
include("histogram.jl")

"""
    CirculationStats{T}

Circulation statistics, including moments and histograms.
"""
struct CirculationStats{Loops, M<:Moments, H<:Histogram}
    Nr         :: Int    # number of loop sizes
    loop_sizes :: Loops  # length Nr
    moments    :: M
    histogram  :: H
end

"""
    CirculationStats(loop_sizes;
                     num_moments=20,
                     hist_edges=LinRange(-10, 10, 42))

Construct and initialise statistics.
"""
function CirculationStats(
        loop_sizes;
        num_moments=20,
        hist_edges=LinRange(-10, 10, 42),
    )
    Nr = length(loop_sizes)
    M = Moments(num_moments, Nr, Float64)
    H = Histogram(hist_edges, Nr, Int)
    CirculationStats(Nr, loop_sizes, M, H)
end

"""
    update!(stats::CirculationStats, Γ, r_ind; to=TimerOutput())

Update circulation data associated to a given loop size (with index `r_ind`)
using circulation data.
"""
function update!(stats::CirculationStats, Γ, r_ind; to=TimerOutput())
    @assert r_ind ∈ 1:stats.Nr
    @timeit to "moments"   update!(stats.moments, Γ, r_ind)
    @timeit to "histogram" update!(stats.histogram, Γ, r_ind)
    stats
end

"""
    finalise!(stats::CirculationStats)

Compute final statistics from collected data.

For instance, moment data is divided by the number of samples to obtain the
actual moments.
"""
function finalise!(stats::CirculationStats)
    finalise!(stats.moments)
    finalise!(stats.histogram)
    stats
end

struct ParamsDataFile
    directory :: String
    index     :: Int

    function ParamsDataFile(dir_base, resolution, idx)
        dir = joinpath(dir_base, string(resolution))
        isdir(dir) || throw(ArgumentError("directory not found: $dir"))
        new(dir, idx)
    end
end

"""
    analyse!(stats::CirculationStats, gp::ParamsGP, data_dir_base;
             data_idx=0, eps_vel=0, to=TimerOutput())

Compute circulation statistics from all possible slices of a GP field.
"""
function analyse!(stats::CirculationStats, gp::ParamsGP,
                  data_dir_base::AbstractString;
                  data_idx=1, kwargs...)
    Ns = gp.dims
    data_params = ParamsDataFile(data_dir_base, Ns[1], data_idx)
    analyse!(stats, gp, data_params; kwargs...)
end

function analyse!(stats::CirculationStats, gp::ParamsGP,
                  data_params::ParamsDataFile; to=TimerOutput(), kwargs...)
    orientations = slice_orientations(gp)
    for or in orientations
        @timeit to "analyse!" analyse!(stats, or, gp, data_params;
                                       to=to, kwargs...)
    end
    finalise!(stats)
end

function analyse!(stats::CirculationStats, orientation::Val, gp::ParamsGP{D},
                  data_params::ParamsDataFile;
                  eps_vel=0, to=TimerOutput()) where {D}
    Nslices = num_slices(gp.dims, orientation)
    @assert Nslices >= 1

    # Slice dimensions.
    # Example: in 3D, if orientation = 2, this is (1, 3).
    i, j = included_dimensions(Val(D), orientation)
    Ni, Nj = gp.dims[i], gp.dims[j]
    Li, Lj = gp.L[i], gp.L[j]

    # Allocate arrays.
    psi = Array{ComplexF64}(undef, Ni, Nj)
    psi_buf = similar(psi)     # buffer for computation of momentum
    ρ = similar(psi, Float64)  # density
    Γ = similar(ρ)             # circulation
    vs = ntuple(_ -> similar(ρ), 2)  # velocity / momentum

    # Allocate integral field for circulation using FFTs.
    Ip = IntegralField2D(vs[1], L=(Li, Lj))

    # TODO
    # - circulation for vreg and v

    for s = 1:Nslices
        # Load ψ at selected slice.
        slice = make_slice(gp.dims, orientation, s)

        @timeit to "load_psi!" GPFields.load_psi!(
            psi, gp, data_params.directory, data_params.index, slice=slice)

        @timeit to "compute_density!" GPFields.compute_density!(ρ, psi)

        # Create parameters associated to slice.
        # This is needed to compute the momentum.
        gp_slice = ParamsGP(gp, slice)

        @timeit to "compute_momentum!" GPFields.compute_momentum!(
            vs, psi, gp_slice, buf=psi_buf)

        # Set integral values with momentum data.
        @timeit to "prepare!" prepare!(Ip, vs)

        for (r_ind, r) in enumerate(stats.loop_sizes)
            @timeit to "circulation!" circulation!(Γ, Ip, (r, r))
            @timeit to "statistics" update!(stats, Γ, r_ind; to=to)
        end
    end

    stats
end

function included_dimensions(::Val{N}, ::Val{s}) where {N,s}
    inds = findall(!=(s), ntuple(identity, Val(N)))  # all dimensions != s
    @assert length(inds) == 2
    inds[1], inds[2]
end

slice_orientations(::ParamsGP{2}) = (Val(3), )       # 2D data -> single z-slice
slice_orientations(::ParamsGP{3}) = Val.((1, 2, 3))  # 3D data

num_slices(::Dims{2}, ::Val{3}) = 1
num_slices(dims::Dims{3}, ::Val{s}) where {s} = dims[s]

function make_slice(dims::Dims{2}, ::Val{3}, i)
    @assert i == 1
    (:, :)
end

function make_slice(dims::Dims{3}, ::Val{s}, i) where {s}
    @assert 1 <= i <= dims[s]
    ntuple(d -> d == s ? i : Colon(), Val(3))
end
