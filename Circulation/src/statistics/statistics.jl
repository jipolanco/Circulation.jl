include("moments.jl")
include("histogram.jl")

module CirculationFields
# Different vector fields from which a circulation can be computed.
@enum VelocityLikeField Velocity RegVelocity Momentum
end

using .CirculationFields

"""
    CirculationStats{T}

Circulation statistics, including moments and histograms.
"""
struct CirculationStats{Loops, M<:Moments, H<:Histogram}
    Nr         :: Int    # number of loop sizes
    loop_sizes :: Loops  # length Nr
    resampling_factor :: Int
    moments    :: M
    histogram  :: H
end

"""
    CirculationStats(loop_sizes;
                     num_moments=20,
                     moments_Nfrac=nothing,
                     resampling_factor=1,
                     hist_edges=LinRange(-10, 10, 42))

Construct and initialise statistics.

# Parameters

- `resampling_factor`: if greater than one, the loaded ψ fields are resampled
  into a finer grid using padding in Fourier space.
  The number of grid points is increased by a factor `resampling_factor` in
  every direction.
"""
function CirculationStats(
        loop_sizes;
        num_moments=20,
        moments_Nfrac=nothing,
        resampling_factor=1,
        hist_edges=LinRange(-10, 10, 42),
    )
    resampling_factor >= 1 || error("resampling_factor must be positive")
    Nr = length(loop_sizes)
    M = Moments(num_moments, Nr, Float64; Nfrac=moments_Nfrac)
    H = Histogram(hist_edges, Nr, Int)
    CirculationStats(Nr, loop_sizes, resampling_factor, M, H)
end

Base.zero(s::CirculationStats) =
    CirculationStats(s.Nr, s.loop_sizes, s.resampling_factor, zero(s.moments),
                     zero(s.histogram))

# Dictionary of statistics.
# Useful for defining circulation statistics related to quantities such as
# velocity, regularised velocity and momentum.
const StatsDict = Dict{CirculationFields.VelocityLikeField,
                       <:CirculationStats}

Base.zero(s::SD) where {SD <: StatsDict} = SD(k => zero(v) for (k, v) in s)

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
    reduce!(stats::CirculationStats, v::AbstractVector{<:CirculationStats})

Reduce values from list of statistics.
"""
function reduce!(stats::CirculationStats, v::AbstractVector{<:CirculationStats})
    @assert all(stats.Nr .== getfield.(v, :Nr))
    @assert all(Ref(stats.loop_sizes) .== getfield.(v, :loop_sizes))
    reduce!(stats.moments, getfield.(v, :moments))
    reduce!(stats.histogram, getfield.(v, :histogram))
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

"""
    reset!(stats::CirculationStats)

Reset all statistics to zero.
"""
function reset!(stats::CirculationStats)
    reset!(stats.moments)
    reset!(stats.histogram)
    stats
end

finalise!(stats::StatsDict) = finalise!.(values(stats))
reset!(s::StatsDict) = reset!.(values(s))

function reduce!(dest::StatsDict, src::AbstractVector{<:StatsDict})
    @assert all(Ref(keys(dest)) .== keys.(src))
    for k in keys(dest)
        src_stats = getindex.(src, k) :: Vector{<:CirculationStats}
        reduce!(dest[k], src_stats)
    end
    dest
end

"""
    write(filename::AbstractString, stats::StatsDict)
    write(ff::Union{HDF5File,HDF5Group}, stats::StatsDict)

Save statistics to HDF5 file.
"""
Base.write(h5filename::AbstractString, stats::StatsDict) =
    h5open(ff -> write(ff, stats), h5filename, "w")

function Base.write(ff::Union{HDF5File,HDF5Group}, stats::StatsDict)
    for (k, v) in stats
        g = g_create(ff, string(k))
        write(g, v)
        close(g)
    end
    ff
end

function Base.write(g::Union{HDF5File,HDF5Group}, stats::CirculationStats)
    g["loop_sizes"] = collect(stats.loop_sizes)
    g["resampling_factor"] = stats.resampling_factor
    write(g_create(g, "Moments"), stats.moments)
    write(g_create(g, "Histogram"), stats.histogram)
    g
end

struct ParamsDataFile
    directory :: String
    index     :: Int

    function ParamsDataFile(dir, idx)
        isdir(dir) || throw(ArgumentError("directory not found: $dir"))
        new(dir, idx)
    end
end

"""
    init_statistics(loop_sizes; which, stats_args...)

Initialise statistics that can be passed to `analyse!`.

## Optional parameters

- `which`: list of quantities of type `CirculationFields.VelocityLikeField` to
  consider. The default is to consider all of them.

- `stats_args`: arguments passed to `CirculationStats`.
"""
function init_statistics(
        loop_sizes;
        which=(CirculationFields.Velocity,
               CirculationFields.RegVelocity,
               CirculationFields.Momentum),
        stats_args...
    )
    if length(unique(which)) != length(which)
        throw(ArgumentError("quantities may not be repeated: $which"))
    end
    Dict(w => CirculationStats(loop_sizes; stats_args...) for w in which)
end

"""
    analyse!(stats::Dict, gp::ParamsGP, data_dir_base;
             data_idx = 1, eps_vel = 0,
             max_slices = typemax(Int),
             to = TimerOutput())

Compute circulation statistics from all possible slices of a GP field.

The `stats` dictionary must contain one `CirculationStats` per quantity of
interest.
The keys of the dictionary are values of type `CirculationFields.VelocityLikeField`.
This dictionary may be generated by calling `init_statistics`.

## Optional parameters

- `data_idx`: "index" of data file. Needed to determine the input data
  filenames.

- `eps_vel`: optional correction applied to `ρ` before computing velocity.

- `max_slices`: maximum number of slices to consider along each direction.
  By default, all available slices are considered.
  This can be useful for testing, to make things faster.

- `to`: a TimerOutput object, useful for measuring performance.
"""
function analyse!(stats::StatsDict, gp::ParamsGP,
                  data_dir_base::AbstractString;
                  data_idx=1, kwargs...)
    data_params = ParamsDataFile(data_dir_base, data_idx)
    analyse!(stats, gp, data_params; kwargs...)
end

# Analyse for all slice orientations (in 3D: x, y and z).
function analyse!(stats::StatsDict, gp::ParamsGP,
                  data_params::ParamsDataFile; to=TimerOutput(), kwargs...)
    orientations = slice_orientations(gp)
    for or in orientations
        @timeit to "analyse!" analyse!(stats, or, gp, data_params;
                                       to=to, kwargs...)
    end
    finalise!(stats)
end

# Analyse for a single slice orientation.
function analyse!(stats::StatsDict, orientation::Val, gp::ParamsGP{D},
                  data_params::ParamsDataFile;
                  max_slices::Int = typemax(Int),
                  eps_vel = 0, to = TimerOutput()) where {D}
    Nslices = min(num_slices(gp.dims, orientation), max_slices) :: Int
    @assert Nslices >= 1
    @assert !isempty(stats)

    resampling_factor = let vals = values(stats)
        r = first(vals).resampling_factor
        @assert(all(getfield.(vals, :resampling_factor) .=== r),
                "all stats must have the same resampling factor")
        @assert r >= 1 "only upscaling is allowed"
        r
    end

    slices = 1:Nslices

    # Slice dimensions.
    # Example: in 3D, if orientation = 2, this is (1, 3).
    i, j = included_dimensions(Val(D), orientation)
    Lij = gp.L[i], gp.L[j]

    Nij_input = gp.dims[i], gp.dims[j]
    Nij_compute = Nij_input .* resampling_factor

    # Determine which statistics to compute.
    stats_keys = keys(stats)
    with_v = CirculationFields.Velocity in stats_keys
    with_vreg = CirculationFields.RegVelocity in stats_keys
    with_p = CirculationFields.Momentum in stats_keys

    Nth = Threads.nthreads()
    slices_per_thread = cld(Nslices, Nth)

    # Allocate arrays (one per thread).
    fields = [allocate_stats_fields(Nij_input, Nij_compute, Lij,
                                    with_v || with_vreg)
              for t = 1:Nth]
    stats_t = [zero(stats) for t = 1:Nth]

    let s = orientation_str(orientation)
        println(stderr)
        @info "Analysing slices $slices along $s..."
    end

    Threads.@threads for s in slices
        t = Threads.threadid()
        F = fields[t]

        if t == 1
            @info "  Thread 1: slice $s/$slices_per_thread"
            flush(stderr)
        end

        # Load ψ at selected slice.
        slice = make_slice(gp.dims, orientation, s)
        timer = t == 1 ? to : TimerOutput()

        analyse_slice!(
            stats_t[t], slice, gp, F, timer, data_params, eps_vel,
            resampling_factor,
            (with_p, with_vreg, with_v)
        )
    end

    @timeit to "reduce!" reduce!(stats, stats_t)

    stats
end

function allocate_stats_fields(Nij_input, Nij_resampled, Lij, with_v)
    FFTW.set_num_threads(1)  # make sure that plans are not threaded!
    ψ_in = Array{ComplexF64}(undef, Nij_input...)
    ψ = if Nij_input === Nij_resampled
        ψ_in  # in this case, the two point to the same data!
    else
        similar(ψ_in, Nij_resampled)
    end
    ρ = similar(ψ, Float64)
    ps = (similar(ρ), similar(ρ))  # momentum
    (
        ψ_in = ψ_in,
        ψ = ψ,
        ψ_buf = similar(ψ),
        ρ = ρ,
        Γ = similar(ρ),
        ps = ps,
        vs = with_v ? similar.(ps) : nothing,
        I = IntegralField2D(ps[1], L=Lij),
        # FFTW plans for computing momentum
        fft_plans_p = GPFields.create_fft_plans_1d!(ψ),
    )
end

function analyse_slice!(
        stats, slice, gp, F, to, data_params,
        eps_vel, resampling_factor,
        (with_p, with_vreg, with_v),
    )
    @assert all(size(F.ψ) .÷ size(F.ψ_in) .== resampling_factor)
    @timeit to "load_psi!" GPFields.load_psi!(
        F.ψ_in, gp, data_params.directory, data_params.index, slice=slice)

    # Create parameters associated to original and resampled slices.
    gp_slice_in = ParamsGP(gp, slice)

    # Note that nξ changes in the resampled field!
    nxi_resample = gp_slice_in.nξ * resampling_factor
    gp_slice = ParamsGP(gp_slice_in, dims=size(F.ψ), nxi=nxi_resample)

    @timeit to "resample ψ" begin
        # Note: resample_field_fourier! works with fields in Fourier space.
        fft!(F.ψ_in)
        GPFields.resample_field_fourier!(F.ψ, F.ψ_in, gp_slice_in)
        ifft!(F.ψ)
    end

    @timeit to "compute_density!" GPFields.compute_density!(F.ρ, F.ψ)

    @timeit to "compute_momentum!" GPFields.compute_momentum!(
        F.ps, F.ψ, gp_slice, buf=F.ψ_buf, fft_plans=F.fft_plans_p)

    if with_p
        compute!(stats[CirculationFields.Momentum], F.Γ, F.I, F.ps, to)
    end

    if with_vreg
        @timeit to "compute_vreg!" map((v, p) -> v .= p ./ sqrt.(F.ρ),
                                       F.vs, F.ps)
        compute!(stats[CirculationFields.RegVelocity], F.Γ, F.I, F.vs, to)
    end

    if with_v
        @timeit to "compute_vel!" map((v, p) -> v .= p ./ (F.ρ .+ eps_vel),
                                      F.vs, F.ps)
        compute!(stats[CirculationFields.Velocity], F.Γ, F.I, F.vs, to)
    end

    nothing
end

function compute!(stats::CirculationStats, Γ, Ip, vs, to)
    # Set integral values with momentum data.
    @timeit to "prepare!" prepare!(Ip, vs)

    for (r_ind, r) in enumerate(stats.loop_sizes)
        s = stats.resampling_factor * r  # loop size in resampled field
        @timeit to "circulation!" circulation!(Γ, Ip, (s, s))
        @timeit to "statistics" update!(stats, Γ, r_ind; to=to)
    end

    stats
end

function included_dimensions(::Val{N}, ::Val{s}) where {N,s}
    inds = findall(!=(s), ntuple(identity, Val(N)))  # all dimensions != s
    @assert length(inds) == 2
    inds[1], inds[2]
end

orientation_str(::Val{s}) where {s} = "xyz"[s]

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
