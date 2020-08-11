module VelocityLikeFields
# Different velocity-like vector fields from which a circulation can be computed.
@enum VelocityLikeField Velocity RegVelocity Momentum
end

using .VelocityLikeFields

abstract type AbstractBaseStats end
include("moments.jl")
include("histogram.jl")

abstract type AbstractFlowStats end
include("circulation.jl")
include("velocity_increments.jl")

# Dictionary of statistics.
# Useful for defining flow statistics related to quantities such as velocity,
# regularised velocity and momentum.
const StatsDict{S} =
    Dict{VelocityLikeFields.VelocityLikeField, S} where {S <: AbstractFlowStats}

Broadcast.broadcastable(s::AbstractBaseStats) = Ref(s)

Base.zero(s::S) where {S <: AbstractFlowStats} =
    S(s.Nr, increments(s), s.resampling_factor, s.resampled_grid,
      zero(s.moments), zero(s.histogram))

Base.zero(s::SD) where {SD <: StatsDict} = SD(k => zero(v) for (k, v) in s)

Base.zero(s::Tuple{Vararg{S}}) where {S <: AbstractBaseStats} = zero.(s)

"""
    update!(stats::AbstractFlowStats, Γ, r_ind; to=TimerOutput())

Update circulation data associated to a given loop size (with index `r_ind`)
using circulation data.
"""
function update!(stats::AbstractFlowStats, Γ, r_ind; to=TimerOutput())
    @assert r_ind ∈ 1:stats.Nr
    for name in statistics(stats)
        @timeit to string(name) update!(getfield(stats, name), Γ, r_ind)
    end
    stats
end

update!(s::Tuple, Γ::Tuple, args...) = map((a, b) -> update!(a, b, args...), s, Γ)

"""
    reduce!(stats::AbstractFlowStats, v::AbstractVector{<:AbstractFlowStats})

Reduce values from list of statistics.
"""
function reduce!(stats::S, v::AbstractVector{<:S}) where {S <: AbstractFlowStats}
    @assert all(stats.Nr .== getfield.(v, :Nr))
    for name in statistics(stats)
        reduce!(getfield(stats, name), getfield.(v, name))
    end
    stats
end

function reduce!(s::NTuple{N,S},
                 inputs::AbstractVector{NTuple{N,S}}) where {N, S <: AbstractBaseStats}
    for n in eachindex(s)
        reduce!(s[n], (v[n] for v in inputs))
    end
    s
end

"""
    finalise!(stats::AbstractFlowStats)

Compute final statistics from collected data.

For instance, moment data is divided by the number of samples to obtain the
actual moments.
"""
function finalise!(stats::AbstractFlowStats)
    for name in statistics(stats)
        finalise!.(getfield(stats, name))
    end
    stats
end

"""
    reset!(stats::AbstractFlowStats)

Reset all statistics to zero.
"""
function reset!(stats::AbstractFlowStats)
    for name in statistics(stats)
        reset!.(getfield(stats, name))
    end
    stats
end

finalise!(stats::StatsDict) = finalise!.(values(stats))
reset!(s::StatsDict) = reset!.(values(s))

function reduce!(dest::StatsDict, src::AbstractVector{<:StatsDict})
    @assert all(Ref(keys(dest)) .== keys.(src))
    for k in keys(dest)
        src_stats = getindex.(src, k) :: Vector{<:AbstractFlowStats}
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

struct ParamsDataFile
    directory :: String
    index     :: Int

    function ParamsDataFile(dir, idx)
        isdir(dir) || throw(ArgumentError("directory not found: $dir"))
        new(dir, idx)
    end
end

"""
    init_statistics(S, loop_sizes; which, stats_args...)

Initialise statistics that can be passed to `analyse!`.

`S` must be the type of statistics (for now, only `CirculationStats` is
supported).

## Optional parameters

- `which`: list of quantities of type `VelocityLikeFields.VelocityLikeField` to
  consider. The default is to consider all of them.

- `stats_args`: arguments passed to `CirculationStats`.
"""
function init_statistics(
        ::Type{S},
        args...;
        which=(VelocityLikeFields.Velocity,
               VelocityLikeFields.RegVelocity,
               VelocityLikeFields.Momentum),
        stats_args...
    ) where {S <: AbstractFlowStats}
    if length(unique(which)) != length(which)
        throw(ArgumentError("quantities may not be repeated: $which"))
    end
    Dict(w => S(args...; stats_args...) for w in which)
end

"""
    analyse!(stats::Dict, gp::ParamsGP, data_dir_base;
             data_idx = 1, eps_vel = 0,
             max_slices = typemax(Int),
             load_velocity = false,
             to = TimerOutput())

Compute circulation statistics from all possible slices of a GP field.

The `stats` dictionary must contain one object of type `AbstractFlowStats` per
quantity of interest.
The keys of the dictionary are values of type `VelocityLikeFields.VelocityLikeField`.
This dictionary may be generated by calling `init_statistics`.

## Optional parameters

- `data_idx`: "index" of data file. Needed to determine the input data
  filenames.

- `eps_vel`: optional correction applied to `ρ` before computing velocity.

- `load_velocity`: load velocity field instead of wave function field.
  In this case, the density `ρ` will be taken as constant and equal to 1.

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
                  load_velocity = false,
                  eps_vel = 0, to = TimerOutput()) where {D}
    Nslices = min(num_slices(gp.dims, orientation), max_slices) :: Int
    @assert Nslices >= 1
    @assert !isempty(stats)

    resampling_factor, resampled_grid = let vals = values(stats)
        r = first(vals).resampling_factor
        grid = first(vals).resampled_grid
        @assert(all(getfield.(vals, :resampling_factor) .=== r),
                "all stats must have the same resampling factor")
        @assert r >= 1 "only upscaling is allowed"
        @assert(all(getfield.(vals, :resampled_grid) .=== grid),
                "all stats must have the same `compute_in_resampled_grid` value")
        r, grid
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
    with_v = VelocityLikeFields.Velocity in stats_keys
    with_vreg = VelocityLikeFields.RegVelocity in stats_keys
    with_p = VelocityLikeFields.Momentum in stats_keys

    Nth = Threads.nthreads()

    # Allocate arrays (one per thread).
    fields = [allocate_fields(first(values(stats)), Nij_input, Nij_compute,
                              with_v || with_vreg, resampled_grid;
                              L=Lij, load_velocity=load_velocity)
              for t = 1:Nth]
    stats_t = [zero(stats) for t = 1:Nth]

    let s = orientation_str(orientation)
        println(stderr)
        @info "Analysing slices $slices along $s..."
    end

    slice_ranges = Iterators.partition(slices, Nth)  # e.g. [1:40, 41:80, 81:120, ...] if Nth = 40
    slices_per_thread = length(slice_ranges)

    for (n_range, slice_range) in enumerate(slice_ranges)
        @info "  Slice $n_range/$slices_per_thread"
        flush(stderr)
        Threads.@threads for s in slice_range
            t = Threads.threadid()
            F = fields[t]

            # Load ψ at selected slice.
            slice = make_slice(gp.dims, orientation, s)
            timer = t == 1 ? to : TimerOutput()

            analyse_slice!(
                stats_t[t], slice, gp, F, timer, data_params, eps_vel,
                resampling_factor, (with_p, with_vreg, with_v),
            )
        end
        GC.gc()  # force garbage collector run to avoid memory issues
    end

    @timeit to "reduce!" reduce!(stats, stats_t)

    stats
end

# Allocate fields common to all stats (circulation, velocity increments).
function allocate_common_fields(Nij_input, Nij_resampled, with_v,
                                compute_in_resampled_grid::Bool;
                                load_velocity)
    FFTW.set_num_threads(1)  # make sure that plans are not threaded!
    if load_velocity
        ψ_in = nothing
        ψ = nothing
        ψ_buf = nothing
        fft_plans_p = nothing
        ρ = ones(Float64, Nij_input...)
    else
        ψ_in = Array{ComplexF64}(undef, Nij_input...)
        ψ = if Nij_input === Nij_resampled
            ψ_in  # in this case, the two point to the same data!
        else
            similar(ψ_in, Nij_resampled)
        end
        ψ_buf = similar(ψ)
        ρ = similar(ψ, Float64)
        fft_plans_p = GPFields.create_fft_plans_1d!(ψ)
    end
    Γ_size = compute_in_resampled_grid ? Nij_resampled : Nij_input
    ps = (similar(ρ), similar(ρ))  # momentum
    (
        ψ_in = ψ_in,
        ψ = ψ,
        ψ_buf = ψ_buf,
        ρ = ρ,
        dims_out = Γ_size,
        ps = ps,
        vs = with_v ? similar.(ps) : nothing,
        # FFTW plans for computing momentum
        fft_plans_p = fft_plans_p,
    )
end

function load_psi_slice!(ψ::AbstractArray, ψ_in::AbstractArray, gp, slice,
                         data_params, resampling_factor, to)
    @assert all(size(ψ) .÷ size(ψ_in) .== resampling_factor)
    @timeit to "load_psi!" GPFields.load_psi!(
        ψ_in, gp, data_params.directory, data_params.index, slice=slice)

    # Create parameters associated to original and resampled slices.
    gp_slice_in = ParamsGP(gp, slice)

    # Note that nξ changes in the resampled field!
    nxi_resample = gp_slice_in.nξ * resampling_factor
    gp_slice = ParamsGP(gp_slice_in, dims=size(ψ), nxi=nxi_resample)

    @timeit to "resample ψ" begin
        # Note: resample_field_fourier! works with fields in Fourier space.
        fft!(ψ_in)
        GPFields.resample_field_fourier!(ψ, ψ_in, gp_slice_in)
        ifft!(ψ)
    end

    gp_slice
end

function load_velocity_slice!(vs::Tuple, gp, slice, data_params, to)
    @timeit to "load_velocity!" GPFields.load_velocity!(
        vs, gp, data_params.directory, data_params.index, slice=slice)
    ParamsGP(gp, slice)
end

function analyse_slice!(
        stats, slice, gp, F, to, data_params,
        eps_vel, resampling_factor,
        (with_p, with_vreg, with_v),
    )
    load_velocity = F.ψ === nothing
    if load_velocity
        if resampling_factor != 1
            error("resampling_factor can't be different from 1 when loading velocity field")
        end
        gp_slice = load_velocity_slice!(F.ps, gp, slice, data_params, to)
        for p in F.ps
            p ./= F.ρ  # this doesn't really do anything, since ρ = 1 everywhere
        end
    else
        gp_slice = load_psi_slice!(F.ψ, F.ψ_in, gp, slice, data_params,
                                   resampling_factor, to)
        @timeit to "compute_density!" GPFields.compute_density!(F.ρ, F.ψ)
        @timeit to "compute_momentum!" GPFields.compute_momentum!(
            F.ps, F.ψ, gp_slice, buf=F.ψ_buf, fft_plans=F.fft_plans_p)
    end

    if with_p
        compute!(stats[VelocityLikeFields.Momentum], F, F.ps, to)
    end

    if with_vreg
        @timeit to "compute_vreg!" map((v, p) -> v .= p ./ sqrt.(F.ρ),
                                       F.vs, F.ps)
        compute!(stats[VelocityLikeFields.RegVelocity], F, F.vs, to)
    end

    if with_v
        @timeit to "compute_vel!" map((v, p) -> v .= p ./ (F.ρ .+ eps_vel),
                                      F.vs, F.ps)
        compute!(stats[VelocityLikeFields.Velocity], F, F.vs, to)
    end

    nothing
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
