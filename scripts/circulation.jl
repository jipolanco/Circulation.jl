#!/usr/bin/env julia

using GPFields
using Circulation

import Pkg.TOML
using TimerOutputs
using HDF5

import Base.Threads

const DEFAULT_CONFIG = joinpath(@__DIR__, "..", "examples", "tangle.toml")

const USAGE =
"""
Usage: julia $(basename(@__FILE__)) CONFIG_FILE.toml

Compute circulation statistics from GP field.

Parameters of the field and of the statistics must be set in a configuration
file."""

function parse_commandline()
    help = any(("--help", "-h") .∈ Ref(ARGS))
    if help
        println(USAGE)
        exit(0)
    end
    config_file = get(ARGS, 1, DEFAULT_CONFIG)
    Dict("parameter-file" => config_file)
end

function params_from_file(filename)
    filename_rel = relpath(filename)
    if !isfile(filename)
        error("parameter file not found: $filename_rel")
    end
    @info "Loading parameters from $filename_rel"
    TOML.parsefile(filename)
end

function parse_params_fields(d::Dict)
    dims = d["N"] :: Vector{Int}
    D = length(dims)
    if D ∉ (2, 3)
        error("`N` parameter must be a vector of 2 or 3 integers")
    end
    L = get(d, "L_2pi", ones(D)) .* 2pi
    res = get(d, "resampling_factor", 1)
    (
        data_dir = replace_env(expanduser(d["data_directory"] :: String)),
        data_idx = d["data_index"] :: Int,
        dims = tuple(dims...) :: NTuple{D,Int},
        L = tuple(L...) :: NTuple{D,Float64},
        resampling_factor = res :: Int,
    )
end

parse_params_physics(d::Dict) = (
    c = Float64(d["c"]),
    nxi = Float64(d["nxi"]),
)

parse_params_output(d::Dict) = (
    statistics = replace_env(d["statistics"] :: String),
)

function parse_params_circulation(d::Dict, dims)
    if !get(d, "enabled", true)
        return nothing  # circulation stats are disabled
    end

    max_slices = let m = get(d, "max_slices", 0) :: Int
        m == 0 ? typemax(Int) : m  # replace 0 -> typemax(Int)
    end
    loop_sizes = parse_loop_sizes(d["loops"], dims) :: AbstractVector{Int}

    stats = d["statistics"]
    moments = stats["moments"]
    hist = stats["histogram"]

    resampled = get(stats, "compute_in_resampled_grid", false)

    Nfrac = get(moments, "N_fractional", 0)

    (
        eps_velocity = d["epsilon_velocity"] :: Real,
        max_slices = max_slices,
        loop_sizes = loop_sizes,
        compute_in_resampled_grid = resampled :: Bool,
        moments_pmax = moments["p_max"] :: Int,
        moments_Nfrac = Nfrac == 0 ? nothing : Nfrac,
        hist_Nedges = hist["num_bin_edges"] :: Int,
        hist_max_kappa = Float64(hist["max_kappa"]),
    )
end

function parse_params_increments(d::Dict, dims)
    if !get(d, "enabled", true)
        return nothing  # increment stats are disabled
    end

    max_slices = let m = get(d, "max_slices", 0) :: Int
        m == 0 ? typemax(Int) : m  # replace 0 -> typemax(Int)
    end
    increments =
        parse_loop_sizes(d["spatial_offsets"], dims) :: AbstractVector{Int}

    stats = d["statistics"]
    moments = stats["moments"]
    hist = stats["histogram"]

    resampled = get(stats, "compute_in_resampled_grid", false)

    Nfrac = get(moments, "N_fractional", 0)

    (
        eps_velocity = d["epsilon_velocity"] :: Real,
        max_slices = max_slices,
        increments = increments,
        compute_in_resampled_grid = resampled :: Bool,
        moments_pmax = moments["p_max"] :: Int,
        moments_Nfrac = Nfrac == 0 ? nothing : Nfrac,
        hist_Nedges = hist["num_bin_edges"] :: Int,
        hist_max_c = Float64(hist["max_c"]),
    )
end

function parse_loop_sizes(d::Dict, dims::Dims)
    type = d["selection_type"]
    if type == "list"
        return parse_loop_sizes(d["sizes"])
    elseif type == "log"
        base = d["log_base"] :: Real
        Rmax = min(dims...) - 1  # max loop size is N - 1
        Nmax = floor(Int, log(base, Rmax))
        return unique(round.(Int, base .^ (0:Nmax)))
    end
    nothing
end

parse_loop_sizes(s::Vector{Int}) = s
parse_loop_sizes(s::Int) = s

function parse_loop_sizes(s::String)
    try
        # We assume the format "start:step:stop".
        sp = split(s, ':', limit=3)
        a, b, c = parse.(Int, sp)
        a:b:c
    catch e
        error("cannot parse range from string '$s'")
    end
end

# Replace environment variables in string.
function replace_env(s::String)
    # Is there an easier way to do this??
    while (m = match(r"\$(\w+)", s)) !== nothing
        var = m[1]
        s = replace(s, "\$$var" => ENV[var])
    end
    s
end

function main(P::NamedTuple)
    params = ParamsGP(
        P.fields.dims,    # resolution: (Nx, Ny) or (Nx, Ny, Nz)
        L = P.fields.L,
        c = P.physics.c,
        nxi = P.physics.nxi,
    )

    num_stats = count(!isnothing, (P.circulation, P.increments))
    if num_stats > 1
        error("cannot enable both circulation and increment statistics!")
    end

    stats_params, stats_type = if P.circulation !== nothing
        @info "Computing circulation statistics"
        loop_sizes = P.circulation.loop_sizes
        output_name = "Circulation"
        P.circulation, CirculationStats
    elseif P.increments !== nothing
        @info "Computing velocity increment statistics"
        loop_sizes = P.increments.increments
        output_name = "Increments"
        P.increments, IncrementStats
    else
        error("all statistics are disabled!")
    end

    resampling = P.fields.resampling_factor
    resampled_grid = stats_params.compute_in_resampled_grid
    @info "Loop sizes: $loop_sizes ($(length(loop_sizes)) sizes)"
    @info "Resampling factor: $resampling"
    if resampling > 1
        @info "Computing in $(resampled_grid ? "resampled" : "original") grid"
    end

    to = TimerOutput()
    stats = let par = stats_params
        Nedges = par.hist_Nedges
        if stats_type === CirculationStats
            κ = params.κ
            M = par.hist_max_kappa
            edges = LinRange(-M * κ, M * κ, Nedges)
        elseif stats_type === IncrementStats
            c = params.c
            M = par.hist_max_c
            edges = LinRange(-M * c, M * c, Nedges)
        end

        Circulation.init_statistics(
            stats_type,
            loop_sizes,
            which=(
                VelocityLikeFields.Velocity,
                VelocityLikeFields.RegVelocity,
                # VelocityLikeFields.Momentum,
            ),
            num_moments=par.moments_pmax,
            moments_Nfrac=par.moments_Nfrac,
            hist_edges=edges,
            resampling_factor=resampling,
            compute_in_resampled_grid=resampled_grid,
        )
    end

    # First run for precompilation (-> accurate timings)
    analyse!(
        stats, params, P.fields.data_dir,
        data_idx=P.fields.data_idx,
        eps_vel=stats_params.eps_velocity,
        to=to,
        max_slices=1,
    )

    reset_timer!(to)
    reset!(stats)

    analyse!(
        stats, params, P.fields.data_dir,
        data_idx=P.fields.data_idx,
        eps_vel=stats_params.eps_velocity,
        to=to,
        max_slices=stats_params.max_slices,
    )

    println(to)

    @info "Saving $(P.output.statistics)"
    h5open(P.output.statistics, "w") do ff
        write(g_create(ff, "ParamsGP"), params)
        write(g_create(ff, output_name), stats)
    end

    nothing
end

function main()
    args = parse_commandline()
    p = params_from_file(args["parameter-file"])

    @info "Using $(Threads.nthreads()) threads"
    if Threads.nthreads() == 1
        @info "Set the JULIA_NUM_THREADS environment variable to change this."
    end

    fields = parse_params_fields(p["fields"])
    circ = parse_params_circulation(p["circulation"], fields.dims)
    incr = parse_params_increments(p["increments"], fields.dims)

    params = (
        fields = fields,
        circulation = circ,
        increments = incr,
        physics = parse_params_physics(p["physics"]),
        output = parse_params_output(p["output"]),
    )

    main(params)
end

main()
