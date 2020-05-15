#!/usr/bin/env julia

using GPFields
using Circulation

import Pkg.TOML
using TimerOutputs
using HDF5

import Base.Threads

const USAGE =
"""
Usage: julia $(basename(@__FILE__)) CONFIG_FILE.toml

Compute circulation statistics from GP field.

Parameters of the field and of the statistics must be set in a configuration
file."""

function parse_commandline()
    help = any(("--help", "-h") .∈ Ref(ARGS))
    if isempty(ARGS) || help
        println(USAGE)
        exit(help ? 0 : 1)
    end
    config_file = ARGS[1]
    Dict("parameter-file" => config_file)
end

function params_from_file(filename)
    if !isfile(filename)
        error("parameter file not found: $filename")
    end
    @info "Loading parameters from $filename"
    TOML.parsefile(filename)
end

function parse_params_fields(d::Dict)
    dims = d["N"] :: Vector{Int}
    if length(dims) ∉ (2, 3)
        error("`N` parameter must be a vector of 2 or 3 integers")
    end
    (
        data_dir = replace_env(expanduser(d["data_directory"] :: String)),
        data_idx = d["data_index"] :: Int,
        dims = tuple(dims...),
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
    max_slices = let m = get(d, "max_slices", 0) :: Int
        m == 0 ? typemax(Int) : m  # replace 0 -> typemax(Int)
    end
    loop_sizes = parse_loop_sizes(d["loops"], dims) :: AbstractVector{Int}

    stats = d["statistics"]
    moments = stats["moments"]
    hist = stats["histogram"]

    Nfrac = get(moments, "N_fractional", 0)

    (
        eps_velocity = d["epsilon_velocity"] :: Real,
        max_slices = max_slices,
        loop_sizes = loop_sizes,
        moments_pmax = moments["p_max"] :: Int,
        moments_Nfrac = Nfrac == 0 ? nothing : Nfrac,
        hist_Nedges = hist["num_bin_edges"] :: Int,
        hist_max_kappa = Float64(hist["max_kappa"]),
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
        c = P.physics.c,
        nxi = P.physics.nxi,
    )

    loop_sizes = P.circulation.loop_sizes
    @info "Loop sizes: $loop_sizes ($(length(loop_sizes)) sizes)"

    to = TimerOutput()
    stats = let par = P.circulation
        M = par.hist_max_kappa
        Nedges = par.hist_Nedges
        κ = params.κ
        edges = LinRange(-M * κ, M * κ, Nedges)

        Circulation.init_statistics(
            loop_sizes,
            which=(
                CirculationFields.Velocity,
                CirculationFields.RegVelocity,
                # CirculationFields.Momentum,
            ),
            num_moments=par.moments_pmax,
            moments_Nfrac=par.moments_Nfrac,
            hist_edges=edges,
        )
    end

    # First run for precompilation (-> accurate timings)
    analyse!(
        stats, params, P.fields.data_dir,
        data_idx=P.fields.data_idx,
        eps_vel=P.circulation.eps_velocity,
        to=to,
        max_slices=1,
    )

    reset_timer!(to)
    reset!(stats)

    analyse!(
        stats, params, P.fields.data_dir,
        data_idx=P.fields.data_idx,
        eps_vel=P.circulation.eps_velocity,
        to=to,
        max_slices=P.circulation.max_slices,
    )

    println(to)

    @info "Saving $(P.output.statistics)"
    h5open(P.output.statistics, "w") do ff
        write(g_create(ff, "ParamsGP"), params)
        write(g_create(ff, "Circulation"), stats)
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

    params = (
        fields = fields,
        circulation = parse_params_circulation(p["circulation"], fields.dims),
        physics = parse_params_physics(p["physics"]),
        output = parse_params_output(p["output"]),
    )

    main(params)
end

main()
