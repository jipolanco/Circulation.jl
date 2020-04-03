#!/usr/bin/env julia

using GPFields
using Circulation

using ArgParse
import Pkg.TOML
using LinearAlgebra: norm
using TimerOutputs

import Base.Threads

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--parameter-file", "-p"
            help = "path to TOML parameter file"
            arg_type = String
            default = "circulation.toml"
    end
    parse_args(s)
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
        data_dir = expanduser(d["data_directory"] :: String),
        data_idx = d["data_index"] :: Int,
        dims = tuple(dims...),
    )
end

parse_params_physics(d::Dict) = (
    c = Float64(d["c"]),
    nxi = Float64(d["nxi"]),
)

parse_params_output(d::Dict) = (
    statistics = d["statistics"] :: String,
)

function parse_params_circulation(d::Dict)
    (
        eps_velocity = d["epsilon_velocity"] :: Real,
        loop_sizes = parse_loop_sizes(d["loop_sizes"]),
    )
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

function main(P::NamedTuple)
    params = ParamsGP(
        P.fields.dims,    # resolution: (Nx, Ny) or (Nx, Ny, Nz)
        c = P.physics.c,
        nxi = P.physics.nxi,
    )

    loop_sizes = P.circulation.loop_sizes
    @info "Loop sizes: $loop_sizes"

    κ = params.κ

    to = TimerOutput()
    stats = Circulation.init_statistics(
        loop_sizes, num_moments=20, hist_edges=LinRange(-20κ, 20κ, 1000))

    analyse!(
        stats, params, P.fields.data_dir,
        data_idx=P.fields.data_idx,
        eps_vel=P.circulation.eps_velocity,
        to=to,
        slice_selection=1:2,
    )

    println(to)

    save_statistics(P.output.statistics, stats)

    nothing
end

function main()
    @info "Using $(Threads.nthreads()) threads"
    if Threads.nthreads() == 1
        @info "Set the JULIA_NUM_THREADS environment variable to change this."
    end

    args = parse_commandline()
    p = params_from_file(args["parameter-file"])

    params = (
        fields = parse_params_fields(p["fields"]),
        circulation = parse_params_circulation(p["circulation"]),
        physics = parse_params_physics(p["physics"]),
        output = parse_params_output(p["output"]),
    )

    main(params)
end

main()
