#!/usr/bin/env julia

using GPFields
using Circulation

using ArgParse
import Pkg.TOML

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
    c = d["c"] :: Float64,
    nxi = d["nxi"] :: Float64,
)

function parse_params_circulation(d::Dict)
    s = get(d, "slice_3d", [0, 0, 1]) :: Vector{Int}
    if length(s) != 3
        error("`slice_3d` parameter must be a vector of 3 integers")
    end
    t = tuple(replace(s, 0 => :)...)
    (
        eps_velocity = d["epsilon_velocity"] :: Float64,
        slice_3d = t,
    )
end

@info "Using $(Threads.nthreads()) threads"
if Threads.nthreads() == 1
    @info "Set the JULIA_NUM_THREADS environment variable to change this."
end

function read_psi(params::ParamsGP, dir_base, resolution, idx)
    datadir = joinpath(dir_base, string(resolution))
    ψ = GPFields.load_psi(params, datadir, idx)
    @info "Loaded $(summary(ψ)) from $datadir"
    ψ
end

# Determine dimensions along which a slice is performed.
# For instance, if the slice is (42, :, :, 15, :) -> dims = (2, 3, 5)
slice_dims() = ()
slice_dims(::Any, etc...) = 1 .+ slice_dims(etc...)
slice_dims(::Colon, etc...) = (1, (1 .+ slice_dims(etc...))...)
slice_dims(t::Tuple) = slice_dims(t...)

@assert slice_dims(42, :, :, 15, :) === (2, 3, 5)
@assert slice_dims(:, 2, :) === (1, 3)
@assert slice_dims(2, :, 1) === (2, )
@assert slice_dims(:, 2) === (1, )
@assert slice_dims(1, 2) === ()

# Select slice.
# Case of 3D input data
function prepare_slice(v::NTuple{3,AbstractArray{T,3}} where T,
                       slice=(:, :, 1))
    dims = slice_dims(slice)
    if length(dims) != 2
        throw(ArgumentError(
            "the given slice should have dimension 2 (got $slice)"
        ))
    end
    vs = v[dims[1]], v[dims[2]]  # select the 2 relevant components
    vsub = view.(vs, slice...)
    @assert all(ndims.(vsub) .== 2) "Slices don't have dimension 2"
    vsub
end

# Case of 2D input data (the `slice` argument is ignored)
prepare_slice(v::NTuple{2,AbstractArray{T,2}} where T, args...) = v

generate_slice(::ParamsGP{2}, args...) = (:, :)     # 2D case
generate_slice(::ParamsGP{3}, slice_3d) = slice_3d  # 3D case

function main(fields, circulation, physics)
    params = ParamsGP(
        fields.dims,    # resolution: (Nx, Ny) or (Nx, Ny, Nz)
        c = physics.c,
        nxi = physics.nxi,
    )

    eps_vel = circulation.eps_velocity

    slice = generate_slice(params, circulation.slice_3d)
    dims = slice_dims(slice) :: Tuple{Int,Int}
    @info "Using slice = $slice (dimensions: $dims)"

    # Load field from file
    psi = read_psi(params, fields.data_dir, params.dims[1], fields.data_idx)

    # Compute different fields (can be 2D or 3D)
    rho = GPFields.compute_density(psi)
    p = GPFields.compute_momentum(psi, params)  # = (px, py, [pz])
    v = map(pp -> pp ./ (rho .+ eps_vel), p)    # = (vx, vy, [vz])
    vreg = map(pp -> pp ./ sqrt.(rho), p)       # regularised velocity

    i, j = dims
    Ls = params.L[i], params.L[j]        # domain size in the slice dimensions
    Ns = params.dims[i], params.dims[j]  # dimensions of slice (N1, N2)
    p_slice = prepare_slice(p, slice)

    # Compute integral fields
    Ip = IntegralField2D(p_slice[1], L=Ls)
    prepare!(Ip, p_slice)

    # Allocate circulation matrix
    circ = similar(rho, Ns...)

    rs = (8, 8)  # rectangle loop dimensions

    # Compute circulation over slice
    circulation!(circ, Ip, rs)
end

function main()
    args = parse_commandline()
    p = params_from_file(args["parameter-file"])

    fields = parse_params_fields(p["fields"])
    circulation = parse_params_circulation(p["circulation"])
    physics = parse_params_physics(p["physics"])

    main(fields, circulation, physics)
end

main()
