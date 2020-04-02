#!/usr/bin/env julia

using GPFields
using Circulation

using ArgParse
import Pkg.TOML
using LinearAlgebra: norm

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

function parse_params_circulation(d::Dict)
    s = get(d, "slice_3d", [0, 0, 1]) :: Vector{Int}
    if length(s) != 3
        error("`slice_3d` parameter must be a vector of 3 integers")
    end
    t = tuple(replace(s, 0 => :)...)
    (
        eps_velocity = d["epsilon_velocity"] :: Real,
        loop_sizes = parse_loop_sizes(d["loop_sizes"]),
        slice_3d = t,
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

function main(P::NamedTuple)
    params = ParamsGP(
        P.fields.dims,    # resolution: (Nx, Ny) or (Nx, Ny, Nz)
        c = P.physics.c,
        nxi = P.physics.nxi,
    )

    eps_vel = P.circulation.eps_velocity

    slice = generate_slice(params, P.circulation.slice_3d)
    dims = slice_dims(slice) :: Tuple{Int,Int}
    @info "Using slice = $slice (dimensions: $dims)"

    # Load field from file
    psi = Array{ComplexF64}(undef, params.dims...)
    let N = params.dims[1]
        dir = joinpath(P.fields.data_dir, string(N))
        GPFields.load_psi!(psi, dir, P.fields.data_idx)
    end

    # Compute different fields (can be 2D or 3D)
    rho = GPFields.compute_density(psi)
    p = GPFields.compute_momentum(psi, params)  # = (px, py, [pz])
    v = map(pp -> pp ./ (rho .+ eps_vel), p)    # = (vx, vy, [vz])
    vreg = map(pp -> pp ./ sqrt.(rho), p)       # regularised velocity

    i, j = dims
    Ls = params.L[i], params.L[j]        # domain size in the slice dimensions
    Ns = params.dims[i], params.dims[j]  # dimensions of slice (N1, N2)
    p_slice = prepare_slice(vreg, slice)

    # Compute integral fields
    Ip = IntegralField2D(p_slice[1], L=Ls)
    prepare!(Ip, p_slice)

    # Allocate circulation matrix (one per thread)
    circ = [similar(rho, Ns...) for n = 1:Threads.nthreads()]

    loop_sizes = P.circulation.loop_sizes
    @info "Loop sizes: $loop_sizes"

    Threads.@threads for r in loop_sizes
        # Compute circulation over slice
        Γ = circ[Threads.threadid()]
        @time circulation!(Γ, Ip, (r, r))
    end

    @show norm(circ[1])
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
    )

    main(params)
end

main()
