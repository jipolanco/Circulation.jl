#!/usr/bin/env julia

# Generate 3D grid of vortices from circulation.

using GPFields

using HDF5
using FFTW
using TimerOutputs

using LinearAlgebra: mul!

using Base.Threads

const Coordinate{N} = NTuple{N,Int32}

function main()
    # ============================================================ #
    # PARAMETERS
    # ============================================================ #

    # Field parameters
    dims = (256, 256, 256)
    Ls = (2π, 2π, 2π)
    gp = ParamsGP(dims; L = Ls, c = 1.0, nxi = 1.5)
    resampling = 2

    # Note: the asterisk is expanded to {Rea,Ima}
    field_names = expanduser(
        "~/Work/data/gGP_samples/tangle/256/fields/*Psi.001.dat"
    )

    # Orientation to analyse (if `nothing`, analyse all orientations).
    # orientation = nothing
    orientation = 1  # analyse X slices only

    # Output HDF5 file
    output_h5 = if orientation === nothing
        "vortices.h5"
    else
        "vortices_dir$orientation.h5"
    end

    # Maximum number of slices to analyse (for testing)
    # If `nothing`, all possible slices are considered.
    max_slices = nothing

    # ============================================================ #
    # END OF PARAMETERS
    # ============================================================ #

    @info "Using $(nthreads()) threads"
    if nthreads() == 1
        @info "Set the JULIA_NUM_THREADS environment variable to change this."
    end

    println("Reading data from: $field_names\n\n", gp)

    params = (; resampling, field_names)

    to = TimerOutput()
    orientations = if orientation === nothing
        slice_orientations(gp)  # autodetect all orientations (in 2D and 3D)
    else
        (Orientation(orientation), )
    end

    # Warmup
    analyse_orientation(Orientation(1), gp, params; to, max_slices=1)
    reset_timer!(to)

    N = ndims(gp)
    coords = (positive = Coordinate{N}[], negative = Coordinate{N}[])

    # This is executed after analysing each slice in analyse_orientation!.
    function postprocess(grid, ::Orientation{dir}, slice) where {dir}
        @timeit to "to_coordinates!" to_coordinates!(coords, grid, slice)
    end

    h5open(output_h5, "w") do ff
        write_params!(ff, gp, params)
        map(orientations) do dir
            empty!.(values(coords))
            @timeit to "analyse_orientation" analyse_orientation(
                postprocess, dir, gp, params; to, max_slices)
            @timeit to "write HDF5" to_hdf5!(ff, dir, coords)
        end
    end

    println(to)

    nothing
end

function to_coordinates!(coords, grid, slice)
    to_coordinates!(coords.positive, grid.positive, slice)
    to_coordinates!(coords.negative, grid.negative, slice)
    coords
end

function to_coordinates!(coords::AbstractVector, grid::AbstractMatrix, slice)
    rlock = ReentrantLock()
    @threads for I in CartesianIndices(grid)
        @inbounds iszero(grid[I]) && continue
        ijk = GPFields.global_index(Tuple(I), slice)
        lock(rlock) do
            push!(coords, ijk)
        end
    end
    coords
end

function write_params!(ff, gp, params)
    g = create_group(ff, "ParamsGP")
    g["resolution"] = collect(size(gp))
    g["box_size"] = collect(box_size(gp))
    g["dx"] = collect(gp.dx)
    g["c"] = gp.c
    g["kappa"] = gp.κ
    g["xi"] = gp.ξ
    g["resampling"] = params.resampling
    g["field_path"] = params.field_names
    ff
end

function to_hdf5!(ff, ::Orientation{dir}, coords) where {dir}
    gname = string("Orientation", "XYZ"[dir])
    g = create_group(ff, gname)
    g["positive"] = as_matrix(coords.positive)
    g["negative"] = as_matrix(coords.negative)
    ff
end

function as_matrix(coords::AbstractVector{Coordinate{N}}) where {N}
    T = eltype(eltype(coords))
    T <: Integer
    x = reinterpret(T, coords)
    reshape(x, N, :)
end

analyse_orientation(args...; kws...) =
    analyse_orientation((stuff...) -> nothing, args...; kws...)

function analyse_orientation(
        postprocess::Function,
        dir::Orientation, gp_in, params;
        to = TimerOutput(),
        max_slices = nothing,
    )
    resampling = params.resampling
    gp_slice_in = let slice = make_slice(size(gp_in), dir, 1)
        ParamsGP(gp_in, slice)
    end

    Ns = size(gp_slice_in) .* resampling  # dimensions of resampled slice
    gp = ParamsGP(gp_slice_in; dims = Ns)

    # Set circulation loop size to grid step of input slice.
    loop_size = min(gp_slice_in.dx...)
    kernel = RectangularKernel(loop_size)
    fields = allocate_fields(gp_slice_in, gp)
    @timeit to "compute kernel" materialise!(fields.g_hat, kernel)

    Nslices = num_slices(size(gp_in), dir, max_slices) :: Int
    slices = 1:Nslices

    let s = string(dir)
        println(stderr)
        @info "Analysing slices $slices along $s..."
    end

    ψ_in = fields.ψ_in

    # Loop size in (resampled) grid step units
    steps = round.(Int, loop_size ./ gp.dx)
    grid = CirculationGrid{Bool}(undef, size(gp) .÷ steps)
    @assert size(grid) == size(gp_slice_in)  # expected by postprocess()

    for s in slices
        @info "  Slice $s/$Nslices"
        flush(stderr)
        slice = make_slice(size(gp_in), dir, s)
        @timeit to "load ψ" load_psi!(ψ_in, gp_in, params.field_names; slice)
        @timeit to "generate grid" generate_grid_slice!(grid, fields, to)
        @timeit to "postprocess" postprocess(grid, dir, slice)
    end

    nothing
end

function resample_psi!(ψ, ψ_in, gp_slice_in, plans)
    plans.fw * ψ_in  # in-place FFT
    GPFields.resample_field_fourier!(ψ, ψ_in, gp_slice_in)
    plans.bw * ψ
    ψ
end

function generate_grid_slice!(grid, F, to)
    gp = F.gp
    ψ = F.ψ
    ρ = F.ρ
    ps = F.ps
    vs = F.ps  # overwrite momentum to compute velocity

    @timeit to "resample ψ" resample_psi!(
        ψ, F.ψ_in, F.gp_in, F.fft_plans_resample)

    @timeit to "density!" GPFields.density!(ρ, ψ)

    @timeit to "momentum!" GPFields.momentum!(
        ps, ψ, gp, buf=F.ψ_buf, fft_plans=F.fft_plans_p)

    @timeit to "velocity!" GPFields.velocity!(vs, ps, ρ)

    plan = F.rplan
    plan_inv = F.rplan_inv
    vF = F.v_hat
    @timeit to "FFT velocity" mul!.(vF, Ref(plan), vs)

    Γ = F.Γ
    Γ_hat = F.Γ_hat
    gF = F.g_hat
    @timeit to "circulation!" circulation!(Γ, vF, gF; buf = Γ_hat, plan_inv)

    method = Grids.BestInteger()
    @timeit to "to_grid!" to_grid!(
        grid, Γ, method;
        κ = gp.κ, force_unity = true, cleanup = true, cell_size = (2, 2),
    )

    nothing
end

function allocate_fields(gp_in, gp, ::Type{T} = Float64) where {T}
    Ns_in = size(gp_in)
    Ns_resampled = size(gp)
    FFTW.set_num_threads(nthreads())  # make sure that FFTs are threaded
    ψ_in = Array{complex(T)}(undef, Ns_in)
    ψ = similar(ψ_in, Ns_resampled)
    ρ = similar(ψ, T)
    fft_plans_resample = (
        fw = plan_fft!(ψ_in, flags=FFTW.MEASURE),
        bw = plan_ifft!(ψ, flags=FFTW.MEASURE),
    )
    fft_plans_p = GPFields.create_fft_plans_1d!(ψ)
    ps = (similar(ρ), similar(ρ))  # momentum
    Γ = similar(ρ)
    ks = GPFields.get_wavenumbers(gp, Val(:r2c))
    ψ_buf = similar(ψ)

    # Reuse buffer for Γ_hat
    Γ_hat = let Ns = length.(ks)
        M = prod(Ns)
        @assert M < length(ψ_buf)
        v = view(vec(ψ_buf), 1:M)
        reshape(v, Ns)
    end

    v_hat = map(_ -> similar(Γ_hat), ps)
    g_hat = DiscreteFourierKernel{T}(undef, ks...)
    rplan = plan_rfft(Γ; flags=FFTW.MEASURE)
    rplan_inv = inv(rplan)

    (;
        gp_in, gp,
        ψ_in, ψ,
        fft_plans_resample,
        ρ, ps, fft_plans_p,
        ks, ψ_buf, Γ, Γ_hat, g_hat, v_hat, rplan, rplan_inv,
    )
end

main()
