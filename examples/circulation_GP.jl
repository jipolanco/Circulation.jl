# Compute circulation statistics from sample 256³ GP data.

using GPFields
using GPStatistics

using TimerOutputs
using HDF5

import Base.Threads

@info "Using $(Threads.nthreads()) threads"
if Threads.nthreads() == 1
    @info "Set the JULIA_NUM_THREADS environment variable to change this."
end

function make_loop_sizes(; base, dims)
    Rmax = min(dims...) - 1  # max loop size is N - 1
    Nmax = floor(Int, log(base, Rmax))
    unique(round.(Int, base .^ (0:Nmax)))
end

function main()
    dims = (256, 256, 256)  # resolution of input fields
    gp = ParamsGP(
        dims,
        L = (2π, 2π, 2π),   # domain size
        c = 1.0,            # speed of sound
        nxi = 1.5,          # normalised healing length
    )

    resampling_factor = 4   # higher is better (but slower and uses more memory)
    compute_in_resampled_grid = false

    # Sizes of square loops to analyse
    loop_sizes = make_loop_sizes(; base = 1.4, dims = dims)

    # Generate convolution kernels associated to square loops
    kernels = RectangularKernel.(loop_sizes .* gp.dx[1])

    data_params = (
        directory = "sample_data/GP",  # directory where data is located
        timestep = 1,  # this corresponds to the "001" in the file names
        load_velocity = false,  # we're loading GP wave function fields, not velocity fields
    )

    output = (
        statistics = "circulation_GP.h5",  # circulation statistics are written to this file
    )

    circulation = (
        max_slices = typemax(Int),  # compute over all possible 2D cuts of the domain (better)
        # max_slices = 4,  # or compute over a subset of slices (faster)
        eps_velocity = 0,  # this is for regularisation of GP velocity fields (0 => no regularisation)

        # Parameters for circulation moments.
        moments = ParamsMoments(
            CirculationField(),
            integer = 10,          # compute moment orders p = 1:10
            fractional = nothing,  # don't compute fractional moments
        ),

        # Parameters for circulation histograms / PDFs.
        histogram = (
            Nedges = 4000,     # number of bin edges
            max_kappa = 30.5,  # histogram extrema in units of quantum of circulation κ
        ),
    )

    @info "Loop sizes: $loop_sizes ($(length(loop_sizes)) sizes)"
    @info "Resampling factor: $resampling_factor"
    if resampling_factor > 1
        @info "Computing in $(compute_in_resampled_grid ? "resampled" :
                              "original") grid"
    end

    to = TimerOutput()
    kwargs = (
        eps_vel = circulation.eps_velocity,
        to = to,
    )

    # Which fields to analyse.
    which = if data_params.load_velocity
        # If a velocity field is loaded
        (
            VelocityLikeFields.Velocity,
        )
    else
        # If a wave function field is loaded
        (
            VelocityLikeFields.Velocity,
            VelocityLikeFields.RegVelocity,
            # VelocityLikeFields.Momentum,
        )
    end

    stats = let par = circulation
        Nedges = par.histogram.Nedges
        κ = gp.κ
        M = par.histogram.max_kappa
        edges = LinRange(-M * κ, M * κ, Nedges)
        histogram = ParamsHistogram(CirculationField(), bin_edges = edges)

        stats_params = (
            histogram,
            par.moments,
        )

        init_statistics(
            CirculationStats,
            kernels,
            stats_params;
            which = which,
            resampling_factor,
            compute_in_resampled_grid,
        )
    end

    # This is just to precompile functions and to get better timings.
    analyse!(stats, gp, data_params; max_slices=1, kwargs...)
    reset_timer!(to)
    reset!(stats)

    # The actual analysis is done here.
    analyse!(
        stats, gp, data_params;
        max_slices = circulation.max_slices,
        kwargs...,
    )

    println(to)

    let outfile = output.statistics
        mkpath(dirname(outfile))
        @info "Saving $(outfile)"
        h5open(outfile, "w") do ff
            write(create_group(ff, "ParamsGP"), gp)
            write(create_group(ff, "Circulation"), stats)
        end
    end

    nothing
end

main()
