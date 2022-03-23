# Compute circulation statistics from sample 64³ synthetic velocity field
# (NS-like data).

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
    # ====================================================================== #
    # Start of parameter section
    # ====================================================================== #

    dims = (64, 64, 64)  # resolution of input fields

    fieldparams = ParamsGP(
        dims,
        L = (2π, 2π, 2π),  # domain size
    )

    # Sizes of square loops to analyse
    loop_sizes = make_loop_sizes(; base = 1.4, dims = dims)

    # Generate convolution kernels associated to square loops
    kernels = RectangularKernel.(loop_sizes .* fieldparams.dx[1])

    data_params = (
        directory = "sample_data/NS",  # directory where data is located
        timestep = 0,          # this corresponds to the "000" in the file names
        load_velocity = true,  # we're loading synthetic velocity fields
    )

    output = (
        statistics = "circulation_NS.h5",  # circulation statistics are written to this file
    )

    circulation = (
        max_slices = typemax(Int),  # compute over all possible 2D cuts of the domain (better)
        # max_slices = 4,  # or compute over a subset of slices (faster)

        # Parameters for circulation moments.
        moments = ParamsMoments(
            CirculationField(),
            integer = 10,     # compute moment orders p = 1:10
            fractional = 10,  # compute fractional moments p = 0.1:0.1:1
        ),

        # Parameters for circulation histograms / PDFs.
        histogram = (
            Nedges = 4000,     # number of bin edges
            max_kappa = 30.5,  # histogram extrema
        ),
    )

    # ====================================================================== #
    # End of parameter section
    # ====================================================================== #

    @info "Loop sizes: $loop_sizes ($(length(loop_sizes)) sizes)"

    to = TimerOutput()
    kwargs = (to = to,)

    # Which fields to analyse.
    which = (VelocityLikeFields.Velocity,)

    stats = let par = circulation
        Nedges = par.histogram.Nedges
        κ = 1.0  # circulation unit (arbitrary for NS...)
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
        )
    end

    # This is just to precompile functions and get better timings.
    analyse!(stats, fieldparams, data_params; max_slices=1, kwargs...)
    reset_timer!(to)
    reset!(stats)

    # The actual analysis is done here.
    analyse!(
        stats, fieldparams, data_params;
        max_slices = circulation.max_slices,
        kwargs...,
    )

    println(to)

    # Write results
    let outfile = output.statistics
        mkpath(dirname(outfile))
        @info "Saving $(outfile)"
        h5open(outfile, "w") do ff
            write(create_group(ff, "ParamsGP"), fieldparams)
            write(create_group(ff, "Circulation"), stats)
        end
    end

    nothing
end

main()

