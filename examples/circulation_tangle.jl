# Compute circulation statistics in turbulent tangle.
#
# Note that the analysis parameters are directly written to this file, and TOML
# parameter files are not used.

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
    dims = (256, 256, 256)
    gp = ParamsGP(dims, L = (2π, 2π, 2π), c = 1.0, nxi = 1.5)
    resampling_factor = 4
    with_convolution = true
    compute_in_resampled_grid = false
    loop_sizes = make_loop_sizes(; base = 1.4, dims = dims)
    kernels = with_convolution ? RectangularKernel.(loop_sizes .* gp.dx[1]) : loop_sizes

    fields = (
        data_dir = expanduser("~/Data/GP/gGP_samples/tangle/256/fields"),
        data_idx = 1,
        load_velocity = false,
    )

    output = (
        statistics = "tangle.h5",
    )

    circulation = (
        # max_slices = typemax(Int),
        max_slices = 4,
        eps_velocity = 0,
        moments = ParamsMoments(integer = 10, fractional = nothing),
        histogram = (Nedges = 4000, max_kappa = 30.5),
    )

    @info "Using convolutions: $with_convolution"
    @info "Loop sizes: $loop_sizes ($(length(loop_sizes)) sizes)"
    @info "Resampling factor: $resampling_factor"
    if resampling_factor > 1
        @info "Computing in $(compute_in_resampled_grid ? "resampled" :
                              "original") grid"
    end

    to = TimerOutput()
    kwargs = (
        data_idx = fields.data_idx,
        load_velocity = fields.load_velocity,
        eps_vel = circulation.eps_velocity,
        to = to,
    )

    # Which fields to analyse.
    which = if kwargs.load_velocity
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
        histogram = ParamsHistogram(bin_edges = edges)

        init_statistics(
            CirculationStats,
            kernels;
            which = which,
            histogram, moments = par.moments,
            resampling_factor,
            compute_in_resampled_grid,
        )
    end

    analyse!(stats, gp, fields.data_dir; max_slices=1, kwargs...)

    reset_timer!(to)
    reset!(stats)

    analyse!(
        stats, gp, fields.data_dir;
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
