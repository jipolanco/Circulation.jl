# Circulation-specific statistics.

"""
    CirculationStats{T} <: AbstractFlowStats

Circulation statistics, including moments and histograms.
"""
struct CirculationStats{Loops, M<:Moments, H<:Histogram} <: AbstractFlowStats
    Nr         :: Int    # number of loop sizes
    loop_sizes :: Loops  # length Nr
    resampling_factor :: Int
    resampled_grid :: Bool
    moments    :: M
    histogram  :: H
end

increments(s::CirculationStats) = s.loop_sizes
statistics(::CirculationStats) = (:moments, :histogram)

"""
    CirculationStats(loop_sizes;
                     hist_edges,
                     num_moments,
                     moments_Nfrac=nothing,
                     resampling_factor=1,
                     compute_in_resampled_grid=false)

Construct and initialise statistics.

# Parameters

- `loop_sizes`: sizes of square circulation loops in grid step units.

- `resampling_factor`: if greater than one, the loaded ψ fields are resampled
  into a finer grid using padding in Fourier space.
  The number of grid points is increased by a factor `resampling_factor` in
  every direction.

- `compute_in_resampled_grid`: if this is `true` and `resampling_factor > 1`,
  the statistics are accumulated over all the possible loops in the resampled
  (finer) grid, instead of the original (coarser) grid.
  This takes more time but may lead to better statistics.
"""
function CirculationStats(
        loop_sizes;
        hist_edges, num_moments,
        moments_Nfrac=nothing,
        resampling_factor=1,
        compute_in_resampled_grid=false,
    )
    resampling_factor >= 1 || error("resampling_factor must be positive")
    Nr = length(loop_sizes)
    M = Moments(num_moments, Nr, Float64; Nfrac=moments_Nfrac)
    H = Histogram(hist_edges, Nr, Int)
    CirculationStats(Nr, loop_sizes, resampling_factor,
                     compute_in_resampled_grid, M, H)
end

function allocate_fields(::CirculationStats, args...; L)
    data = allocate_common_fields(args...)
    (;
        data...,
        Γ = similar(data.ρ, data.dims_out),
        I = IntegralField2D(data.ps[1], L=L),
    )
end

function compute!(stats::CirculationStats, fields, vs, to)
    Γ = fields.Γ
    Ip = fields.I

    # Set integral values with momentum data.
    @timeit to "prepare!" prepare!(Ip, vs)

    resampling = stats.resampling_factor
    grid_step = stats.resampled_grid ? 1 : resampling
    @assert grid_step .* size(Γ) == size(vs[1]) "incorrect dimensions of Γ"

    for (r_ind, r) in enumerate(increments(stats))
        s = resampling * r  # loop size in resampled field
        @timeit to "circulation!" circulation!(
            Γ, Ip, (s, s), grid_step=grid_step)
        @timeit to "statistics" update!(stats, Γ, r_ind; to=to)
    end

    stats
end

function Base.write(g::Union{HDF5File,HDF5Group}, stats::CirculationStats)
    g["loop_sizes"] = collect(increments(stats))
    g["resampling_factor"] = stats.resampling_factor
    g["resampled_grid"] = stats.resampled_grid
    write(g_create(g, "Moments"), stats.moments)
    write(g_create(g, "Histogram"), stats.histogram)
    g
end
