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

"""
    CirculationStats(loop_sizes;
                     num_moments=20,
                     moments_Nfrac=nothing,
                     resampling_factor=1,
                     compute_in_resampled_grid=false,
                     hist_edges=LinRange(-10, 10, 42))

Construct and initialise statistics.

# Parameters

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
        num_moments=20,
        moments_Nfrac=nothing,
        resampling_factor=1,
        compute_in_resampled_grid=false,
        hist_edges=LinRange(-10, 10, 42),
    )
    resampling_factor >= 1 || error("resampling_factor must be positive")
    Nr = length(loop_sizes)
    M = Moments(num_moments, Nr, Float64; Nfrac=moments_Nfrac)
    H = Histogram(hist_edges, Nr, Int)
    CirculationStats(Nr, loop_sizes, resampling_factor,
                     compute_in_resampled_grid, M, H)
end

Base.zero(s::CirculationStats) =
    CirculationStats(s.Nr, s.loop_sizes, s.resampling_factor, s.resampled_grid,
                     zero(s.moments), zero(s.histogram))

function compute!(stats::CirculationStats, Γ, Ip, vs, to)
    # Set integral values with momentum data.
    @timeit to "prepare!" prepare!(Ip, vs)

    resampling = stats.resampling_factor
    grid_step = stats.resampled_grid ? 1 : resampling
    @assert grid_step .* size(Γ) == size(vs[1]) "incorrect dimensions of Γ"

    for (r_ind, r) in enumerate(stats.loop_sizes)
        s = resampling * r  # loop size in resampled field
        @timeit to "circulation!" circulation!(
            Γ, Ip, (s, s), grid_step=grid_step)
        @timeit to "statistics" update!(stats, Γ, r_ind; to=to)
    end

    stats
end

function Base.write(g::Union{HDF5File,HDF5Group}, stats::CirculationStats)
    g["loop_sizes"] = collect(stats.loop_sizes)
    g["resampling_factor"] = stats.resampling_factor
    g["resampled_grid"] = stats.resampled_grid
    write(g_create(g, "Moments"), stats.moments)
    write(g_create(g, "Histogram"), stats.histogram)
    g
end
