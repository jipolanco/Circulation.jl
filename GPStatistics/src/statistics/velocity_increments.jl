# Spatial velocity increment statistics.

"""
    IncrementStats{T} <: AbstractFlowStats

Spatial increment statistics, including moments and histograms.
"""
struct IncrementStats{Increments,
                      M <: NTuple{2,Moments},
                      H <: NTuple{2,Histogram}} <: AbstractFlowStats
    Nr         :: Int         # number of increments to consider
    increments :: Increments  # length = Nr
    resampling_factor :: Int  # TODO resampling may not make much sense here...
    resampled_grid    :: Bool
    moments    :: M  # tuple (longitudinal, transverse)
    histogram  :: H  # tuple (longitudinal, transverse)
end

function Base.zero(s::IncrementStats)
    IncrementStats(s.Nr, s.increments, s.resampling_factor, s.resampled_grid,
                   zero(s.moments), zero(s.histogram))
end

increments(s::IncrementStats) = s.increments
statistics(::IncrementStats) = (:moments, :histogram)

"""
    IncrementStats(increments;
                   hist_edges,
                   num_moments,
                   moments_Nfrac=nothing,
                   resampling_factor=1,
                   compute_in_resampled_grid=false)

Construct and initialise statistics.

# Parameters

- `increments`: spatial increments in grid steps units.

See [`CirculationStats`](@ref) for other parameters.
"""
function IncrementStats(
        increments;
        hist_edges, num_moments,
        moments_Nfrac=nothing,
        resampling_factor=1,
        compute_in_resampled_grid=false,
    )
    resampling_factor >= 1 || error("resampling_factor must be positive")
    Nr = length(increments)
    # TODO
    M = Moments(num_moments, Nr, Float64; Nfrac=moments_Nfrac)
    H = Histogram(hist_edges, Nr, Int)
    IncrementStats(Nr, increments, resampling_factor, compute_in_resampled_grid,
                   (M, zero(M)), (H, zero(H)))
end

function allocate_fields(S::IncrementStats, args...; L, kwargs...)
    data = allocate_common_fields(args...; kwargs...)
    dv = similar(data.ρ, data.dims_out)  # velocity increments field
    (;
        data...,
        dv_para = dv,        # longitudinal increments
        dv_perp = copy(dv),  # transverse increments
    )
end

function compute!(stats::IncrementStats, fields, vs, to)
    dv_para = fields.dv_para
    dv_perp = fields.dv_perp

    resampling = stats.resampling_factor
    grid_step = stats.resampled_grid ? 1 : resampling
    @assert grid_step .* size(dv_para) == size(vs[1]) "incorrect dimensions of dv_para"

    # For now, we compute longitudinal increments along the two directions.
    # Note that this is redundant, as each longitudinal increment is computed
    # twice when all slice directions are considered!
    # TODO
    # - compute along one of the slice dimensions (this requires information on
    #   which slice we're on, to avoid repeating data)

    for (r_ind, r) in enumerate(increments(stats))
        s = resampling * r  # increment in resampled field
        for i = 1:2
            # Compute longitudinal and transverse velocity increments of velocity v[i].
            v = vs[i]
            @timeit to "increments" begin
                velocity_increments!(dv_para, v, s, i, grid_step=grid_step)
                j = 3 - i  # transverse direction
                velocity_increments!(dv_perp, v, s, j, grid_step=grid_step)
            end
            @timeit to "statistics" update!(stats, (dv_para, dv_perp), r_ind; to=to)
        end
    end

    stats
end

"""
    velocity_increments!(dv, v, r, dim)

Compute increments from velocity component `v` along dimension `dim`.
"""
function velocity_increments!(Γ::AbstractMatrix, v::AbstractMatrix,
                              r::Integer, dim::Integer; grid_step = 1)
    @assert dim ∈ (1, 2)
    if grid_step .* size(Γ) != size(v)
        throw(DimensionMismatch("incompatible size of output array"))
    end
    inds_v = map(axes(v)) do ax
        range(first(ax), last(ax), step=grid_step)
    end
    # Note: increment `r` is in the resampled (input) field.
    δi, δj = dim == 1 ? (r, 0) : (0, r)  # increment in 2D input grid
    Ni, Nj = size(v)
    @inbounds for j_out ∈ axes(Γ, 2), i_out ∈ axes(Γ, 1)
        i = inds_v[1][i_out]
        j = inds_v[2][j_out]
        v1 = v[i, j]
        i = mod1(i + δi, Ni)
        j = mod1(j + δj, Nj)
        v2 = v[i, j]
        Γ[i_out, j_out] = v2 - v1
    end
    Γ
end

function Base.write(gbase::Union{HDF5.File,HDF5.Group}, stats::IncrementStats)
    names = ("Longitudinal", "Transverse")
    for (i, name) in enumerate(names)
        g = create_group(gbase, name)
        g["increments"] = collect(increments(stats))
        g["resampling_factor"] = stats.resampling_factor
        g["resampled_grid"] = stats.resampled_grid
        write(create_group(g, "Moments"), stats.moments[i])
        write(create_group(g, "Histogram"), stats.histogram[i])
    end
    gbase
end
