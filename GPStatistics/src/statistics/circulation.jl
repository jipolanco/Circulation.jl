# Circulation-specific statistics.

using GPFields.Circulation.Kernels: AbstractKernel
using LinearAlgebra: mul!

"""
    CirculationStats <: AbstractFlowStats

Circulation statistics, including moments and histograms.
"""
struct CirculationStats{
        Loops,
        StatsTuple <: BaseStatsTuple,
    } <: AbstractFlowStats
    Nr           :: Int    # number of loop sizes
    loop_sizes   :: Loops  # vector of length Nr (can be integers or convolution kernels)
    resampling_factor :: Int
    resampled_grid :: Bool
    stats          :: StatsTuple
end

function Base.zero(s::CirculationStats)
    CirculationStats(s.Nr, s.loop_sizes, s.resampling_factor,
                     s.resampled_grid, zero.(s.stats))
end

@inline sampling_method(::Type{CirculationStats}, ::Type{<:Integer}) =
    PhysicalMethod()
@inline sampling_method(::Type{CirculationStats}, ::Type{<:AbstractKernel}) =
    ConvolutionMethod()
@inline sampling_method(::Type{<:CirculationStats{Loops}}) where {Loops} =
    sampling_method(CirculationStats, eltype(Loops))
@inline sampling_method(s::CirculationStats) = sampling_method(typeof(s))

"""
    CirculationStats(
        loop_sizes_or_kernels, stats_params;
        histogram::s.ParamsHistogram,
        moments::ParamsMoments,
        resampling_factor = 1,
        compute_in_resampled_grid = false,
    )

Construct and initialise statistics.

# Parameters

- `loop_sizes_or_kernels`: sizes of square circulation loops in grid step units.

  It may also be an array of convolution kernels (subtypes of `AbstractKernel`).
  All kernels must be of the same type (e.g. `RectangularKernel`). They will
  typically differ on their characteristic lengthscale (e.g. the size of a
  rectangular kernel).

- `stats_params`: tuple with parameters for different statistics.

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
        loop_sizes::AbstractArray{T} where {T <: Union{Real, AbstractKernel}},
        stats_params::Tuple{Vararg{BaseStatsParams}};
        resampling_factor=1,
        compute_in_resampled_grid=false,
    )
    resampling_factor >= 1 || error("resampling_factor must be positive")
    Nr = length(loop_sizes)
    stats = init_statistics.(stats_params, Nr)
    CirculationStats(Nr, loop_sizes, resampling_factor,
                     compute_in_resampled_grid, stats)
end

function allocate_fields(::PhysicalMethod, s::CirculationStats, args...; L, kwargs...)
    data = allocate_common_fields(args...; kwargs...)
    data_fields = allocate_fields(scalar_fields(s), data; with_ffts = false)
    (;
        data..., data_fields...,
        I = IntegralField2D(data.ps[1], L=L),
    )
end

function allocate_fields(
        ::ConvolutionMethod, s::CirculationStats, args...; L, kwargs...,
    )
    # Note that Γ is always computed in the resampled grid.
    # This overrides a possible `compute_in_resampled_grid` in kwargs, which
    # only applies to computation of final statistics (histograms, ...).
    data = allocate_common_fields(
        args...; kwargs..., compute_in_resampled_grid = true)
    Ns = data.dims_out
    ks = map((f, n, L) -> f(n, 2π * n / L), (rfftfreq, fftfreq), Ns, L)
    Ms = length.(ks)
    T = Float64
    FFTW.set_num_threads(nthreads())  # make sure that FFTs are threaded
    g_hat = DiscreteFourierKernel{T}(undef, ks...)
    plan = plan_rfft(data.vs[1], flags=FFTW.MEASURE)
    plan_inv = inv(plan)
    v_hat = map(similar_fft, data.vs)
    data_fields = allocate_fields(scalar_fields(s), data; with_ffts = true)
    (;
        data..., data_fields...,
        ks, plan, plan_inv, g_hat, v_hat,
    )
end

allocate_fields(s::CirculationStats, args...; kwargs...) =
    allocate_fields(sampling_method(s), s, args...; kwargs...)

function compute!(s::AbstractVector{S}, args...) where {S <: CirculationStats}
    compute!(sampling_method(S), s, args...)
end

function compute!(
        ::PhysicalMethod, stats_t::AbstractVector{<:CirculationStats},
        fields, vs, to,
    )
    Γ = fields.Γ
    Ip = fields.I
    fields_stats = (; Γ)

    # Set integral values with momentum data.
    @timeit to "prepare!" prepare!(Ip, vs)

    st1 = first(stats_t)

    with_dissipation = DissipationField() ∈ scalar_fields(st1)
    if with_dissipation
        throw(ArgumentError(
            "dissipation conditioning must be done with convolution method"
        ))
    end

    resampling = st1.resampling_factor
    grid_step = st1.resampled_grid ? 1 : resampling
    @assert grid_step .* size(Γ) == size(vs[1]) "incorrect dimensions of Γ"

    for (r_ind, r) in enumerate(st1.loop_sizes)
        s = resampling * r  # loop size in resampled field
        @timeit to "circulation!" circulation!(
            Γ, Ip, (s, s), grid_step=grid_step)
        @timeit to "statistics" update!(stats_t, fields_stats, r_ind; to=to)
    end

    stats_t
end

function compute!(
        ::ConvolutionMethod, stats_t::AbstractVector{<:CirculationStats},
        fields, vs, to,
    )
    Γ = fields.Γ
    g_hat = fields.g_hat
    v_hat = fields.v_hat

    @assert length(v_hat) == length(vs) == 2
    @timeit to "FFT(v)" map((v, vF) -> mul!(vF, fields.plan, v), vs, v_hat)
    @assert size(Γ) == size(vs[1]) "incorrect dimensions of Γ"

    st1 = first(stats_t)
    with_dissipation = DissipationField() ∈ scalar_fields(st1)

    if with_dissipation
        @timeit to "FFT(ε)" mul!(fields.ε_hat, fields.plan, fields.ε)
        ε_coarse = fields.ε_coarse
        fields_stats = (; Γ, ε = ε_coarse)
    else
        resampling = st1.resampling_factor
        Γ_stats = if resampling == 1 || st1.resampled_grid
            Γ
        else
            view(Γ, map(ax -> range(first(ax), last(ax), step=resampling),
                        axes(Γ))...)
        end
        fields_stats = (; Γ = Γ_stats)
    end

    for (r_ind, kernel) in enumerate(st1.loop_sizes)
        @timeit to "kernel!" materialise!(g_hat, kernel)
        @timeit to "circulation!" circulation!(
            Γ, v_hat, g_hat;
            buf = fields.Γ_hat, plan_inv = fields.plan_inv,
        )
        if with_dissipation
            @timeit to "convolve dissipation" convolve!(
                ε_coarse, fields.ε_hat, g_hat;
                buf = fields.Γ_hat, plan_inv = fields.plan_inv,
            )
        end
        @timeit to "statistics" update!(stats_t, fields_stats, r_ind; to=to)
    end

    stats_t
end

function Base.write(g::Union{HDF5.File,HDF5.Group}, stats::CirculationStats)
    write_loop_sizes(g, stats.loop_sizes)
    g["resampling_factor"] = stats.resampling_factor
    g["resampled_grid"] = stats.resampled_grid
    map(statistics(stats)) do s
        _write_stats_group(g, s)
    end
    g
end

function _write_stats_group(g, s::AbstractBaseStats)
    name = string(nameof(typeof(s)))
    gg = create_group(g, name)
    write(gg, s)
    g
end

write_loop_sizes(g, loop_sizes) = g["loop_sizes"] = collect(loop_sizes)

function write_loop_sizes(g, kernels::AbstractVector{<:AbstractKernel})
    g["kernel_shape"] = eltype(kernels) |> nameof |> string
    Ls = lengthscales.(kernels)
    g["kernel_lengths"] = as_array(Ls)
    As = Kernels.area.(kernels)
    g["kernel_area"] = As
    g["kernel_size"] = sqrt.(As)
    g
end

function as_array(x::AbstractArray{NTuple{N,T}}) where {N,T}
    r = reinterpret(T, x)
    reshape(r, N, size(x)...)
end
