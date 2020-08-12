module Circulation

using FFTW
using HDF5
using LinearAlgebra: mul!
using TimerOutputs
using Base.Threads

using GPFields

export IntegralField2D
export prepare!
export circulation, circulation!
export analyse!
export CirculationStats, IncrementStats, VelocityLikeFields, reset!

include("loops/rectangle.jl")
include("statistics/statistics.jl")

# Type definitions
const ComplexArray{T,N} = AbstractArray{Complex{T},N} where {T<:Real,N}
const RealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

"""
    IntegralField2D{T}

Contains arrays required to compute the integral of a 2D vector field
`(vx, vy)`.

Also contains FFT plans and array buffers for computation of integral data.

---

    IntegralField2D(Nx, Ny, [T = Float64]; L)

Allocate `IntegralField2D` of dimensions `(Nx, Ny)`.

The domain size must be given as a tuple `L = (Lx, Ly)`.

---

    IntegralField2D(A::AbstractMatrix{<:AbstractFloat}; L)

Allocate `IntegralField2D` having dimensions and type compatibles with input
matrix.
"""
struct IntegralField2D{T, PlansFW, PlansBW}
    N :: NTuple{2,Int}      # Nx, Ny
    L :: NTuple{2,Float64}  # domain size: Lx, Ly

    # Mean value Ux(y), Uy(x).
    U :: NTuple{2,Vector{T}}  # lengths: Ny, Nx

    # Integral fields wx(x, y), wy(x, y).
    w :: NTuple{2,Matrix{T}}  # [Nx, Ny]

    # FFTW plans (plan_x, plan_y)
    plans_fw :: PlansFW  # forwards
    plans_bw :: PlansBW  # backwards

    # Buffer arrays for FFTs (one per thread).
    bufs :: NTuple{2,Matrix{T}}  # lengths: Nx, Ny
    bufs_f :: NTuple{2,Matrix{Complex{T}}}

    # Wave numbers
    ks :: NTuple{2,Frequencies{Float64}}

    function IntegralField2D(Nx, Ny, ::Type{T} = Float64; L) where {T}
        Ns = (Nx, Ny)
        U = Vector{T}.(undef, (Ny, Nx))
        w = ntuple(_ -> Matrix{T}(undef, Nx, Ny), 2)

        fs = 2pi .* Ns ./ L  # sampling frequency
        ks = rfftfreq.(Ns, fs)

        Nth = nthreads()
        bufs = Array{T}.(undef, Ns, Nth)
        bufs_f = Array{Complex{T}}.(undef, length.(ks), Nth)

        # Make sure that plans are NOT threaded (these are small transforms)
        FFTW.set_num_threads(1)
        plans_fw = plan_rfft.(view.(bufs, :, 1), flags=FFTW.MEASURE)
        plans_bw = plan_irfft.(view.(bufs_f, :, 1), Ns, flags=FFTW.MEASURE)

        Pfw = typeof(plans_fw)
        Pbw = typeof(plans_bw)

        new{T, Pfw, Pbw}(Ns, L, U, w, plans_fw, plans_bw, bufs, bufs_f, ks)
    end

    IntegralField2D(A::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat} =
        IntegralField2D(size(A)..., T; kwargs...)
end

Base.ndims(::IntegralField2D) = 2
Base.size(I::IntegralField2D) = I.N
Base.eltype(::Type{IntegralField2D{T}}) where {T} = T

"""
    LoopIterator

Iterator over a set of rectangular loops of a fixed size on a 2D field.
"""
struct LoopIterator{
        Inds <: Tuple{Vararg{AbstractRange,2}}
    } <: AbstractMatrix{Rectangle{Int}}
    rs   :: Dims{2}  # rectangle size (rx, ry)
    step :: Int      # step between grid points (if 1, iterate over all grid points)
    inds :: Inds
    function LoopIterator(inds_all::Tuple{Vararg{AbstractRange,2}},
                          rs::Dims{2}, step = 1)
        # Half radius (truncated if rs has odd numbers...).
        # This is just to make sure that the element Γ[i, j] corresponds to the
        # loop centred at (x[i], y[j]), which is nice for plotting.
        rs_half = rs .>> 1
        inds = map(inds_all, rs_half) do ax, offset
            # This formula is to make sure that on refined grids (when step > 1),
            # the circulation is computed around exactly the same loops as the
            # original grid.
            range(first(ax), last(ax), step=step) .- offset
        end
        # inds = Iterators.product(inds_loops...)
        Inds = typeof(inds)
        new{Inds}(rs, step, inds)
    end
end

LoopIterator(x, args...) = LoopIterator(axes(x), args...)

Base.size(l::LoopIterator) = length.(l.inds)

@inline function Base.getindex(l::LoopIterator, i, j)
    a, b = l.inds
    @boundscheck checkbounds(a, i)
    @boundscheck checkbounds(b, j)
    @inbounds Rectangle((a[i], b[j]), l.rs)
end

"""
    prepare!(I::IntegralField2D{T}, v)

Set values of the integral fields from 2D vector field `v = (vx, vy)`.
"""
function prepare!(I::IntegralField2D{T},
                  v::NTuple{2,AbstractMatrix{T}}) where {T}
    Ns = size(I)
    if any(Ref(Ns) .!= size.(v))
        throw(DimensionMismatch("incompatible array sizes"))
    end
    prepare!(I, v[1], Val(1))
    prepare!(I, v[2], Val(2))
    I
end

function prepare!(I::IntegralField2D, u, ::Val{c}) where {c}
    @assert c in (1, 2)
    U = I.U[c]
    k = I.ks[c]
    plan_fw = I.plans_fw[c]
    plan_bw = I.plans_bw[c]
    ubuf_t = I.bufs[c]
    uf_t = I.bufs_f[c]
    w = I.w[c]

    @assert k[1] == 0
    Nk = length(k)

    Ns = size(w)
    @assert size(ubuf_t, 1) == size(u, c) == Ns[c]

    Nc, Nother = (c === 1) ? (Ns[1], Ns[2]) : (Ns[2], Ns[1])

    @threads for j = 1:Nother
        t = threadid()
        ubuf = view(ubuf_t, :, t)
        uf = view(uf_t, :, t)

        for i = 1:Nc
            ind = (c === 1) ? CartesianIndex((i, j)) : CartesianIndex((j, i))
            ubuf[i] = u[ind]
        end

        mul!(uf, plan_fw, ubuf)  # apply FFT

        # Copy mean value and then set it to zero.
        # Note: the mean value must be normalised by the input data length.
        U[j] = Real(uf[1]) / Nc
        uf[1] = 0

        for i = 2:Nk
            uf[i] /= im * k[i]  # w(k) -> w(k) / ik
        end

        mul!(ubuf, plan_bw, uf)  # apply inverse FFT

        for i = 1:Nc
            ind = (c === 1) ? CartesianIndex((i, j)) : CartesianIndex((j, i))
            w[ind] = ubuf[i]
        end
    end

    I
end

"""
    circulation!(Γ::AbstractMatrix, I::IntegralField2D, rs; grid_step=1)

Compute circulation on a 2D slice around loops with a fixed rectangle shape.

## Parameters

- `Γ`: output real matrix containing the circulation associated to each point of
  the physical grid.

- `I`: `IntegralField2D` containing integral information of 2D vector field.

- `rs = (r_x, r_y)`: rectangle dimensions.

- `grid_step`: optional parameter allowing to visit only a subset of grid
  points. For instance, if `grid_step = 2`, one out of two grid points is
  considered in every direction. Note that the dimensions of `Γ` must be
  consistent with this parameter.

"""
function circulation!(
        Γ::AbstractMatrix{<:Real}, I::IntegralField2D, rs::NTuple{2,Int};
        grid_step::Int = 1,
    )
    if grid_step .* size(Γ) != size(I)
        throw(DimensionMismatch("incompatible size of output array"))
    end
    loops = LoopIterator(I, rs, grid_step)
    @threads for j ∈ axes(Γ, 2)
        for i ∈ axes(Γ, 1)
            @inbounds loop = loops[i, j]
            @inbounds Γ[i, j] = circulation(loop, I)
        end
    end
    Γ
end

"""
    circulation(loop::Rectangle, I::IntegralField2D)

Compute circulation around the given loop on a 2D slice.

The rectangle must be given in integer coordinates (grid indices).

See also `circulation!`.
"""
function circulation(loop::Rectangle{Int}, I::IntegralField2D)
    # Coordinates (including end point).
    xs = LinRange.(0, I.L, I.N .+ 1)

    # Indices of rectangle points.
    ia, ja = loop.x
    ib, jb = loop.x .+ loop.r

    # Coordinates of rectangle corners, taking periodicity into account.
    (ia, xa), (ja, ya) = _make_coordinate.(xs, (ia, ja))
    (ib, xb), (jb, yb) = _make_coordinate.(xs, (ib, jb))

    dx = xb - xa
    dy = yb - ya

    int_x_ya = I.U[1][ja] * dx + I.w[1][ib, ja] - I.w[1][ia, ja]
    int_x_yb = I.U[1][jb] * dx + I.w[1][ib, jb] - I.w[1][ia, jb]

    int_y_xa = I.U[2][ia] * dy + I.w[2][ia, jb] - I.w[2][ia, ja]
    int_y_xb = I.U[2][ib] * dy + I.w[2][ib, jb] - I.w[2][ib, ja]

    int_x_ya + int_y_xb - int_x_yb - int_y_xa
end

function _make_coordinate(x::AbstractVector, i::Int)
    # Return coordinate x[i] taking periodicity into account.
    @assert x[1] == 0
    N = length(x) - 1
    L = x[N + 1]
    x0 = zero(L)
    while i <= 0
        i += N
        x0 -= L
    end
    while i > N
        i -= N
        x0 += L
    end
    i, x0 + x[i]
end

end
