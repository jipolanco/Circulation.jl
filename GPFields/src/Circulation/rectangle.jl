"""
    Rectangle{T}

Defines a rectangular loop in a two-dimensional geometry.

---

    Rectangle(x::NTuple{2}, r::NTuple{2})

Define rectangle passing through corners `x` and `x + r`.

"""
struct Rectangle{T}   # T: coordinate type (e.g. Int, Float64)
    x :: NTuple{2,T}  # lower-left corner
    r :: NTuple{2,T}  # length along each orientation

    Rectangle(x::NTuple{2,T}, r) where {T} = new{T}(x, r)
end

Base.eltype(::Type{Rectangle{T}}) where {T} = T

# Scale loop
Base.:*(a, loop::Rectangle) = Rectangle(a .* loop.x, a .* loop.r)
Base.:*(loop::Rectangle, a) = a * loop
Base.:/(loop::Rectangle, a) = loop * inv(a)

# Shift loop
Base.:+(a, loop::Rectangle) = Rectangle(a .+ loop.x, loop.r)
Base.:+(loop::Rectangle, a) = a + loop
Base.:-(loop::Rectangle, a) = -a + loop

function get_corners(loop::Rectangle)
    x1, y1 = loop.x
    x2, y2 = loop.x .+ loop.r
    p0 = (x1, y1)
    p1 = (x2, y1)
    p2 = (x2, y2)
    p3 = (x1, y2)
    p0, p1, p2, p3
end

get_centre(loop::Rectangle) = loop.x .+ loop.r ./ 2

function Base.show(io::IO, loop::Rectangle)
    xc = get_centre(loop)
    print(io, "Rectangle of size ", loop.r, " centred at ", xc)
end

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
                          rs::Dims{2}, step = 1; centre_cells = false)
        rs_offset = if centre_cells
            # Half radius (truncated if rs has odd numbers...).
            # This is just to make sure that the element Γ[i, j] corresponds to the
            # loop centred at (x[i], y[j]), which is nice for plotting.
            rs .>> 1
        else
            (0, 0)
        end :: Dims{2}
        inds = map(inds_all, rs_offset) do ax, offset
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

LoopIterator(x, args...; kws...) = LoopIterator(axes(x), args...; kws...)

Base.size(l::LoopIterator) = length.(l.inds)

@inline function Base.getindex(l::LoopIterator, i, j)
    a, b = l.inds
    @boundscheck checkbounds(a, i)
    @boundscheck checkbounds(b, j)
    @inbounds Rectangle((a[i], b[j]), l.rs)
end
