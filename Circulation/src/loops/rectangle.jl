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

function Base.show(io::IO, loop::Rectangle)
    xc = loop.x .+ loop.r ./ 2
    print(io, "Rectangle of size ", loop.r, " centred at ", xc)
end
