"""
    Grids

Functions for construction of circulation "grids".
"""
module Grids

export CirculationGrid, to_grid, to_grid!
import Base: @kwdef
import Base.Threads: @threads

const IntOrBool = Union{Integer, Bool}

# This can save some memory, but will probably cause problems when running with
# threads.
const USE_BITARRAY = false

"""
    CirculationGrid{T}

Circulation of individual vortices on a grid.

The grid representation is such that positive and negative vortices are split
into two separate matrices.
"""
struct CirculationGrid{T, Mat <: AbstractMatrix{T}}
    positive :: Mat
    negative :: Mat
    function CirculationGrid(pos::Mat, neg::Mat) where {Mat}
        size(pos) == size(neg) ||
            throw(ArgumentError("matrices must have the same dimensions"))
        T = eltype(Mat)
        new{T, Mat}(pos, neg)
    end
end

function CirculationGrid{T}(init, dims...) where {T}
    Mat = (USE_BITARRAY && T === Bool) ? BitArray{2} : Array{T,2}
    pos = Mat(init, dims...)
    neg = Mat(init, dims...)
    CirculationGrid(pos, neg)
end

Base.eltype(::Type{<:CirculationGrid{T}}) where {T} = T
Base.size(g::CirculationGrid) = size(g.positive)
Base.CartesianIndices(g::CirculationGrid) = CartesianIndices(g.positive)

matrices(g::CirculationGrid) = (g.positive, g.negative)

Base.similar(g::CirculationGrid, etc...) =
    CirculationGrid(map(x -> similar(x, etc...), matrices(g))...)

function Base.fill!(g::CirculationGrid, x)
    map(A -> fill!(A, x), matrices(g))
    g
end

const DEFAULT_INT_THRESHOLD = 0.05

"""
    FindIntMethod

Abstract type representing an integer finding method within an array of floating
point values.
"""
abstract type FindIntMethod end

"""
    DiagonalSearch <: FindIntMethod

Find the first value that is sufficiently close to an integer.

Cells are searched diagonally starting from index `(1, 1)`.

Takes a threshold (should be in ``[0, 0.5]``) as an optional parameter:

    DiagonalSearch(; int_threshold = $DEFAULT_INT_THRESHOLD)

"""
@kwdef struct DiagonalSearch <: FindIntMethod
    int_threshold :: Float64 = DEFAULT_INT_THRESHOLD
end

Base.show(io::IO, d::DiagonalSearch) = print(io, "DiagonalSearch(", d.int_threshold, ")")

"""
    BestInteger <: FindIntMethod

Find the value that is closest to an integer among values of a cell.

This is done by determining the floating point value that is closest to an
integer.

Gives priority to values on the lower left corner of the cell, assigning
increasingly larger error weights as the distance from this corner increases.
"""
struct BestInteger <: FindIntMethod end

Base.show(io::IO, ::BestInteger) = print(io, "BestInteger()")

"""
    RoundAverage <: FindIntMethod

Compute average value within a cell, then round it to the nearest integer.
"""
struct RoundAverage <: FindIntMethod end

Base.show(io::IO, ::RoundAverage) = print(io, "RoundAverage()")

"""
    find_int(method::FindIntMethod, cell::AbstractArray; ?? = 1)

Find "best" integer value within array of floating point values.

Values are divided by ?? before searching for integers.
"""
function find_int end

function find_int(method::DiagonalSearch, cell::AbstractArray; ?? = 1)
    # The first candidate is the corner (1, 1). Then we start looking by
    # diagonally increasing the position.
    s = zero(Int)
    good = false
    Base.require_one_based_indexing(cell)
    I = first(CartesianIndices(cell))  # = (1, 1)
    dI = I  # diagonal increment
    M = min(size(cell)...)
    for i = 1:M
        ?? = cell[I] / ??
        I += dI
        s = round(Int, ??)
        isint = abs(?? - s) ??? method.int_threshold  # value is considered an integer
        if isint
            good = true
            break
        end
    end
    if !good
        error("""couldn't find ??/?? ??? {-1, 0, 1}.
              Try increasing the loop size or int_threshold.""")
    end
    s
end

function find_int(::BestInteger, cell::AbstractArray; ?? = 1)
    s = zero(Int)
    err_best = Inf
    for n in eachindex(cell)
        ?? = cell[n] / ??
        ??_int = round(Int, ??)
        err = abs(?? - ??_int)
        if err < err_best
            err_best = err :: Float64
            s = ??_int
        end
    end
    s
end

function find_int(::RoundAverage, cell::AbstractArray; ?? = 1)
    mean = sum(cell) / (length(cell) * ??)
    round(Int, mean)
end

"""
    to_grid!(g::CirculationGrid, ??::AbstractMatrix, method = BestInteger();
             ?? = 1, cell_size = (2, 2), force_unity = false, cleanup = false)

Convert small-scale circulation field to its grid representation.

The `method` argument determines the way integer values of `?? / ??` are identified.

The `cell_size` optional argument determines the maximum dimensions of a
subcell, from which the circulation of a grid cell will be determined. For
instance, if `cell_size = (2, 2)` (the default), 4 neighbouring points in `??`
are taken into account to decide on each value of the grid `g`.

If `force_unity = true`, then values of `?? / ??` in `{-1, 0, 1}` will be
enforced. For instance, a measured value of 4 will be brought down to 1. This
probably makes sense when the loop sizes used to compute the circulation are
very small.

If `cleanup = true`, removes vortices that were possibly identified twice or more.
See [`cleanup_grid!`](@ref) for details.

See also [`to_grid`](@ref).
"""
function to_grid!(g::CirculationGrid, ??::AbstractMatrix,
                  method::FindIntMethod = BestInteger(); kws...)
    steps = size(??) .?? size(g)
    to_grid!(g, ??, steps, method; kws...)
end

function to_grid!(g::CirculationGrid, ??::AbstractMatrix, steps::Dims,
                  method::FindIntMethod = BestInteger();
                  cleanup = false, force_unity = false,
                  cell_size = (2, 2), kws...)
    T = eltype(g)
    @assert steps .* size(g) == size(??)
    fill!(g, zero(T))
    @threads for I in CartesianIndices(g)
        cell = make_cell(??, I, steps, cell_size)
        grid, val = cell_spin(g, cell, method; kws...)
        if force_unity && val > 1
            val = one(val)
        end
        @inbounds grid[I] = val
    end
    if cleanup
        cleanup_grid!(g)
    end
    g
end

"""
    cleanup_grid!(g::AbstractMatrix)
    cleanup_grid!(g::CirculationGrid)

Remove possible duplicates from grid of positive or negative vortices.

Assumes that two vortices of same sign cannot be direct neighbours.
"""
function cleanup_grid!(g::AbstractMatrix)
    increments = (
        CartesianIndex(1, 0),
        CartesianIndex(0, 1),
        # CartesianIndex(1, 1),
    )
    @inbounds for I in CartesianIndices(g)
        v = g[I]
        iszero(v) && continue
        for dI in increments
            J = I + dI
            if checkbounds(Bool, g, J) && g[J] == v
                g[I] = 0  # remove current vortex
                break
            end
        end
    end
    g
end

function cleanup_grid!(g::CirculationGrid)
    map(cleanup_grid!, matrices(g))
    g
end

function cell_spin(g, cell::AbstractArray, method; kws...)
    s = find_int(method, cell; kws...)
    if s ??? 0
        g.positive, s
    else
        g.negative, -s
    end
end

function make_cell(??, I, steps, cell_size)
    ranges = map(Tuple(I), steps, cell_size) do i, ??, c
        h = min(c, ??)  # only consider corner values: (1:2, 1:2)
        j = (i - 1) * ??
        (j + 1):(j + h)
    end
    view(??, ranges...)
end

"""
    to_grid(??::AbstractMatrix, steps, [method::FindIntMethod = BestInteger()],
            [::Type{T} = Bool]; kws...)

Convert small-scale circulation field to its grid representation.

The `steps` represent the integer step size of the grid. For instance, if `steps
= (8, 8)`, the output grid is 8??8 times coarser than the dimensions of `??`.

The algorithm divides `??` into cells of size given by `steps` (8??8 cells in the
example). Then, according to the chosen [`FindIntMethod`](@ref), the circulation
within that cell is given a single integer value.

Ideally, the grid step should be equal to the loop size used to compute the
circulation. Moreover, if `T` is `Bool`, the loop size must be small enough so
that circulations are within the set `{0, ????}`.

See [`to_grid!`] for details and for possible keyword arguments.
"""
function to_grid(??::AbstractMatrix, steps::Dims,
                 method::FindIntMethod = BestInteger(),
                 ::Type{T} = Bool; kws...) where {T}
    g = CirculationGrid{T}(undef, size(??) .?? steps)
    to_grid!(g, ??, steps, method; kws...)
end

@inline to_grid(??::AbstractMatrix, steps::Dims, ::Type{T}; kws...) where {T} =
    to_grid(??, steps, BestInteger(), T; kws...)

end
