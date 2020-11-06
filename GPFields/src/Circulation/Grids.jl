"""
    Grids

Functions for construction of circulation "grids".
"""
module Grids

export to_grid, to_grid!
import Base: @kwdef

const IntOrBool = Union{Integer, Bool}

"""
    CirculationGrid{T}

Circulation of individual vortices on a grid.

The grid representation is such that positive and negative vortices are split
into two separate matrices.
"""
const CirculationGrid{T} = NTuple{2,AbstractMatrix{T}} where {T <: IntOrBool}

const POSITIVE = 1
const NEGATIVE = 2

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
    find_int(method::FindIntMethod, cell::AbstractArray; κ = 1)

Find "best" integer value within array of floating point values.

Values are divided by κ before searching for integers.
"""
function find_int end

function find_int(method::DiagonalSearch, cell::AbstractArray; κ = 1)
    # The first candidate is the corner (1, 1). Then we start looking by
    # diagonally increasing the position.
    s = zero(Int)
    good = false
    Base.require_one_based_indexing(cell)
    I = first(CartesianIndices(cell))  # = (1, 1)
    dI = I  # diagonal increment
    M = min(size(cell)...)
    for i = 1:M
        Γ = cell[I] / κ
        I += dI
        s = round(Int, Γ)
        isint = abs(Γ - s) ≤ method.int_threshold  # value is considered an integer
        if isint
            good = true
            break
        end
    end
    if !good
        error("""couldn't find Γ/κ ∈ {-1, 0, 1}.
              Try increasing the loop size or int_threshold.""")
    end
    s
end

function find_int(::BestInteger, cell::AbstractArray; κ = 1)
    s = zero(Int)
    err_best = Inf
    for n in eachindex(cell)
        Γ = cell[n] / κ
        Γ_int = round(Int, Γ)
        err = abs(Γ - Γ_int)
        if err < err_best
            err_best = err :: Float64
            s = Γ_int
        end
    end
    s
end

function find_int(::RoundAverage, cell::AbstractArray; κ = 1)
    mean = sum(cell) / (length(cell) * κ)
    round(Int, mean)
end

"""
    to_grid!(g::CirculationGrid, Γ::AbstractMatrix, method = BestInteger();
             κ = 1, cell_size = (2, 2), cleanup = false)

Convert small-scale circulation field to its grid representation.

The `method` argument determines the way integer values of `Γ / κ` are identified.

The `cell_size` optional argument determines the maximum dimensions of a
subcell, from which the circulation of a grid cell will be determined. For
instance, if `cell_size = (2, 2)` (the default), 4 neighbouring points in `Γ`
are taken into account to decide on each value of the grid `g`.

If `cleanup = true`, removes vortices that were possibly identified twice or more.
See [`cleanup_grid!`](@ref) for details.

See also [`to_grid`](@ref).
"""
function to_grid!(g::CirculationGrid, Γ::AbstractMatrix,
                  method::FindIntMethod = BestInteger(); kws...)
    steps = size(Γ) .÷ size(g[POSITIVE])
    to_grid!(g, Γ, steps, method; kws...)
end

function to_grid!(g::CirculationGrid{T}, Γ::AbstractMatrix, steps::Dims,
                  method::FindIntMethod = BestInteger();
                  cleanup = false, cell_size = (2, 2), kws...) where {T}
    gpos = g[POSITIVE]
    gneg = g[NEGATIVE]
    @assert size(gpos) == size(gneg)
    @assert steps .* size(gpos) == size(Γ)
    fill!.(g, zero(T))
    for I in CartesianIndices(gpos)
        cell = make_cell(Γ, I, steps, cell_size)
        sign, val = cell_spin(cell, method; kws...)
        @inbounds g[sign][I] = val
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

cleanup_grid!(g::CirculationGrid) = map(cleanup_grid!, g)

function cell_spin(cell::AbstractArray, method; kws...)
    s = find_int(method, cell; kws...)
    if s ≥ 0
        POSITIVE, s
    else
        NEGATIVE, -s
    end
end

function make_cell(Γ, I, steps, cell_size)
    ranges = map(Tuple(I), steps, cell_size) do i, δ, c
        h = min(c, δ)  # only consider corner values: (1:2, 1:2)
        j = (i - 1) * δ
        (j + 1):(j + h)
    end
    view(Γ, ranges...)
end

"""
    to_grid(Γ::AbstractMatrix, steps, [method::FindIntMethod = BestInteger()],
            [::Type{T} = Bool]; kws...)

Convert small-scale circulation field to its grid representation.

The `steps` represent the integer step size of the grid. For instance, if `steps
= (8, 8)`, the output grid is 8×8 times coarser than the dimensions of `Γ`.

The algorithm divides `Γ` into cells of size given by `steps` (8×8 cells in the
example). Then, according to the chosen [`FindIntMethod`](@ref), the circulation
within that cell is given a single integer value.

Ideally, the grid step should be equal to the loop size used to compute the
circulation. Moreover, if `T` is `Bool`, the loop size must be small enough so
that circulations are within the set `{0, ±κ}`.

See [`to_grid!`] for details and for possible keyword arguments.
"""
function to_grid(Γ::AbstractMatrix, steps::Dims,
                 method::FindIntMethod = BestInteger(),
                 ::Type{T} = Bool; kws...) where {T}
    gpos = Array{T}(undef, size(Γ) .÷ steps)
    g = (gpos, similar(gpos))
    to_grid!(g, Γ, steps, method; kws...)
end

@inline to_grid(Γ::AbstractMatrix, steps::Dims, ::Type{T}; kws...) where {T} =
    to_grid(Γ, steps, BestInteger(), T; kws...)

end
