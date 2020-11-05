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

Base.show(io::IO, d::DiagonalSearch) = print(io, "DiagonalSearch(d.int_threshold)")

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
    good, s
end

function find_int(::BestInteger, cell::AbstractArray; κ = 1)
    s = zero(Int)
    err_best = Inf
    for I in CartesianIndices(cell)
        Γ = cell[I] / κ
        ω = sum(n -> abs2(n - 1), Tuple(I))  # radial weight
        Γ_int = round(Int, Γ)
        err = abs(Γ - Γ_int) * (1 + ω)
        if err < err_best
            err_best = err :: Float64
            s = Γ_int
        end
    end
    good = true
    good, s
end

function find_int(::RoundAverage, cell::AbstractArray; κ = 1)
    mean = sum(cell) / (length(cell) * κ)
    good = true
    good, round(Int, mean)
end

"""
    to_grid!(g::CirculationGrid, Γ::AbstractMatrix, method = BestInteger(); κ = 1)

Convert small-scale circulation field to its grid representation.

The `method` argument determines the way integer values of `Γ / κ` are identified.

See also [`to_grid`](@ref).
"""
function to_grid!(g::CirculationGrid, Γ::AbstractMatrix,
                  method::FindIntMethod = BestInteger(); kws...)
    steps = size(Γ) .÷ size(g[POSITIVE])
    to_grid!(g, Γ, steps, method; kws...)
end

function to_grid!(g::CirculationGrid{T}, Γ::AbstractMatrix, steps::Dims,
                  method::FindIntMethod = BestInteger(); kws...) where {T}
    gpos = g[POSITIVE]
    gneg = g[NEGATIVE]
    @assert size(gpos) == size(gneg)
    @assert steps .* size(gpos) == size(Γ)
    fill!.(g, zero(T))
    for I in CartesianIndices(gpos)
        cell = make_cell(Γ, I, steps)
        sign, val = cell_spin(cell, method; kws...)
        @inbounds g[sign][I] = val
    end
    g
end

function cell_spin(cell::AbstractArray, method; kws...)
    good, s = find_int(method, cell; kws...)
    if !good
        error("""couldn't find Γ/κ ∈ {-1, 0, 1}.
              Try increasing the loop size or int_threshold.""")
    end
    if s ≥ 0
        POSITIVE, s
    else
        NEGATIVE, -s
    end
end

function make_cell(Γ, I, steps)
    ranges = map(Tuple(I), steps) do i, δ
        h = δ >> 1  # half the total step
        j = (i - 1) * δ
        (j + 1):(j + h)
    end
    view(Γ, ranges...)
end

"""
    to_grid(Γ::AbstractMatrix, steps, [method = BestInteger()], [::Type{T} = Bool];
            kws...)

Convert small-scale circulation field to its grid representation.

The `steps` represent the integer step size of the grid. For instance, if `steps
= (8, 8)`, the output grid is 8×8 times coarser than the dimensions of `Γ`.

The algorithm divides `Γ` into cells of size given by `steps` (8×8 cells in the
example). Only the lower left quarter of each cell (a 4×4 matrix) is considered
to determine the circulation of that cell. This is to avoid duplicated
identification of vortices that influence multiple neighbouring cells.

Ideally, the grid step should be equal to the loop size used to compute the
circulation. Moreover, if `T` is `Bool`, the loop size must be small enough so
that circulations are within the set `{0, ±κ}`.

See [`to_grid!`] for possible keyword arguments.
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
