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

"""
    BestInteger <: FindIntMethod

Find the value that is closest to an integer among values of a cell.

Only half of the cell is considered along each dimension. For instance, if the
cell has dimensions 8×8, only the lower left corner of dimensions 4×4 is
considered. This is to avoid single vortices from being identified multiple
times, by neighbouring cells.
"""
struct BestInteger <: FindIntMethod end

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
    Base.require_one_based_indexing(cell)
    hs = size(cell) .>> 1  # half size of the cell
    subcell = view(cell, Base.OneTo.(hs)...)
    s = zero(Int)
    err_best = 1.0
    for v in subcell
        Γ = v / κ
        Γ_int = round(Int, Γ)
        err = abs(Γ - Γ_int)
        if err < err_best
            err_best = err
            s = Γ_int
        end
    end
    good = true
    good, s
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
        j = (i - 1) * δ
        (j + 1):(j + δ)
    end
    view(Γ, ranges...)
end

"""
    to_grid(Γ::AbstractMatrix, steps, [method = BestInteger()], [::Type{T} = Bool];
            kws...)

Convert small-scale circulation field to its grid representation.

The `steps` represent the integer step size of the grid. For instance, if `steps
= (4, 4)`, the output grid is 4×4 times coarser than the dimensions of `Γ`.

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