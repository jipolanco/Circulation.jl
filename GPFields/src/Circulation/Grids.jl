"""
    Grids

Functions for construction of circulation "grids".
"""
module Grids

export to_grid, to_grid!

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

"""
    to_grid!(g::CirculationGrid, Γ::AbstractMatrix; κ = 1, int_threshold = 0.05)

Convert small-scale circulation field to its grid representation.

The `int_threshold` argument determines the threshold below which a real value
is interpreted as an integer. For instance, 2.07 is rounded to 2 only if
`int_threshold ≥ 0.07`.

See also [`to_grid`](@ref).
"""
function to_grid!(g::CirculationGrid, Γ::AbstractMatrix; kws...)
    steps = size(Γ) .÷ size(g[POSITIVE])
    to_grid!(g, Γ, steps; kws...)
end

function to_grid!(g::CirculationGrid{T}, Γ::AbstractMatrix, steps::Dims;
                  kws...) where {T}
    gpos = g[POSITIVE]
    gneg = g[NEGATIVE]
    @assert size(gpos) == size(gneg)
    @assert steps .* size(gpos) == size(Γ)
    fill!.(g, zero(T))
    for I in CartesianIndices(gpos)
        cell = make_cell(Γ, I, steps)
        sign, val = cell_spin(cell; kws...)
        @inbounds g[sign][I] = val
    end
    g
end

function cell_spin(cell::AbstractArray; κ = 1, int_threshold = 0.05)
    # Search for the first circulation in {-1, 0, 1} within the cell.
    # This allows to skip spurious values.
    # The first candidate is the corner (1, 1). Then we start looking by
    # diagonally increasing the position.
    s = zero(Int)
    good = false
    M = min(size(cell)...)
    Base.require_one_based_indexing(cell)
    I = first(CartesianIndices(cell))  # = (1, 1)
    dI = I  # diagonal increment
    for i = 1:M
        Γ = cell[I] / κ
        I += dI
        s = round(Int, Γ)
        isint = abs(Γ - s) ≤ int_threshold  # value is considered an integer
        if isint && abs(s) ≤ 1
            good = true
            break
        end
    end
    if !good
        error("couldn't find Γ/κ ∈ {-1, 0, 1}. Maybe try modifying the loop size?")
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
    to_grid(Γ::AbstractMatrix, steps, ::Type{T} = Bool; kws...)

Convert small-scale circulation field to its grid representation.

The `steps` represent the integer step size of the grid. For instance, if `steps
= (4, 4)`, the output grid is 4×4 times coarser than the dimensions of `Γ`.

Ideally, the grid step should be equal to the loop size used to compute the
circulation. Moreover, if `T` is `Bool`, the loop size must be small enough so
that circulations are within the set `{0, ±κ}`.

See [`to_grid!`] for possible keyword arguments.
"""
function to_grid(Γ::AbstractMatrix, steps::Dims, ::Type{T} = Bool;
                 kws...) where {T}
    gpos = Array{T}(undef, size(Γ) .÷ steps)
    g = (gpos, similar(gpos))
    to_grid!(g, Γ, steps; kws...)
end

end
