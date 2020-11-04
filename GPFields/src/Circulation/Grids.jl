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
    to_grid!(g::CirculationGrid, Γ::AbstractMatrix; κ = 1)

Convert small-scale circulation field to its grid representation.

See also [`to_grid`](@ref).
"""
function to_grid!(g::CirculationGrid, Γ::AbstractMatrix; kws...)
    steps = size(Γ) .÷ size(g[POSITIVE])
    to_grid!(g, Γ, steps; kws...)
end

function to_grid!(g::CirculationGrid{T}, Γ::AbstractMatrix, steps::Dims;
                  κ = 1) where {T}
    gpos = g[POSITIVE]
    gneg = g[NEGATIVE]
    @assert size(gpos) == size(gneg)
    @assert steps .* size(gpos) == size(Γ)
    fill!.(g, zero(T))
    for I in CartesianIndices(gpos)
        cell = make_cell(Γ, I, steps)
        sign, val = cell_spin(cell; κ)
        @inbounds g[sign][I] = val
    end
    g
end

function cell_spin(cell::AbstractArray; κ)
    # Search for the first circulation in {-1, 0, 1} within the cell.
    # This allows to skip spurious values.
    Γ = zero(Int)
    good = false
    for val in cell
        Γ = round(Int, val / κ)
        if abs(Γ) ≤ 1
            good = true
            break
        end
    end
    if !good
        error("couldn't find Γ/κ ∈ {-1, 0, 1}. Maybe try decreasing the loop size?")
    end
    if Γ ≥ 0
        POSITIVE, Γ
    else
        NEGATIVE, -Γ
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
    to_grid(Γ::AbstractMatrix, steps, ::Type{T} = Bool; κ = 1)

Convert small-scale circulation field to its grid representation.

The `steps` represent the integer step size of the grid. For instance, if `steps
= (4, 4)`, the output grid is 4×4 times coarser than the dimensions of `Γ`.

Ideally, the grid step should be equal to the loop size used to compute the
circulation. Moreover, if `T` is `Bool`, the loop size must be small enough so
that circulations are within the set `{0, ±κ}`.
"""
function to_grid(Γ::AbstractMatrix, steps::Dims, ::Type{T} = Bool;
                 kws...) where {T}
    gpos = Array{T}(undef, size(Γ) .÷ steps)
    g = (gpos, similar(gpos))
    to_grid!(g, Γ, steps; kws...)
end

end
