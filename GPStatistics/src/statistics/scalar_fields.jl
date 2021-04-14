export CirculationField, DissipationField

abstract type AbstractScalarField end
@inline Base.fieldname(f::AbstractScalarField, suffix) =
    Symbol(fieldname(f), suffix)

struct CirculationField <: AbstractScalarField end
Base.fieldname(::CirculationField) = :Γ

struct DissipationField <: AbstractScalarField end
Base.fieldname(::DissipationField) = :ε
