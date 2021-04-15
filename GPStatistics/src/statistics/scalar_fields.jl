export CirculationField, DissipationField

abstract type AbstractScalarField end
@inline Base.fieldname(f::AbstractScalarField, suffix) =
    Symbol(fieldname(f), suffix)

struct CirculationField{divide_by_area} <: AbstractScalarField
    @inline CirculationField(; divide_by_area::Bool = false) =
        new{divide_by_area}()
end

divide_by_area(::CirculationField{D}) where {D} = D
Base.fieldname(::CirculationField) = :Γ

struct DissipationField <: AbstractScalarField end
Base.fieldname(::DissipationField) = :ε
divide_by_area(::DissipationField) = true  # dissipation is always divided by area

function find_field(
        ::Type{F},
        fields::Tuple{Vararg{AbstractScalarField}},
    ) where {F <: AbstractScalarField}
    _find_field(F, fields...)
end

function _find_field(::Type{F}, field, etc...) where {F}
    if field isa F
        field
    else
        _find_field(F, etc...)
    end
end

_find_field(::Type) = nothing
