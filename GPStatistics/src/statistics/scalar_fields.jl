export CirculationField, DissipationField

using LinearAlgebra: ldiv!

abstract type AbstractScalarField{divide_by_area} end

@inline Base.fieldname(f::AbstractScalarField, suffix) =
    Symbol(fieldname(f), suffix)

divide_by_area(::AbstractScalarField{D}) where {D} = D

struct CirculationField{divide_by_area} <: AbstractScalarField{divide_by_area}
    @inline CirculationField(; divide_by_area::Bool = false) =
        new{divide_by_area}()
end

Base.fieldname(::CirculationField) = :Γ

metadata(f::CirculationField) = (
    "divided_by_area" => divide_by_area(f),
)

struct DissipationField{divide_by_area} <: AbstractScalarField{divide_by_area}
    inplane :: Bool
    ν :: Float64
    @inline function DissipationField(;
            divide_by_area::Bool = true,
            inplane_only::Bool = false,
            ν = 1.0,
        )
        new{divide_by_area}(inplane_only, ν)
    end
end

compute_inplane(f::DissipationField) = f.inplane
Base.fieldname(::DissipationField) = :ε

metadata(f::DissipationField) = (
    "divided_by_area" => divide_by_area(f),
    "inplane (2D)" => compute_inplane(f),
    "viscosity" => f.ν,
)

"""
    compute_from_velocity!(
        field::DissipationField, ε::AbstractArray{<:Real,2},
        v_hat::NTuple{2, AbstractArray{<:Complex,2}};
        ks, fft_plan, buf, buf_hat,
    )

Compute in-plane dissipation field, ``ε_z``, from in-plane velocity field in
Fourier space, `(\\hat{v}_x, \\hat{v}_y)``.

Definition:

```math
ε_z = 2ν (S_{xx}^2 + 2 S_{xy}^2 + S_{xz}^2)
```

where ``S_{ij} = (𝜕_i v_j + 𝜕_j v_i) / 2``.
"""
function compute_from_velocity!(
        field::DissipationField, ε::AbstractArray{<:Real,2},
        v_hat::NTuple{2, AbstractArray{<:Complex,2}};
        ks, fft_plan, buf, buf_hat,
    )
    @assert compute_inplane(field)
    @assert size(buf) == size(ε)
    @assert size(buf_hat) == size(v_hat[1]) == length.(ks)

    ν = field.ν

    # 1. Sxx = 𝜕_x v_x
    @inbounds for (I, vx) in pairs(IndexCartesian(), v_hat[1])
        i = Tuple(I)[1]
        kx = ks[1][i]
        buf_hat[I] = im * kx * vx
    end
    ldiv!(buf, fft_plan, buf_hat)
    ε .= buf.^2

    # 2. Syy = 𝜕_y v_y
    @inbounds for (I, vy) in pairs(IndexCartesian(), v_hat[2])
        j = Tuple(I)[2]
        ky = ks[2][j]
        buf_hat[I] = im * ky * vy
    end
    ldiv!(buf, fft_plan, buf_hat)
    ε .+= buf.^2

    # 3. Sxy = (𝜕_x v_y + 𝜕_y v_x) / 2
    @inbounds for I in CartesianIndices(buf_hat)
        kx, ky = getindex.(ks, Tuple(I))
        buf_hat[I] = im * (kx * v_hat[2][I] + ky * v_hat[1][I]) / 2
    end
    ldiv!(buf, fft_plan, buf_hat)

    ε .= (ε .+ 2 .* buf.^2) .* 2ν

    ε
end

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
