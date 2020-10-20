"""
Kernels for computation of circulation using convolution with vorticity field.

Different kernels correspond to different loop shapes (e.g. rectangular, circular).
"""
module Kernels

export EllipsoidalKernel, RectangularKernel
export materialise, materialise!

using SpecialFunctions: besselj1

abstract type AbstractKernel end

wavenumbers(g::AbstractKernel) = g.ks

"""
    EllipsoidalKernel{T}

Describes a kernel for convolution with ellipsoidal step function on periodic
domain.

Note that the associated convolution kernel in Fourier space, constructed via
[`materialise!`](@ref), is based on Bessel functions.

---

    EllipsoidalKernel((Dx, Dy), (kx, ky))

Construct ellipsoidal kernel with diameters `(Dx, Dy)` and Fourier wave
numbers `(kx, ky)`.
"""
struct EllipsoidalKernel{T <: AbstractFloat, WaveNumbers} <: AbstractKernel
    diameters :: NTuple{2,T}
    ks :: NTuple{2,WaveNumbers}
    function EllipsoidalKernel(Ds::NTuple{2,T}, ks) where {T}
        new{float(T), typeof(first(ks))}(Ds, ks)
    end
end

"""
    EllipsoidalKernel(D, (kx, ky))

Construct circular kernel with diameter `D`.
"""
EllipsoidalKernel(D::Real, ks) = EllipsoidalKernel((D, D), ks)

"""
    RectangularKernel{T}

Describes a kernel for convolution with 2D rectangular step function on periodic
domain.

Note that the associated convolution kernel in Fourier space, constructed via
[`materialise!`](@ref), is a product of `sinc` functions.

---

    RectangularKernel((Rx, Ry), (kx, ky))

Construct rectangular kernel with sides `(Rx, Ry)` and Fourier wave numbers
`(kx, ky)`.
"""
struct RectangularKernel{T <: AbstractFloat, WaveNumbers} <: AbstractKernel
    sides :: NTuple{2,T}
    ks :: NTuple{2,WaveNumbers}
    function RectangularKernel(Rs::NTuple{2,T}, ks) where {T}
        new{float(T), typeof(first(ks))}(Rs, ks)
    end
end

"""
    RectangularKernel(R, (kx, ky))

Construct square kernel with side `R`.
"""
RectangularKernel(R, ks) = RectangularKernel((R, R), ks)

"""
    materialise!(u::AbstractMatrix, kernel::AbstractKernel)

Fill kernel array in Fourier space.

See also [`materialise`](@ref).
"""
function materialise! end

"""
    materialise(kernel::AbstractKernel, [T = Float64])

Create new kernel array in Fourier space.

See also [`materialise!`](@ref).
"""
function materialise(g::AbstractKernel, ::Type{T} = Float64) where {T}
    Ns = length.(wavenumbers(g))
    u = Array{T}(undef, Ns)
    materialise!(u, g)
end

function materialise!(u::AbstractMatrix, g::RectangularKernel)
    ks = wavenumbers(g)
    if length.(ks) != size(u)
        throw(DimensionMismatch("size of `u` inconsistent with number of wave numbers"))
    end
    Ls = 2π ./ getindex.(ks, 2)  # domain size: L = 2π / k[2]
    Rs = g.sides ./ Ls
    area = prod(g.sides)
    @inbounds for I in CartesianIndices(u)
        kvec = getindex.(ks, Tuple(I))
        u[I] = area * prod(sinc, kvec .* Rs)  # = A * sinc(kx * rx / Lx) * sinc(ky * ry / Ly)
    end
    u
end

function materialise!(u::AbstractMatrix, g::EllipsoidalKernel)
    ks = wavenumbers(g)
    if length.(ks) != size(u)
        throw(DimensionMismatch("size of `u` inconsistent with number of wave numbers"))
    end
    Ls = 2π ./ getindex.(ks, 2)  # domain size: L = 2π / k[2]
    Rs = g.diameters ./ Ls
    area = π * prod(g.diameters) / 4
    @inbounds for I in CartesianIndices(u)
        kvec = getindex.(ks, Tuple(I))
        kr = sqrt(sum(abs2, kvec .* Rs))  # = √[(kx * rx / Lx)^2 + (ky * ry / Ly)^2]
        u[I] = area * J1norm(kr)
    end
    u
end

"""
    J1norm(x)

Normalised Bessel function of the first kind and first order.

Works similarly to `sinc(x)`: returns ``2 J_1(π x) / (π x)`` if ``x ≠ 0``, or
``1`` otherwise, where ``J_1`` is the Bessel function of the first kind and
first order.
"""
function J1norm(x)
    T = promote_type(typeof(x), typeof(π))
    if iszero(x)
        one(T)
    else
        y = π * x
        2 * besselj1(y) / y
    end :: T
end

end
