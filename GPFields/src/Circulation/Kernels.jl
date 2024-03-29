"""
Kernels for computation of circulation using convolution with vorticity field.

Different kernels correspond to different loop shapes (e.g. rectangular, circular).
"""
module Kernels

export EllipsoidalKernel, RectangularKernel, DiscreteFourierKernel
export materialise!, lengthscales, kernel

using Base.Threads
using SpecialFunctions: besselj1

"""
    AbstractKernel

Abstract type for 2D convolution kernels.
"""
abstract type AbstractKernel end

"""
    lengthscales(kernel::AbstractKernel) -> NTuple

Returns the characteristic length scales of a kernel.

## Examples

- for a [`RectangularKernel`](@ref), these are the two sides.

- for an [`EllipsoidalKernel`](@ref), these are the two diameters (or axes).
"""
function lengthscales end

"""
    area(kernel::AbstractKernel)

Returns the physical area of a kernel.
"""
function area end

"""
    EllipsoidalKernel{T} <: AbstractKernel

Describes a kernel for convolution with ellipsoidal step function on periodic
domain.

Note that the associated convolution kernel in Fourier space, constructed via
[`materialise!`](@ref), is based on Bessel functions.

---

    EllipsoidalKernel(Dx, Dy)

Construct ellipsoidal kernel with diameters `(Dx, Dy)`.
"""
struct EllipsoidalKernel{T <: AbstractFloat} <: AbstractKernel
    diameters :: NTuple{2,T}
    function EllipsoidalKernel(Dx::T, Dy::T) where {T}
        new{float(T)}((Dx, Dy))
    end
end

lengthscales(kern::EllipsoidalKernel) = kern.diameters
area(kern::EllipsoidalKernel) = prod(kern.diameters) * π / 4

"""
    EllipsoidalKernel(D)

Construct circular kernel with diameter `D`.
"""
EllipsoidalKernel(D) = EllipsoidalKernel(D, D)

"""
    RectangularKernel{T} <: AbstractKernel

Describes a kernel for convolution with 2D rectangular step function on periodic
domain.

Note that the associated convolution kernel in Fourier space, constructed via
[`materialise!`](@ref), is a product of `sinc` functions.

---

    RectangularKernel(Rx, Ry)

Construct rectangular kernel with sides `(Rx, Ry)`.
"""
struct RectangularKernel{T <: AbstractFloat} <: AbstractKernel
    sides :: NTuple{2,T}
    function RectangularKernel(Rx::T, Ry::T) where {T}
        new{float(T)}((Rx, Ry))
    end
end

lengthscales(kern::RectangularKernel) = kern.sides
area(kern::RectangularKernel) = prod(kern.sides)

"""
    RectangularKernel(R)

Construct square kernel with side `R`.
"""
RectangularKernel(R) = RectangularKernel(R, R)

"""
    DiscreteFourierKernel{T, WaveNumbers}

Represents a convolution kernel matrix materialised in Fourier space.

---

    DiscreteFourierKernel{T}(undef, kx, ky)

Construct uninitialised kernel matrix in Fourier space, for the given wave
numbers `(kx, ky)`.
"""
struct DiscreteFourierKernel{
        T, WaveNumbers,
        Kernel <: Union{Nothing, AbstractKernel},
    }
    mat :: Array{T,2}
    ks  :: NTuple{2,WaveNumbers}
    buf :: Vector{T}
    kernel :: Kernel
end

function DiscreteFourierKernel{T}(init, ks...; kernel = nothing) where {T}
    Ns = length.(ks)
    mat = Array{T}(init, Ns)
    buf = Vector{T}(undef, 0)
    # WaveNumbers = typeof(first(ks))
    DiscreteFourierKernel(mat, ks, buf, kernel)
end

DiscreteFourierKernel{T}(init, ks) where {T} = DiscreteFourierKernel{T}(init, ks...)

DiscreteFourierKernel(g::DiscreteFourierKernel, kernel::AbstractKernel) =
    DiscreteFourierKernel(g.mat, g.ks, g.buf, kernel)

"""
    DiscreteFourierKernel{T}(kernel::AbstractKernel, (kx, ky))
    DiscreteFourierKernel(kernel::AbstractKernel, (kx, ky), [T = Float64])

Construct and initialise discretised kernel in Fourier space, for the given wave numbers
`(kx, ky)`.
"""
function DiscreteFourierKernel{T}(kernel::AbstractKernel, ks) where {T}
    u = DiscreteFourierKernel{T}(undef, ks...)
    materialise!(u, kernel)
end

DiscreteFourierKernel(kernel::AbstractKernel, ks, ::Type{T} = Float64) where {T} =
    DiscreteFourierKernel{T}(kernel, ks)

wavenumbers(u::DiscreteFourierKernel) = u.ks
data(u::DiscreteFourierKernel) = u.mat
kernel(u::DiscreteFourierKernel) = u.kernel

function kernel_check(u::DiscreteFourierKernel)
    ker = kernel(u)
    isnothing(ker) && error("DiscreteFourierKernel has no attached kernel")
    ker
end

area(u::DiscreteFourierKernel) = area(kernel_check(u))

"""
    materialise!(u::DiscreteFourierKernel, kernel::AbstractKernel)

Fill discretised kernel in Fourier space.

See also [`materialise`](@ref).
"""
function materialise! end

function materialise!(kf::DiscreteFourierKernel, g::RectangularKernel)
    ks = wavenumbers(kf)
    u = data(kf)
    # Ls = 2π ./ getindex.(ks, 2)  # domain size: L = 2π / k[2]
    rs = g.sides
    A = area(g)

    # Write sinc evaluations to buffer
    sincs, offsets = _eval_sincs!(kf.buf, ks, rs)

    @inbounds for I in CartesianIndices(u)
        inds = Tuple(I) .+ offsets
        p = prod(i -> sincs[i], inds)
        u[I] = A * p
    end

    DiscreteFourierKernel(kf, g)
end

function _eval_sincs!(buf, ks, rs)
    offsets_long = (0, cumsum(length.(ks))...)
    offsets = ntuple(d -> offsets_long[d], Val(length(ks)))
    Nk = sum(length, ks)
    resize!(buf, Nk)
    @inbounds for i = 1:length(ks)
        off = offsets[i]
        kvec = ks[i]
        v = view(buf, (off + 1):(off + length(kvec)))
        _eval_sincs_1D!(v, kvec, rs[i])
    end
    buf, offsets
end

function _eval_sincs_1D!(buf, ks, r)
    N = length(ks)
    R = r / 2π
    @inbounds for n = 1:N
        buf[n] = sinc(ks[n] * R)
    end
    buf
end

function materialise!(kf::DiscreteFourierKernel, g::EllipsoidalKernel)
    ks = wavenumbers(kf)
    u = data(kf)
    # Ls = 2π ./ getindex.(ks, 2)  # domain size: L = 2π / k[2]
    rs = g.diameters
    Rs = rs ./ 2π
    A = area(g)
    @inbounds @threads for I in CartesianIndices(u)
        kvec = getindex.(ks, Tuple(I))
        kr = sqrt(sum(abs2, kvec .* Rs))  # = √[(kx * rx / 2π)^2 + (ky * ry / 2π)^2]
        u[I] = A * J1norm(kr)
    end
    DiscreteFourierKernel(kf, g)
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
