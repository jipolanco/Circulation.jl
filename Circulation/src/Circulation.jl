module Circulation

import Base.Threads

export circulation, circulation!

include("loops/rectangle.jl")

# Type definitions
const ComplexArray{T,N} = AbstractArray{Complex{T},N} where {T<:Real,N}
const RealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

"""
    circulation!(Γ, vf, rs, ks)

Compute circulation on a 2D slice around loops with a fixed rectangle shape.

## Parameters

- `Γ`: output real matrix containing the circulation associated to each point of
  the physical grid.

- `vf = (vf_x, vf_y)`: two-component 2D vector field of complex values.
  The `i`-th component must have been Fourier-transformed along the `i`-th
  direction. They should be generated using a real-to-complex transform (e.g.
  `FFTW.rfft`).

- `rs = (r_x, r_y)`: rectangle dimensions.

- `ks = (k_x, k_y)`: non-negative Fourier wave numbers.
  They are typically generated using `rfftfreq` (from `FFTW` package).

"""
function circulation!(Γ::AbstractMatrix{<:Real},
                      vf::NTuple{2,ComplexArray},
                      rs::NTuple{2,Int},
                      ks::NTuple{2,AbstractVector},
                     )
    Ns = size(Γ)
    Nv_expected = div.(Ns, 2) .+ 1

    if length.(ks) != Nv_expected
        @show Ns
        @show Nv_expected
        @show size.(ks)
        throw(ArgumentError(
            "incompatible sizes of output array and wave numbers"
        ))
    end

    Nx, Ny = Ns
    loop_base = Rectangle((0, 0), rs)
    rs_half = rs .>> 1  # half radius (truncated if rs has odd numbers...)

    Threads.@threads for j = 1:Ny
        for i = 1:Nx
            x0 = (i, j) .- rs_half  # lower left corner of loop
            loop = loop_base + x0
            Γ[i, j] = circulation(loop, vf, ks, Ns)
        end
    end

    Γ
end

"""
    circulation(loop::Rectangle, vf, ks, Ns)

Compute circulation around the given loop on a 2D slice.

The rectangle must be given in integer coordinates (grid indices).

The tuple `Ns = (Nx, Ny)` must contain the dimensions of the original real data.

See also `circulation!`.

"""
function circulation(loop::Rectangle{Int},
                     vf::NTuple{2,ComplexArray{T,2}} where {T},
                     ks::NTuple{2,AbstractVector},
                     Ns::NTuple{2,Int},
                    )
    # Determine domain size from first positive wave number.
    Ls = 2pi ./ getindex.(ks, 2)  # (Lx, Ly)

    # Coordinates (including end point).
    xs = LinRange.(0, Ls, Ns .+ 1)

    # Indices of rectangle points.
    ia, ja = loop.x
    ib, jb = loop.x .+ loop.r

    # Coordinates of rectangle corners, taking periodicity into account.
    (ia, xa), (ja, ya) = _make_coordinate.(xs, (ia, ja))
    (ib, xb), (jb, yb) = _make_coordinate.(xs, (ib, jb))

    # Array views along each line.
    vx_a = @view vf[1][:, ja]
    vx_b = @view vf[1][:, jb]

    vy_a = @view vf[2][ia, :]
    vy_b = @view vf[2][ib, :]

    kx, ky = ks
    Nx, Ny = Ns

    Ix_a, Ix_b = integrate((vx_a, vx_b), kx, xa, xb)
    Iy_a, Iy_b = integrate((vy_a, vy_b), ky, ya, yb)

    int_x = (Ix_a - Ix_b) / Nx
    int_y = (Iy_b - Iy_a) / Ny

    # Slower version:
    # int_x = (integrate(vx_a, kx, xa, xb) - integrate(vx_b, kx, xa, xb)) / Nx
    # int_y = (integrate(vy_b, ky, ya, yb) - integrate(vy_a, ky, ya, yb)) / Ny

    int_x + int_y
end

function _make_coordinate(x::AbstractVector, i::Int)
    # Return coordinate x[i] taking periodicity into account.
    @assert x[1] == 0
    N = length(x) - 1
    L = x[N + 1]
    x0 = zero(L)
    while i <= 0
        i += N
        x0 -= L
    end
    while i > N
        i -= N
        x0 += L
    end
    i, x0 + x[i]
end

"""
    integrate(vf, k, x1, x2)

Integrate Fourier-transformed vector `vf` between `x1` and `x2`.

It is assumed that data was transformed using a real-to-complex FFT (`rfft`),
and therefore only the positive wave numbers `k` are needed.

Note that if the FFT was not normalised, then the result should be divided by
the length of the *original real data* to get the actual integral.

`vf` can also be a tuple of complex vectors (`vfA`, `vfB`, ...).
In that case, the integral of each vector is returned.
This can be useful for performance, since trigonometric functions will be computed just once for all vectors.
"""
function integrate(vf::Tuple{Vararg{AbstractVector{<:Complex}}},
                   k::AbstractVector, x1, x2)
    Nk = length(k)

    all(length.(vf) .== Nk) || throw(ArgumentError("incompatible vector lengths"))
    k[1] == 0 || throw(ArgumentError("first wave number should be zero"))
    k[end] > 0 || throw(ArgumentError("negative wave numbers must not be passed"))

    Δx = x2 - x1
    int = @. real(getindex(vf, 1) * Δx)  # zero mode

    # Sum over k != 0
    @inbounds for n = 2:Nk
        # Because of Hermitian symmetry, the real parts for k and -k are equal,
        # so we multiply stuff by two. The imaginary parts have opposite sign
        # and cancel out.
        kn = k[n]

        # Original version (may need modifications to work...).
        # α = im * kn
        # vn = getindex.(vf, n)
        # int += 2 * real(vn / α * (exp(α * x2) - exp(α * x1)))

        # Optimised version, avoiding computation of imaginary part.
        vn = getindex.(vf, n)
        kx1 = kn * x1
        kx2 = kn * x2
        sx = sin(kx2) - sin(kx1)
        cx = cos(kx2) - cos(kx1)
        int = @. int + (2 / kn) * (real(vn) * sx + imag(vn) * cx)
    end

    int
end

# If a vector is passed instead of a tuple of vectors...
integrate(vf::AbstractVector, args...) = first(integrate((vf, ), args...))

end
