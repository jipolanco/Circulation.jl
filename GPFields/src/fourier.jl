"""
    create_fft_plans_1d!(ψ::ComplexArray{T,N}) -> (plans_1, plans_2, ...)

Create in-place complex-to-complex FFT plans.

Returns `N` pairs of forward/backward plans along each dimension.
"""
function create_fft_plans_1d!(ψ::ComplexArray{T,D}) where {T,D}
    FFTW.set_num_threads(nthreads())
    ntuple(Val(D)) do d
        p = plan_fft!(ψ, d, flags=FFTW.MEASURE)
        (fw=p, bw=inv(p))
    end
end

"""
    curlF!(ω̂, v̂, gp::ParamsGP)

Compute curl of real vector field (such as momentum) in Fourier space.
"""
function curlF! end

# 2D version
function curlF!(ωzF::ComplexArray{T,2}, vF::ComplexVector{T,2},
                gp::ParamsGP{2}) where {T}
    Nx, Ny = size(gp)
    ks = get_wavenumbers(gp, Val(:r2c))
    Nk = length.(ks)
    if any(size.((ωzF, vF...)) .!= Ref(Nk))
        throw(DimensionMismatch("incorrect dimensions of fields"))
    end
    @inbounds for I in CartesianIndices(vF[1])
        kx, ky = getindex.(ks, Tuple(I))
        ωzF[I] = im * (kx * vF[2][I] - ky * vF[1][I])
    end
    ωzF
end
