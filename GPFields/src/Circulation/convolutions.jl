"""
    convolve!(
        u::AbstractMatrix, uF::AbstractMatrix, kernel::DiscreteFourierKernel;
        buf = similar(uF), plan_inv = plan_irfft(buf, size(Γ, 1); flags=FFTW.MEASURE),
    )

Convolve 2D slice (given in Fourier space) with the given kernel.

See also [`circulation!`](@ref).
"""
function convolve!(
        u::AbstractMatrix{<:Real}, uF::AbstractMatrix{<:Complex},
        kernel::DiscreteFourierKernel;
        buf = similar(uF),
        plan_inv = plan_irfft(buf, size(Γ, 1); flags=FFTW.MEASURE),
    )
    ks = Kernels.wavenumbers(kernel)
    gF = Kernels.data(kernel)
    u_hat = buf
    check_convolution(u, u_hat, uF, gF, ks)
    u_hat .= uF .* gF
    mul!(u, plan_inv, u_hat)
    u
end

function check_convolution(u, u_hat, uF, gF, ks)
    if size(uF) != length.(ks)
        throw(DimensionMismatch("kernel wave numbers incompatible with size of `vF` arrays"))
    end
    if size(uF) != size(gF)
        throw(DimensionMismatch("incompatible size of kernel array"))
    end
    if size(uF) != size(u_hat)
        throw(DimensionMismatch("incompatible size of buffer array"))
    end
    if ((size(u, 1) >> 1) + 1, size(u, 2)) != size(gF)
        throw(DimensionMismatch("incompatible size of output array"))
    end
    nothing
end
