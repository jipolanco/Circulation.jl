"""
    resample_field_fourier!(
        dest::AbstractArray, src::AbstractArray, params_src::ParamsGP,
    )

Resample complex field by zero-padding in Fourier space.

The resampling factor is determined from the dimensions of the two arrays.
It must be the same along all dimensions.

For now, the resampling factor must also be a non-negative power of two.

Resampling is performed in Fourier space.
No transforms are performed in this function, meaning that the input and output
are also in Fourier space.
"""
function resample_field_fourier!(dst::ComplexArray{T,N}, src::ComplexArray{T,N},
                                 p_src::ParamsGP{N}) where {T,N}
    if size(src) === size(dst)
        if src !== dst
            copy!(dst, src)
        end
        return dst
    end

    p_dst = ParamsGP(p_src, dims=size(dst))

    ksrc = get_wavenumbers(p_src)
    kdst = get_wavenumbers(p_dst)

    kmap = _wavenumber_map.(ksrc, kdst)

    # The coefficients are scaled by this ratio, to make sure that the
    # normalised inverse FFT (e.g. with ifft) has the good magnitude.
    scale = length(dst) / length(src)

    # 1. Set everything to zero.
    @threads for n in eachindex(dst)
        @inbounds dst[n] = 0
    end

    # 2. Copy all modes in src.
    @threads for I in CartesianIndices(src)
        is = Tuple(I)
        js = getindex.(kmap, is)
        @inbounds dst[js...] = scale * src[I]
    end

    dst
end

# Maps ki index to ko index, such that ki[n] = ko[kmap[n]].
function _wavenumber_map(ki::Frequencies, ko::Frequencies)
    Base.require_one_based_indexing.((ki, ko))
    Ni = length(ki)
    No = length(ko)
    if Ni > No
        error("downscaling (Fourier truncation) is not allowed")
    end
    if any(isodd.((Ni, No)))
        error("data length must be even (got $((Ni, No)))")
    end
    if ki[Ni] > 0 || ko[No] > 0
        error("negative wave numbers must be included")
    end
    h = Ni >> 1
    kmap = similar(ki, Int)
    for n = 1:h
        kmap[n] = n
        kmap[Ni - n + 1] = No - n + 1
    end
    # Verification
    for n in eachindex(ki)
        @assert ki[n] ≈ ko[kmap[n]]
    end
    kmap
end
