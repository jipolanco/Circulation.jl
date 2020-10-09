module GPFields

export ParamsGP
export get_coordinates
export
    load_psi,
    load_psi!,
    load_psi_resampled,
    density,
    momentum,
    density!,
    momentum!,
    curlF!,
    resample_field_fourier!

using FFTW
using HDF5
using Printf: @sprintf
import Mmap
using Base.Threads
using Reexport

# Type definitions
const ComplexArray{T,N} = AbstractArray{Complex{T},N} where {T<:Real,N}
const RealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

const ComplexVector{T,N} = NTuple{N, ComplexArray{T,N}} where {T<:Real,N}
const RealVector{T,N} = NTuple{N, RealArray{T,N}} where {T<:Real,N}

# Defines a slice in N dimensions.
const Slice{N} = Tuple{Vararg{Union{Int,Colon}, N}} where {N}

include("slices.jl")
include("params.jl")
include("resampling.jl")
include("io.jl")
include("fourier.jl")
include("fields.jl")

include("Circulation/Circulation.jl")
@reexport using .Circulation

end
