# Check size of complex scalar field data.
function check_size(::Type{T}, dims, io_r, io_c) where {T <: Complex}
    size_r = stat(io_r).size
    size_i = stat(io_c).size
    size_r == size_i || error("files have different sizes")
    N = prod(dims)
    if sizeof(T) * N != size_r + size_i
        sr = size_r ÷ sizeof(T)
        error(
            """
            given GP dimensions are inconsistent with file sizes
                given dimensions:    $N  $dims
                expected from files: $sr
            """
        )
    end
    nothing
end

# Check size of real scalar field data.
function check_size(::Type{T}, dims, io) where {T}
    size_io = stat(io).size
    N = prod(dims)
    if sizeof(T) * N != size_io
        sr = size_io ÷ sizeof(T)
        error(
            """
            given GP dimensions are inconsistent with file sizes
                given dimensions:    $N $dims
                expected from files: $sr
            """
        )
    end
    nothing
end

# Read the full data
function load_slice!(psi::ComplexArray{T}, vr::RealArray{T}, vi::RealArray{T},
                     slice::Nothing) where {T}
    @assert length(psi) == length(vr) == length(vi)
    @threads for n in eachindex(psi)
        @inbounds psi[n] = Complex{T}(vr[n], vi[n])
    end
    psi
end

# Read a data slice
function load_slice!(psi::ComplexArray{T}, vr::RealArray{T}, vi::RealArray{T},
                     slice::Slice) where {T}
    inds = view(CartesianIndices(vr), slice...)
    @assert size(vr) == size(vi)
    if size(psi) != size(inds)
        throw(DimensionMismatch(
            "output array has different dimensions from slice: " *
            "$(size(psi)) ≠ $(size(inds))"
        ))
    end
    @threads for n in eachindex(inds)
        @inbounds I = inds[n]
        @inbounds psi[n] = Complex{T}(vr[I], vi[I])
    end
    psi
end

# Variants for real values.
function load_slice!(vs::RealArray{T}, vin::RealArray{T},
                     slice::Nothing) where {T}
    @assert size(vs) == size(vin)
    @threads for n in eachindex(vs)
        @inbounds vs[n] = vin[n]
    end
    vs
end

function load_slice!(vs::RealArray{T,N}, vin::RealArray{T,M},
                     slice::Slice) where {T,N,M}
    inds = view(CartesianIndices(vin), slice...)
    if size(vs) != size(inds)
        throw(DimensionMismatch(
            "output array has different dimensions from slice: " *
            "$(size(vs)) ≠ $(size(inds))"
        ))
    end
    @threads for n in eachindex(inds)
        @inbounds I = inds[n]
        @inbounds vs[n] = vin[I]
    end
    vs
end

"""
    load_psi!(psi, gp::ParamsGP, datadir, field_index; slice = nothing)
    load_psi!(psi, gp::ParamsGP, filename_pat; kws...)

Load complex ψ(x) field from files for `ψ_r` and `ψ_c`.

Writes data to preallocated output `psi`.

The optional `slice` parameter may designate a slice of the domain,
such as `(:, 42, :)`.

In the second variant, a filename pattern should be passed.
The pattern must contain a `*` placeholder that will be replaced by "Rea" and
"Ima", for the real and imaginary parts of ψ.

Returns the parameters of the loaded field, as a [`ParamsGP`](@ref). In
particular, if a slice of a 3D field is loaded, then the parameters are those of
the resulting 2D field.
"""
function load_psi!(psi::ComplexArray, gp::ParamsGP,
                   datadir::AbstractString, field_index::Integer;
                   kw...)
    ts = @sprintf "%03d" field_index  # e.g. "007" if field_index = 7
    load_psi!(psi, gp, joinpath(datadir, "*Psi.$ts.dat"); kw...)
end

function load_psi!(psi::ComplexArray{T}, gp::ParamsGP{N},
                   filename_pat::AbstractString;
                   slice::Union{Nothing,Slice{N}} = nothing) where {T,N}
    if '*' ∉ filename_pat
        throw(ArgumentError("filename pattern must contain '*'"))
    end
    fname_r = replace(filename_pat, '*' => "Rea")
    fname_i = replace(filename_pat, '*' => "Ima")

    for fname in (fname_r, fname_i)
        isfile(fname) || error("file not found: $fname")
    end

    check_size(Complex{T}, gp.dims, fname_r, fname_i)

    # Memory-map data from file.
    # That is, data is not loaded into memory until needed.
    vr = Mmap.mmap(fname_r, Array{T,N}, gp.dims)
    vi = Mmap.mmap(fname_i, Array{T,N}, gp.dims)

    load_slice!(psi, vr, vi, slice)

    # Parameters of the slice
    gp_slice = slice === nothing ? gp : ParamsGP(gp, slice)

    gp_slice
end

"""
    load_psi(gp::ParamsGP, datadir, field_index; slice = nothing, resampling = nothing)
    load_psi(gp::ParamsGP, filename_pat; kws...)

Load complex ψ(x) field from files for `ψ_r` and `ψ_c`.

Allocates and returns loaded `ψ` field.

The field may be optionally resampled into a higher resolution grid, by
zero-padding in Fourier space.
This is done via the `resampling` argument, which should be a resampling factor
larger than 1.
If this option is passed, a tuple `(gp_out, ψ_out)` is returned, where `gp_out`
is a [`ParamsGP`](@ref) containing the parameters of the resampled field.

See also [`load_psi!`](@ref).
"""
function load_psi(gp::ParamsGP, args...; slice = nothing, resampling = nothing)
    ψ = Array{ComplexF64}(undef, _loaded_dims(size(gp), slice))
    gp_slice = load_psi!(ψ, gp, args...; slice=slice)
    resample_psi(gp_slice, ψ, resampling)
end

# Note that `params` is not returned for backwards compatibility.
resample_psi(params, ψ, ::Nothing) = ψ

function resample_psi(params, ψ_input, resampling::Integer)
    if resampling == 1
        return params, ψ_input
    end
    dims_in = size(ψ_input)
    dims = resampling .* dims_in
    gp = ParamsGP(params; dims = dims)
    ψ = similar(ψ_input, dims)
    fft!(ψ_input)
    resample_field_fourier!(ψ, ψ_input, params)
    ifft!(ψ)
    gp, ψ
end

"""
    load_velocity!(v, gp::ParamsGP, datadir, field_index;
                   incompressible=true, slice=nothing)
    load_velocity!(v, gp::ParamsGP, filename_pat; slice=nothing)

Load velocity vector field `v = (v1, v2, ...)` from binary file.

In the first variant, data must be in the file `\$datadir/Vel.\$field_index.dat`
(where `field_index` is formatted using 3 digits, as in "042").

In the second variant, `filename_pat` is a pattern such as
`path/to/velocity_*.bin`, where the `*` will be replaced by `x`, `y` and
`z` for the three velocity components.

In the case of a slice, only the in-plane velocity components are loaded.

See also [`load_psi!`](@ref).
"""
function load_velocity!(
        vs::RealVector, gp::ParamsGP, filename_pat::AbstractString; kws...,
    )
    load_vector_field!(vs, gp, filename_pat; kws...)
end

function load_velocity!(
        vs, gp, datadir::AbstractString, field_index::Integer;
        incompressible=true, kws...,
    )
    prefix = joinpath(datadir, incompressible ? "VI" : "VC")
    suffix = @sprintf "_d.%03d.dat" field_index
    pat = string(prefix, '*', suffix)
    load_velocity!(vs, gp, pat; kws...)
end

function load_vector_field!(
        vs::NTuple{N}, gp::ParamsGP{M}, filename_pat::AbstractString;
        slice = nothing,
    ) where {N,M}

    if '*' ∉ filename_pat
        throw(ArgumentError("filename pattern must contain '*'"))
    end

    components = dims_slice(Val(M), slice)
    @assert length(components) == N

    for (v, c) in zip(vs, components)
        filename = replace(filename_pat, '*' => "xyz"[c])
        load_scalar_field!(v, gp, filename; slice)
    end

end

function load_scalar_field!(
        u::AbstractArray{T}, gp::ParamsGP{M}, filename::AbstractString;
        slice = nothing,
    ) where {T,M}
    isfile(filename) || error("file not found: $filename")
    check_size(T, gp.dims, filename)
    vmap = Mmap.mmap(filename, Array{T,M}, gp.dims)
    load_slice!(u, vmap, slice)
end

# Example: dims_slice(Val(3), (:, 42, :)) = (1, 3).
@inline dims_slice(::Val{N}, ::Nothing) where {N} = ntuple(identity, Val(N))
@inline dims_slice(::Val{N}, s::Slice{N}) where {N} = dims_slice(1, s...)
@inline dims_slice(n::Int, ::Colon, etc...) = (n, dims_slice(n + 1, etc...)...)
@inline dims_slice(n::Int, ::Integer, etc...) = dims_slice(n + 1, etc...)
@inline dims_slice(n::Int) = ()

"""
    load_velocity(gp::ParamsGP, datadir, field_index;
                  slice=nothing, incompressible=true)

Load full vector velocity field from file.
"""
function load_velocity(gp::ParamsGP{N}, args...;
                       slice=nothing, kwargs...) where {N}
    dims = _loaded_dims(size(gp), slice)
    Nc = length(dims)  # number of velocity components to load
    v = ntuple(d -> Array{Float64}(undef, dims), Val(Nc))
    load_velocity!(v, gp, args...; slice=slice, kwargs...) :: RealVector
end

_loaded_dims(dims, slice::Nothing) = dims
_loaded_dims(dims::Dims{N}, slice::Slice{N}) where {N} =
    size(CartesianIndices(dims)[slice...])
