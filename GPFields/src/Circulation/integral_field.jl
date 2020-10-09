"""
    IntegralField2D{T}

Contains arrays required to compute the integral of a 2D vector field
`(vx, vy)`.

Also contains FFT plans and array buffers for computation of integral data.

---

    IntegralField2D(Nx, Ny, [T = Float64]; L)

Allocate `IntegralField2D` of dimensions `(Nx, Ny)`.

The domain size must be given as a tuple `L = (Lx, Ly)`.

---

    IntegralField2D(A::AbstractMatrix{<:AbstractFloat}; L)

Allocate `IntegralField2D` having dimensions and type compatibles with input
matrix.
"""
struct IntegralField2D{T, PlansFW, PlansBW}
    N :: NTuple{2,Int}      # Nx, Ny
    L :: NTuple{2,Float64}  # domain size: Lx, Ly

    # Mean value Ux(y), Uy(x).
    U :: NTuple{2,Vector{T}}  # lengths: Ny, Nx

    # Integral fields wx(x, y), wy(x, y).
    w :: NTuple{2,Matrix{T}}  # [Nx, Ny]

    # FFTW plans (plan_x, plan_y)
    plans_fw :: PlansFW  # forwards
    plans_bw :: PlansBW  # backwards

    # Buffer arrays for FFTs (one per thread).
    bufs :: NTuple{2,Matrix{T}}  # lengths: Nx, Ny
    bufs_f :: NTuple{2,Matrix{Complex{T}}}

    # Wave numbers
    ks :: NTuple{2,Frequencies{Float64}}

    function IntegralField2D(Nx, Ny, ::Type{T} = Float64; L) where {T}
        Ns = (Nx, Ny)
        U = Vector{T}.(undef, (Ny, Nx))
        w = ntuple(_ -> Matrix{T}(undef, Nx, Ny), 2)

        fs = 2pi .* Ns ./ L  # sampling frequency
        ks = rfftfreq.(Ns, fs)

        Nth = nthreads()
        bufs = Array{T}.(undef, Ns, Nth)
        bufs_f = Array{Complex{T}}.(undef, length.(ks), Nth)

        # Make sure that plans are NOT threaded (these are small transforms)
        FFTW.set_num_threads(1)
        plans_fw = plan_rfft.(view.(bufs, :, 1), flags=FFTW.MEASURE)
        plans_bw = plan_irfft.(view.(bufs_f, :, 1), Ns, flags=FFTW.MEASURE)

        Pfw = typeof(plans_fw)
        Pbw = typeof(plans_bw)

        new{T, Pfw, Pbw}(Ns, L, U, w, plans_fw, plans_bw, bufs, bufs_f, ks)
    end

    IntegralField2D(A::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat} =
        IntegralField2D(size(A)..., T; kwargs...)
end

"""
    IntegralField2D(v; L)

Initialise and set values of the integral fields from 2D vector field
`v = (vx, vy)`.

See also [`prepare!`](@ref).
"""
function IntegralField2D(v::NTuple{2,<:AbstractMatrix}; kwargs...)
    I = IntegralField2D(v[1]; kwargs...)
    prepare!(I, v)
    I
end

Base.ndims(::IntegralField2D) = 2
Base.size(I::IntegralField2D) = I.N
Base.eltype(::Type{IntegralField2D{T}}) where {T} = T

"""
    prepare!(I::IntegralField2D{T}, v)

Set values of the integral fields from 2D vector field `v = (vx, vy)`.
"""
function prepare!(I::IntegralField2D{T},
                  v::NTuple{2,AbstractMatrix{T}}) where {T}
    Ns = size(I)
    if any(Ref(Ns) .!= size.(v))
        throw(DimensionMismatch("incompatible array sizes"))
    end
    prepare!(I, v[1], Val(1))
    prepare!(I, v[2], Val(2))
    I
end

function prepare!(I::IntegralField2D, u, ::Val{c}) where {c}
    @assert c in (1, 2)
    U = I.U[c]
    k = I.ks[c]
    plan_fw = I.plans_fw[c]
    plan_bw = I.plans_bw[c]
    ubuf_t = I.bufs[c]
    uf_t = I.bufs_f[c]
    w = I.w[c]

    @assert k[1] == 0
    Nk = length(k)

    Ns = size(w)
    @assert size(ubuf_t, 1) == size(u, c) == Ns[c]

    Nc, Nother = (c === 1) ? (Ns[1], Ns[2]) : (Ns[2], Ns[1])

    @threads for j = 1:Nother
        t = threadid()
        ubuf = view(ubuf_t, :, t)
        uf = view(uf_t, :, t)

        for i = 1:Nc
            ind = (c === 1) ? CartesianIndex((i, j)) : CartesianIndex((j, i))
            ubuf[i] = u[ind]
        end

        mul!(uf, plan_fw, ubuf)  # apply FFT

        # Copy mean value and then set it to zero.
        # Note: the mean value must be normalised by the input data length.
        U[j] = Real(uf[1]) / Nc
        uf[1] = 0

        for i = 2:Nk
            uf[i] /= im * k[i]  # w(k) -> w(k) / ik
        end

        mul!(ubuf, plan_bw, uf)  # apply inverse FFT

        for i = 1:Nc
            ind = (c === 1) ? CartesianIndex((i, j)) : CartesianIndex((j, i))
            w[ind] = ubuf[i]
        end
    end

    I
end
