struct DataFileParams
    directory :: String
    index     :: Int
end

"""
    analyse(gp::ParamsGP, loop_sizes, data_dir_base;
            data_idx=0, eps_vel=0, to=TimerOutput())

Compute circulation statistics from all possible slices of a GP field.
"""
function analyse(gp::ParamsGP, loop_sizes, data_dir_base;
                 data_idx=1, to=TimerOutput(), kwargs...)
    Ns = gp.dims

    data_dir = joinpath(data_dir_base, string(Ns[1]))
    isdir(data_dir) || throw(ArgumentError("directory not found: $data_dir"))
    data_params = DataFileParams(data_dir, data_idx)

    orientations = slice_orientations(gp)

    for or in orientations
        @timeit to "analyse" analyse(or, gp, loop_sizes, data_params;
                                     to=to, kwargs...)
    end
end

function analyse(orientation::Val, gp::ParamsGP{D}, loop_sizes,
                 data_params::DataFileParams;
                 eps_vel=0, to=TimerOutput()) where {D}
    Nslices = num_slices(gp.dims, orientation)
    @assert Nslices >= 1

    # Example: in 3D, if orientation = 2, this is (1, 3).
    i, j = included_dimensions(Val(D), orientation)
    Ni, Nj = gp.dims[i], gp.dims[j]
    Li, Lj = gp.L[i], gp.L[j]

    # Allocate arrays.
    psi = Array{ComplexF64}(undef, Ni, Nj)
    vs = ntuple(_ -> Array{Float64}(undef, Ni, Nj), 2)  # velocity, momentum
    Γ = similar(vs[1])  # circulation

    # Allocate integral field for circulation using FFTs.
    Ip = IntegralField2D(vs[1], L=(Li, Lj))

    # TODO
    # - circulation for vreg and v
    # - TimerOutputs
    # - statistics

    for s = 1:Nslices
        # Load ψ at selected slice.
        slice = make_slice(gp.dims, orientation, s)

        @timeit to "load_psi!" GPFields.load_psi!(
            psi, gp, data_params.directory, data_params.index, slice=slice)

        # Create parameters associated to slice.
        # This is needed to compute the momentum.
        gp_slice = ParamsGP(gp, slice)

        @timeit to "compute_momentum!" GPFields.compute_momentum!(
            vs, psi, gp_slice)

        # Set integral values with momentum data.
        @timeit to "prepare!" prepare!(Ip, vs)

        @timeit to "circulation!" for r in loop_sizes
            circulation!(Γ, Ip, (r, r))
        end
    end
end

function included_dimensions(::Val{N}, ::Val{s}) where {N,s}
    inds = findall(!=(s), ntuple(identity, Val(N)))  # all dimensions != s
    @assert length(inds) == 2
    inds[1], inds[2]
end

slice_orientations(::ParamsGP{2}) = (Val(3), )       # 2D data -> single z-slice
slice_orientations(::ParamsGP{3}) = Val.((1, 2, 3))  # 3D data

num_slices(::Dims{2}, ::Val{3}) = 1
num_slices(dims::Dims{3}, ::Val{s}) where {s} = dims[s]

function make_slice(dims::Dims{2}, ::Val{3}, i)
    @assert i == 1
    (:, :)
end

function make_slice(dims::Dims{3}, ::Val{s}, i) where {s}
    @assert 1 <= i <= dims[s]
    ntuple(d -> d == s ? i : Colon(), Val(3))
end
