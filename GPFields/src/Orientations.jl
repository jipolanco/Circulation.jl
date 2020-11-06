module Orientations

using ..GPFields: ParamsGP

export Orientation, slice_orientations, num_slices, make_slice, included_dimensions

struct Orientation{N}
    Orientation{N}() where {N} = (N::Int; new{N}())
end
@inline Orientation(N) = Orientation{N}()

Base.show(io::IO, ::Orientation{s}) where {s} = print(io, "xyz"[s])

slice_orientations(::ParamsGP{2}) = (Orientation(3), )       # 2D data -> single z-slice
slice_orientations(::ParamsGP{3}) = Orientation.((1, 2, 3))  # 3D data

function included_dimensions(::Val{N}, ::Orientation{s}) where {N,s}
    inds = findall(!=(s), ntuple(identity, Val(N)))  # all dimensions != s
    @assert length(inds) == 2
    inds[1], inds[2]
end

num_slices(::Dims{2}, ::Orientation{3}) = 1
num_slices(dims::Dims{3}, ::Orientation{s}) where {s} = dims[s]

num_slices(a, b, ::Nothing) = num_slices(a, b)
num_slices(a, b, max_slices) = min(num_slices(a, b), max_slices)

function make_slice(dims::Dims{2}, ::Orientation{3}, i)
    @assert i == 1
    (:, :)
end

function make_slice(dims::Dims{3}, ::Orientation{s}, i) where {s}
    @assert 1 <= i <= dims[s]
    ntuple(d -> d == s ? i : Colon(), Val(3))
end

end
