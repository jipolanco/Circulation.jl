# Defines a slice in N dimensions.
const Slice{N} = Tuple{Vararg{Union{Int,Colon}, N}} where {N}

"""
    slice_dimensions(slice::Slice)

Determine dimensions along which a slice is performed.

For instance, if `slice = (:, 42, :)`, returns `(1, 3)`.
"""
slice_dimensions(slice::Slice) = slice_dimensions(slice...)
slice_dimensions(::Colon, etc...) = (1, (1 .+ slice_dimensions(etc...))...)
slice_dimensions(::Int, etc...) = 1 .+ slice_dimensions(etc...)
slice_dimensions() = ()

@assert slice_dimensions((:, 42, :, 31)) === (1, 3)

"""
    global_index(ind::NTuple{M}, slice::Slice{N})       -> NTuple{N}
    global_index(I::CartesianIndex{M}, slice::Slice{N}) -> CartesianIndex{N}

Return index in global dataset from index in slice.

For instance, if `slice = (:, 42, :)` and `ind = (3, 6)`, returns `(3, 42, 6)`.
"""
function global_index end

global_index(I::CartesianIndex, slice::Slice) =
    CartesianIndex(global_index(Tuple(I), slice))

global_index(ind::NTuple, slice::Slice) = global_index(ind, slice...)

global_index(ind::NTuple, ::Colon, etc...) =
    (first(ind), global_index(Base.tail(ind), etc...)...)
global_index(::Tuple{}, ::Colon, etc...) =
    throw(ArgumentError("some slice indices are missing"))

global_index(ind::NTuple, i::Integer, etc...) = (i, global_index(ind, etc...)...)
global_index(::NTuple) = throw(ArgumentError("too many slice indices"))

global_index(::Tuple{}) = ()
