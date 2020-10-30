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
