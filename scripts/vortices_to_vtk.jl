#!/usr/bin/env julia

using HDF5
using WriteVTK

function to_vtk(ff, outfile, orientation)
    @assert orientation ∈ 1:3
    gname = string("Orientation", "XYZ"[orientation])
    pts = NTuple{3,Int32}[]
    r = Colon()
    ranges = map(("positive", "negative")) do name
        @time grid = ff["/$gname/$name"][r, r, r] :: Array{Bool,3}
        a = length(pts) + 1
        @time gather_points!(pts, grid)
        b = length(pts)
        a:b
    end
    if isempty(pts)
        @warn "No points found ($(filename(ff)))"
        return
    end
    Γ = zeros(Int8, length(pts))
    Γ[ranges[1]] .= 1
    Γ[ranges[2]] .= -1
    cells = [MeshCell(PolyData.Verts(), (i, )) for i = 1:length(pts)]
    xyz = let x = pts[1] :: NTuple
        reshape(reinterpret(eltype(x), pts), length(x), :)
    end
    vtk_grid(outfile, xyz, cells) do vtk
        vtk["circulation"] = Γ
    end
    nothing
end

function gather_points!(points, g::AbstractArray)
    for I in CartesianIndices(g)
        @inbounds iszero(g[I]) && continue
        push!(points, Tuple(I))
    end
    points
end

gather_points(g::AbstractArray) = gather_points!(NTuple{3,Int32}[], g)

function main()
    files = ("vortices_x2.h5", "vortices_x4.h5", "vortices_x8.h5")
    orientation = 1
    for fname in files
        outfile = splitext(fname)[1]
        @assert fname != outfile
        h5open(ff -> to_vtk(ff, outfile, orientation), fname, "r")
    end
    nothing
end

main()
