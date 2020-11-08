#!/usr/bin/env julia

using HDF5
using WriteVTK

function to_vtk(ff, outfile)
    vtk_multiblock(outfile) do vtm
        dirs = ('X', 'Y', 'Z')
        signs = ("positive", "negative")
        iter = Iterators.product(dirs, signs)
        map(Iterators.product(dirs, signs)) do it
            dir, sign = it
            dname = "/Orientation$dir/$sign"
            points = ff[dname][:, :] :: Matrix{Int32}
            cells = [MeshCell(PolyData.Verts(), (i, ))
                     for i = 1:length(points)]
            fname = "$(outfile)_$(dir)_$(sign)"
            vtk_grid(identity, vtm, fname, Float32.(points), cells;
                     compress=false, append=false)
        end
    end
    nothing
end

function main()
    files = ("vortices_x2.h5", "vortices_x4.h5", "vortices_x8.h5")
    orientation = 1
    for fname in files
        outfile = splitext(fname)[1]
        @assert fname != outfile
        h5open(ff -> to_vtk(ff, outfile), fname, "r")
    end
    nothing
end

main()
