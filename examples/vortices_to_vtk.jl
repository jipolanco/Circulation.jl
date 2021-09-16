using HDF5
using WriteVTK

function to_vtk(ff, outfile)
    files = vtk_multiblock(outfile) do vtm
        dirs = ('X', 'Y', 'Z')
        signs = ("positive", "negative")
        iter = Iterators.product(dirs, signs)
        map(iter) do it
            dir, sign = it
            dname = "/Orientation$dir/$sign"
            points = ff[dname][:, :] :: Matrix{Int32}
            cells = [
                MeshCell(PolyData.Verts(), (i, ))
                for i ∈ axes(points, 2)
            ]
            fname = "$(outfile)_$(dir)_$(sign)"
            vtk_grid(identity, vtm, fname, Float32.(points), cells;
                     compress=false, append=true)
        end
    end
    @info "Saved $files"
    nothing
end

function main()
    # files = ("vortices_x8.h5", "vortices_x16.h5")
    files = ("vortices.h5",)
    for fname in files
        outfile = splitext(fname)[1]
        @assert fname != outfile
        h5open(ff -> to_vtk(ff, outfile), fname, "r")
    end
    nothing
end

main()
