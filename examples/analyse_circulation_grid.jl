#!/usr/bin/env julia

using HDF5

function load_vortex_grid!(group, grid_pos, grid_neg)
    ind_pos = group["positive"][:, :] :: Matrix{Int32}
    ind_neg = group["negative"][:, :] :: Matrix{Int32}
    indices_to_grid!(grid_pos, ind_pos)
    indices_to_grid!(grid_neg, ind_neg)
    grid_pos, grid_neg
end

# Convert from indices in input files, to grid of zeros and ones.
function indices_to_grid!(grid, indices)
    fill!(grid, 0)  # set everything to zero
    M, N = size(indices)
    @assert M == 3
    @inbounds for n = 1:N
        i = indices[1, n]
        j = indices[2, n]
        k = indices[3, n]
        grid[i, j, k] = 1  # there's a vortex at this index
    end
    grid
end

# This function takes a grid of positive and negative vortices on a 2D slice.
function analyse_plane(plane_pos, plane_neg)
    # For example, compute total circulation and total number of vortices on the
    # plane.
    # Npos = sum(plane_pos)  # number of positive vortices
    # Nneg = sum(plane_neg)  # number of negative vortices
    # num_vort = Npos + Nneg
    # circ_sum = Npos - Nneg
    nothing
end

# Analyse grid[i, :, :] planes
function analyse_grid_x(grid_pos, grid_neg)
    Nx, Ny, Nz = size(grid_pos)
    for i = 1:Nx
        plane_pos = @view grid_pos[i, :, :]
        plane_neg = @view grid_neg[i, :, :]
        analyse_plane(plane_pos, plane_neg)
    end
    nothing
end

# Analyse grid[:, j, :] planes
function analyse_grid_y(grid_pos, grid_neg)
    Nx, Ny, Nz = size(grid_pos)
    for j = 1:Ny
        plane_pos = @view grid_pos[:, j, :]
        plane_neg = @view grid_neg[:, j, :]
        analyse_plane(plane_pos, plane_neg)
    end
    nothing
end

# Analyse grid[:, :, k] planes
function analyse_grid_z(grid_pos, grid_neg)
    Nx, Ny, Nz = size(grid_pos)
    for k = 1:Nz
        plane_pos = @view grid_pos[:, :, k]
        plane_neg = @view grid_neg[:, :, k]
        analyse_plane(plane_pos, plane_neg)
    end
    nothing
end

function analyse_file(ff)
    Nx, Ny, Nz = get_resolution(ff)
    grid_pos = falses(Nx, Ny, Nz)
    grid_neg = copy(grid_pos)

    load_vortex_grid!(ff["OrientationX"], grid_pos, grid_neg)
    analyse_grid_x(grid_pos, grid_neg)

    load_vortex_grid!(ff["OrientationY"], grid_pos, grid_neg)
    analyse_grid_y(grid_pos, grid_neg)

    load_vortex_grid!(ff["OrientationZ"], grid_pos, grid_neg)
    analyse_grid_z(grid_pos, grid_neg)

    nothing
end

function get_resolution(ff)
    Ns = ff["/ParamsGP/resolution"][:] :: Vector{Int}
    @assert length(Ns) == 3
    Ns[1], Ns[2], Ns[3]
end

function main()
    filename_in = "results/GP_vortices/vortices_x16.h5"
    h5open(filename_in, "r") do ff
        analyse_file(ff)
    end
    nothing
end

main()
