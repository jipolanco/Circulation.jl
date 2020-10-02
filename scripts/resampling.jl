#!/usr/bin/env julia

using GPFields
using Circulation

using FFTW
using WriteVTK

import PyPlot
const plt = PyPlot

using LinearAlgebra

get_params() = (
    data_directory = expanduser("~/Dropbox/circulation/data/tangle/256"),
    data_index = 1,
    N = (256, 256, 256),
    L = (2pi, 2pi, 2pi),
    slice = (:, :, 1),
    c = 1.0,
    nxi = 1.5,
    resampling_factor = 4,
    loop_size = 8,
)

vecdiff(x::AbstractArray, y::AbstractArray, yslice) =
    norm(x .- view(y, yslice...))

vecdiff(x::Tuple, y::Tuple, yslice) =
    map((x, y) -> vecdiff(x, y, yslice), x, y)

function main(params)
    gp_3D = ParamsGP(
        params.N,
        L = params.L,
        c = params.c,
        nxi = params.nxi,
    )

    gp_in = ParamsGP(gp_3D, params.slice)

    Nres = params.resampling_factor

    gp_res = ParamsGP(
        gp_in,
        dims = Nres .* size(gp_in),
        nxi = Nres * gp_in.nξ,
    )

    ψ_in = GPFields.load_psi(gp_3D, params.data_directory,
                             params.data_index, slice=params.slice)
    ψ = similar(ψ_in, Nres .* size(ψ_in))

    let ψ_f = fft(ψ_in)
        GPFields.resample_field_fourier!(ψ, ψ_f, gp_in)
        ifft!(ψ)
    end

    # Subset of resampled field corresponding to grid points of original field.
    slice_res = range.(1, size(ψ), step=Nres)

    ρ_in = GPFields.density(ψ_in)
    ρ = GPFields.density(ψ)
    @show vecdiff(ρ_in, ρ, slice_res)

    p_in = GPFields.momentum(ψ_in, gp_in)
    p = GPFields.momentum(ψ, gp_res)
    @show vecdiff(p_in, p, slice_res)

    vreg_in = map(p -> p ./ sqrt.(ρ_in), p_in)
    vreg = map(p -> p ./ sqrt.(ρ), p)
    @show vecdiff(vreg_in, vreg, slice_res)

    v_in = map(p -> p ./ ρ_in, p_in)
    v = map(p -> p ./ ρ, p)
    @show vecdiff(v_in, v, slice_res)

    # Compare integral field used for computing circulation.
    I_in = IntegralField2D(v_in[1], L=gp_in.L)
    prepare!(I_in, v_in)

    I = IntegralField2D(v[1], L=gp_res.L)
    prepare!(I, v)

    @show vecdiff(I_in.w, I.w, slice_res)
    @show norm.(I_in.w, Inf)
    @show norm.(I.w, Inf)

    # plot_integral_fields(gp_in, I_in.w, I.w, slice_res, ρ_in)

    test_circulation(gp_in, gp_res, ρ_in, ρ, I_in, I, slice_res,
                     loop_size=params.loop_size)

    nothing
end

function test_circulation(gp_in, gp_res, ρ_in, ρ, I_in, I, slice_res; loop_size)
    κ = gp_in.κ

    Γ_in = Array{Float64}(undef, size(I_in))
    Γ = similar(Γ_in)                # computed on original grid from resampled field
    Γ_full = similar(Γ_in, size(I))  # computed on the whole refined grid

    let r = loop_size
        resampling_factor = first(Int.(size(I) ./ size(I_in)))
        rs = (r, r)
        rs_res = resampling_factor .* rs
        circulation!(Γ_in, I_in, rs, grid_step=1)
        circulation!(Γ, I, rs_res, grid_step=resampling_factor)
        circulation!(Γ_full, I, rs_res, grid_step=1)
        Γ_in ./= κ
        Γ ./= κ
        Γ_full ./= κ
        @show extrema(Γ_in)
        @show extrema(Γ)
        @show extrema(Γ_full)
    end

    xy = GPFields.get_coordinates(gp_in)
    xy_fine = GPFields.get_coordinates(gp_res)

    let (fig, axes) = plt.subplots(1, 2, figsize=(8, 3.5),
                                   sharex=true, sharey=true)
        levels = [0.1, 0.9, 1.1, 1.9, 2.1]
        prepend!(levels, reverse(-levels))
        kwargs = (
            levels = levels,
            # levels = LinRange(-2.5, 2.5, 10),
            cmap = plt.ColorMap("RdBu"),
            extend = :both,
        )
        let ax = axes[1]
            ax.set_aspect(:equal)
            ax.contourf(xy..., Γ_in'; kwargs...)
            plot_vortices!(ax, xy, ρ_in)
        end
        let ax = axes[2]
            ax.set_aspect(:equal)
            # cf = ax.contourf(xy..., Γ'; kwargs...)
            cf = ax.contourf(xy_fine..., Γ_full'; kwargs...)
            fig.colorbar(cf, ax=ax)
            # plot_vortices!(ax, xy, ρ_in)
            plot_vortices!(ax, xy_fine, ρ)
        end
    end

    let (fig, ax) = plt.subplots()
        ax.set_aspect(:equal)
        Γ_diff = Γ_in - Γ
        A = ceil(maximum(abs, Γ_diff))
        levels = LinRange(-A, A, 12)
        cf = ax.contourf(xy..., Γ_diff', levels=levels,
                         cmap=plt.ColorMap("RdBu"))
        fig.colorbar(cf, ax=ax)
        plot_vortices!(ax, xy, ρ_in)
    end
end

main() = main(get_params())

plot_vortices!(ax, xy, ρ; level=0.1) =
    ax.contour(xy..., ρ', levels=[level], colors=:black, linewidths=0.5)

function plot_integral_fields(gp_in, w_in, w, slice, ρ_in; component=2)
    xy = GPFields.get_coordinates(gp_in)
    κ = gp_in.κ

    w_orig = w_in[component]' ./ κ
    w_fine = view(w[component], slice...)' ./ κ

    let (fig, axes) = plt.subplots(1, 2, figsize=(8, 3.5),
                                   sharex=true, sharey=true)
        levels = -1.2:0.05:1.2
        let ax = axes[1]
            ax.set_aspect(:equal)
            ax.contourf(xy..., w_orig, levels=levels)
            plot_vortices!(ax, xy, ρ_in)
        end
        let ax = axes[2]
            ax.set_aspect(:equal)
            cf = ax.contourf(xy..., w_fine, levels=levels)
            fig.colorbar(cf, ax=ax)
            plot_vortices!(ax, xy, ρ_in)
        end
    end

    let (fig, ax) = plt.subplots()
        ax.set_aspect(:equal)
        cf = ax.contourf(xy..., w_fine .- w_orig)
        fig.colorbar(cf, ax=ax)
        plot_vortices!(ax, xy, ρ_in)
    end
end

main()
