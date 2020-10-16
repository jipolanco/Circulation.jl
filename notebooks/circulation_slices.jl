### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 16106ec4-04b8-11eb-032e-072f8a2efdcc
begin
	using Pkg
	Pkg.activate(".")
	using Revise
	using GPFields
	using FFTW
	using LinearAlgebra
end

# ╔═╡ 9040b526-04c6-11eb-3b36-77a97c91ed7a
begin
	ENV["MPLBACKEND"] = "Agg"
	import PyPlot
	using LaTeXStrings
	const plt = PyPlot.plt
	const mpl = plt.matplotlib
	plt.ion()
end;

# ╔═╡ 5e07c81e-04c2-11eb-2eaf-3d6136e0ed3e
begin
	ENV["GRDISPLAY"] = "pluto"
	using GR
	GR.js.init_pluto()
end

# ╔═╡ 93f9d4f2-04be-11eb-2bda-c35b6e65b22d
md"# Circulation"

# ╔═╡ 24fb6616-06ed-11eb-202a-0966bf3cd31b
function circulation_filter!(Γ)
	@inbounds for n in eachindex(Γ)
		Γ[n] = round(Γ[n])  # round to the nearest integer
	end
	Γ
end

# ╔═╡ c3f9b778-0705-11eb-0d57-87de6a741735
function coarse_grain(Γ_fine, r)
	N_fine = size(Γ_fine)
	if any(N_fine .< r)
		error("loop radius is too large: ($r, $r) > $N_fine")
	end
	@assert all(N_fine .% r .== 0)
	Ns = N_fine .÷ r
	Γ = similar(Γ_fine, Ns)
	fill!(Γ, 0)
	for I in CartesianIndices(Γ)
		ij_in = map(Tuple(I)) do i
			((i - 1) * r + 1):(i * r)
		end
		Γ[I] = sum(view(Γ_fine, ij_in...))
	end
	Γ
end

# ╔═╡ c63357ce-0719-11eb-3222-219e0ceecdd5
function make_transparent_cmap(cmap_base)
	# This only works if cmap_base is a matplotlib.colors.LinearSegmentedColormap
	# https://matplotlib.org/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html
	cdict = cmap_base._segmentdata
	r, g, b = getindex.(Ref(cdict), ("red", "green", "blue"))
	alpha = similar(r)
	T = eltype(r[1])
	N = length(alpha)
	Nh = (N - 1) / 2
	for n = 1:N
		dn = n - 1
		x = dn / (N - 1)
		y0 = (Nh - dn)^4 / Nh^4
		y1 = y0
		alpha[n] = (x, y0, y1)
	end
	name = cmap_base.name * "_alpha"
	cmap = PyPlot.ColorMap(name, r, g, b, alpha)
	cmap
end

# ╔═╡ 2a8579aa-0715-11eb-2182-01856621ec2c
function plot_vorticity!(ax, x, y, ω; kws...)
	M = round(sqrt(maximum(abs2, ω))) / 2
	levels = range(-M, M, length=4)
	cmap = make_transparent_cmap(plt.cm.bwr)
	cf = ax.contourf(
		x, y, ω';
		cmap = cmap,
		levels,
		extend = :both,
		# vmin = -M, vmax = M,
		kws...,
	)
	cf
end

# ╔═╡ 731478ec-070b-11eb-28b6-573261e0bb10
function plot_density!(ax, x, y, ρ; kws...)
	cf = ax.contourf(
		x, y, ρ;
		cmap = plt.cm.gray_r,
		kws...,
	)
	cf
end

# ╔═╡ e139a206-06ee-11eb-08a5-295e7d32d4f2
function plot_circulation!(ax, x, y, Γ; kws...)
	cf = ax.pcolormesh(
		x, y, Γ';
		shading = :flat,
		cmap = plt.cm.seismic,
		edgecolors = "tab:gray",
		lw = 0.1, kws...,
	)
	cf
end

# ╔═╡ 0f407512-072b-11eb-0383-6df8c19eaec0
md"# Plotting tools"

# ╔═╡ 0e027790-072b-11eb-123f-63c2ff71c321
function plot_colourmap(cmap)
	fig, ax = plt.subplots(figsize = (6, 1))
	ax.set_aspect(:auto)
	M = 255
	cols = 8
	data = reshape(repeat(collect(0:1:M), cols), :, cols)
	ax.imshow(data'; cmap, zorder=4)
  	# Draw background line to check transparency
	ax.axhline(cols >> 1, color="tab:green", zorder=3)
	fig
end

# ╔═╡ 287f2998-071b-11eb-1530-596effb8e8b3
plot_colourmap(make_transparent_cmap(plt.cm.RdBu))

# ╔═╡ 04437242-04b8-11eb-0d5b-857b5dc34b5d
md"# Setup"

# ╔═╡ 35ae4ae0-04b9-11eb-26c3-6b7a66da4acd
resampling = 1

# ╔═╡ 6531fd6e-0fab-11eb-063c-895e7d5ae867
from_convolution = false  # true -> testing!!

# ╔═╡ e260a816-06ec-11eb-3416-9f11f837c6cd
resolution = Val(1024)  # 256, 1024 or 2048

# ╔═╡ afb6568e-0935-11eb-3b94-c10bedef706a
get_value(::Val{N}) where {N} = N

# ╔═╡ 184e0186-04c1-11eb-0d3a-892888819d53
# Size of smallest circulation loop (= step of circulation grid)
grid_step = if from_convolution
	1
else
	resampling * get_value(resolution) >> 5
end

# ╔═╡ 65ed010e-06e6-11eb-3975-ef0217acb6c0
function compute_fields(gp, ψ)
	ρ = density(ψ)
	ps = momentum(ψ, gp)
	vs = map(p -> p ./ ρ, ps)
	(; ψ, ρ, ps, vs)
end

# ╔═╡ 9a742c84-04c5-11eb-1123-093a9f87394e
begin
	
function load_psi_tangle(::Val{256})
	filename_fmt = expanduser("~/Work/Shared/data/gGP_samples/tangle/256/fields/*Psi.001.dat")
	gp_in = ParamsGP((256, 256, 256); L = (2π, 2π, 2π), c = 1.0, nxi = 1.5)
	slice = (:, :, 3)
	gp = ParamsGP(gp_in, slice)
	gp, load_psi(gp_in, filename_fmt; slice)
end

function load_psi_tangle(::Val{1024})
	filename_fmt = expanduser("~/Dropbox/circulation/data/1024/2D/*2D_1024_slice2_t100.bin")
	gp = ParamsGP((1024, 1024); L = (2π, 2π), c = 1.0, nxi = 1.5)
	gp, load_psi(gp, filename_fmt)
end
	
function load_psi_tangle(::Val{2048})
	filename_fmt = expanduser("~/Dropbox/Data/gGP/Slices/Tangle2048/*2D_2048_slice340_t18.bin")
	gp = ParamsGP((2048, 2048); L = (2π, 2π), c = 1.0, nxi = 1.5)
	ψ = load_psi(gp, filename_fmt)
	gp, ψ
end

end

# ╔═╡ d7fb95b0-06de-11eb-3fb9-ed2c3a2981d4
gp_in, ψ_in = load_psi_tangle(resolution);

# ╔═╡ 4f067bc4-04c7-11eb-02cc-47b79114208c
function resample_fields(gp_a, ψ_a, factor)
	factor == 1 && return (gp_a, ψ_a)
	plan_a = plan_fft(ψ_a; flags = FFTW.ESTIMATE | FFTW.PRESERVE_INPUT)
	dims = factor .* size(gp_a)
	ψ = similar(ψ_a, dims)
	gp = ParamsGP(gp_a, dims=dims)
	resample_field_fourier!(ψ, plan_a * ψ_a, gp_in)
	ifft!(ψ)
	gp, ψ
end

# ╔═╡ a67e27d4-06e6-11eb-1881-f9bdd88a7a07
gp, ψ = resample_fields(gp_in, ψ_in, resampling);

# ╔═╡ c4721330-04ba-11eb-0dc4-a7a6244c08bd
gp

# ╔═╡ 9663eb22-06e6-11eb-3a2d-4d4912f69b4e
fields = compute_fields(gp, ψ);

# ╔═╡ 7a7efc84-04be-11eb-1b00-f783f2e7ac47
vint = IntegralField2D(fields.vs, L = gp.L);

# ╔═╡ 304081f6-04c1-11eb-1533-a3699c44cdd2
# Compute circulation on cells of circulation grid
Γ = if from_convolution
	let r = 0.1
		Ns = size(gp)
		Γ = Array{Float64}(undef, Ns...)
		vs = fields.vs
		vF = rfft.(vs)
		circulation!(Γ, vF, r)
	end
else
	let r = grid_step
		Ns = size(gp) .÷ grid_step
		Γ = Array{Float64}(undef, Ns...)
		circulation!(Γ, vint, (r, r); grid_step, centre_cells=false)
		Γ ./= gp.κ
		# circulation_filter!(Γ)
		Γ
	end
end;

# ╔═╡ 33ca1bc0-06dd-11eb-1bbc-0fca09e652dc
extrema(Γ)

# ╔═╡ a287cf14-06e0-11eb-23a9-9b78529a7ff5
let
	fig, ax = plt.subplots()
	ax.set_yscale(:log)
	ax.hist(vec(Γ), bins=200)
	fig
end

# ╔═╡ ec362e78-06e5-11eb-0627-3d164ec60a17
Γ_nans = count(isnan, Γ);

# ╔═╡ 7bf498b4-06ea-11eb-2ebf-3f08748feb08
md"NaN circulation values: $Γ_nans"

# ╔═╡ 5b6111a8-06e1-11eb-28ff-1f80244d1529
@assert maximum(abs2, ψ) > 0

# ╔═╡ ac9a4426-06dd-11eb-241f-efd33a18c513
extrema(abs2, ψ_in), extrema(abs2, ψ)

# ╔═╡ 4fa45496-0714-11eb-362e-9bbe41380d02
begin

	function load_vorticity_slice(::Val, gp, fields; momentum = true)
		vs = if momentum
			fields.ps
		else
			# Estimate vorticity from regularised velocity
			map(p -> p ./ sqrt.(fields.ρ), fields.ps)
		end
		vsF = rfft.(vs)
		ωF = similar(vsF[1])
		curlF!(ωF, vsF, gp)
		ω = irfft(ωF, size(vs[1], 1))
		x, y = get_coordinates(gp)
		(; x, y, ω)
	end

	function load_vorticity_slice(::Val{2048}, args...; kws...)
		filename = expanduser("~/Dropbox/Data/gGP/Slices/Tangle2048/2048Vort2048t18_slice340_x4.dat")
		resampling = 4
		N = 2048 * resampling
		ω = Array{Float64}(undef, N, N)
		@assert sizeof(ω) == stat(filename).size
		read!(filename, ω)
		x = range(0, 2π, length = N + 1)[1:end-1]
		y = x
		(; x, y, ω)
	end
	
end

# ╔═╡ f8ca32e6-0714-11eb-2dd2-614e83339175
vort = load_vorticity_slice(resolution, gp, fields; momentum = true);

# ╔═╡ 49293646-04c2-11eb-19fb-05609f24f2c6
fig_slice = let
	fig, ax = plt.subplots()
	ax.set_aspect(:equal)
	ax.set_xlabel(L"x")
	ax.set_ylabel(L"y")
	r0 = grid_step
	x, y = map(get_coordinates(gp), (r0, r0)) do x, r
		# Extend range to include x = 2π
		step = x[2] - x[1]
		z = range(first(x), length = length(x) + 1, step = step)
		z[1:r:end]
	end
	@assert last(x) ≈ 2π
	@assert length(x) == size(Γ, 1) + 1
	
	radii = reshape([
		8,  # lower left
		4,  # lower right
		1,   # upper left
		2,   # upper right
	], 2, 2)
	
	if from_convolution
		radii .*= 8
	end
	
	Hs = Int.(size(Γ) ./ size(radii))
	radii .= min.(radii, min(Hs...) >> 1)  # fixes low resolutions
	
	vmin, vmax = -20, 20
	
	cf = nothing
	
	for I in CartesianIndices(radii)
		ij = Tuple(I)
		r = radii[I]
		
		ij_start = @. Hs * (ij - 1) + 1
		ij_end = @. Hs * ij + 1
		range_in_grid = (:).(ij_start, ij_end)
		range_in_cell = map(x -> x[1:end-1], range_in_grid)
		Γ_in = view(Γ, range_in_cell...)
		xy_in = getindex.((x, y), range_in_grid)
		
		Γ_coarse = coarse_grain(Γ_in, r)
		xy_coarse = map(x -> x[1:r:end], xy_in)
		# cf = plot_circulation!(ax, xy_in..., Γ_in; vmin, vmax)
		cf = plot_circulation!(
                        ax, xy_coarse..., Γ_coarse;
			vmin, vmax,
			alpha = 0.6,
		)
	end
	
	fig.colorbar(cf, ax = ax, label = L"Γ / κ")
	
	if vort === nothing
		let (x, y) = get_coordinates(gp)
			plot_density!(ax, x, y, fields.ρ; alpha = 0.2)
		end
	else
		let step = 1
			is = step:step:length(vort.x)
			js = step:step:length(vort.y)
			@views cf = plot_vorticity!(
				ax, vort.x[is], vort.y[js], vort.ω[is, js];
			)
		end
		# fig.colorbar(cf, ax = ax, label = L"ω")
	end

	fig
end

# ╔═╡ 48963a30-06eb-11eb-30e3-d1a52d9a8f8a
fig_slice.savefig("circulation_slice.svg")

# ╔═╡ Cell order:
# ╟─93f9d4f2-04be-11eb-2bda-c35b6e65b22d
# ╠═7a7efc84-04be-11eb-1b00-f783f2e7ac47
# ╠═184e0186-04c1-11eb-0d3a-892888819d53
# ╠═24fb6616-06ed-11eb-202a-0966bf3cd31b
# ╠═304081f6-04c1-11eb-1533-a3699c44cdd2
# ╠═33ca1bc0-06dd-11eb-1bbc-0fca09e652dc
# ╟─a287cf14-06e0-11eb-23a9-9b78529a7ff5
# ╟─7bf498b4-06ea-11eb-2ebf-3f08748feb08
# ╟─ec362e78-06e5-11eb-0627-3d164ec60a17
# ╠═49293646-04c2-11eb-19fb-05609f24f2c6
# ╠═c3f9b778-0705-11eb-0d57-87de6a741735
# ╟─c63357ce-0719-11eb-3222-219e0ceecdd5
# ╟─2a8579aa-0715-11eb-2182-01856621ec2c
# ╟─287f2998-071b-11eb-1530-596effb8e8b3
# ╟─731478ec-070b-11eb-28b6-573261e0bb10
# ╟─e139a206-06ee-11eb-08a5-295e7d32d4f2
# ╠═48963a30-06eb-11eb-30e3-d1a52d9a8f8a
# ╟─0f407512-072b-11eb-0383-6df8c19eaec0
# ╟─0e027790-072b-11eb-123f-63c2ff71c321
# ╟─04437242-04b8-11eb-0d5b-857b5dc34b5d
# ╟─c4721330-04ba-11eb-0dc4-a7a6244c08bd
# ╠═35ae4ae0-04b9-11eb-26c3-6b7a66da4acd
# ╠═6531fd6e-0fab-11eb-063c-895e7d5ae867
# ╠═e260a816-06ec-11eb-3416-9f11f837c6cd
# ╟─afb6568e-0935-11eb-3b94-c10bedef706a
# ╠═a67e27d4-06e6-11eb-1881-f9bdd88a7a07
# ╠═9663eb22-06e6-11eb-3a2d-4d4912f69b4e
# ╠═65ed010e-06e6-11eb-3975-ef0217acb6c0
# ╠═d7fb95b0-06de-11eb-3fb9-ed2c3a2981d4
# ╠═f8ca32e6-0714-11eb-2dd2-614e83339175
# ╠═5b6111a8-06e1-11eb-28ff-1f80244d1529
# ╠═ac9a4426-06dd-11eb-241f-efd33a18c513
# ╠═4f067bc4-04c7-11eb-02cc-47b79114208c
# ╠═9a742c84-04c5-11eb-1123-093a9f87394e
# ╠═4fa45496-0714-11eb-362e-9bbe41380d02
# ╠═16106ec4-04b8-11eb-032e-072f8a2efdcc
# ╠═9040b526-04c6-11eb-3b36-77a97c91ed7a
# ╠═5e07c81e-04c2-11eb-2eaf-3d6136e0ed3e
