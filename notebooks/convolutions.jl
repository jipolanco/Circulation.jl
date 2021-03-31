### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 42dd1e28-0fe3-11eb-2ec1-8dba1f719df6
begin
	using Pkg
	Pkg.activate(".")
	using Revise
	using GPFields
	using FFTW
	using LinearAlgebra
	using SpecialFunctions
	using BenchmarkTools
	
	ENV["MPLBACKEND"] = "Agg"
	import PyPlot
	PyPlot.matplotlib.interactive(false)
	using LaTeXStrings
	const plt = PyPlot.plt
	const mpl = plt.matplotlib
end;

# ╔═╡ 1ee3cc7a-0fe7-11eb-2a43-59039c95d09e
md"# Kernels in 1D"

# ╔═╡ 18ab49a6-11e9-11eb-2b3b-e5f61dfa0e58
# 2 * J1(πx) / πx — analogous of sinc(x)
function besselj1_norm(x)
	T = promote_type(typeof(x), typeof(π))
	if iszero(x)
		one(T)
	else
		y = π * x
		2 * besselj1(y) / y
	end :: T
end

# ╔═╡ b3d8987a-11e4-11eb-0992-2770e3477dcc
let
	fig, ax = plt.subplots()
	x = 0:0.1:10
	ax.plot(x, sinc.(x))
	ax.plot(x, besselj1_norm.(x))
	fig
end

# ╔═╡ 2ff03896-0fec-11eb-2c47-edd0e6364f8c
md"# Kernels in 2D"

# ╔═╡ 62ba911c-1042-11eb-0742-3b0fd972393e
function kernel_square_fourier(kx, ky, r; L = 2π)
	R = r / L
	gs = map(k -> sinc(k * R), (kx, ky))
	r^2 * prod(gs)
end

# ╔═╡ 844a98c2-1042-11eb-0252-1fd8f242ea50
begin
	function kernel_circle_fourier(kx, ky, r; L = 2π)
		R = r / L
		k = sqrt(kx^2 + ky^2)
		(π * r^2 / 4) * besselj1_norm(k * R)
	end
	
	# Generalisation for ellipse
	function kernel_circle_fourier(kx, ky, rs::NTuple{2}; L = (2π, 2π))
		Rs = rs ./ L
		ks = (kx, ky)
		kr = sqrt(sum(abs2, ks .* Rs))
		(π * prod(rs) / 4) * besselj1_norm(kr)
	end
end

# ╔═╡ 9c83146a-0ff1-11eb-3c9e-0fd52493550d
md"# Circulation"

# ╔═╡ d7ea2b1c-1f67-11eb-10f0-d728a65f971a
grid_method = Grids.BestInteger()

# ╔═╡ c22a4a50-1eae-11eb-346c-3b5ccb501a5f
function plot_detected_vortices!(ax, grid, gp; xylims = nothing)
	pos, neg = grid.positive, grid.negative
	colours = Dict(pos => "tab:blue", neg => "tab:red")
	Ns = size(grid)
	xy = map((N, L) -> range(0, L, length = N + 1), Ns, gp.L)
	kws = (marker = :s, markeredgewidth = 1, alpha = 0.6)
	for I in CartesianIndices(pos)
		for mat in (pos, neg)
			val = mat[I]
			iszero(val) && continue
			xy_local = map(xy, Tuple(I)) do x, i
				# (x[i] + x[i + 1]) / 2
				x[i]
			end
			if xylims !== nothing && !all(xylims[1] .≤ xy_local .≤ xylims[2])
				continue
			end
			markersize = 6 * abs(val)
			ax.plot(xy_local...; kws..., color = colours[mat], markersize)
		end
	end
	ax
end

# ╔═╡ e9d8781c-1f7e-11eb-139f-f75a8ef5f634
function plot_grid!(ax, xy, steps; subgrid = false)
	let kws = (lw = 0.5, c = "0.6")
		x, y = map((x, dx) -> x[1:dx:end], xy, steps)
		ax.axhline.(x; kws...)
		ax.axvline.(y; kws...)
	end
	if !subgrid
		return ax
	end
	let kws = (lw = 0.4, c = "0.8")
		x, y = map((x, dx) -> x[((dx >> 1) + 1):dx:end], xy, steps)
		ax.axhline.(x; kws...)
		ax.axvline.(y; kws...)
	end
	ax
end

# ╔═╡ 389c2466-711a-11eb-2df2-53eb482eba9e
function index_range(xs, (a, b); remove_end = false)
	i, j = map(x -> searchsortedlast(xs, x), (a, b))
	i:(j - remove_end)
end

# ╔═╡ 191c1b62-0fe7-11eb-2a03-0f3402de0f1a
md"# Setup"

# ╔═╡ 6161103a-0fe7-11eb-1186-29294ece4afd
step_function(x, a, b) = a ≤ x < b

# ╔═╡ 2927c7a4-0fe7-11eb-38de-41f91e223653
let
	N = 512
	L = 2π
	r = L / 4
	x = range(0, L, length=N + 1)[1:N]
	# g = step_function.(x, 0, r)
	g = step_function.(x, 0, r/2) .+ step_function.(x, L - r/2, L)
	kx = rfftfreq(N, 2π * N / L)
	kmax = maximum(kx)
	gF = rfft(g) ./ N
	smoothing(k) = exp(-k^2 / kmax^2)
	# smoothing(k) = 1  # no smoothing
	fig, axes = plt.subplots(1, 2, figsize=(6, 3))
	let ax = axes[1]
		wF = @. sinc(kx * r / L) * (r / L) * smoothing(kx)
		ax.plot(x, brfft(wF, N), ls=:solid, lw=1)
		# wF = @. besselj1(π * kx * r / L) / (kx * L / π)
		# wF[1, 1] = r / 2L
		wF = @. besselj1_norm(kx * r / L) * (r / L) * smoothing(kx)
		ax.plot(x, brfft(wF, N), ls=:solid, lw=1)
		ax.plot(x, g, ls=:dotted, lw=2)
	end
	let ax = axes[2]
		ax.plot(kx, real.(gF), lw=1, label = "real")
		ax.plot(kx, imag.(gF), lw=1, label = "imag")
		ax.plot(kx, sinc.(kx .* r / L) .* (r / L) .* smoothing.(kx), ls=:dashed, lw=1, label = "sinc")
		ax.plot(kx, besselj1_norm.(kx .* r / L) .* (r / L) .* smoothing.(kx), label = "Bessel")
		ax.set_xlim(-2, 160)
		ax.legend()
	end
	fig
end

# ╔═╡ 431af800-0ff3-11eb-3a85-c39c489879f6
kernel_square(x, y, r; L = 2π) =
	[L^2 * step_function(x < π ? x : 2π - x, 0, r/2) *
	 step_function(y < π ? y : 2π - y, 0, r/2) for x in x, y in y]

# ╔═╡ 1bd240ea-0fef-11eb-293b-c9e8f84b4e06
circle_step(x, y, r) = x^2 + y^2 ≤ r^2

# ╔═╡ 67e2449a-0ff3-11eb-1481-8d020ffdb184
kernel_circle(x, y, r; L = 2π) =
	[L^2 * circle_step(x < π ? x : 2π - x, y < π ? y : 2π - y, r/2)
	 for x in x, y in y]

# ╔═╡ 390ed57a-0fec-11eb-09fe-6b20f5825dd3
let
	N = 512
	r = 1.0
	xlim = 0.55r
	Ls = (2π, 2π)
	Atotal = prod(Ls)
	xy = map(L -> range(0, L, length=N + 1)[1:N], Ls)
	x, y = xy
	# g = kernel_square(x, y, r)
	g = kernel_circle(x, y, r)
	kx = rfftfreq(N, N)
	ky = fftfreq(N, N)
	ks = (kx, ky)
	kmax = max(maximum.(ks)...)
	gF = rfft(g) ./ length(g)
	fig, axes = plt.subplots(2, 2, figsize=(6, 6))
	levels_phys = -0.2:0.05:1.2
	let ax = axes[1, 1]
		ax.set_aspect(:equal)
		cf = ax.contourf(x, y, g' ./ Atotal; levels = levels_phys)
		fig.colorbar(cf, ax=ax)
		ax.set_xlim(0, xlim)
		ax.set_ylim(0, xlim)
	end
	let ax = axes[1, 2]
		ax.set_aspect(:equal)
		cf = ax.contourf(kx, ky, log10.(abs.(gF')), levels = -4:0.5:-1)
		xmax = 50
		ax.set_xlim(0, xmax)
		ax.set_ylim(-xmax, xmax)
		# ax.grid(true)
		fig.colorbar(cf, ax=ax)
	end
	let ax = axes[2, 1]
		ax.set_aspect(:equal)
		kernel = EllipsoidalKernel(r)
		wF = DiscreteFourierKernel(kernel, ks).mat
		w = brfft(wF, N) ./ Atotal
		ax.set_xlim(0, xlim)
		ax.set_ylim(0, xlim)
		cf = ax.contourf(x, y, w'; levels = levels_phys)
		fig.colorbar(cf, ax=ax)
	end
	let ax = axes[2, 2]
		kr = kx .* r
		ax.plot(kx, real.(gF[:, 1]), label = "real")
		ax.plot(kx, imag.(gF[:, 1]), label = "imag")
		ax.plot(kx, kernel_square_fourier.(kx, 0, r; L=Ls[1]) ./ Atotal, "--", label = "sinc")
		ax.plot(kx, kernel_circle_fourier.(kx, 0, r; L=Ls[1]) ./ Atotal, "--", label = "Bessel") 
		ax.set_xlim(-2, 8π / r)
		ax.legend()
	end
	fig
end

# ╔═╡ 2aa86a5c-1fff-11eb-1ec0-55209eccdc17
benchmarks = true

# ╔═╡ 182f1376-0ff6-11eb-350e-a7265f281d7b
resampling = 4

# ╔═╡ 7ac7dda2-1f4f-11eb-0a96-897e81ea4742
hostname = Symbol(replace(gethostname(), '.' => '_'))

# ╔═╡ 53b02da8-1ec0-11eb-3d1d-0d8ee9eeefb4
begin
	function load_psi_resolution(::Val{256}, ::Val{host}, resampling) where {host}
		N = 256
		slice = (:, :, 4)
		workdir = host == :thinkpad ? "~/Work" : "~/Work/Shared"
		filenames = expanduser("$workdir/data/gGP_samples/tangle/256/fields/*Psi.001.dat")
		gp_input = ParamsGP((N, N, N); L = (2pi, 2pi, 2pi), c = 1, nxi = 1.5)
		load_psi(gp_input, filenames; slice, resampling)
	end
	
	function load_psi_resolution(::Val{1024}, ::Val{host}, resampling) where {host}
		N = 1024
		slice = 4  # in [0, 9]
		workdir = host == :thinkpad ? nothing : "~/Work"
		filenames = expanduser("$workdir/data/Circulation/gGP/1024/2D/*2D_1024_slice$(slice)_t100.bin")
		gp_input = ParamsGP((N, N); L = (2pi, 2pi), c = 1, nxi = 1.5)
		load_psi(gp_input, filenames; resampling)
	end
	
	function load_psi_resolution(::Val{1024}, ::Val{:castor_cluster_local}, resampling)
		N = 1024
		slice = (:, :, 4)
		# filenames = expanduser("$workdir/data/Circulation/gGP/1024/2D/*2D_1024_slice$(slice)_t100.bin")
		filenames = joinpath(ENV["SCRATCHDIR"], "..", "nmuller", "circulation",
			"1024", "dataGP", "*Psi.300.dat")
		gp_input = ParamsGP((N, N, N); L = (2pi, 2pi, 2pi), c = 1, nxi = 1.5)
		load_psi(gp_input, filenames; slice, resampling)
	end
end

# ╔═╡ 53b315bc-0fe4-11eb-30dc-a3785444134f
gp, ψ = load_psi_resolution(
	Val(1024),
	Val(hostname),
	resampling,
);

# ╔═╡ dcf5911a-1043-11eb-0a2e-8778d99f43fb
loop_size = 
	# gp.dx[1] * resampling * 1
	gp.dx[1] * 2

# ╔═╡ e4589a9c-1f57-11eb-264d-014214a80357
cell_step = round.(Int, loop_size ./ gp.dx)

# ╔═╡ 796b337c-0ff1-11eb-2ba9-97f399308235
begin
	plt.close(:all)
	ρ = GPFields.density(ψ)
	ps = momentum(ψ, gp)
	vs = map(p -> p ./ ρ, ps)
	plan2D = plan_rfft(vs[1], flags=FFTW.MEASURE)
	vF = Ref(plan2D) .* vs
	xy = coordinates(gp)
	xy_in = map(x -> x[1:end-1], xy)
	ks = GPFields.get_wavenumbers(gp, Val(:r2c))
end;

# ╔═╡ 3233339a-0ff3-11eb-098c-51070a1c30d1
Γ = let
	r = loop_size
	Ns = length.(xy) .- 1
	Γhat = similar(vF[1])
	plan_inv = plan_irfft(Γhat, Ns[1]; flags=FFTW.MEASURE)
	kernel = RectangularKernel(r)
	# kernel = EllipsoidalKernel(r * sqrt(2))
	gF = DiscreteFourierKernel{Float64}(undef, ks...)
	materialise!(gF, kernel)
	Γ = Matrix{Float64}(undef, Ns)
	print("Using convolution...    ")
	@time circulation!(Γ, vF, gF; buf = Γhat, plan_inv)
	Γ ./= gp.κ
end;

# ╔═╡ c23d43da-1e77-11eb-10b2-5553330282c3
grid = to_grid(Γ, cell_step, grid_method, Bool;
			   κ = 1, force_unity = true, cleanup = true, cell_size = (2, 2));

# ╔═╡ e71bacde-1eb1-11eb-2aa3-d3f0fb5fdf61
md"Positive / negative vortices found: $(sum(grid.positive), sum(grid.negative))"

# ╔═╡ 0491c7c6-1ec1-11eb-2621-19f081b1024f
grid_circulation = grid.positive .- grid.negative;

# ╔═╡ d35e2c60-0ff4-11eb-08c5-1f045d375c4f
let
	plt.close(:all)
	fig, ax = plt.subplots()
	ax.set_yscale(:log)
	M = 4.2
	bins = range(-M, M, length=100)
	# bins = 400
	kws = (; bins, density = true)
	ax.hist(vec(Γ); kws..., label=:convolution)
	# ax.hist(vec(Γ_orig); kws..., alpha=0.5, label=:original)
	ax.hist(vec(grid_circulation); kws..., alpha=0.4, label=:grid)
	# ax.axvline.(-3:3, ls=:dotted, color="tab:grey", zorder=-1)
	ax.legend()
	fig
end

# ╔═╡ c2236452-7119-11eb-28d1-8fd53c040dc4
let
	xlim = (1.5, 2) .* π
	ylim = xlim
	lims = (xlim, ylim)

	plt.close(:all)
	fig = plt.figure(dpi = 200)
	ax = fig.subplots()
	ax.set_aspect(:equal)
	
	let
		inds = index_range.(xy, lims)
		inds_cells = map(x -> x[1:end-1], inds)
		xy_lims = view.(xy, inds)
		Γ_lims = view(Γ, inds_cells...)
		vmax = 1
		cf = ax.pcolormesh(
			xy_lims..., Γ_lims'; vmax, vmin=-vmax,
			cmap=plt.cm.RdBu, shading=:flat,
		)
		fig.colorbar(cf; ax, label=L"Γ / κ")
	end
	
	let
		inds = index_range.(xy_in, lims; remove_end=true)
		xy_lims = view.(xy_in, inds)
		ρ_lims = view(ρ, inds...)
		ax.contour(xy_lims..., ρ_lims'; linewidths = 0.5, colors = "0.5", levels=[0.01, ])
	end
	
	xylims = broadcast((f, x) -> f(x), (first, last), lims)
	plot_detected_vortices!(ax, grid, gp; xylims)
	
	fig
end

# ╔═╡ 90dd9228-1043-11eb-3ec4-4f98237703e2
Γ_orig = let
	grid_step = 1
	rs = round.(Int, loop_size .* size(gp) ./ (2π))
	vint = IntegralField2D(vs[1]; L = gp.L)
	Γ = similar(vs[1], size(vs[1]) .÷ grid_step)
	print("Using original method...")
	@time begin
		prepare!(vint, vs)
		circulation!(Γ, vint, rs; grid_step)
	end
	Γ ./= gp.κ
end;

# ╔═╡ 03a544c0-103c-11eb-2e30-1d6e829715c1
extrema(ps[1])

# ╔═╡ 3711a71a-0fe5-11eb-0698-0d3290290f00
let
	plt.close(:all)
	fig, ax = plt.subplots(dpi=100)
	ax.imshow(ρ')
	# ax.quiver(ps...)
	fig
end

# ╔═╡ Cell order:
# ╟─1ee3cc7a-0fe7-11eb-2a43-59039c95d09e
# ╠═18ab49a6-11e9-11eb-2b3b-e5f61dfa0e58
# ╠═b3d8987a-11e4-11eb-0992-2770e3477dcc
# ╠═2927c7a4-0fe7-11eb-38de-41f91e223653
# ╟─2ff03896-0fec-11eb-2c47-edd0e6364f8c
# ╠═390ed57a-0fec-11eb-09fe-6b20f5825dd3
# ╠═431af800-0ff3-11eb-3a85-c39c489879f6
# ╠═67e2449a-0ff3-11eb-1481-8d020ffdb184
# ╠═62ba911c-1042-11eb-0742-3b0fd972393e
# ╠═844a98c2-1042-11eb-0252-1fd8f242ea50
# ╟─9c83146a-0ff1-11eb-3c9e-0fd52493550d
# ╠═3233339a-0ff3-11eb-098c-51070a1c30d1
# ╟─e71bacde-1eb1-11eb-2aa3-d3f0fb5fdf61
# ╠═e4589a9c-1f57-11eb-264d-014214a80357
# ╠═d7ea2b1c-1f67-11eb-10f0-d728a65f971a
# ╠═c23d43da-1e77-11eb-10b2-5553330282c3
# ╠═0491c7c6-1ec1-11eb-2621-19f081b1024f
# ╠═c22a4a50-1eae-11eb-346c-3b5ccb501a5f
# ╠═e9d8781c-1f7e-11eb-139f-f75a8ef5f634
# ╠═c2236452-7119-11eb-28d1-8fd53c040dc4
# ╠═389c2466-711a-11eb-2df2-53eb482eba9e
# ╠═d35e2c60-0ff4-11eb-08c5-1f045d375c4f
# ╠═90dd9228-1043-11eb-3ec4-4f98237703e2
# ╟─191c1b62-0fe7-11eb-2a03-0f3402de0f1a
# ╠═6161103a-0fe7-11eb-1186-29294ece4afd
# ╠═1bd240ea-0fef-11eb-293b-c9e8f84b4e06
# ╠═2aa86a5c-1fff-11eb-1ec0-55209eccdc17
# ╠═182f1376-0ff6-11eb-350e-a7265f281d7b
# ╠═dcf5911a-1043-11eb-0a2e-8778d99f43fb
# ╠═7ac7dda2-1f4f-11eb-0a96-897e81ea4742
# ╠═53b315bc-0fe4-11eb-30dc-a3785444134f
# ╠═53b02da8-1ec0-11eb-3d1d-0d8ee9eeefb4
# ╠═03a544c0-103c-11eb-2e30-1d6e829715c1
# ╠═3711a71a-0fe5-11eb-0698-0d3290290f00
# ╠═796b337c-0ff1-11eb-2ba9-97f399308235
# ╠═42dd1e28-0fe3-11eb-2ec1-8dba1f719df6
