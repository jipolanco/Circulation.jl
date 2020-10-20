### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 79fd1b1a-0fe3-11eb-1a2b-6573aa57ee46
begin
	ENV["MPLBACKEND"] = "Agg"
	import PyPlot
	using LaTeXStrings
	const plt = PyPlot.plt
	const mpl = plt.matplotlib
	mpl.interactive(false)
end;

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
end

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
		kernel = EllipsoidalKernel(r, ks)
		wF = materialise(kernel)
		# wF = [
		# 	# kernel_square_fourier(k..., r)
		# 	kernel_circle_fourier(k..., (r, r); L = Ls)
		# 	for k in Iterators.product(ks...)
		# ]
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

# ╔═╡ 182f1376-0ff6-11eb-350e-a7265f281d7b
resampling = 4

# ╔═╡ dcf5911a-1043-11eb-0a2e-8778d99f43fb
loop_size = π / 4

# ╔═╡ 53b315bc-0fe4-11eb-30dc-a3785444134f
ψ_in, gp_in = let slice = (:, :, 4)
	N = 256
	filenames = expanduser("~/Work/Shared/data/gGP_samples/tangle/256/fields/*Psi.001.dat")
	gp_input = ParamsGP((N, N, N); L = (2pi, 2pi, 2pi), c = 1, nxi = 1.5)
	ψ = load_psi(gp_input, filenames; slice)
	gp = ParamsGP(gp_input, slice)
	ψ, gp
end;

# ╔═╡ 2a81a4d4-0ff2-11eb-2b97-21f0822618c2
function resample_psi(ψ_in, gp_in; resampling)
	if resampling == 1
		return ψ_in, gp_in
	end
	dims = resampling .* size(ψ_in)
	ψ = similar(ψ_in, resampling .* size(ψ_in))
	resample_field_fourier!(ψ, fft(ψ_in), gp_in)
	ifft!(ψ)
	gp = ParamsGP(gp_in; dims = dims)
	ψ, gp
end

# ╔═╡ dbcc43f0-0fe5-11eb-2eb0-9be38e54907e
ψ, gp = resample_psi(ψ_in, gp_in, resampling = resampling);

# ╔═╡ 796b337c-0ff1-11eb-2ba9-97f399308235
begin
	ρ = density(ψ)
	ps = momentum(ψ, gp)
	vs = map(p -> p ./ ρ, ps)
	plan2D = plan_rfft(vs[1])
	vF = Ref(plan2D) .* vs
	xy = get_coordinates(gp)
	ks = GPFields.get_wavenumbers(gp, Val(:r2c))
end;

# ╔═╡ 3233339a-0ff3-11eb-098c-51070a1c30d1
Γ = let
	r = loop_size
	Ns = length.(xy)
	Γhat = similar(vF[1])
	plan_inv = plan_irfft(Γhat, Ns[1])
	kernel = RectangularKernel(r, ks)
	gF = materialise(kernel)
	Γ = Matrix{Float64}(undef, Ns)
	print("Using convolution...")
	@time circulation!(Γ, vF, kernel, gF; buf = Γhat, plan_inv)
	Γ ./= gp.κ
end;

# ╔═╡ 3cf9e6d0-0ff5-11eb-23c0-4d20ff2e03b9
let
	fig, ax = plt.subplots(dpi=120)
	ax.set_aspect(:equal)
	vmax = 4
	cf = ax.pcolormesh(xy..., Γ'; vmax, vmin=-vmax,
		cmap=plt.cm.RdBu, shading=:auto)
	fig.colorbar(cf; ax, label=L"Γ / κ")
	ax.contour(xy..., ρ', levels=[0.1, ])
	ax.set_title("r = $(loop_size / π) π")
	fig
end

# ╔═╡ 90dd9228-1043-11eb-3ec4-4f98237703e2
Γ_orig = let
	grid_step = resampling
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

# ╔═╡ d35e2c60-0ff4-11eb-08c5-1f045d375c4f
let
	fig, ax = plt.subplots(dpi=120)
	ax.set_yscale(:log)
	M = 4.2
	bins = range(-M, M, length=400)
	# bins = 400
	ax.hist(vec(Γ); bins, label=:convolution, density=true)
	ax.hist(vec(Γ_orig); bins, alpha=0.5, label=:original, density=true)
	# ax.axvline.(-3:3, ls=:dotted, color="tab:grey", zorder=-1)
	ax.legend()
	fig
end

# ╔═╡ 03a544c0-103c-11eb-2e30-1d6e829715c1
extrema(ps[1])

# ╔═╡ 3711a71a-0fe5-11eb-0698-0d3290290f00
let
	fig, ax = plt.subplots(dpi=100)
	ax.imshow(ρ)
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
# ╠═3cf9e6d0-0ff5-11eb-23c0-4d20ff2e03b9
# ╠═d35e2c60-0ff4-11eb-08c5-1f045d375c4f
# ╠═90dd9228-1043-11eb-3ec4-4f98237703e2
# ╟─191c1b62-0fe7-11eb-2a03-0f3402de0f1a
# ╠═6161103a-0fe7-11eb-1186-29294ece4afd
# ╠═1bd240ea-0fef-11eb-293b-c9e8f84b4e06
# ╠═182f1376-0ff6-11eb-350e-a7265f281d7b
# ╠═dcf5911a-1043-11eb-0a2e-8778d99f43fb
# ╠═53b315bc-0fe4-11eb-30dc-a3785444134f
# ╠═2a81a4d4-0ff2-11eb-2b97-21f0822618c2
# ╠═dbcc43f0-0fe5-11eb-2eb0-9be38e54907e
# ╠═03a544c0-103c-11eb-2e30-1d6e829715c1
# ╠═3711a71a-0fe5-11eb-0698-0d3290290f00
# ╠═796b337c-0ff1-11eb-2ba9-97f399308235
# ╠═79fd1b1a-0fe3-11eb-1a2b-6573aa57ee46
# ╠═42dd1e28-0fe3-11eb-2ec1-8dba1f719df6
