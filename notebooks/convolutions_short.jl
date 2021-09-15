### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 960fe2a4-c51b-11eb-0756-65f8adcbc92d
begin
	using Pkg
	Pkg.activate(".")
	using Revise
	using PyPlot
	using GPFields
	using FFTW
	using SpecialFunctions
end

# ╔═╡ 8909c44c-3f7c-484b-9e78-0a155296ca9a
step_function(x, a, b) = a ≤ x < b

# ╔═╡ efb16f8d-5f10-403c-80b3-4f38908f27d8
kernel_square(x, y, r; L = 2π) =
	[L^2 * step_function(x < π ? x : 2π - x, 0, r/2) *
	 step_function(y < π ? y : 2π - y, 0, r/2) for x in x, y in y]

# ╔═╡ d5121adb-af36-4d96-b6fe-c62690fb8fb7
circle_step(x, y, r) = x^2 + y^2 ≤ r^2

# ╔═╡ 3d78ded8-bafb-4980-8e27-1af8531cd989
kernel_circle(x, y, r; L = 2π) =
	[L^2 * circle_step(x < π ? x : 2π - x, y < π ? y : 2π - y, r/2)
	 for x in x, y in y]

# ╔═╡ 4bcc66b3-2ac2-4f33-9403-daa79f7de3e0
function kernel_square_fourier(kx, ky, r)
	R = r / 2π
	gs = map(k -> sinc(k * R), (kx, ky))
	r^2 * prod(gs)
end

# ╔═╡ b390126c-df50-4ce1-82a9-fa991172186e
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

# ╔═╡ f22a31eb-4942-4959-9ed4-5c7033de6b5a
function kernel_circle_fourier(kx, ky, r)
	R = r / 2π
	k = sqrt(kx^2 + ky^2)
	(π * r^2 / 4) * besselj1_norm(k * R)
end

# ╔═╡ 013d8e30-e27b-4388-8c35-4dbd3e278cd6
let
	N = 512
	r = 1.0
	xlim = 0.55r
	Ls = (π, π)
        Atotal = prod(Ls)
	xy = map(L -> range(0, L, length=N + 1)[1:N], Ls)
	x, y = xy
	# g = kernel_square(x, y, r)
	g = kernel_circle(x, y, r)
        kx = rfftfreq(N, 2π * N / Ls[1])
        ky = fftfreq(N, 2π * N / Ls[2])
	ks = (kx, ky)
	kmax = max(maximum.(ks)...)
	gF = rfft(g) ./ length(g)
	fig, axes = plt.subplots(2, 2, figsize=(6, 6) .* 0.8)
	levels_phys = -0.2:0.05:1.2
	let ax = axes[1, 1]
		ax.set_aspect(:equal)
                cf = ax.contourf(x, y, g' ./ (2π)^2; levels = levels_phys)
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
		# kernel = RectangularKernel(r)
		wF = DiscreteFourierKernel(kernel, ks).mat
		w = brfft(wF, N)
		ax.set_xlim(0, xlim)
		ax.set_ylim(0, xlim)
		cf = ax.contourf(x, y, w' ./ Atotal; levels = levels_phys)
		fig.colorbar(cf, ax=ax)
	end
	let ax = axes[2, 2]
		kr = kx .* r
		ax.plot(kx, real.(gF[:, 1]), label = "real")
		ax.plot(kx, imag.(gF[:, 1]), label = "imag")
		ax.plot(kx, kernel_square_fourier.(kx, 0, r) ./ Atotal, "--", label = "sinc")
		ax.plot(kx, kernel_circle_fourier.(kx, 0, r) ./ Atotal, "--", label = "Bessel") 
		ax.set_xlim(-2, 8π / r)
		ax.legend()
	end
	fig
end

# ╔═╡ Cell order:
# ╠═013d8e30-e27b-4388-8c35-4dbd3e278cd6
# ╠═efb16f8d-5f10-403c-80b3-4f38908f27d8
# ╠═3d78ded8-bafb-4980-8e27-1af8531cd989
# ╠═8909c44c-3f7c-484b-9e78-0a155296ca9a
# ╠═d5121adb-af36-4d96-b6fe-c62690fb8fb7
# ╠═4bcc66b3-2ac2-4f33-9403-daa79f7de3e0
# ╠═f22a31eb-4942-4959-9ed4-5c7033de6b5a
# ╠═b390126c-df50-4ce1-82a9-fa991172186e
# ╠═960fe2a4-c51b-11eb-0756-65f8adcbc92d
