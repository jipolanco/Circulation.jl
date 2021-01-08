### A Pluto.jl notebook ###
# v0.12.15

using Markdown
using InteractiveUtils

# ╔═╡ 50ddeda0-30cb-11eb-0845-575a6fa6f018
begin
	# using Pkg
	# pkg"activate ."
	import PyPlot: plt
	using HDF5
	using VoronoiDelaunay
	using GeometricalPredicates
	using Statistics
	using StatsBase
	using LinearAlgebra: normalize!, normalize
	using Distributions
end

# ╔═╡ e717c624-30cb-11eb-29e0-274eda111bef
data_file = "../results/GP_vortices/vortices_x16.h5"

# ╔═╡ 1d456d76-327e-11eb-2cec-f31dc9b4743a
begin
	function normalise_coord(x::Real)
		a = VoronoiDelaunay.min_coord
		b = VoronoiDelaunay.max_coord
		L = b - a
		N = 1024
		z = (x - 1) / N * L
		@assert 0 ≤ z < L
		z + a
	end
	
	normalise_coord(x::Tuple) = map(normalise_coord, x)
end

# ╔═╡ 20623002-30d3-11eb-3de5-29c1c003c71d
function voronoi(points_in)
	tess = DelaunayTessellation(length(points_in))
	points = map(points_in) do xy
		Point(normalise_coord.(xy)...)
	end
	push!(tess, points)
	tess
end

# ╔═╡ ee31e1ba-327e-11eb-1095-071198749d9c
function plot_voronoi!(ax, tess; kws...)
	xy = getplotxy(voronoiedges(tess))
	ax.plot(xy...; kws...)
	let a = VoronoiDelaunay.min_coord, b = VoronoiDelaunay.max_coord
		ax.set_xlim(a, b)
		ax.set_ylim(a, b)
	end
	
	return ax
	
	# Plot a single edge
	local edge
	for (n, e) in enumerate(voronoiedges(tess))
		edge = e
		n == 4 && break
	end
	let
		a = geta(edge)
		b = getb(edge)
		gena = getgena(edge)
		genb = getgenb(edge)
		ax.plot(getx(a), gety(a), "o"; color = "tab:red")
		ax.plot(getx(b), gety(b), "s"; color = "tab:red")
		ax.plot(getx(gena), gety(gena), "o"; color = "tab:green")
		ax.plot(getx(genb), gety(genb), "s"; color = "tab:green")
	end
	
	ax
end

# ╔═╡ 723cbf82-30ce-11eb-29f3-f5d0229000a4
function filter_slice_x(coords, i::Integer)
	@assert size(coords, 1) == 3
	coords_slice = Dims{2}[]
	for j in axes(coords, 2)
		coords[1, j] == i || continue
		push!(coords_slice, (coords[2, j], coords[3, j]))
	end
	coords_slice
end

# ╔═╡ fbf90966-30cd-11eb-31c0-cdb3352d6c6d
locations = h5open(data_file, "r") do ff
	gname = "OrientationX"
	pos = ff["$gname/positive"][:, :] :: Array{Int32,2}
	neg = ff["$gname/negative"][:, :] :: Array{Int32,2}
	i = 42
	(
		positive = filter_slice_x(pos, i),
		negative = filter_slice_x(neg, i),
	)
end

# ╔═╡ 3048279e-30d3-11eb-2bb3-750761ba8dc8
tess = (
	positive = voronoi(locations.positive),
	negative = voronoi(locations.negative),
)

# ╔═╡ f04b1750-3283-11eb-3265-f7f34ea4684f
areas = let
	areas = Dict{Point2D,Float64}()
	inlims(x) = VoronoiDelaunay.min_coord < x < VoronoiDelaunay.max_coord
	for edge in voronoiedges(tess.positive)
		a = geta(edge)
		b = getb(edge)
		for c in (getgena(edge), getgenb(edge))
			if !(inlims(getx(c)) && inlims(gety(c)))
				continue
			end
			triangle = Primitive(a, b, c)
			A = area(triangle)
			Aprev = get(areas, c, 0.0)  # TODO optimise...
			areas[c] = Aprev + A
		end
	end
	areas
end

# ╔═╡ 1299b5be-30cf-11eb-0f99-9d833180bf04
let
	fig, ax = plt.subplots()
	ax.set_aspect(:equal)
	pos = normalise_coord.(locations.positive)
	neg = normalise_coord.(locations.negative)
	ax.plot(getindex.(pos, 1), getindex.(pos, 2), "+"; color = :C0)
	# ax.plot(getindex.(neg, 1), getindex.(neg, 2), "x"; color = :C1)
	plot_voronoi!(ax, tess.positive; color = :C0, lw = 1)
	# plot_voronoi!(ax, tess.negative; color = :C1, lw = 1)
	for (c, A) in pairs(areas)
		A < 0.1 || continue
		ax.plot(getx(c), gety(c), "o", color = "0.5"; markersize = 1000A)
	end
	fig
end

# ╔═╡ a4c3b5e0-32d7-11eb-2089-bd0b55ed0790
let
	fig, ax = plt.subplots()
	dx = 5 / 100
	bins = range(0, 5, step = dx)
	As = filter!(<(0.04), sort(collect(values(areas))))
	σ = std(As)
	h = fit(Histogram, As ./ σ, bins)
	hnorm = normalize(h, mode = :pdf)
	# normalize(h, mode = :pdf)
	# ax.hist(As ./ σ; bins)
	ax.bar(hnorm.edges[1][1:end-1], hnorm.weights; width = dx, align = :edge)

	# Expected Gamma distribution from Poisson distribution in 2D (Ferenc & Néda 2007)
	let
		x = 0.5 .* (bins[1:end-1] .+ bins[2:end])
		p = (343 / 15) * sqrt(7 / 2pi) * x.^(5/2) .* exp.(-(7/2) .* x)
		ax.plot(x, p; color = :black, ls = :dashed)
	end
	
	# ax.set_yscale(:log)
	
	fig
end

# ╔═╡ Cell order:
# ╠═50ddeda0-30cb-11eb-0845-575a6fa6f018
# ╠═e717c624-30cb-11eb-29e0-274eda111bef
# ╠═fbf90966-30cd-11eb-31c0-cdb3352d6c6d
# ╠═3048279e-30d3-11eb-2bb3-750761ba8dc8
# ╠═1d456d76-327e-11eb-2cec-f31dc9b4743a
# ╠═20623002-30d3-11eb-3de5-29c1c003c71d
# ╠═f04b1750-3283-11eb-3265-f7f34ea4684f
# ╠═ee31e1ba-327e-11eb-1095-071198749d9c
# ╠═723cbf82-30ce-11eb-29f3-f5d0229000a4
# ╠═1299b5be-30cf-11eb-0f99-9d833180bf04
# ╠═a4c3b5e0-32d7-11eb-2089-bd0b55ed0790
