#!/usr/bin/env julia

using Circulation
using FFTW
using Test

function test_integration()
    L = 4pi
    N = 32
    x = LinRange(0, L, N + 1)[1:end-1]

    v = @. 0.2 + sin(x) + cos(2x)
    vI = @. 0.2x - cos(x) + sin(2x) / 2

    vf = rfft(v)
    k = rfftfreq(N, 2pi * N / L)

    a = searchsortedlast(x, 0)
    b = searchsortedlast(x, π / 4)

    int1 = vI[b] - vI[a]
    int2 = Circulation.integrate(vf, k, x[a], x[b]) / N

    @test int1 ≈ int2
end

function main()
    test_integration()
end

main()
