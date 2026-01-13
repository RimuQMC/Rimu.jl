using Test
using Random
using Rimu
using Rimu.InterfaceTests: test_observable_interface, test_operator_interface,
    test_hamiltonian_interface, test_hamiltonian_structure
using Rimu.Interfaces: LOStructure, IsHermitian, IsDiagonal, AdjointKnown,
    AdjointUnknown
using Rimu.Hamiltonians: TestOneParticleDensity, GradOneParticleDensity,
    TestTwoParticleDensity, GradTwoParticleDensity, index

@testset "ReducedDensityMatrix" begin
    dvec_f = PDVec(FermiFS{2,4}(1, 1, 0, 0) => 0.5, FermiFS{2,4}(0, 0, 1, 1) => 0.5)
    dvec_b = PDVec(BoseFS{4,4}(0, 0, 2, 2) => 0.5, BoseFS{4,4}(2, 2, 0, 0) => 0.5)
    op = ReducedDensityMatrix(1)
    spd_b = zeros(4, 4)
    spd_f = zeros(4, 4)
    for i in 1:4, j in 1:4
        spd_b[i, j] = dot(dvec_b, SingleParticleExcitation(i, j), dvec_b)
        spd_f[i, j] = dot(dvec_f, SingleParticleExcitation(i, j), dvec_f)
    end
    tpd_f = zeros(6, 6)
    t1 = 0
    t2 = 0
    for i in 1:4, j in i+1:4
        t1 += 1
        t2 = 0
        for k in 1:4, l in k+1:4
            t2 += 1
            tpd_f[t1, t2] = dot(dvec_f, TwoParticleExcitation(i, j, k, l), dvec_f)
        end
    end
    @test dot(dvec_f, op, dvec_f) == spd_f
    @test dot(dvec_b, op, dvec_b) == spd_b
    @test dot(dvec_f, ReducedDensityMatrix(2), dvec_f) == tpd_f
    @test_throws ArgumentError dot(dvec_b, ReducedDensityMatrix(2), dvec_b)
    @test LOStructure(op) isa IsHermitian
    test_observable_interface(ReducedDensityMatrix(1), BoseFS{4,4}(2, 2, 0, 0))
    test_observable_interface(ReducedDensityMatrix(2), FermiFS{2,4}(1, 1, 0, 0))
    for r in (ReducedDensityMatrix(1), ReducedDensityMatrix{ComplexF32}(2))
        # Check that the result of show can be pasted into the REPL
        @test eval(Meta.parse(repr(r))) == r
    end
    # complex hermitian Hamiltonian still produces approx hermitian RDM
    H = HubbardReal1D(BoseFS(0, 1, 2, 0); t=1 + im)
    res = solve(ExactDiagonalizationProblem(H))
    gs = res.vectors[1]
    rdm = ReducedDensityMatrix{ComplexF64}(1)
    m = dot(gs, rdm, gs)
    @test all(x -> abs(x) < √eps(Float64), m - m') # hermitian up to floating point noise

    # a global relative phase in the vectors results in a global phase in the RDM
    m_phase = dot(im * gs, rdm, gs)
    @test all(x -> abs(x) < √eps(Float64), m_phase + im * m)

    # complex non-hermitian Hamiltonian still produces approx hermitian RDM
    Hc = HubbardReal1D(BoseFS(0, 1, 2, 0); u=1 + im)
    resc = solve(ExactDiagonalizationProblem(Hc))
    gsc = resc.vectors[1]
    mc = dot(gsc, rdm, gsc)
    @test all(x -> abs(x) < √eps(Float64), mc - mc') # hermitian up to floating point noise
end

@testset "TestOneParticleDensity" begin
    address = BoseFS(1, 1, 1, 1, 0, 0, 0, 0)
    x = ones(num_modes(address))
    opd = TestOneParticleDensity(x)
    @test LOStructure(opd) isa IsHermitian
    test_operator_interface(opd, address)

    h = HubbardRealSpace(address; t=1, u=0.2)
    ep = ExactDiagonalizationProblem(h)
    epsol = solve(ep)
    gs = epsol.vectors[1]
    onerdmop = ReducedDensityMatrix(1)
    onerdm = dot(gs, onerdmop, gs)
    evs = eigvals(Hermitian(onerdm))
    @test dot(gs, opd, gs) ≈ maximum(evs)

    Random.seed!(1234)
    opdrand = TestOneParticleDensity(rand(num_modes(address)))
    @test minimum(evs) ≤ dot(gs, opdrand, gs) ≤ maximum(evs)

    # Check that the result of show can be pasted into the REPL
    opd2 = TestOneParticleDensity(x; normalize=false)
    @test eval(Meta.parse(repr(opd2))) == opd2
end

@testset "GradOneParticleDensity" begin
    address = BoseFS(1, 1, 1, 0, 0, 0)

    h = HubbardRealSpace(address; t=1, u=0.2)
    ep = ExactDiagonalizationProblem(h)
    epsol = solve(ep)
    gs = epsol.vectors[1]
    x1 = zeros(Float64, num_modes(address))
    x2 = zeros(Float64, 2, num_modes(address))
    para = [1.0, 0.0]
    for i in 1:num_modes(address)
        x1[i] = para[1]*exp(-para[2]*(i-3)^2)
        x2[1, i] = exp(-para[2]*(i-3)^2)
        x2[2, i] = -para[1] * (i-3)^2 * exp(-para[2]*(i-3)^2)
    end
    x = (x1, x2,)
    opd = GradOneParticleDensity(x; zeta = dot(gs,TestOneParticleDensity(x[1]),gs))
    @test LOStructure(opd) isa IsHermitian
    test_operator_interface(opd, address)
    @test round(sum(dot(gs, opd, gs)), digits = 12) == 0.0

    # Check that the result of show can be pasted into the REPL
    opd2 = GradOneParticleDensity(x; normalize=false)
    @test eval(Meta.parse(repr(opd2))) == opd2
end

@testset "TestTwoParticleDensity" begin
    address = FermiFS(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)

    h = HubbardRealSpace(address; t=1, w=-1.0)
    ep = ExactDiagonalizationProblem(h)
    epsol = solve(ep)
    gs = epsol.vectors[1]
    onerdmop = ReducedDensityMatrix(2)
    onerdm = dot(gs, onerdmop, gs)
    evs, evc = eigen(Hermitian(onerdm))
    x = evc[:, end];
    opd = TestTwoParticleDensity(x)
    @test LOStructure(opd) isa IsHermitian
    test_operator_interface(opd, address)
    @test dot(gs, opd, gs) ≈ 2 * maximum(evs)

    Random.seed!(1234)
    opdrand = TestTwoParticleDensity(rand(45))
    @test minimum(evs) ≤ dot(gs, opdrand, gs) ≤ 2 * maximum(evs)

    # Check that the result of show can be pasted into the REPL
    opd2 = TestTwoParticleDensity(zero(x).+1.0 ; normalize=false)
    @test eval(Meta.parse(repr(opd2))) == opd2
end

@testset "GradTwoParticleDensity" begin
    address = FermiFS(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)

    h = HubbardRealSpace(address; t=1, w=-1.0)
    ep = ExactDiagonalizationProblem(h)
    epsol = solve(ep)
    gs = epsol.vectors[1]
    M = num_modes(address)
    y = zeros(Float64, 3, binomial(M, 2));
    onerdmop = ReducedDensityMatrix(2)
    onerdm = dot(gs, onerdmop, gs)
    evs, evc = eigen(Hermitian(onerdm))
    #gradient of 2-pdm w.r.t. parameters assumming the functional form: p[1]*exp(-p[2]*(i-j))
    y = zeros(eltype(evc[:,end]), 2, length(evc[:,end]))
    y[1, :] = evc[:,end]
    for i in 1:M, j in 1:i-1
        # gradient w.r.t. parameter
        if (i-j) <= M/2
            y[2, index((i, j))] = -(i-j) * y[1, index((i, j))]
        else
            y[2, index((i, j))] = -(8 - (i-j)) * y[1, index((i, j))]
        end
    end
    x = (evc[:, end], y,)    
    opd = GradTwoParticleDensity(x; 
        zeta = dot(gs,TestTwoParticleDensity(evc[:, end]),gs))
    @test LOStructure(opd) isa IsHermitian
    test_operator_interface(opd, address)
    @test round(sum(dot(gs, opd, gs)), digits = 10) == 0.0

    # Check that the result of show can be pasted into the REPL
    opd2 = GradTwoParticleDensity((zero(x[1]).+1.0, zero(x[2]).+1,); normalize=false)
    @test eval(Meta.parse(repr(opd2))) == opd2
end
