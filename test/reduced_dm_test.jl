using Test
using Random
using Rimu
using Rimu.InterfaceTests: test_observable_interface, test_operator_interface,
    test_hamiltonian_interface, test_hamiltonian_structure
using Rimu.Interfaces: LOStructure, IsHermitian, IsDiagonal, AdjointKnown,
    AdjointUnknown
using Rimu.Hamiltonians: TestOneParticleDensity, GradOneParticleDensity,
    TestTwoParticleDensity, GradTwoParticleDensity, fulleigvectwoparticledensity

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
    x = zeros(Float64, 3, num_modes(address))
    para = [1.0, 0.0]
    for i in 1:num_modes(address)
        x[1, i] = para[1]*exp(-para[2]*(i-3)^2)
        x[2, i] = exp(-para[2]*(i-3)^2)
        x[3, i] = -para[1] * (i-3)^2 * exp(-para[2]*(i-3)^2)
    end
    opd = GradOneParticleDensity(x; zeta = dot(gs,TestOneParticleDensity(x[1,:]),gs))
    @test LOStructure(opd) isa IsHermitian
    test_operator_interface(opd, address)
    @test sum(abs(gs, opd, gs)) ≈ 0.0

    # Check that the result of show can be pasted into the REPL
    opd2 = GradOneParticleDensity(x; normalize=false)
    @test eval(Meta.parse(repr(opd2))) == opd2
end

@testset "TestTwoParticleDensity" begin
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)

    h = HubbardRealSpace(address; t=1, w=-1.0)
    ep = ExactDiagonalizationProblem(h)
    epsol = solve(ep)
    gs = epsol.vectors[1]
    onerdmop = ReducedDensityMatrix(2)
    onerdm = dot(gs, onerdmop, gs)
    evs, evc = eigen(Hermitian(onerdm))
    x = fulleigvectwoparticledensity(evc[:,end], 8)
    opd = TestTwoParticleDensity(x)
    @test LOStructure(opd) isa IsHermitian
    test_operator_interface(opd, address)
    @test dot(gs, opd, gs) ≈ maximum(evs)

    Random.seed!(1234)
    opdrand = TestOneParticleDensity(rand(num_modes(address)))
    @test minimum(evs) ≤ dot(gs, opdrand, gs) ≤ maximum(evs)

    # Check that the result of show can be pasted into the REPL
    opd2 = TestTwoParticleDensity(x; normalize=false)
    @test eval(Meta.parse(repr(opd2))) == opd2
end

@testset "GradOneParticleDensity" begin
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)

    h = ExtendedHubbardReal1D(address; t=1, v=-2, boundary_condition=:twisted)
    ep = ExactDiagonalizationProblem(h)
    epsol = solve(ep)
    gs = epsol.vectors[1]
    M = num_modes(address)
    x = zeros(Float64, 3, M, M);
    onerdmop = ReducedDensityMatrix(2)
    onerdm = dot(gs, onerdmop, gs)
    evs, evc = eigen(Hermitian(onerdm))
    x[1, :, :] = fulleigvectwoparticledensity(evc[:,end], 8)
    #gradient of 2-pdm w.r.t. parameters assumming the functional form: p[1]*exp(-p[2]*(i-j))
    x[2, :, :] = x[1, :, :]
    for i in 1:M, j in 1:i-1
        # gradient w.r.t. parameter
        if (i-j) <= 4
            x[3, i, j] = -(i-j) * x[1, i, j]
        else
            x[3, i, j] = -(8 - (i-j)) * x[1, i, j]
        end
        x[3, j, i] = -x[3, i, j]
    end
    opd = GradTwoParticleDensity(x; zeta = dot(gs,TestTwoParticleDensity(x[1,:, :]),gs))
    @test LOStructure(opd) isa IsHermitian
    test_operator_interface(opd, address)
    @test sum(dot(gs, opd, gs)) ≈ 0.0

    # Check that the result of show can be pasted into the REPL
    opd2 = GradOneParticleDensity(x; normalize=false)
    @test eval(Meta.parse(repr(opd2))) == opd2
end
