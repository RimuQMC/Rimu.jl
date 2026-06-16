using Test
using Rimu
using Rimu: StepAction, OperatorOverlaps, StrictPairIter, SingleState,
    SpectralState, CoefficientVectorOverlaps, ParticleDensityGradientOverlap,
    OptimizationAction
using StaticArrays: SVector

@testset "StrictPairIter" begin
    spi = StrictPairIter(4)
    pairs = collect(spi)
    @test pairs == [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    @test length(spi) == 6
end

@testset "OperatorOverlaps and CoefficientVectorOverlaps" begin
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)
    h = ExtendedHubbardReal1D(address, v=2)
    # Operator overlaps test
    oops = OperatorOverlaps(h; name=:test_overlaps)
    @test oops == OperatorOverlaps(h, :test_overlaps)
    p = ProjectorMonteCarloProblem(h; n_replicas=3, step_actions=(oops,))
    res = solve(p)
    @test res.df.test_overlaps isa Vector{Matrix{Float64}}
    # Coefficient vector overlaps test
    cvos = CoefficientVectorOverlaps()
    replica_strategy = AllOverlaps(4; operator=h)
    p2 = ProjectorMonteCarloProblem(
        h;
        replica_strategy,
        step_actions=(cvos,oops)
    )
    res2 = solve(p2)
    df2 = res2.df
    @test first.(df2.coefficient_vector_overlaps) ≈ df2.r1s1_dot_r2s1
    @test first.(df2.test_overlaps) ≈ df2.r1s1_Op1_r2s1
end

@testset "OptimizationAction" begin
    address = FermiFS(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)

    h = HubbardRealSpace(address; w=-1.0)
    ep = ExactDiagonalizationProblem(h)
    epsol = solve(ep)
    gs = epsol.vectors[1]
    M = num_modes(address)
    onerdmop = ReducedDensityMatrix(2)
    onerdm = dot(gs, onerdmop, gs)
    evs, evc = eigen(Hermitian(onerdm))   
    # Operator overlaps test
    optimizationparameter = [SVector{binomial(M,2),eltype(evs)}(evc[:,end])]

    @testset "Optimization with ParticleDensityGradientOverlap" begin
        gop = ParticleDensityGradientOverlap((TestTwoParticleDensity,
            TestTwoParticleDensityGradient); testfunction = nothing, optimizationparameter)
        oops = OptimizationAction(gop; optimizationstep = 5, threshold = 1e-2)

        p = ProjectorMonteCarloProblem(h; n_replicas=3, step_actions=(oops,))
        res = solve(p)
        df = res.df
        @test (length(unique(df.optimizationparameter[1:5])),
                length(unique(df.optimizationparameter[1:10]))) == (2,3)
        @test res.df.optimizationparameter isa Vector{typeof(optimizationparameter)}
        @test res.df.gradient isa Vector{typeof(optimizationparameter)}
        # Coefficient vector overlaps test
        cvos = CoefficientVectorOverlaps()
        replica_strategy = AllOverlaps(4)
        p2 = ProjectorMonteCarloProblem(
            h;
            replica_strategy,
            step_actions=(oops,)
        )
        res2 = solve(p2)
        df2 = res2.df
    end
end
