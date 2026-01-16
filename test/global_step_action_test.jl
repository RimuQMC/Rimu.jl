using Test
using Rimu
using Rimu: GlobalStepAction, OperatorOverlaps, StrictPairIter, SingleState,
    SpectralState, CoefficientVectorOverlaps, ParticleDensityGradientOverlap,
    OverlapwithOptimization
using StaticArrays : SVector

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
    p = ProjectorMonteCarloProblem(h; n_replicas=3, global_step_actions=(oops,))
    res = solve(p)
    @test res.df.test_overlaps isa Vector{Matrix{Float64}}
    # Coefficient vector overlaps test
    cvos = CoefficientVectorOverlaps()
    replica_strategy = AllOverlaps(4; operator=h)
    p2 = ProjectorMonteCarloProblem(
        h;
        replica_strategy,
        global_step_actions=(cvos,oops)
    )
    res2 = solve(p2)
    df2 = res2.df
    @test first.(df2.coefficient_vector_overlaps) ≈ df2.r1s1_dot_r2s1
    @test first.(df2.test_overlaps) ≈ df2.r1s1_Op1_r2s1
end

@testset "ParticleDensityGradientOverlaps" begin
    address = FermiFS(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)

    h = HubbardRealSpace(address; t=1, w=-1.0)
    ep = ExactDiagonalizationProblem(h)
    epsol = solve(ep)
    gs = epsol.vectors[1]
    M = num_modes(address)
    onerdmop = ReducedDensityMatrix(2)
    onerdm = dot(gs, onerdmop, gs)
    evs, evc = eigen(Hermitian(onerdm))   
    # Operator overlaps test
    parameter = [SVector{binomial(M,2),eltype(evs)}(evc[:,end])]

    @testset "ParticleDensityGradientOverlaps" begin
        oops = ParticleDensityGradientOverlap((TestTwoParticleDensity,
            TwoParticleDensityGradient); name=(:gradient_test_overlaps,
            :coefficient_vector_overlaps), test_vector_function = nothing, parameter)

        p = ProjectorMonteCarloProblem(h; n_replicas=3, global_step_actions=(oops,))
        res = solve(p)
        @test res.df.gradient_test_overlaps isa Vector{Matrix{eltype(parameter)}}
        @test res.df.coefficient_vector_overlaps isa Vector{Matrix{Float64}}
        # Coefficient vector overlaps test
        cvos = CoefficientVectorOverlaps()
        replica_strategy = AllOverlaps(4)
        p2 = ProjectorMonteCarloProblem(
            h;
            replica_strategy,
            global_step_actions=(oops,)
        )
        res2 = solve(p2)
        df2 = res2.df
        @test first.(df2.coefficient_vector_overlaps) ≈ df2.r1s1_dot_r2s1
    end

    @testset "OverlapwithOptimization " begin
        gop = ParticleDensityGradientOverlap((TestTwoParticleDensity,
            TwoParticleDensityGradient); name=(:gradient_test_overlaps,
            :coefficient_vector_overlaps), test_vector_function = nothing, parameter)
        oops = OverlapwithOptimization(gop; name = :parameter, step = 5, 
            threshold = 1e-2, method = Adam(0.1))

        p = ProjectorMonteCarloProblem(h; n_replicas=3, global_step_actions=(oops,))
        res = solve(p)
        df = res.df
        @test (length(unique(df.parameter[1:5])),
                length(unique(df.parameter[1:10]))) == (1,2)
        @test res.df.parameter isa Vector{typeof(parameter)}
        @test res.df.gradient_test_overlaps isa Vector{Matrix{eltype(parameter)}}
        @test res.df.coefficient_vector_overlaps isa Vector{Matrix{Float64}}
        # Coefficient vector overlaps test
        cvos = CoefficientVectorOverlaps()
        replica_strategy = AllOverlaps(4)
        p2 = ProjectorMonteCarloProblem(
            h;
            replica_strategy,
            global_step_actions=(oops,)
        )
        res2 = solve(p2)
        df2 = res2.df
        @test first.(df2.coefficient_vector_overlaps) ≈ df2.r1s1_dot_r2s1
    end
end
