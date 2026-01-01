using Test
using Rimu
using Rimu: GlobalStepAction, OperatorOverlaps, StrictPairIter, SingleState,
                SpectralState

@testset "StrictPairIter" begin
    spi = StrictPairIter(4)
    pairs = collect(spi)
    @test pairs == [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    @test length(spi) == 6
end

@testset "OperatorOverlaps" begin
    # two simple sparse vectors (DVec) with overlapping keys
    v1 = DVec(:a => 1.0, :b => 2.0)
    v2 = DVec(:a => 3.0, :b => 4.0)

    # working memories / previous vectors
    wm1 = working_memory(v1)
    wm2 = working_memory(v2)

    # single states (minimal placeholders for hamiltonian and algorithm)
    s1 = SingleState(nothing, nothing, v1, zerovector(v1), wm1, 0.0, "_r1")
    s2 = SingleState(nothing, nothing, v2, zerovector(v2), wm2, 0.0, "_r2")

    # spectral states (single spectral state per replica)
    ss1 = SpectralState((s1,), GramSchmidt(1), "")
    ss2 = SpectralState((s2,), GramSchmidt(1), "")

    # build a minimal ReplicaState holding two replicas with one spectral state each
    spectral_states = (ss1, ss2)
    state = ReplicaState(
        spectral_states,
        Ref(10),
        Ref(1),
        SimulationPlan(),
        ReportDFAndInfo(),
        (), # post_step_strategy
        NoStats(2),
        ()  # global_step_actions
    )

    # use the identity operator so the overlap is the plain inner product
    op = IdentityOperator()
    oo = OperatorOverlaps(op; name = :operator_overlaps)

    res = oo(state)
    @test haskey(res, :operator_overlaps)

    # expected overlap: 1*3 + 2*4 = 11
    @test res.operator_overlaps == [11.0]
end
