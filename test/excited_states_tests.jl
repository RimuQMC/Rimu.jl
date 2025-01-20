using Rimu
using Test


@testset "excited state energies" begin
    ham = HubbardReal1D(BoseFS(1,1,1,1,1))
    pr = ExactDiagonalizationProblem(ham)
    result = solve(pr)
    vals = result.values
    vecs = result.vectors
    g2s = [rayleigh_quotient(G2RealCorrelator(0),vecs[i]) for i in 1:3]

    spectral_strategy = GramSchmidt(3)
    last_step=2000
    style = IsDeterministic()
    p = ProjectorMonteCarloProblem(ham; spectral_strategy, last_step, style)
    df = DataFrame(solve(p))
    energy1 = shift_estimator(df, shift="shift_r1s1", skip=1000)
    energy2 = shift_estimator(df, shift="shift_r1s2", skip=1000)
    energy3 = shift_estimator(df, shift="shift_r1s3", skip=1000)

    @test energy1.mean ≈ vals[1]
    @test energy2.mean ≈ vals[2]
    @test energy3.mean ≈ vals[3]

    n_replicas = 2
    p = ProjectorMonteCarloProblem(ham; spectral_strategy, last_step, style, n_replicas)
    df = DataFrame(solve(p))
    energy1 = shift_estimator(df, shift="shift_r1s1", skip=1000)
    energy2 = shift_estimator(df, shift="shift_r1s2", skip=1000)
    energy3 = shift_estimator(df, shift="shift_r1s3", skip=1000)
    energy4 = shift_estimator(df, shift="shift_r2s1", skip=1000)
    energy5 = shift_estimator(df, shift="shift_r2s2", skip=1000)
    energy6 = shift_estimator(df, shift="shift_r2s3", skip=1000)

    @test energy1.mean ≈ vals[1]
    @test energy2.mean ≈ vals[2]
    @test energy3.mean ≈ vals[3]
    @test energy4.mean ≈ vals[1]
    @test energy5.mean ≈ vals[2]
    @test energy6.mean ≈ vals[3]

    replica_strategy = AllOverlaps(n_replicas; operator=G2RealCorrelator(0), mixed_spectral_overlaps=true)
    p = ProjectorMonteCarloProblem(ham; spectral_strategy, last_step, style, replica_strategy)
    df = DataFrame(solve(p))
    for state in 1:3
        r = rayleigh_replica_estimator(df; spectral_state=state)
        @test r.f ≈ g2s[state] atol=0.01
    end
    num_overlaps = length(filter(startswith(r"r[0-9]+s[0-9]+_dot"), names(df)))
    @test num_overlaps == 15
end
