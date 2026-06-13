using LinearAlgebra
using Rimu
using Test
using DataFrames

@testset "Interface basics" begin
    @test eltype(StyleUnknown{String}()) == String
    @test StochasticStyle(['a', 'b']) == StyleUnknown{Char}()

    vector = [1, 2, 3]
    deposit!(vector, 1, 1, 1 => 1)
    @test vector == [2, 2, 3]
    @test storage(vector) ≡ vector
    @test localpart(vector) ≡ vector

    zerovector!(vector)
    @test vector == [0, 0, 0]

    @test_throws ArgumentError Interfaces.dot_from_right(1, 2, 3)
end

@testset "DataFrame interfaces" begin
    @test_throws ArgumentError num_replicas(DataFrame())
    @test_throws ArgumentError num_spectral_states(DataFrame())
    @test_throws ArgumentError num_overlaps(DataFrame())
end

# using lomc! with a matrix was removed in Rimu.jl v0.12.0
@testset "lomc! with matrix" begin
    ham = [1 1 2 3 2;
           2 0 2 2 3;
           0 0 0 3 2;
           0 0 1 1 2;
           0 1 0 1 0]
    vector = ones(5)

    # rephrase with MatrixHamiltonian
    mh = MatrixHamiltonian(ham)
    sv = DVec(pairs(vector))
    post_step_strategy = ProjectedEnergy(mh, sv)

    # solve with new API
    p = ProjectorMonteCarloProblem(mh; start_at=sv, last_step=10_000, post_step_strategy)
    sm = solve(p)
    df = DataFrame(sm)

    eigs = eigen(ham)

    @test df.shift[end] ≈ eigs.values[1] rtol=0.01
    @test df.hproj[end] / df.vproj[end] ≈ eigs.values[1] rtol=0.01
    @test normalize(state_vectors(sm)[1]) ≈ DVec(pairs(eigs.vectors[:, 1])) rtol = 0.01
end

@testset "apply_operator!" begin
    ham = [1 1 2 3 2;
           2 0 2 2 3;
           0 0 0 3 2;
           0 0 1 1 2;
           0 1 0 1 0]
    vector = ones(5)

    mh = MatrixHamiltonian(ham)
    sv = DVec(pairs(vector))
    wm = working_memory(sv)

    stat_names, stats, wm, target = apply_operator!(wm, zerovector(sv), sv, mh)
    @test target == DVec(pairs(ham * vector))

    mpi_seed!(123)
    h = HubbardReal1D(BoseFS(3, 1, 2))
    basis = build_basis(h)
    style = IsDeterministic(StochasticStyles.ThresholdCompression())

    # apply_operator! with DVec
    dv = DVec([basis[i] => 0.1*rand() for i in 1:length(basis)]; style)
    wm = working_memory(dv)
    # turn off compression
    stat_names, stats, wm, target3 = apply_operator!(NoCompression(), wm, zerovector(dv), dv, h, 1)
    @test target3 == DVec(pairs(h * dv))
    # turn on compression
    stat_names, stats, wm, target2 = apply_operator!(wm, zerovector(dv), dv, h)
    @test length(target2) < length(target3)

    # apply_operator! with PDVec
    pdv = PDVec([basis[i] => 0.1*rand() for i in 1:length(basis)]; style)
    wm = working_memory(pdv)
    # turn off compression
    stat_names, stats, wm, target3 = apply_operator!(NoCompression(), wm, zerovector(pdv), pdv, h, 1)
    @test target3 == PDVec(pairs(h * pdv))

    # turn on compression
    stat_names, stats, wm, target2 = apply_operator!(wm, zerovector(pdv), pdv, h)
    @test length(target2) < length(target3)
end
