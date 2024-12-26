using Rimu
using Test, Suppressor
import Random

using Rimu: is_finalized
using Rimu.DictVectors: FrozenDVec
using OrderedCollections: freeze

@testset "ProjectorMonteCarloProblem" begin
    h = HubbardReal1D(BoseFS(1,3))
    p = ProjectorMonteCarloProblem(h; threading=true)
    @test p.hamiltonian == h

    @test Rimu.num_replicas(p) == 1
    @test startswith(sprint(show, p), "ProjectorMonteCarloProblem with 1 replica(s)")
    @test eval(Meta.parse(repr(p.simulation_plan))) == p.simulation_plan

    simulation = init(p)
    @test simulation.hamiltonian == h
    @test only(state_vectors(simulation)) isa PDVec
    sp = only(simulation.state).shift_parameters
    @test sp.shift == diagonal_element(h, starting_address(h))
    @test sp.pnorm == walkernumber(only(state_vectors(simulation)))
    @test sp.pnorm isa Float64
    @test p.max_length == 2 * p.algorithm.shift_strategy.target_walkers + 100

    ps = ProjectorMonteCarloProblem(h; initial_shift_parameters=sp, threading=false)
    @test ps.initial_shift_parameters === sp
    @test ps.start_at isa AbstractFockAddress
    sm = init(ps)
    @test only(state_vectors(sm)) isa DVec

    p = ProjectorMonteCarloProblem(h; n_replicas = 3, threading=false, initiator=Initiator())
    @test Rimu.num_replicas(p) == 3
    sm = init(p)
    @test Rimu.num_replicas(sm) == 3
    @test size(state_vectors(sm)) == (3, 1)
    @test only(state_vectors(sm.state.spectral_states[1])) == first(state_vectors(sm))
    dv = first(state_vectors(sm))
    @test dv isa InitiatorDVec
    @test collect(pairs(dv)) == [starting_address(h) => 10.0]
    sm2 = init(ProjectorMonteCarloProblem(h; start_at=dv, n_replicas=3))
    @test state_vectors(sm) == state_vectors(sm2)
    sv = state_vectors(sm2)
    @test sv[1] !== sv[2] !== sv[3] !== sv[1]
    sm3 = init(ProjectorMonteCarloProblem(h; start_at=dv), copy_vectors=false)
    @test state_vectors(sm3)[1] === dv
    p4 = ProjectorMonteCarloProblem(h; start_at=sv, n_replicas=3)
    sm4 = init(p4, copy_vectors=false)
    sv4 = state_vectors(sm4)
    @test sv4[1] === sv[1] && sv4[2] === sv[2] && sv4[3] === sv[3]

    dv = DVec(BoseFS(1, 3) => 1, BoseFS(2, 2) => 3)
    p = ProjectorMonteCarloProblem(h; start_at=freeze(dv), n_replicas=3)
    sm = init(p)
    @test state_vectors(sm)[1] == dv
    @test ProjectorMonteCarloProblem(h; shift=2).initial_shift_parameters.shift == 2

    # passing PDVec to ProjectorMonteCarloProblem
    dv = PDVec(starting_address(h)=>3; style=IsDynamicSemistochastic())
    p = ProjectorMonteCarloProblem(h; n_replicas=3, start_at=dv)
    sm = init(p)
    @test first(state_vectors(sm)) == dv
    @test first(state_vectors(sm)) !== dv
    @test first(sm.state).pv !== dv

    # copy_vectors = false
    dv1 = deepcopy(dv)
    dv2 = deepcopy(dv)
    p = ProjectorMonteCarloProblem(h; n_replicas=2, start_at = [dv1, dv2])
    sm = init(p; copy_vectors=false)
    @test state_vectors(sm)[1] === dv1
    @test state_vectors(sm)[2] === dv2
    @test_throws BoundsError sm.state.spectral_states[3]

    # complex Hamiltonian
    h = HubbardReal1D(BoseFS(1, 3); u=1.0 + 1.0im)
    @test scalartype(h) <: Complex
    @test_throws ArgumentError ProjectorMonteCarloProblem(h)
end

@testset "PMCSimulation" begin
    h = HubbardReal1D(BoseFS(1, 3))
    @testset "init" begin
        p = ProjectorMonteCarloProblem(
            h;
            shift=[1, 2],
            start_at=[BoseFS(1, 3), BoseFS(3, 1)],
            replica_strategy=AllOverlaps(2)
        )
        sm = init(p)
        @test sm.modified[] == false == sm.aborted[] == sm.success[]
        @test size(DataFrame(sm)) == (0, 0)
        @test sm.state[1].shift_parameters.shift ≡ 1.0
        @test sm.state[2].shift_parameters.shift ≡ 2.0
        @test state_vectors(sm.state)[1][BoseFS(1, 3)] == 10
        @test state_vectors(sm.state)[2][BoseFS(3, 1)] == 10
        @test startswith(sprint(show, sm.state), "2×1 Rimu.ReplicaState")
        @test startswith(sprint(show, sm.state[1]), "Rimu.SingleState")
        @test startswith(sprint(show, sm.state.spectral_states),
            "(1-element Rimu.SpectralState"
        )
        @test startswith(sprint(show, sm.state.spectral_states[1]),
            "1-element Rimu.SpectralState"
        )
    end

    @testset "Default DVec" begin
        address = BoseFS(2, 3)
        H = HubbardReal1D(address; u=20)
        sm = init(ProjectorMonteCarloProblem(H; threading=false))
        @test only(state_vectors(sm)) isa DVec
        @test StochasticStyle(only(state_vectors(sm))) isa IsDynamicSemistochastic

        sm = init(ProjectorMonteCarloProblem(H; threading=true))
        @test only(state_vectors(sm)) isa PDVec
        @test StochasticStyle(only(state_vectors(sm))) isa IsDynamicSemistochastic

        sm = init(ProjectorMonteCarloProblem(H; threading=false, initiator=true))
        @test only(state_vectors(sm)) isa InitiatorDVec
        @test StochasticStyle(only(state_vectors(sm))) isa IsDynamicSemistochastic
    end


    @testset "random seeds" begin
        p = ProjectorMonteCarloProblem(h) # generates random_seed
        @test p.random_seed isa UInt64

        @testset "default gives reproducible random numbers" begin
            sm = init(p) # seeds RNG
            r = rand(Int)
            init(p) # re-seeds RNG with same seed
            @test r == rand(Int)
        end
        @testset "but ProjectorMonteCarloProblem will re-seed" begin
            Random.seed!(127)
            p = ProjectorMonteCarloProblem(h)
            sm = init(p)
            r = rand(Int)
            Random.seed!(127)
            p = ProjectorMonteCarloProblem(h)
            sm = init(p)
            @test r ≠ rand(Int)
        end
        @testset "unless seeding in ProjectorMonteCarloProblem is disabled" begin
            Random.seed!(127)
            p = ProjectorMonteCarloProblem(h; random_seed=false)
            @test isnothing(p.random_seed)
            sm = init(p)
            r = rand(Int)
            Random.seed!(127)
            p = ProjectorMonteCarloProblem(h; random_seed=false)
            sm = init(p)
            @test r == rand(Int)
        end
        @testset "or if the seed is provided" begin
            p = ProjectorMonteCarloProblem(h; random_seed=123)
            @test p.random_seed == 123
            sm = init(p)
            r = rand(Int)
            p = ProjectorMonteCarloProblem(h; random_seed=123)
            sm = init(p)
            @test r == rand(Int)
        end
    end
end

using Rimu: num_replicas, num_spectral_states
@testset "step! and solve!" begin
    h = HubbardReal1D(BoseFS(1, 3))
    p = ProjectorMonteCarloProblem(h; threading=true, n_replicas=3)
    sm = init(p)
    @test sm.modified == false == sm.aborted == sm.success
    @test is_finalized(sm.report) == false
    @test startswith(sprint(show, sm), "PMCSimulation with 3 replica(s) and 1 spectral")

    @test step!(sm) isa Rimu.PMCSimulation
    @test sm.modified == true
    @test is_finalized(sm.report) == false
    @test size(DataFrame(sm))[1] == sm.state.step[]

    @test solve!(sm) isa Rimu.PMCSimulation
    @test sm.modified == true
    @test sm.success == true
    @test is_finalized(sm.report) == true
    @test size(DataFrame(sm))[1] == sm.state.step[]
    @test num_replicas(sm) == num_replicas(p) == num_replicas(sm.state)
    @test num_spectral_states(sm) == num_spectral_states(p) == num_spectral_states(sm.state)
    @test size(state_vectors(sm)) == (num_replicas(sm), num_spectral_states(sm))
    @test size(sm.state) == (num_replicas(sm), num_spectral_states(sm))
    @test sm.state[1, 1] === sm.state.spectral_states[1][1]
    @test length(sm.state.spectral_states[1]) == num_spectral_states(sm)

    df, state = sm # deconstruction for backward compatibility
    @test df == DataFrame(sm) == sm.df
    @test state == sm.state

    # Tables.jl interface
    @test Tables.istable(sm)
    @test Tables.columnaccess(sm)
    @test Tables.schema(sm) == Tables.schema(DataFrame(sm))
    @test map(NamedTuple, Tables.rows(sm)) == map(NamedTuple, Tables.rows(df))

    # continue simulation
    @test sm.state.step[] == 100
    solve!(sm; last_step=200)
    @test sm.state.step[] == 200
    @test sm.success == true == parse(Bool, (Rimu.get_metadata(sm.report, "success")))

    # time out
    p = ProjectorMonteCarloProblem(h; last_step=500, wall_time=1e-3)
    sm = init(p)
    @test_logs (:warn, Regex("(Wall time)")) solve!(sm)
    @test sm.success == false
    @test sm.aborted == true
    @test sm.message == "Wall time limit reached."

    sm2 = solve!(sm; wall_time=1.0)
    @test sm2 === sm
    @test sm.success == true
    @test sm.state.step[] == 500 == size(sm.df)[1]

    # continue simulation and change strategies
    sm3 = solve!(sm;
        last_step = 600,
        post_step_strategy = Rimu.Timer(),
        metadata = Dict(:test => 1)
    )
    @test sm3 === sm
    @test sm.success == true
    @test sm.state.step[] == 600
    @test size(sm.df)[1] == 100 # the report was emptied
    @test parse(Int, Rimu.get_metadata(sm.report, "test")) == 1
    @test Rimu.get_metadata(sm.report, "post_step_strategy") == "(Rimu.Timer(),)"

    # continue simulation and change replica strategy
    @test_throws ArgumentError solve!(sm; replica_strategy = NoStats(3))

    p = ProjectorMonteCarloProblem(h; last_step=100, replica_strategy=NoStats(3))
    sm = init(p; copy_vectors=false)
    sv = state_vectors(sm)
    @test sv[1] !== sv[2] !== sv[3] !== sv[1]

    @test solve!(sm) === sm
    @test solve!(sm; last_step=200, replica_strategy=AllOverlaps(3)) === sm
    @test size(sm.df)[1] == 100 # the report was emptied
    @test solve!(sm; last_step=300, reporting_strategy=ReportDFAndInfo()) === sm
    @test size(sm.df)[1] == 200 # the report was not emptied
end

@testset "Dead population" begin
    address = BoseFS{5,2}((2, 3))
    H = HubbardReal1D(address; u=20)
    dv = DVec(address => 10; style=IsStochasticInteger())

    # Only population is dead.
    p = ProjectorMonteCarloProblem(H; start_at=dv, last_step=100, shift=0.0, random_seed=7)
    sim = @suppress_err solve(p)
    @test sim.aborted == true
    @test sim.success == false
    @test sim.modified == true
    @test sim.message == "Aborted in step 5."
    @test size(sim.df, 1) < 100

    # population does not die with sensible default shift
    p = ProjectorMonteCarloProblem(H; start_at=dv, last_step=100, random_seed=7)
    sim = solve(p)
    @test sim.aborted == false
    @test sim.success == true
    @test sim.modified == true
    @test size(sim.df, 1) == 100

    # Populations in replicas are dead.
    p = ProjectorMonteCarloProblem(
        H;
        start_at=dv, n_replicas=3, last_step=100, shift=0.0, random_seed=7
    )
    sim = @suppress_err solve(p)
    @test sim.aborted == true
    @test sim.success == false
    @test sim.modified == true
    @test sim.message == "Aborted in step 3."
    @test size(sim.df, 1) < 100
end

@testset "max_length" begin
    # walker number blows up when time_step is too large
    h = HubbardReal1D(BoseFS(1, 3, 5, 2, 1))
    p = ProjectorMonteCarloProblem(h; time_step=0.1, target_walkers=100, random_seed=7)
    sm = init(p)
    @test size(DataFrame(sm)) == (0, 0)
    @test size(sm.df) == (0, 0)
    @test sm.state.max_length[] > 100 # default max_length
    @test sm.state.step[] == 0
    @test @suppress_err solve!(sm) === sm
    @test sm.modified == true
    @test sm.success == false
    @test sm.aborted == true
    @test sm.message == "Aborted in step 6."
    @test is_finalized(sm.report) == true
    @test @suppress_err step!(sm) === sm # no effect, aborted


    # runs fine with a smaller time_step
    p = ProjectorMonteCarloProblem(h; time_step=0.01, target_walkers=100, random_seed=7)
    sm = solve!(init(p))
    @test sm.success == true
    @test sm.aborted == false
    @test size(sm.df, 1) == 100
    @test @suppress_err step!(sm) === sm # no effect, already finalized
end

@testset "deprecated keyword arguments" begin
    h = HubbardReal1D(BoseFS(1, 3))
    p = @suppress_err ProjectorMonteCarloProblem(
        h; shift=1.0, targetwalkers=100, maxlength=200, walltime=23
    )
    @test p.algorithm.shift_strategy.target_walkers == 100
    @test p.max_length == 200
    @test p.simulation_plan.wall_time == 23
end
