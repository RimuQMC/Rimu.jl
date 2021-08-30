using DataFrames
using Rimu
using LinearAlgebra
using SafeTestsets
using Statistics
using Suppressor
using Test
using Rimu.StatsTools, Rimu.RimuIO

# assuming VERSION ≥ v"1.6"
# the following is needed because random numbers of collections are computed
# differently after version 1.6, and thus the results of many tests change
# for Golden Master Testing (@https://en.wikipedia.org/wiki/Characterization_test)
@assert VERSION ≥ v"1.6"

@safetestset "Interfaces" begin
    include("Interfaces.jl")
end

@safetestset "StatsTools" begin
    include("StatsTools.jl")
end

@safetestset "BitStringAddresses" begin
    include("BitStringAddresses.jl")
end

@safetestset "DictVectors.jl" begin
    include("DictVectors.jl")
end

using Rimu.ConsistentRNG
@testset "ConsistentRNG.jl" begin
    seedCRNG!(127) # uses `RandomNumbers.Xorshifts.Xoshiro256StarStar()`
    @test cRand(UInt128) == 0x50f0f296b239b257a8c2ac2f11d6d2cb

    @test rand(ConsistentRNG.CRNGs[][1],UInt128) == 0xba97314c00e092e448993a2bef41d28d
    # Only looks at first element of the `NTuple`. This should be reproducible
    # regardless of `numthreads()`.
    @test rand(trng(),UInt16) == 0x4e65
    @test rand(newChildRNG(),UInt16) == 0x03aa
    @test ConsistentRNG.check_crng_independence(0) == Threads.nthreads()
end

@testset "Hamiltonians.jl" begin
    include("Hamiltonians.jl")
end

@safetestset "lomc!" begin
    include("lomc.jl")
end

@testset "MemoryStrategy" begin
    # Define the initial Fock state with n particles and m modes
    n = m = 9
    add = near_uniform(BoseFS{n,m})
    H = HubbardReal1D(add; u = 6.0, t = 1.0)
    dv = DVec(add => 1; style=IsStochasticWithThreshold(1.0))
    s_strat = DoubleLogUpdate(targetwalkers=100)

    @testset "NoMemory" begin
        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=NoMemory(), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2907 atol=1
    end

    @testset "DeltaMemory" begin
        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=DeltaMemory(1), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2907 atol=1

        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=DeltaMemory(10), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2317 atol=1
    end

    @testset "DeltaMemory2" begin
        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=Rimu.DeltaMemory2(1), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2907 atol=1

        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=Rimu.DeltaMemory2(10), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2646 atol=1
    end

    @testset "ShiftMemory" begin
        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=ShiftMemory(1), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2907 atol=1

        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=ShiftMemory(10), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 1719 atol=1
    end
end

@testset "IsDeterministic with Vector" begin
    ham = HubbardReal1D(BoseFS((1, 1, 1, 1)))
    dim = dimension(ham)
    sm, basis = Rimu.Hamiltonians.build_sparse_matrix_from_LO(ham, starting_address(ham))
    @test dim == length(basis)
    # run lomc! in deterministic mode with Matrix and Vector
    a = lomc!(sm, ones(dim); threading=true).df # no actual threading is done, though
    b = lomc!(sm, ones(dim); threading=false).df
    @test a.shift ≈ b.shift
    # run lomc! in deterministic mode with Hamiltonian and DVec
    v = DVec(k=>1.0 for k in basis) # corresponds to `ones(dim)`
    c = lomc!(ham, v).df
    @test a.shift ≈ c.shift
end

@testset "helpers" begin
    v = [1,2,3]
    @test walkernumber(v) == norm(v,1)
    dvc= DVec(:a=>2-5im)
    @test StochasticStyle(dvc) isa DictVectors.IsStochastic2Pop
    @test walkernumber(dvc) == 2.0 + 5.0im
    Rimu.purge_negative_walkers!(dvc)
    @test walkernumber(dvc) == 2.0 + 0.0im
    dvi= DVec(:a=>Complex{Int32}(2-5im))
    @test StochasticStyle(dvi) isa DictVectors.IsStochastic2Pop
    dvr = DVec(i => cRandn() for i in 1:100; capacity = 100)
    @test walkernumber(dvr) ≈ norm(dvr,1)
end

using Rimu.Blocking
@testset "Blocking.jl" begin
    n=10
    a = rand(n)
    m = mean(a)
    @test m == sum(a)/n
    myvar(a,m) = sum((a .- m).^2)/n
    @test var(a) == sum((a .- m).^2)/(n-1)
    @test var(a, corrected=false) == sum((a .- m).^2)/n == myvar(a,m)
    @test var(a, corrected=false) == var(a, corrected=false, mean = m)
    # @benchmark myvar($a, $m)
    # @benchmark var($a, corrected=false, mean = $m)
    # evaluating the above shows that the library function is faster and avoids
    # memory allocations completely

    # test se
    a = collect(1:10)
    @test Rimu.Blocking.se(a) ≈ 0.9574271077563381
    @test Rimu.Blocking.se(a;corrected=false) ≈ 0.9082951062292475
    # test autocovariance
    @test autocovariance(a,1) ≈ 6.416666666666667
    @test autocovariance(a,1;corrected=false) ≈ 5.775
    # test covariance
    b = collect(2:11)
    @test covariance(a,b) ≈ 9.166666666666666
    @test covariance(a,b;corrected=false) ≈ 8.25
    c = collect(2:20) # should be truncated
    @test covariance(a,b) == covariance(a,c)
    @test covariance(a,b;corrected=false) == covariance(a,c;corrected=false)

    # Define the initial Fock state with n particles and m modes
    n = m = 9
    aIni = near_uniform(BoseFS{n,m})
    ham = HubbardReal1D(aIni; u = 6.0, t = 1.0)
    pa = RunTillLastStep(laststep = 1000)

    # standard fciqmc
    s = DoubleLogUpdate(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2))
    StochasticStyle(svec)
    vs = copy(svec)
    post_step = ProjectedEnergy(ham, svec)
    τ_strat = ConstantTimeStep()

    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    # @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @time rdfs = lomc!(
        ham, vs; params = pa, s_strat = s, post_step, τ_strat, wm = similar(vs),
    ).df
    r = autoblock(rdfs, start=101)
    @test r.s̄ ≈ -4.78 atol=0.1
    @test r.σs ≈ 0.27 atol=0.1
    @test r.ē ≈ -5.81 atol=0.1
    @test r.σe ≈ 0.39 atol=0.1
    @test r.k == 6

    g = growthWitness(rdfs, b=50)
    # @test sum(g) ≈ -5725.3936298329545
    @test length(g) == nrow(rdfs)
    g = growthWitness(rdfs, b=50, pad = :false)
    @test length(g) == nrow(rdfs) - 50
    @test_throws AssertionError growthWitness(rdfs.norm, rdfs.shift[1:end-1],rdfs.dτ[1])
end

@testset "RimuIO" begin
    @testset "save_df, load_df" begin
        file = joinpath(@__DIR__, "tmp.arrow")
        rm(file; force=true)

        df = DataFrame(a=[1, 2, 3], b=Complex{Float64}[1, 2, 3+im], d=rand(Complex{Int}, 3))
        RimuIO.save_df(file, df)
        df2 = RimuIO.load_df(file)
        @test df == df2

        rm(file)
    end
    @testset "save_dvec, load_dvec" begin
        file1 = joinpath(@__DIR__, "tmp1.bson")
        file2 = joinpath(@__DIR__, "tmp2.bson")
        rm(file1; force=true)
        rm(file2; force=true)

        add = BoseFS2C((1,1,0,1), (1,1,0,0))
        dv = InitiatorDVec(add => 1.0, style=IsDynamicSemistochastic(abs_threshold=3.5))
        H = BoseHubbardMom1D2C(add)

        _, state = lomc!(H, dv; replica=NoStats(2))
        RimuIO.save_dvec(file1, state.replicas[1].v)
        RimuIO.save_dvec(file2, state.replicas[2].v)

        dv1 = RimuIO.load_dvec(file1)
        dv2 = RimuIO.load_dvec(file2)

        @test dv1 == state.replicas[1].v
        @test typeof(dv2) == typeof(state.replicas[1].v)
        @test StochasticStyle(dv1) == StochasticStyle(state.replicas[1].v)
        @test storage(dv2) == storage(state.replicas[2].v)

        rm(file1; force=true)
        rm(file2; force=true)
    end
end

using Rimu.EmbarrassinglyDistributed
@testset "EmbarrassinglyDistributed" begin
    add = BoseFS((1,1,0,1))
    v = DVec(add => 2; capacity = 200)
    ham = HubbardReal1D(add, u=4.0)
    @test setup_workers(4) == 4 # add workers and load code (Rimu and its modules)
    seedCRNGs_workers!(127)     # seed rgns on workers deterministically
    nt = d_lomc!(ham, v; eqsteps = 1_000, laststep = 21_000) # perform parallel lomc!
    @test [size(df)[1] for df in nt.dfs] == [6000, 6000, 6000, 6000]
    ntc = combine_dfs(nt) # combine results into one DataFrame
    @test size(ntc.df)[1] == 20997
    energies = autoblock(ntc) # perform `autoblock()` discarding `eqsteps` time steps
    # in a single line:
    # energies = d_lomc!(ham, v; eqsteps = 1_000, laststep = 21_000) |> combine_dfs |> autoblock
    @test ismissing(energies.ē) && ismissing(energies.σe)
    # golden master test on results because qmc evolution is deterministic
    @test energies.s̄ ≈ -4.1 atol=0.1
    @test energies.σs ≈ 0.006 atol=1e-3
end

@testset "BoseFS2C" begin
    bfs2c = BoseFS2C(BoseFS((1,2,0,4)),BoseFS((4,0,3,1)))
    @test typeof(bfs2c) <: BoseFS2C{7,8,4}
    @test num_occupied_modes(bfs2c.bsa) == 3
    @test num_occupied_modes(bfs2c.bsb) == 3
    @test onr(bfs2c.bsa) == [1,2,0,4]
    @test onr(bfs2c.bsb) == [4,0,3,1]
    @test Hamiltonians.bose_hubbard_2c_interaction(bfs2c) == 8 # n_a*n_b over all sites
end

@testset "TwoComponentBosonicHamiltonian" begin
    aIni2cReal = BoseFS2C(BoseFS((1,1,1,1)),BoseFS((1,1,1,1))) # real space two-component
    Ĥ2cReal = BoseHubbardReal1D2C(aIni2cReal; ua = 6.0, ub = 6.0, ta = 1.0, tb = 1.0, v= 6.0)
    hamA = HubbardReal1D(BoseFS((1,1,1,1)); u=6.0, t=1.0)
    hamB = HubbardReal1D(BoseFS((1,1,1,1)); u=6.0)
    @test hamA == Ĥ2cReal.ha
    @test hamB == Ĥ2cReal.hb
    @test num_offdiagonals(Ĥ2cReal,aIni2cReal) == 16
    @test num_offdiagonals(Ĥ2cReal,aIni2cReal) == num_offdiagonals(Ĥ2cReal.ha,aIni2cReal.bsa)+num_offdiagonals(Ĥ2cReal.hb,aIni2cReal.bsb)
    @test dimension(Ĥ2cReal) == 1225
    @test dimension(Float64, Ĥ2cReal) == 1225.0

    hp2c = offdiagonals(Ĥ2cReal,aIni2cReal)
    @test length(hp2c) == 16
    @test hp2c[1][1] == BoseFS2C(BoseFS((0,2,1,1)), BoseFS((1,1,1,1)))
    @test hp2c[1][2] ≈ -1.4142135623730951
    @test diagonal_element(Ĥ2cReal,aIni2cReal) ≈ 24.0 # from the V term

    aIni2cMom = BoseFS2C(BoseFS((0,4,0,0)),BoseFS((0,4,0,0))) # momentum space two-component
    Ĥ2cMom = BoseHubbardMom1D2C(aIni2cMom; ua = 6.0, ub = 6.0, ta = 1.0, tb = 1.0, v= 6.0)
    @test num_offdiagonals(Ĥ2cMom,aIni2cMom) == 9
    @test dimension(Ĥ2cMom) == 1225
    @test dimension(Float64, Ĥ2cMom) == 1225.0

    hp2cMom = offdiagonals(Ĥ2cMom,aIni2cMom)
    @test length(hp2cMom) == 9
    @test hp2cMom[1][1] == BoseFS2C(BoseFS((1,2,1,0)), BoseFS((0,4,0,0)))
    @test hp2cMom[1][2] ≈ 2.598076211353316

    smat2cReal, adds2cReal = Hamiltonians.build_sparse_matrix_from_LO(Ĥ2cReal,aIni2cReal)
    eig2cReal = eigen(Matrix(smat2cReal))
    smat2cMom, adds2cMom = Hamiltonians.build_sparse_matrix_from_LO(Ĥ2cMom,aIni2cMom)
    eig2cMom = eigen(Matrix(smat2cMom))
    @test eig2cReal.values[1] ≈ eig2cMom.values[1]
end

@safetestset "KrylovKit" begin
    include("KrylovKit.jl")
end
@safetestset "RMPI" begin
    include("RMPI.jl")
end

@testset "deprecated" begin
    @test @capture_err(EveryTimeStep()) ≠ ""
    @test @capture_err(EveryKthStep()) ≠ ""
    @suppress_err begin
        @test EveryTimeStep().k == 1
        @test EveryTimeStep().writeinfo == false

        @test EveryKthStep(k=1000).k == 1000
        @test EveryKthStep().k == 10
        @test EveryKthStep().writeinfo == false
    end
end

# Note: This last test is set up to work on Pipelines, within a Docker
# container, where everything runs as root. It should also work locally,
# where typically mpi is not (to be) run as root.
@testset "MPI" begin
    # read name of mpi executable from environment variable if defined
    # necessary for allow-run-as root workaround for Pipelines
    mpiexec = haskey(ENV, "JULIA_MPIEXEC") ? ENV["JULIA_MPIEXEC"] : "mpirun"
    is_local = !haskey(ENV, "CI")

    # use user installed julia executable if available
    if isfile(joinpath(homedir(),"bin/julia"))
        juliaexec = joinpath(homedir(),"bin/julia")
    else
        juliaexec = "julia"
    end
    run(`which $mpiexec`)

    if is_local
        rr = run(`$mpiexec -np 2 $juliaexec -t 1 mpi_runtests.jl`)
        @test rr.exitcode == 0
    else
        @info "not testing MPI on CI"
    end
end

@testset "example script" begin
    include("../scripts/BHM-example.jl")
    dfr = load_df("fciqmcdata.arrow")
    qmcdata = last(dfr,steps_measure)
    (qmcShift,qmcShiftErr) = mean_and_se(qmcdata.shift)
    @test qmcShift ≈ -4.11255936332424

    # clean up
    rm("fciqmcdata.arrow", force=true)
end

@safetestset "doctests" begin
    include("doctests.jl")
end
