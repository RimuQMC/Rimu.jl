using DataFrames
using Rimu
using LinearAlgebra
using SafeTestsets
using StaticArrays
using Statistics
using Suppressor
using Logging, TerminalLoggers
using TOML
using Test
using Rimu.StatsTools, Rimu.RimuIO


# assuming VERSION ≥ v"1.6"
# the following is needed because random numbers of collections are computed
# differently after version 1.6, and thus the results of many tests change
# for Golden Master Testing (@https://en.wikipedia.org/wiki/Characterization_test)
@assert VERSION ≥ v"1.6"

@test Rimu.PACKAGE_VERSION == VersionNumber(TOML.parsefile(pkgdir(Rimu, "Project.toml"))["version"])

@safetestset "Interfaces" begin
    include("Interfaces.jl")
end

@safetestset "StatsTools" begin
    include("StatsTools.jl")
end

@safetestset "BitStringAddresses" begin
    include("BitStringAddresses.jl")
end

@safetestset "StochasticStyles" begin
    include("StochasticStyles.jl")
end

@safetestset "DictVectors" begin
    include("DictVectors.jl")
end

@testset "Hamiltonians" begin
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
        Random.seed!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=NoMemory(), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2195 atol=1
    end

    @testset "DeltaMemory" begin
        Random.seed!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=DeltaMemory(1), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2195 atol=1

        Random.seed!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=DeltaMemory(10), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2140 atol=1
    end

    @testset "DeltaMemory2" begin
        Random.seed!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=Rimu.DeltaMemory2(1), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2195 atol=1

        Random.seed!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=Rimu.DeltaMemory2(10), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 1913 atol=1
    end

    @testset "ShiftMemory" begin
        Random.seed!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=ShiftMemory(1), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2195 atol=1

        Random.seed!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=ShiftMemory(10), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 1872 atol=1
    end
end

@testset "helpers" begin
    @testset "walkernumber" begin
        v = [1,2,3]
        @test walkernumber(v) == norm(v,1)
        dvc = DVec(:a => 2-5im)
        @test StochasticStyle(dvc) isa StochasticStyles.IsStochastic2Pop
        @test walkernumber(dvc) == 2.0 + 5.0im
        Rimu.purge_negative_walkers!(dvc)
        @test walkernumber(dvc) == 2.0 + 0.0im
        dvi= DVec(:a=>Complex{Int32}(2-5im))
        @test StochasticStyle(dvi) isa StochasticStyles.IsStochastic2Pop
        dvr = DVec(i => randn() for i in 1:100; capacity = 100)
        @test walkernumber(dvr) ≈ norm(dvr,1)
    end
    @testset "MultiScalar" begin
        a = Rimu.MultiScalar(1, 1.0, SVector(1))
        @test a[1] ≡ 1
        @test a[2] ≡ 1.0
        @test a[3] ≡ SVector(1)
        @test length(a) == 3
        @test collect(a) == [1, 1.0, SVector(1)]
        b = Rimu.MultiScalar(SVector(2, 3.0, SVector(4)))
        for op in (+, min, max)
            c = op(a, b)
            @test op(a[1], b[1]) == c[1]
            @test op(a[2], b[2]) == c[2]
            @test op(a[2], b[2]) == c[2]
        end
        @test_throws MethodError a + Rimu.MultiScalar(1, 1, 1)
    end
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
        # BSON is currently broken on 1.8
        if VERSION ≤ v"1.7"
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

@testset "Logging" begin
    default_logger()
    l = Base.global_logger()
    @test l isa Logging.ConsoleLogger
    sl = smart_logger()
    if isdefined(Main, :IJulia) && Main.IJulia.inited
        @test sl isa ConsoleProgressMonitor.ProgressLogRouter
        @info "Jupyter progress bar" sl
    elseif isa(stderr, Base.TTY) && (get(ENV, "CI", nothing) ≠ true)
        @test sl isa TerminalLoggers.TerminalLogger
        @info "Terminal progress bar" sl
    else
        @test sl isa Logging.ConsoleLogger
        @info "No progress bar" sl
    end
    @test default_logger() isa Logging.ConsoleLogger
end

@safetestset "doctests" begin
    include("doctests.jl")
end

# Note: Running Rimu with several MPI ranks is tested seperately on GitHub CI and not here.
