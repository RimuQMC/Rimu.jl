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
using Rimu.StatsTools
using ExplicitImports: check_no_implicit_imports


@test Rimu.PACKAGE_VERSION == VersionNumber(TOML.parsefile(pkgdir(Rimu, "Project.toml"))["version"])

@safetestset "ExplicitImports" begin
    using Rimu
    using ExplicitImports
    # Check that no implicit imports are used in the Rimu module.
    # See https://ericphanson.github.io/ExplicitImports.jl/stable/
    @test check_no_implicit_imports(Rimu; skip=(Rimu, Base, Core, VectorInterface)) === nothing
    # If this test fails, make your import statements explicit.
    # For example, replace `using Foo` with `using Foo: bar, baz`.
end

@safetestset "doctests" begin
    include("doctests.jl")
end

@safetestset "excited states" begin
    include("excited_states_tests.jl")
end

@safetestset "Interfaces" begin
    include("Interfaces.jl")
end

@safetestset "ExactDiagonalization" begin
    include("ExactDiagonalization.jl")
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

@safetestset "projector_monte_carlo_problem" begin
    include("projector_monte_carlo_problem.jl")
end

# @safetestset "lomc!" begin
#     include("lomc.jl")
# end

@safetestset "RimuIO" begin
    include("RimuIO.jl")
end

@safetestset "StatsTools" begin
    include("StatsTools.jl")
end

using Rimu: replace_keys, delete_and_warn_if_present, clean_and_warn_if_others_present
@testset "helpers" begin
    @testset "walkernumber" begin
        v = [1,2,3]
        @test walkernumber(v) == norm(v,1)
        dvc = DVec(:a => 2-5im)
        @test StochasticStyle(dvc) isa StochasticStyles.IsStochastic2Pop
        @test walkernumber(dvc) == 2.0 + 5.0im
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

    @testset "keyword helpers" begin
        nt = (; a=1, b=2, c = 3, d = 4)
        nt2 = replace_keys(nt, (:a => :x, :b => :y, :u => :v))
        @test nt2 == (c=3, d=4, x=1, y=2)
        nt3 = @test_logs((:warn, "The keyword(s) \"a\", \"b\" are unused and will be ignored."),
            delete_and_warn_if_present(nt, (:a, :b, :u)))
        @test nt3 == (; c = 3, d = 4)
        nt4 = @test_logs((:warn, "The keyword(s) \"c\", \"d\" are unused and will be ignored."),
            clean_and_warn_if_others_present(nt, (:a, :b, :u)))
        @test nt4 == (; a = 1, b = 2)
    end
end


@safetestset "KrylovKit" begin
    include("KrylovKit.jl")
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

# Note: Running Rimu with several MPI ranks is tested seperately on GitHub CI and not here.
