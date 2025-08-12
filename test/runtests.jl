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
    @test check_no_implicit_imports(
        Rimu; skip=(Rimu, Base, Core, LinearAlgebra, VectorInterface)
    ) === nothing
    # If this test fails, make your import statements explicit.
    # For example, replace `using Foo` with `using Foo: bar, baz`.
end

@safetestset "Helpers" begin
    include("helpers.jl")
end

@safetestset "Interfaces" begin
    include("Interfaces.jl")
end

@safetestset "excited states" begin
    include("excited_states_tests.jl")
end

@safetestset "StochasticStyles" begin
    include("StochasticStyles.jl")
end

@safetestset "projector_monte_carlo_problem" begin
    include("projector_monte_carlo_problem.jl")
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

@safetestset "RimuIO" begin
    include("RimuIO.jl")
end

@safetestset "StatsTools" begin
    include("StatsTools.jl")
end

@safetestset "BitStringAddresses" begin
    include("BitStringAddresses.jl")
end

# We test loading of extension packages here (and load them), so
# this test should run before other tests that use them (like "doctests").
@safetestset "ExactDiagonalization" begin
    include("ExactDiagonalization.jl")
end

@safetestset "doctests" begin
    include("doctests.jl")
end

@safetestset "DictVectors" begin
    include("DictVectors.jl")
end

@testset "Hamiltonians" begin
    include("Hamiltonians.jl")
end

@safetestset "KrylovKit" begin
    include("KrylovKit.jl")
end

@suppress_err @safetestset "lomc!" begin
    include("lomc.jl")
end

# Note: Running Rimu with several MPI ranks is tested seperately on GitHub CI and not here.
