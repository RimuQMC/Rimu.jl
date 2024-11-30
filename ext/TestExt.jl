module TestExt

using Test: Test, @test, @testset, @test_throws
using Rimu: Rimu, DVec, Interfaces, LOStructure, IsHermitian, IsDiagonal, AdjointKnown,
    Hamiltonians, num_offdiagonals
using VectorInterface: scalartype
using LinearAlgebra: dot

@show "TestExt.jl: Loading TestExt module"


function Rimu.Hamiltonians.test_observable_interface(obs, addr)
    @testset "Observable interface: $(nameof(typeof(obs)))" begin
        @testset "three way dot" begin # this works with vector valued operators
            v = DVec(addr => scalartype(obs)(2))
            @test dot(v, obs, v) isa eltype(obs)
            @test dot(v, obs, v) ≈ Interfaces.dot_from_right(v, obs, v)
        end
        @testset "LOStructure" begin
            @test LOStructure(obs) isa LOStructure
            if LOStructure(obs) isa IsHermitian
                @test obs' === obs
            elseif LOStructure(obs) isa IsDiagonal
                @test num_offdiagonals(obs, addr) == 0
                if scalartype(obs) <: Real
                    @test obs' === obs
                end
            elseif LOStructure(obs) isa AdjointKnown
                @test begin
                    obs'
                    true
                end # make sure no error is thrown
            else
                @test_throws ArgumentError obs'
            end
        end
    end
end

end
