"""
The module `Rimu.InterfaceTests` provides functions to test compliance with the
[`AbstractObservable`](@ref), [`AbstractOperator`](@ref), and [`AbstractHamiltonian`](@ref)
interfaces. Load the module with `using Rimu.InterfaceTests`.

The module exports the following functions:
- [`test_observable_interface`](@ref Rimu.InterfaceTests.test_observable_interface)
- [`test_operator_interface`](@ref Rimu.InterfaceTests.test_operator_interface)
- [`test_hamiltonian_interface`](@ref Rimu.InterfaceTests.test_hamiltonian_interface)
- [`test_hamiltonian_structure`](@ref Rimu.InterfaceTests.test_hamiltonian_structure)
"""
module InterfaceTests

using Test: Test, @test, @testset, @test_throws
using Rimu: Rimu, DVec, Interfaces, LOStructure, IsHermitian, IsDiagonal, AdjointKnown,
    Hamiltonians, num_offdiagonals, allows_address_type, offdiagonals, random_offdiagonal,
    diagonal_element, dimension, dot_from_right, IsDeterministic, starting_address, PDVec,
    sparse, scale!, scalartype
using Rimu.Hamiltonians: AbstractHamiltonian, AbstractOperator, AbstractObservable,
    AbstractOffdiagonals
using LinearAlgebra: dot, mul!, isdiag, ishermitian

export test_observable_interface, test_operator_interface, test_hamiltonian_interface,
    test_hamiltonian_structure

"""
    test_observable_interface(obs, addr)

This function tests compliance with the [`AbstractObservable`](@ref) interface for an
observable `obs` at address `addr` (typically
[`<: AbstractFockAddress`](@ref Rimu.BitStringAddresses.AbstractFockAddress)) by checking
that all required methods are defined.

The following properties are tested:
- `dot(v, obs, v)` returns a value of the same type as the `eltype` of the observable
- `LOStructure` is set consistently

### Example
```julia-doctest
julia> using Rimu.InterfaceTests

julia> test_observable_interface(ReducedDensityMatrix(2), FermiFS(1,0,1,1));
Test Summary:                              | Pass  Total  Time
Observable interface: ReducedDensityMatrix |    4      4  0.0s
```

See also [`AbstractObservable`](@ref), [`test_operator_interface`](@ref),
[`test_hamiltonian_interface`](@ref).
"""
function test_observable_interface(obs, addr)
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

"""
    test_operator_interface(op, addr; test_spawning=true)

This function tests compliance with the [`AbstractOperator`](@ref) interface for an operator
`op` at address `addr` (typically
[`<: AbstractFockAddress`](@ref Rimu.BitStringAddresses.AbstractFockAddress)) by
checking that all required methods are defined.

If `test_spawning` is `true`, tests are performed that require `offdiagonals` to return an
`Hamiltonians.AbstractOffDiagonals`, which is a prerequisite for using the `spawn!`
function. Otherwise, the spawning tests are skipped.

The following properties are tested:
- `diagonal_element` returns a value of the same type as the `eltype` of the operator
- `offdiagonals` behaves like an `AbstractVector`
- `num_offdiagonals` returns the correct number of offdiagonals
- `random_offdiagonal` returns a tuple with the correct types
- `mul!` and `dot` work as expected
- `dimension` returns a consistent value
- the [`AbstractObservable`](@ref) interface is tested

### Example
```julia-doctest
julia> using Rimu.InterfaceTests

julia> test_operator_interface(SuperfluidCorrelator(3), BoseFS(1, 2, 3, 1));
Test Summary:                              | Pass  Total  Time
Observable interface: SuperfluidCorrelator |    4      4  0.0s
Test Summary:       | Pass  Total  Time
allows_address_type |    1      1  0.0s
Test Summary:                            | Pass  Total  Time
Operator interface: SuperfluidCorrelator |    9      9  0.0s
```

See also [`AbstractOperator`](@ref), [`test_observable_interface`](@ref),
[`test_hamiltonian_interface`](@ref).
"""
function test_operator_interface(op, addr; test_spawning=true)
    test_observable_interface(op, addr)

    @testset "allows_address_type" begin
        @test allows_address_type(op, addr)
    end
    @testset "Operator interface: $(nameof(typeof(op)))" begin
        @testset "diagonal_element" begin
            @test diagonal_element(op, addr) isa eltype(op)
            @test eltype(diagonal_element(op, addr)) == scalartype(op)
        end
        @testset "offdiagonals" begin
            # `get_offdiagonal` is not mandatory and thus not tested
            ods = offdiagonals(op, addr)
            vec_ods = collect(ods)
            eltype(vec_ods) == Tuple{typeof(addr),eltype(op)} == eltype(ods)
            @test length(vec_ods) ≤ num_offdiagonals(op, addr)
        end
        if test_spawning
            @testset "spawning" begin
                ods = offdiagonals(op, addr)
                @test ods isa AbstractOffdiagonals{typeof(addr),eltype(op)}
                @test ods isa AbstractVector
                @test size(ods) == (num_offdiagonals(op, addr),)
                if length(ods) > 0
                    @test random_offdiagonal(op, addr) isa Tuple{typeof(addr),<:Real,eltype(op)}
                end
            end
        end
        @testset "mul!" begin # this works with vector valued operators
            v = DVec(addr => scalartype(op)(2))
            w = empty(v, eltype(op); style=IsDeterministic{scalartype(op)}())
            mul!(w, op, v) # operator vector product
            @test dot(v, op, v) ≈ Interfaces.dot_from_right(v, op, v) ≈ dot(v, w)
        end
        @testset "dimension" begin
            @test dimension(addr) ≥ dimension(op, addr)
        end
    end
end

"""
    test_hamiltonian_interface(h, addr=starting_address(h); test_spawning=true)

The main purpose of this test function is to check that all required methods of the
[`AbstractHamiltonian`](@ref) interface are defined and work as expected.

Set `test_spawning=false` to skip tests that require [`offdiagonals`](@ref) to return an
`AbstractVector`.

This function also tests the following properties of the Hamiltonian:
- `dimension(h) ≥ dimension(h, addr)`
- `scalartype(h) === eltype(h)`
- Hamiltonian action on a vector <: `AbstractDVec`
- `starting_address` returns an [`allows_address_type`](@ref) address
- `LOStructure` is one of `IsDiagonal`, `IsHermitian`, `AdjointKnown`
- the [`AbstractOperator`](@ref) interface is tested
- the [`AbstractObservable`](@ref) interface is tested

### Example
```julia-doctest
julia> using Rimu.InterfaceTests

julia> test_hamiltonian_interface(HubbardRealSpace(BoseFS(2,0,3,1)));
Test Summary:                          | Pass  Total  Time
Observable interface: HubbardRealSpace |    4      4  0.0s
Test Summary:       | Pass  Total  Time
allows_address_type |    1      1  0.0s
Test Summary:                        | Pass  Total  Time
Operator interface: HubbardRealSpace |    9      9  0.0s
Test Summary:       | Pass  Total  Time
allows_address_type |    1      1  0.0s
Test Summary:                                 | Pass  Total  Time
Hamiltonians-only tests with HubbardRealSpace |    6      6  0.0s
```

See also [`test_operator_interface`](@ref), [`test_observable_interface`](@ref).
"""
function test_hamiltonian_interface(h, addr=starting_address(h); test_spawning=true)
    test_operator_interface(h, addr; test_spawning)

    @testset "allows_address_type" begin
        @test allows_address_type(h, addr)
    end
    @testset "Hamiltonians-only tests with $(nameof(typeof(h)))" begin
        # starting_address is specific to Hamiltonians
        @test allows_address_type(h, starting_address(h))

        @test dimension(h) ≥ dimension(h, addr)

        # Hamiltonians can only have scalar eltype
        @test scalartype(h) === eltype(h)

        # Hamiltonian action on a vector
        v = DVec(addr => scalartype(h)(2))
        v1 = similar(v)
        mul!(v1, h, v)
        v2 = h * v
        v3 = similar(v)
        h(v3, v)
        v4 = h(v)
        @test v1 == v2 == v3 == v4
        v5 = DVec(addr => diagonal_element(h, addr))
        for (addr, val) in offdiagonals(h, addr)
            v5[addr] += val
        end
        scale!(v5, scalartype(h)(2))
        v5[addr] = v5[addr] # remove possible 0.0 from the diagonal
        @test v5 == v1

        if test_spawning && scalartype(h) <: Real
            # applying an operator on a PDVec uses spawn!, which requires
            # offdiagonals to be an AbstractVector
            # currently this only works for real operators as spawn! is not
            # implemented for complex operators
            pv = PDVec(addr => scalartype(h)(2))
            pv1 = h(pv)
            @test dot(pv1, h, pv) ≈ Interfaces.dot_from_right(pv1, h, pv) ≈ dot(v1, v1)
        end

    end
end

"""
    test_hamiltonian_structure(h::AbstractHamiltonian; sizelim=20)

Test the `LOStructure` of a small Hamiltonian `h` by instantiating it as a sparse matrix and
checking whether the structure of the matrix is constistent with the result of
`LOStructure(h)` and the `eltype` is consistent with `eltype(h)`.

This function is intended to be used in automated test for small Hamiltonians where
instantiating the matrix is quick. A warning will print if the dimension of the Hamiltonian
is larger than `20`.

### Example
```julia-doctest
julia> using Rimu.InterfaceTests

julia> test_hamiltonian_structure(HubbardRealSpace(BoseFS(2,0,1)));
Test Summary: | Pass  Total  Time
structure     |    4      4  0.0s
```
"""
function test_hamiltonian_structure(h::AbstractHamiltonian; sizelim=20)
    @testset "Hamiltonian structure" begin
        d = dimension(h)
        d > 20 && @warn "This function is intended for small Hamiltonians. The dimension is $d."
        m = sparse(h; sizelim)
        @test eltype(m) === eltype(h)
        if LOStructure(h) == IsDiagonal()
            @test isdiag(m)
        elseif LOStructure(h) == IsHermitian()
            @test h' == h
            @test h' === h
            @test ishermitian(m)
        elseif LOStructure(h) == AdjointKnown()
            @test m' == sparse(h')
        end
        if !ishermitian(m)
            @test LOStructure(h) != IsHermitian()
            if LOStructure(h) == IsDiagonal()
                @test !isreal(m)
                @test !(eltype(h) <: Real)
            end
        end
    end
end


end
