###
### This file contains abstract types, interfaces and traits.
###
"""
    AbstractObservable{T}

Most permissive supertype for operators in the type hierarchy:

    AbstractHamiltonian{T} <: AbstractOperator{T} <: AbstractObservable{T}

`AbstractObservable` provides an interface for operators that can appear in a three-way dot
product [`dot(x, op, y)`](@ref LinearAlgebra.dot) with two vectors of type
[`AbstractDVec`](@ref). The result is a value of type `T`, which is also returned by the
[`eltype`](@ref) function. This may be a vector type associated with a scalar type returned
by the [`scalartype`](@ref) function.

The `AbstractObservable` type is useful for defining observables that can be calculated in
the context of a [`ProjectorMonteCarloProblem`](@ref) using
[`AllOverlaps`](@ref Main.Hamiltonians).

# Interface

Basic interface methods to implement:
- [`Interfaces.dot_from_right(x, op, y)`](@ref)
- [`allows_address_type(op, type)`](@ref)

Optional additional methods to implement:
- [`VectorInterface.scalartype(op)`](@ref): defaults to `eltype(eltype(op))`
- [`LOStructure(::Type{typeof(op)})`](@ref LOStructure): defaults to `AdjointUnknown`

See also [`AbstractOperator`](@ref), [`AbstractHamiltonian`](@ref), [`Interfaces`](@ref).
"""
abstract type AbstractObservable{T} end

"""
    eltype(op::AbstractObservable)
Return the type of the elements of the operator. This can be a vector value. For the
underlying scalar type use [`scalartype`](@ref).

Part of the [`AbstractObservable`](@ref) interface.
!!! note
    New types do not have to implement this method explicitly. An implementation is provided based on the [`AbstractObservable`](@ref)'s type parameter.
"""
Base.eltype(::Type{<:AbstractObservable{T}}) where {T} = T # could be vector value

"""
    scalartype(op::AbstractObservable)
Return the type of the underlying scalar field of the operator. This may be different from
the element type of the operator returned by [`eltype`](@ref), which can be a vector value.

Part of the [`AbstractObservable`](@ref) interface.
!!! note
    New types do not have to implement this method explicitly. An implementation is provided based on the [`AbstractObservable`](@ref)'s type parameter.
"""
VectorInterface.scalartype(::Type{<:AbstractObservable{T}}) where {T} = eltype(T)

"""
    AbstractOperator{T} <: AbstractObservable{T}

Supertype that provides an interface for linear operators over a linear space with elements
of type `T` (returned by [`eltype`](@ref)) and general (custom type) indices called
'addresses'.

`AbstractOperator` instances operate on vectors of type [`AbstractDVec`](@ref) from the
module `DictVectors` and work well with addresses of type
[`AbstractFockAddress`](@ref Main.BitStringAddresses.AbstractFockAddress)
from the module `BitStringAddresses`.

The defining feature of an `AbstractOperator` is that it can be applied to a vector with
[`mul!(y, op, x)`](@ref LinearAlgebra.mul!) and that three-way dot products can be
calculated with [`dot(x, op, y)`](@ref LinearAlgebra.dot).

The `AbstractOperator` type is useful for defining operators that are not necessarily
Hamiltonians, but that can be used in the context of a [`ProjectorMonteCarloProblem`](@ref)
as observable operators in a [`ReplicaStrategy`](@ref Rimu.ReplicaStrategy), e.g. for
defining correlation functions. In contrast to [`AbstractHamiltonian`](@ref)s,
`AbstractOperator`s do not need to have a [`starting_address`](@ref). Moreover, the
`eltype` of an `AbstractOperator` can be a vector value whereas
[`AbstractHamiltonian`](@ref)s requre a scalar `eltype`.

    AbstractHamiltonian{T} <: AbstractOperator{T} <: AbstractObservable{T}

The `AbstractOperator` type is part of the [`AbstractObservable`](@ref) hierarchy. It is
more restrictive than `AbstractObservable` in that it requires the interface for the
generation of diagonal and off-diagonal elements.

For concrete implementations see [`Hamiltonians`](@ref Main.Hamiltonians). In order to
implement a Hamiltonian for use in [`ProjectorMonteCarloProblem`](@ref) or
[`ExactDiagonalizationProblem`](@ref) use the type [`AbstractHamiltonian`](@ref) instead.

# Interface

Basic interface methods to implement:
- [`allows_address_type(op, type)`](@ref)
- [`diagonal_element(op, address)`](@ref)
- [`num_offdiagonals(op, address)`](@ref) and
- [`get_offdiagonal(op, address, chosen)`](@ref) or [`offdiagonals`](@ref)

Optional additional methods to implement:
- [`VectorInterface.scalartype(op)`](@ref): defaults to `eltype(eltype(op))`
- [`LOStructure(::Type{typeof(op)})`](@ref LOStructure): defaults to `AdjointUnknown`
- [`dimension(op, addr)`](@ref Main.Hamiltonians.dimension): defaults to dimension of
  address space

In order to calculate observables efficiently, it may make sense to implement custom methods
for [`Interfaces.dot_from_right(x, op, y)`](@ref) and [`LinearAlgebra.mul!(y, op, x)`](@ref).

See also [`AbstractHamiltonian`](@ref), [`Interfaces`](@ref).
"""
abstract type AbstractOperator{T} <: AbstractObservable{T} end

@doc """
    LinearAlgebra.mul!(w::AbstractDVec, op::AbstractOperator, v::AbstractDVec)
In place multiplication of `op` with `v` and storing the result in `w`. The result is
returned. Note that `w` needs to have a `valtype` that can hold a product of instances
of `eltype(op)` and `valtype(v)`. Moreover, the [`StochasticStyle`](@ref) of `w` needs to
be [`<:IsDeterministic`](@ref Rimu.StochasticStyles.IsDeterministic).

Part of the [`AbstractOperator`](@ref) interface.

The default implementation relies of [`diagonal_element`](@ref) and [`offdiagonals`](@ref)
to access the elements of the operator. The function can be overloaded for custom operators.
"""
LinearAlgebra.mul!

@doc """
    dot(w, op::AbstractObservable, v)

Evaluate `w⋅op(v)` minimizing memory allocations.
"""
LinearAlgebra.dot

@doc """
    dot_from_right(w, op::AbstractObservable, v)

Internal function evaluates the 3-argument `dot()` function in order from right
to left.
"""
function dot_from_right(::W, ::O, ::V) where {W, O, V}
    throw(ArgumentError("dot_from_right not implemented for types $W, $O, $V"))
end

"""
    AbstractHamiltonian{T} <: AbstractOperator{T}

Supertype that provides an interface for linear operators over a linear space with scalar
type `T` that are suitable for FCIQMC (with
[`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem)). Indexing is done
with addresses (typically not integers) from an address space that may be large (and will
not need to be completely generated).

`AbstractHamiltonian` instances operate on vectors of type [`AbstractDVec`](@ref) from the
module `DictVectors` and work well with addresses of type
[`AbstractFockAddress`](@ref Main.BitStringAddresses.AbstractFockAddress)
from the module `BitStringAddresses`. The type works well with the external package
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).

For available implementations see [`Hamiltonians`](@ref Main.Hamiltonians).

# Interface

Basic interface methods to implement:

* [`starting_address(::AbstractHamiltonian)`](@ref)
* [`diagonal_element(::AbstractHamiltonian, address)`](@ref)
* [`num_offdiagonals(::AbstractHamiltonian, address)`](@ref)
* [`get_offdiagonal(::AbstractHamiltonian, address, chosen::Integer)`](@ref) (optional, see
    below)

Optional additional methods to implement:

* [`LOStructure(::Type{typeof(lo)})`](@ref LOStructure): defaults to `AdjointUnknown`
* [`dimension(::AbstractHamiltonian, addr)`](@ref Main.Hamiltonians.dimension): defaults to
  dimension of address space
* [`allows_address_type(h::AbstractHamiltonian, type)`](@ref): defaults to
  `type :< typeof(starting_address(h))`
* [`momentum(::AbstractHamiltonian)`](@ref Main.Hamiltonians.momentum): no default

Provides the following functions and methods:

* [`offdiagonals`](@ref): iterator over reachable off-diagonal matrix elements
* [`random_offdiagonal`](@ref): function to generate random off-diagonal matrix element
* `*(H, v)`: deterministic matrix-vector multiply (allocating)
* `H(v)`: equivalent to `H * v`.
* `mul!(w, H, v)`: mutating matrix-vector multiply.
* [`dot(x, H, v)`](@ref Main.Hamiltonians.dot): compute `x⋅(H*v)` minimizing allocations.
* `H[address1, address2]`: indexing with `getindex()` - mostly for testing purposes (slow!)
* [`BasisSetRepresentation`](@ref Main.ExactDiagonalization.BasisSetRepresentation):
  construct a basis set repesentation
* [`sparse`](@ref Main.ExactDiagonalization.sparse), [`Matrix`](@ref): construct a (sparse)
  matrix representation

Alternatively to the above, [`offdiagonals`](@ref) can be implemented instead of
[`get_offdiagonal`](@ref). Sometimes this can be done efficiently. In this case
[`num_offdiagonals`](@ref) should provide an upper bound on the number of elements obtained
when iterating [`offdiagonals`](@ref).

See also [`Hamiltonians`](@ref Main.Hamiltonians), [`Interfaces`](@ref),
[`AbstractOperator`](@ref), [`AbstractObservable`](@ref).
"""
abstract type AbstractHamiltonian{T} <: AbstractOperator{T} end

"""
    allows_address_type(operator, addr_or_type)
Returns `true` if `addr_or_type` is a valid address for `operator`. Otherwise, returns
`false`.

Part of the [`AbstractHamiltonian`](@ref) interface.

# Extended help
Defaults to `addr_or_type <: typeof(starting_address(operator))`. Overload this function if
the operator can be used with addresses of different types.
"""
@inline function allows_address_type(hamiltonian, ::Type{A}) where {A}
    return A <: typeof(starting_address(hamiltonian))
end
function allows_address_type(op, address)
    allows_address_type(op, typeof(address))
end

"""
    diagonal_element(ham, address)

Compute the diagonal matrix element of the linear operator `ham` at
address `address`.

# Example

```jldoctest
julia> address = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(address);


julia> diagonal_element(H, address)
8.666666666666664
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
diagonal_element(m::AbstractMatrix, i) = m[i, i]

"""
    num_offdiagonals(ham, address)

Compute the number of number of reachable configurations from address `address`.

# Example

```jldoctest
julia> address = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(address);


julia> num_offdiagonals(H, address)
10
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
num_offdiagonals(m::AbstractMatrix, i) = length(offdiagonals(m, i))

"""
    newadd, me = get_offdiagonal(ham, address, chosen)

Compute value `me` and new address `newadd` of a single (off-diagonal) matrix element in a
Hamiltonian `ham`. The off-diagonal element is in the same column as address `address` and is
indexed by integer index `chosen`.

# Example

```jldoctest
julia> addr = BoseFS(3, 2, 1);

julia> H = HubbardMom1D(addr);

julia> get_offdiagonal(H, addr, 3)
(BoseFS{6,3}(2, 1, 3), 1.0)
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
get_offdiagonal(m::AbstractMatrix, i, n) = offdiagonals(m, i)[n]

"""
    starting_address(h)

Return the starting address for Hamiltonian `h`. When called on an `AbstractMatrix`,
`starting_address` returns the index of the lowest diagonal
element.

# Example

```jldoctest
julia> address = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(address);


julia> address == starting_address(H)
true
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
starting_address(m::AbstractMatrix) = findmin(real.(diag(m)))[2]

"""
    offdiagonals(h::AbstractHamiltonian, address)

Return an iterator over nonzero off-diagonal matrix elements of `h` in the same column as
`address`. Will iterate over pairs `(newaddress, matrixelement)`.

# Example

```jldoctest
julia> address = BoseFS(3,2,1);


julia> H = HubbardReal1D(address);


julia> h = offdiagonals(H, address)
6-element Rimu.Hamiltonians.Offdiagonals{BoseFS{6, 3, BitString{8, 1, UInt8}}, Float64, HubbardReal1D{Float64, BoseFS{6, 3, BitString{8, 1, UInt8}}, 1.0, 1.0}}:
 (fs"|2 3 1⟩", -3.0)
 (fs"|2 2 2⟩", -2.449489742783178)
 (fs"|3 1 2⟩", -2.0)
 (fs"|4 1 1⟩", -2.8284271247461903)
 (fs"|4 2 0⟩", -2.0)
 (fs"|3 3 0⟩", -1.7320508075688772)
```
Part of the [`AbstractHamiltonian`](@ref) interface.

# Extemded help

`offdiagonals` return and iterator of type `<:AbstractOffdiagonals`. It defaults to
returning `Offdiagonals(h, a)`

See also [`Offdiagonals`](@ref Main.Hamiltonians.Offdiagonals),
[`AbstractOffdiagonals`](@ref Main.Hamiltonians.AbstractOffdiagonals).

"""
function offdiagonals(m::AbstractMatrix, i)
    pairs = collect(zip(axes(m, 1), view(m, :, i)))
    return filter!(pairs) do ((k, v))
        k ≠ i && v ≠ 0
    end
end

"""
    random_offdiagonal(offdiagonals::AbstractOffdiagonals)
    random_offdiagonal(ham::AbstractHamiltonian, address)
    -> newaddress, probability, matrixelement

Generate a single random excitation, i.e. choose from one of the accessible off-diagonal
elements in the column corresponding to `address` in the Hamiltonian matrix represented
by `ham`. Alternatively, pass as argument an iterator over the accessible matrix elements.

Part of the [`AbstractHamiltonian`](@ref) interface.
"""
function random_offdiagonal(offdiagonals::AbstractVector)
    nl = length(offdiagonals) # check how many sites we could get_offdiagonal to
    chosen = rand(1:nl) # choose one of them
    naddress, melem = offdiagonals[chosen]
    return naddress, 1.0/nl, melem
end

function random_offdiagonal(ham, address)
    return random_offdiagonal(offdiagonals(ham, address))
end

@doc """
    LOStructure(op::AbstractHamiltonian)
    LOStructure(typeof(op))

Return information about the structure of the linear operator `op`.
`LOStructure` is used as a trait to speficy symmetries or other properties of the linear
operator `op` that may simplify or speed up calculations. Implemented instances are:

* `IsDiagonal()`: The operator is diagonal.
* `IsHermitian()`: The operator is complex and Hermitian or real and symmetric.
* `AdjointKnown()`: The operator is not Hermitian, but its
    [`adjoint`](@ref Main.Hamiltonians.adjoint) is implemented.
* `AdjointUnknown()`: [`adjoint`](@ref Main.Hamiltonians.adjoint) for this operator is not
    implemented.

Part of the [`AbstractHamiltonian`](@ref) interface.

In order to define this trait for a new linear operator type, define a method for
`LOStructure(::Type{<:MyNewLOType}) = …`.
"""
abstract type LOStructure end

struct IsDiagonal <: LOStructure end
struct IsHermitian <: LOStructure end
struct AdjointKnown <: LOStructure end
struct AdjointUnknown <: LOStructure end

# defaults
LOStructure(op) = LOStructure(typeof(op))
LOStructure(::Type) = AdjointUnknown()
LOStructure(::AbstractMatrix) = AdjointKnown()

# diagonal matrices have zero offdiagonal elements
function num_offdiagonals(h::H, addr) where {H<:AbstractOperator}
    return num_offdiagonals(LOStructure(H), h, addr)
end
num_offdiagonals(::IsDiagonal, _, _) = 0

"""
    has_adjoint(op)

Return true if `adjoint` is defined on `op`.

Part of the [`AbstractHamiltonian`](@ref) interface.

See also [`LOStructure`](@ref Main.Hamiltonians.LOStructure).
"""
has_adjoint(op) = has_adjoint(LOStructure(op))
has_adjoint(::AdjointUnknown) = false
has_adjoint(::LOStructure) = true
