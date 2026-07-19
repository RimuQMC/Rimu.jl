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
- [`has_random_offdiagonal(::Type{typeof(op)})`](@ref has_random_offdiagonal): defaults to
  `false`
- [`has_iterable_offdiagonals(::Type{typeof(op)})`](@ref has_iterable_offdiagonals):
  defaults to `false`

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
[`AbstractHamiltonian`](@ref)s require a scalar `eltype`.

    AbstractHamiltonian{T} <: AbstractOperator{T} <: AbstractObservable{T}

The `AbstractOperator` type is part of the [`AbstractObservable`](@ref) hierarchy. It is
more restrictive than `AbstractObservable` in that it requires the interface for the
generation of diagonal and off-diagonal elements.

For concrete implementations see [`Hamiltonians`](@ref Main.Hamiltonians). In order to
implement a Hamiltonian for use in [`ProjectorMonteCarloProblem`](@ref) or
[`ExactDiagonalizationProblem`](@ref) use the type [`AbstractHamiltonian`](@ref) instead.

# Interface

Mandatory methods to implement:
- [`allows_address_type(op, type)`](@ref)
- [`operator_column(op, address)`](@ref)
- [`diagonal_element(column)`](@ref)
- [`num_offdiagonals(column)`](@ref) (this can be an upper bound)
- [`offdiagonals(column)`](@ref) required for deterministic operations, see
    [`has_iterable_offdiagonals(::Type{typeof(op)})`](@ref has_iterable_offdiagonals) below

Optional additional methods to implement:
- [`VectorInterface.scalartype(ham)`](@ref): defaults to `eltype(eltype(ham))`
- [`LOStructure(::Type{typeof(ham)})`](@ref LOStructure): defaults to `AdjointUnknown`
- [`dimension(ham, addr)`](@ref Main.Hamiltonians.dimension): defaults to dimension of
  address space
- [`undo_transform(ham, op)`](@ref): the default implementation returns `op`.
- [`has_iterable_offdiagonals(::Type{typeof(op)})`](@ref has_iterable_offdiagonals):
  defaults to `true`
- [`has_random_offdiagonal(::Type{typeof(op)})`](@ref has_random_offdiagonal): defaults to
  `false`. If this set to `true`, the method [`random_offdiagonal(column)`](@ref) needs to
  be implemented.

## Alternative Interface (deprecated)

If the number of non-zero matrix elements that can be reached from any address is known,
and they can be separately generated:
- [`allows_address_type(op, type)`](@ref)
- [`diagonal_element(op, address)`](@ref)
- [`num_offdiagonals(op, address)`](@ref) and
- [`get_offdiagonal(op, address, chosen)`](@ref) or [`offdiagonals`](@ref) returning an
    `AbstractVector` object.

In order to calculate observables efficiently, it may make sense to implement custom methods
for [`Interfaces.dot_from_right(x, ham, y)`](@ref) and
[`LinearAlgebra.mul!(y, ham, x)`](@ref).

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

Mandatory methods to implement:

* [`starting_address(op::AbstractHamiltonian)`](@ref)
* [`operator_column(op, address)`](@ref) returns an [`AbstractOperatorColumn`](@ref)

[`AbstractOperatorColumn`](@ref) has its own interface methods:
* [`parent_operator(column)`](@ref) returns the operator of the column
* [`diagonal_element(column)`](@ref) returns the diagonal element of the column
* [`num_offdiagonals(column)`](@ref) (this can be an upper bound) returns the number of
    off-diagonal elements in the column
* [`offdiagonals(column)`](@ref) returns an object representing the off-diagonal elements
    of the column
* [`random_offdiagonal(column)`](@ref) returns a random off-diagonal element in the column

Optional additional methods to implement:

* [`LOStructure(::Type{typeof(op)})`](@ref LOStructure): defaults to `AdjointUnknown`
* [`has_random_offdiagonal(::Type{typeof(op)})`](@ref has_random_offdiagonal): defaults to
  `true`.
* [`has_iterable_offdiagonals(::Type{typeof(op)})`](@ref has_iterable_offdiagonals):
  defaults to `true`.
* [`dimension(::AbstractHamiltonian, addr)`](@ref Main.Hamiltonians.dimension): defaults to
  dimension of address space
* [`allows_address_type(h::AbstractHamiltonian, type)`](@ref): defaults to
  `type :< typeof(starting_address(h))`
* [`momentum(::AbstractHamiltonian)`](@ref Main.Hamiltonians.momentum): no default

## Alternative Interface (deprecated)

* [`starting_address(::AbstractHamiltonian)`](@ref)
* [`diagonal_element(op, address)`](@ref)
* [`num_offdiagonals(op, address)`](@ref) and
* [`get_offdiagonal(op, address, chosen)`](@ref) or [`offdiagonals(op, address)`](@ref)

Provides the following functions and methods:

* [`ProjectorMonteCarloProblem(H)`](@ref Main.ProjectorMonteCarloProblem):
  use this Hamiltonian for FCIQMC
* [`ExactDiagonalizationProblem(H)`](@ref Main.ExactDiagonalizationProblem):
  use this Hamiltonian for exact diagonalization
* `*(H, v)`: deterministic matrix-vector multiply (allocating)
* `H(v)`: equivalent to `H * v`.
* `mul!(w, H, v)`: mutating matrix-vector multiply.
* [`dot(x, H, v)`](@ref Main.Hamiltonians.dot): compute `x⋅(H*v)` minimizing allocations.
* `H[address1, address2]`: indexing with `getindex()` - mostly for testing purposes (slow!)
* [`BasisSetRepresentation`](@ref Main.ExactDiagonalization.BasisSetRepresentation):
  construct a basis set representation
* [`sparse`](@ref Main.ExactDiagonalization.sparse), [`Matrix`](@ref): construct a (sparse)
  matrix representation

See also [`Hamiltonians`](@ref Main.Hamiltonians), [`Interfaces`](@ref),
[`AbstractOperatorColumn`](@ref), [`AbstractOperator`](@ref), [`AbstractObservable`](@ref).
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
    diagonal_element(column::AbstractOperatorColumn)
    diagonal_element(ham, address) # (deprecated)

Compute the diagonal matrix element of the linear operator `ham` at
address `address`, where `column = operator_column(ham, address)`.

# Example

```jldoctest
julia> H = HubbardMom1D(BoseFS(3, 2, 1));

julia> column = operator_column(H, starting_address(H));

julia> diagonal_element(column)
8.666666666666664
```
Part of the [`AbstractHamiltonian`](@ref) interface. See also
[`AbstractOperatorColumn`](@ref) and [`operator_column`](@ref).
"""
diagonal_element

"""
    num_offdiagonals(column::AbstractOperatorColumn)
    num_offdiagonals(ham, address) # (deprecated)

Compute the number of number of reachable configurations from address `address`,
where `column = operator_column(ham, address)`. If necessary, this may be an upper bound.

# Example

```jldoctest
julia> H = HubbardMom1D(BoseFS(3, 2, 1));

julia> column = operator_column(H, starting_address(H));

julia> num_offdiagonals(column)
10
```
Part of the [`AbstractHamiltonian`](@ref) interface. See also
[`AbstractOperatorColumn`](@ref) and [`operator_column`](@ref).
"""
num_offdiagonals

"""
    newadd, me = get_offdiagonal(ham, address, chosen) # (deprecated)

Compute value `me` and new address `newadd` of a single (off-diagonal) matrix element in a
Hamiltonian `ham`. The off-diagonal element is in the same column as address `address` and is
indexed by integer index `chosen`.

# Example

```jldoctest
julia> addr = BoseFS(3, 2, 1);

julia> H = HubbardMom1D(addr);

julia> get_offdiagonal(H, addr, 3)
(BoseFS(2, 1, 3), 1.0)
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
get_offdiagonal(m, i, n) = offdiagonals(m, i)[n]

"""
    starting_address(h::AbstractHamiltonian)
    starting_address(column::AbstractOperatorColumn)

Return the starting address for the Hamiltonian `h`, or for the `column`.

# Example

```jldoctest
julia> address = BoseFS((3, 2, 1));

julia> H = HubbardMom1D(address);

julia> address == starting_address(H)
true
```
Part of the [`AbstractHamiltonian`](@ref) interface. See also
[`AbstractOperatorColumn`](@ref).
"""
starting_address

"""
    undo_transform(ham::AbstractHamiltonian, operator)

Modify the `operator` in a way that undoes the transform that was applied to `ham`.
See [`TransformUndoer`](@ref Main.Hamiltonians).
"""
undo_transform(::AbstractHamiltonian, op::AbstractObservable) = op

"""
    offdiagonals(column)
    offdiagonals(h::AbstractHamiltonian, address) # (deprecated)

Return an iterator over nonzero off-diagonal matrix elements of `h` in the same column as
`address`. Will iterate over pairs `(newaddress, matrixelement)` or `newaddress => matrixelement`.

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

See also [`Offdiagonals`](@ref Main.Hamiltonians.Offdiagonals),
[`AbstractOffdiagonals`](@ref Main.Hamiltonians.AbstractOffdiagonals).

"""
function offdiagonals(m::AbstractMatrix, i)
    pairs = collect(zip(axes(m, 1), view(m, :, i)))
    return filter!(pairs) do ((k, v))
        k ≠ i && v ≠ 0
    end
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

"""
    AbstractOperatorColumn{A,T,O}

Abstract type for operator columns returned by [`operator_column(operator, address)`](@ref)
or `operator * address`. The type parameters represent the address type (`A`), the eltype
(`T`), and the type of the operator (`O`).

In quantum notation, the column represents the object
```math
    Ĥ|α⟩ = ∑ᵦ|β⟩⟨β|Ĥ|α⟩,
```
where ``α`` is the  `address` and ``β`` represents all reachable addresses with nonzero
matrix element ``⟨β|Ĥ|α⟩`` of the `operator` ``Ĥ``.

# Interface

The default constructor for `AbstractOperatorColumn` is
`operator_column(operator, address)`. The interface for `AbstractOperatorColumn` is defined
by the following functions:

* [`parent_operator(c::AbstractOperatorColumn)`](@ref) - returns the `operator`,
* [`starting_address(c::AbstractOperatorColumn)`](@ref) - returns the starting `address`,
* [`diagonal_element(c::AbstractOperatorColumn)`](@ref) - returns the diagonal element of
    the column `c`, ``⟨α|Ĥ|α⟩``, where ``α`` is the starting address,
* [`num_offdiagonals(c::AbstractOperatorColumn)`](@ref) - returns an upper bound on the
    number of off-diagonal elements in the column `c`,
* [`offdiagonals(c::AbstractOperatorColumn)`](@ref) - returns an object representing the
    off-diagonal elements of the column `c`,
* [`random_offdiagonal(c::AbstractOperatorColumn)`](@ref) - returns a random off-diagonal
    element in the column `c`.

Methods for these functions need to be implemented for a new type of
[`AbstractOperator`](@ref).

Part of the  [`AbstractHamiltonian`](@ref) and  [`AbstractOperator`](@ref) interface.
See also [`operator_column`](@ref).
"""
abstract type AbstractOperatorColumn{A,T,O} end

function Base.show(io::IO, c::AbstractOperatorColumn{A,T,O}) where {A,T,O}
    io = IOContext(io, :compact => true)
    show(io, parent_operator(c))
    print(io, " * ", starting_address(c))
end

"""
    OffdiagonalsOperatorColumn <: AbstractOperatorColumn

Default column, using the deprecated interface with [`offdiagonals(op, address)`](@ref) and
[`diagonal_element(op, address)`](@ref).

See also [`operator_column`](@ref).
"""
struct OffdiagonalsOperatorColumn{A,T,O<:AbstractOperator{T},OD} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    ods::OD
    diagonal::T
end

"""
    parent_operator(c::AbstractOperatorColumn)
Return the operator of the column `c`.  This is the operator that was used to
construct the column with [`operator_column`](@ref).

Part of the [`AbstractHamiltonian`](@ref) and [`AbstractOperator`](@ref) interface.
See also [`AbstractOperatorColumn`](@ref) and [`operator_column`](@ref).
"""
parent_operator(::AbstractOperatorColumn)

parent_operator(c::OffdiagonalsOperatorColumn) = c.operator

"""
    operator_column(operator::AbstractOperator, address) -> column <: AbstractOperatorColumn
    *(o::AbstractOperator, a::AbstractFockAddress) -> column

Return a (lazy) object representing the column of `operator` given by `address`. In quantum
notation, the `column` represents the object
```math
    Ĥ|α⟩ = ∑ᵦ|β⟩⟨β|Ĥ|α⟩,
```
where ``α`` is the  `address` and ``β`` represents all reachable addresses with nonzero
matrix element ``⟨β|Ĥ|α⟩`` of the `operator` ``Ĥ``. Columns iterate over pairs
`address => matrix_element`.

For more information and the definition of an interface see
[`AbstractOperatorColumn`](@ref).

# Example
```jldoctest
julia> address = BoseFS(0,1,0);

julia> H = HubbardReal1D(address; u=6.0);

julia> column = H * address # construct column with `*` operator
HubbardReal1D(fs"|0 1 0⟩"; u=6.0, t=1.0) * fs"|0 1 0⟩"

julia> operator_column(H, address) == column
true

julia> diagonal_element(column) == column[address] # access elements with `getindex`
true

julia> DVec(column) # materialise as a `DVec`
DVec{BoseFS{1, 3, BitString{3, 1, UInt8}},Float64} with 2 entries, style = IsDeterministic{Float64}()
  fs"|1 0 0⟩" => -1.0
  fs"|0 0 1⟩" => -1.0
```

Part of the [`AbstractHamiltonian`](@ref) interface. See also
[`AbstractOperatorColumn`](@ref), [`parent_operator`](@ref), [`starting_address`](@ref),
[`diagonal_element`](@ref), [`offdiagonals`](@ref), [`num_offdiagonals`](@ref),
[`random_offdiagonal`](@ref).
"""
operator_column(o, a) = OffdiagonalsOperatorColumn(o, a, offdiagonals(o,a),
    eltype(o)(diagonal_element(o,a)))

@doc (@doc operator_column) Base.:*
function Base.:*(o::AbstractObservable, a::A) where {A<:Union{AbstractFockAddress,Integer}}
    if allows_address_type(o, A)
        return operator_column(o, a)
    elseif A <: Number
        throw(ArgumentError("`$A` is not a valid address type for `$o`. To scale the operator, use `$a * $o`."))
    else
        throw(ArgumentError("`$A` is not a valid address type for `$o`."))
    end
end

starting_address(c::OffdiagonalsOperatorColumn) = c.address
diagonal_element(c::OffdiagonalsOperatorColumn) = c.diagonal
num_offdiagonals(c::OffdiagonalsOperatorColumn) = num_offdiagonals(c.operator, c.address)
offdiagonals(c::OffdiagonalsOperatorColumn) = c.ods

"""
    getindex(column::AbstractOperatorColumn, address)

Return the matrix element of the operator column `column` at `address`.

!!! warning
    This method is provided for convenience only. It is not efficient and should not be
    used in performance-critical code. Use [`diagonal_element`](@ref) and
    [`offdiagonals`](@ref) instead to access the diagonal and off-diagonal
    elements of the operator column.

See also [`operator_column`](@ref), [`AbstractOperatorColumn`](@ref).
"""
function Base.getindex(
    column::AbstractOperatorColumn{A,T}, address::A
) where {A,T}
    if address == starting_address(column)
        return diagonal_element(column)::T
    else
        for (newaddress, matrixelement) in offdiagonals(column)
            if newaddress == address
                return matrixelement::T
            end
        end
    end
    return zero(T)
end

"""
    dot(dv, column::AbstractOperatorColumn)

Compute the dot product of the vector `dv` with the operator column `column`.

See also [`operator_column`](@ref), [`AbstractOperatorColumn`](@ref),
and [`AbstractDVec`](@ref).
"""
function LinearAlgebra.dot(
    dv, column::AbstractOperatorColumn
)
    res = conj(dv[starting_address(column)]) * diagonal_element(column)
    for (newaddress, matrixelement) in offdiagonals(column)
        res += conj(dv[newaddress]) * matrixelement
    end
    return res
end
function LinearAlgebra.dot(column::AbstractOperatorColumn, a)
    return adjoint(LinearAlgebra.dot(a, column))
end

"""
    offdiagonals(column::AbstractOperatorColumn)
    offdiagonals(h::AbstractHamiltonian, address) # (deprecated)

Return an iterator over nonzero off-diagonal matrix elements of `h` in the same column as
`address`. Will iterate over pairs `(newaddress, matrixelement)` or
`newaddress => matrixelement`.

# Example

```jldoctest
julia> address = BoseFS(3,2,1);

julia> H = HubbardReal1D(address);

julia> h = offdiagonals(H * address)
6-element Rimu.Hamiltonians.Offdiagonals{BoseFS{6, 3, BitString{8, 1, UInt8}}, Float64, HubbardReal1D{Float64, BoseFS{6, 3, BitString{8, 1, UInt8}}, 1.0, 1.0}}:
 (fs"|2 3 1⟩", -3.0)
 (fs"|2 2 2⟩", -2.449489742783178)
 (fs"|3 1 2⟩", -2.0)
 (fs"|4 1 1⟩", -2.8284271247461903)
 (fs"|4 2 0⟩", -2.0)
 (fs"|3 3 0⟩", -1.7320508075688772)
```
Part of the [`AbstractHamiltonian`](@ref) interface. See also
[`AbstractOperatorColumn`](@ref) and [`operator_column`](@ref).
"""
offdiagonals

# Iteration interface for AbstractOperatorColumn
@inline function Base.iterate(col::AbstractOperatorColumn)
    return starting_address(col) => diagonal_element(col), nothing
end
@inline function Base.iterate(col::AbstractOperatorColumn, state)
    result = if state === nothing
        iterate(offdiagonals(col))
    else
        iterate(offdiagonals(col), state)
    end
    if result === nothing
        return nothing
    end
    (address, value), newstate = result
    return address => value, newstate
end
Base.IteratorSize(::AbstractOperatorColumn) = Base.SizeUnknown()
Base.eltype(::AbstractOperatorColumn{A,T}) where {A,T} = Pair{A,T}

"""
    random_offdiagonal(column::AbstractOperatorColumn)
    random_offdiagonal(ham::AbstractHamiltonian, address) # deprecated
    -> newaddress, probability, matrixelement

Generate a single random excitation, i.e. choose from one of the accessible off-diagonal
elements in the column `ham * address`.
The result is a tuple of the new address `newaddress`, the probability of this excitation
`probability`, and the matrix element `matrixelement` of the excitation.

Part of the [`AbstractHamiltonian`](@ref) interface. See also
[`AbstractOperatorColumn`](@ref) and [`operator_column`](@ref).
"""
function random_offdiagonal(column::OffdiagonalsOperatorColumn)
    offdiags = offdiagonals(column)
    nl = length(offdiags) # check how many sites we could reach
    chosen = rand(1:nl) # choose one of them
    naddress, melem = offdiags[chosen]
    return naddress, 1.0/nl, melem
end

function random_offdiagonal(ham, address)
    return random_offdiagonal(operator_column(ham, address))
end

"""
    has_iterable_offdiagonals(operatortype::Type)::Bool

Return `true` if the operator's columns have iterable
[`offdiagonals`](@ref).

## Example
```jldoctest
julia> using Rimu.Interfaces

julia> h = HubbardReal1D(BoseFS(1,2,3));

julia> has_iterable_offdiagonals(typeof(h))
true
```

When extending the interface, implement a method for
`has_iterable_offdiagonals(::Type{<:MyNewOperator})`.

Part of the [`AbstractHamiltonian`](@ref) interface. See also
[`AbstractOperatorColumn`](@ref) and [`operator_column`](@ref).
"""
has_iterable_offdiagonals(op::AbstractObservable) = has_iterable_offdiagonals(typeof(op))
has_iterable_offdiagonals(::AbstractOperatorColumn{A,T,O}) where {A,T,O} =
    has_iterable_offdiagonals(O)

# default traits for operators and observables
has_iterable_offdiagonals(::Type{<:AbstractOperator}) = true
has_iterable_offdiagonals(::Type{<:AbstractObservable}) = false

"""
    has_random_offdiagonal(operatortype::Type)::Bool

Return `true` if the operator's columns have a
[`random_offdiagonal`](@ref) method implemented.

## Example
```jldoctest
julia> using Rimu.Interfaces

julia> h = HubbardReal1D(BoseFS(1,2,3));

julia> has_random_offdiagonal(typeof(h))
true
```

When extending the interface, implement a method for
`Rimu.Interfaces.has_random_offdiagonal(::Type{<:MyNewOperator})`.

Part of the [`AbstractHamiltonian`](@ref) interface. See also
[`AbstractOperatorColumn`](@ref) and [`operator_column`](@ref).
"""
has_random_offdiagonal(op::AbstractObservable) = has_random_offdiagonal(typeof(op))
has_random_offdiagonal(::AbstractOperatorColumn{A,T,O}) where {A,T,O} =
    has_random_offdiagonal(O)

# default traits for operators and observables
has_random_offdiagonal(::Type{<:AbstractHamiltonian}) = true
has_random_offdiagonal(::Type{<:AbstractObservable}) = false
