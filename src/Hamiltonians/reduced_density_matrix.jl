"""
    SingleParticleExcitation(i, j) <: AbstractOperator

Represent the ``{i,j}`` element of the single-particle reduced density matrix:

```math
ρ̂^{(1)}_{i,j} = â^†_{i} â_{j}
```

where `i <: Int` and `j <: Int` specify the mode numbers.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`TwoParticleExcitation`](@ref)
"""
struct SingleParticleExcitation{I,J} <: AbstractOperator{Float64}
end

SingleParticleExcitation(I::Int,J::Int) = SingleParticleExcitation{I,J}()

function Base.show(io::IO, ::SingleParticleExcitation{I,J}) where {I,J}
    print(io, "SingleParticleExcitation($(I), $(J))")
end

LOStructure(::Type{<:SingleParticleExcitation}) = AdjointUnknown()
function allows_address_type(::SingleParticleExcitation{I,J}, ::Type{A}) where {I,J,A}
    return A <: SingleComponentFockAddress && I ≤ num_modes(A) && J ≤ num_modes(A)
end

function diagonal_element(
    ::SingleParticleExcitation{I,J}, addr::SingleComponentFockAddress
) where {I,J}
    if I != J
        return 0.0
    else
        src = find_mode(addr, J)
        return src.occnum
    end
end

function num_offdiagonals(
    ::SingleParticleExcitation{I,J}, ::SingleComponentFockAddress
) where {I,J}
    if I == J
        return 0
    else
        return 1
    end
end

function get_offdiagonal(
    ::SingleParticleExcitation{I,J}, addr::SingleComponentFockAddress, _
) where {I,J}
    src = find_mode(addr, J)
    dst = find_mode(addr, I)
    address, value = excitation(addr, (dst,), (src,))
    return address, value
end

"""
    TwoParticleExcitation(i, j, k, l) <: AbstractOperator

Represent the ``{ij, kl}`` element of the two-particle reduced density matrix:

```math
ρ̂^{(2)}_{ij, kl} =  â^†_{i} â^†_{j} â_{l} â_{k}
```

where `i`, `j`, `k`, and `l` (all `<: Int`) specify the mode numbers.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`SingleParticleExcitation`](@ref)
"""
struct TwoParticleExcitation{I,J,K,L} <: AbstractOperator{Float64}
end

TwoParticleExcitation(I::Int,J::Int,K::Int,L::Int) = TwoParticleExcitation{I,J,K,L}()

function Base.show(io::IO, ::TwoParticleExcitation{I,J,K,L}) where {I,J,K,L}
    print(io, "TwoParticleExcitation($(I), $(J), $(K), $(L))")
end

LOStructure(::Type{<:TwoParticleExcitation}) = AdjointUnknown()
function allows_address_type(::TwoParticleExcitation{I,J,K,L}, ::Type{A}) where {I,J,K,L,A}
    return A <: SingleComponentFockAddress && I ≤ num_modes(A) && J ≤ num_modes(A) &&
            K ≤ num_modes(A) && L ≤ num_modes(A)
end

function diagonal_element(
    ::TwoParticleExcitation{I,J,K,L}, addr::SingleComponentFockAddress
) where {I,J,K,L}
    if (I, J) == (K, L) || (I, J) == (L, K)
        src = find_mode(addr, (L, K))
        dst = find_mode(addr, (I, J))
        return excitation(addr, dst, src)[2]
    else
        return 0.0
    end
end

function num_offdiagonals(
    ::TwoParticleExcitation{I,J,K,L}, ::SingleComponentFockAddress
) where {I,J,K,L}
    if (I, J) == (K, L) || (I, J) == (L, K)
        return 0
    else
        return 1
    end
end

function get_offdiagonal(
    ::TwoParticleExcitation{I,J,K,L}, addr::SingleComponentFockAddress, _
) where {I,J,K,L}
    src = find_mode(addr, (L, K))
    dst = find_mode(addr, (I, J))
    address, value = excitation(addr, dst, src)
    return address, value
end

"""
    ReducedDensityMatrix{T=Float64}(p) <: AbstractObservable{Matrix{T}}

A matrix-valued operator that can be used to calculate the `p`-particle reduced density
matrix. The matrix elements are defined as:

```math
\\hat{ρ}^{(p)}_{j_1,...,j_1,k_1,...,k_p} =  \\prod_{i=1}^{p} â^†_{j_i} \\prod_{l=p}^{1} â_{k_{l}}
```

The integer indices `j_i` and `k_i` represent single particle modes. For efficiency they are
chosen to be distinct and ordered:

```math
j_1 < j_2 < \\ldots < j_{p} \\quad \\land \\quad k_1 < k_2 < \\ldots < k_{p}
```
`ReducedDensityMatrix` can be used to construct the single-particle reduced density matrix
(with `p == 1`) for fermionic and bosonic Fock spaces with address types
[`<: SingleComponentFockAddress`](@ref SingleComponentFockAddress).
For higher order reduced density matrices with `p > 1` only fermionic Fock addresses
([`FermiFS`](@ref)) are supported due to the ordering of indices.

`ReducedDensityMatrix` can be used with [`dot`](@ref) or [`AllOverlaps`](@ref) to calculate
the whole matrix in one go.

# Examples

```jldoctest
julia> dvec_b = PDVec(BoseFS(1,1) => 0.5, BoseFS(2,0) => 0.5)
2-element PDVec: style = IsDeterministic{Float64}()
  fs"|2 0⟩" => 0.5
  fs"|1 1⟩" => 0.5

julia> Op1 = ReducedDensityMatrix(1)
ReducedDensityMatrix{Float64}(1)

julia> dot(dvec_b, Op1, dvec_b)
2×2 Matrix{Float64}:
 0.75      0.353553
 0.353553  0.25

julia> Op2 = ReducedDensityMatrix{Float32}(2)
ReducedDensityMatrix{Float32}(2)

julia> dvec_f = PDVec(FermiFS(1,1,0,0) => 0.5, FermiFS(0,1,1,0) => 0.5)
2-element PDVec: style = IsDeterministic{Float64}()
  fs"|↑↑⋅⋅⟩" => 0.5
  fs"|⋅↑↑⋅⟩" => 0.5

julia> dot(dvec_f, Op2, dvec_f)
6×6 Matrix{Float32}:
 0.25  0.0  0.25  0.0  0.0  0.0
 0.0   0.0  0.0   0.0  0.0  0.0
 0.25  0.0  0.25  0.0  0.0  0.0
 0.0   0.0  0.0   0.0  0.0  0.0
 0.0   0.0  0.0   0.0  0.0  0.0
 0.0   0.0  0.0   0.0  0.0  0.0
```
See also [`single_particle_density`](@ref), [`SingleParticleDensity`](@ref),
[`SingleParticleExcitation`](@ref), [`TwoParticleExcitation`](@ref).
"""
struct ReducedDensityMatrix{T, P} <: AbstractObservable{Matrix{T}} end

ReducedDensityMatrix(p) = ReducedDensityMatrix{Float64}(p)
ReducedDensityMatrix{T}(P::Int) where T = ReducedDensityMatrix{T, P}()

function Base.show(io::IO, ::ReducedDensityMatrix{T, P}) where {T, P}
    print(io, "ReducedDensityMatrix{$T}($P)")
end

LOStructure(::Type{<:ReducedDensityMatrix}) = IsHermitian()

function Interfaces.dot_from_right(
    left::AbstractDVec, op::ReducedDensityMatrix{TT, P}, right::AbstractDVec
) where {TT, P}
    if P > 1 && !(keytype(left) <: FermiFS && keytype(right) <: FermiFS)
         throw(ArgumentError("ReducedDensityMatrix(p) with `p > 1` requires `FermiFS` addresses"))
    end
    dim = binomial(num_modes(keytype(left)), P)
    ρ = sum_mutating!(
        zeros(TT, (dim, dim)),
        ReducedDensityMatrixCalculcator!{TT,P}(left, dim),
        pairs(right)
    )
    return ρ
end

# This struct is used to calculate matrix elements of `ReducedDensityMatrix`
# It was introduced because passing a function to `sum` in `dot_from_right` was causing
# type instabilites.
"""
    calc! = ReducedDensityMatrixCalculator!{P}(left, dim)
Instantiate a `ReducedDensityMatrixCalculator!{P}` object to calculate matrix elements of
`ReducedDensityMatrix`.

    calc!(rdm, pair)

Add the contribution of `pair` to the reduced density matrix to `rdm`.
"""
struct ReducedDensityMatrixCalculcator!{TT,P,D}
    left::D
    dim::Int

    ReducedDensityMatrixCalculcator!{TT,P}(left, dim) where {TT,P} = new{TT,P,typeof(left)}(left, dim)
end

function (calc!::ReducedDensityMatrixCalculcator!{TT, P})(result, pair) where {TT, P}
    addr_right, val_right = pair
    left = calc!.left

    for j in axes(result, 2)
        dsts = find_mode(addr_right, vertices(j, Val(P)))
        for i in axes(result, 1)
            srcs = reverse(find_mode(addr_right, vertices(i, Val(P))))

            addr_left, elem = excitation(addr_right, dsts, srcs)
            result[i, j] += TT(conj(left[addr_left]) * elem * val_right)
        end
    end
    return result
end

"""
    TestOneParticleDensity(v; normalize=true) <: AbstractOperator{eltype(v)}
A one particle operator constructed from a provided test vector `v`. An expectation value
with this operator yields a lower bound on the largest eigenvalue (and an upper bound on the
smallest eigenvalue) of the one-particle density matrix.
If `normalize` is true (default), the vector is normalized before use.

```math
    ρ̂ = ∑_{ij} v_i^* v_j â^†_{i} â_{j}
```
"""
struct TestOneParticleDensity{T,V<:SVector{<:Any,T},M} <: AbstractOperator{T}
    test_vector::V
end
function TestOneParticleDensity(v; normalize=true)
    if normalize
        v = v / norm(v)
    end
    M = length(v)
    T = float(eltype(v))
    tv = SVector{M,T}(v)
    return TestOneParticleDensity{T,typeof(tv),M}(tv)
end

function Base.show(io::IO, topd::TestOneParticleDensity)
    print(io, "TestOneParticleDensity(", topd.test_vector)
    if !(norm(topd.test_vector) ≈ 1.0)
        print(io, "; normalize=false")
    end
    print(io, ")")
end

Interfaces.LOStructure(::Type{<:TestOneParticleDensity}) = IsHermitian()
function Interfaces.allows_address_type(
    ::TestOneParticleDensity{T,A,M}, ::Type{B}
) where {T,A,M,B}
    B <: SingleComponentFockAddress && num_modes(B) == M
end

struct TestOneParticleDensityColumn{A,T,O,OMM} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    omm::OMM
end
function Interfaces.operator_column(o::TestOneParticleDensity, add::A) where {A}
    allows_address_type(o, A) || throw(ArgumentError("Address type not allowed for this operator"))
    omm = occupied_mode_map(add)
    return TestOneParticleDensityColumn{A,eltype(o),typeof(o),typeof(omm)}(o, add, omm)
end
Interfaces.parent_operator(c::TestOneParticleDensityColumn) = c.operator
Interfaces.starting_address(c::TestOneParticleDensityColumn) = c.address
function Interfaces.diagonal_element(c::TestOneParticleDensityColumn{<:Any,T}) where {T}
    val = zero(T)
    @inbounds for idx in c.omm
        val += abs2(c.operator.test_vector[idx.mode]) * idx.occnum
    end
    return val
end
function Interfaces.num_offdiagonals(c::TestOneParticleDensityColumn)
    return length(c.omm) * (num_modes(c.address) - 1)
end
function Interfaces.offdiagonals(c::TestOneParticleDensityColumn{A,T}) where {A,T}
    TestOneParticleDensityOffdiagonals{A,T,typeof(c)}(c)
end
struct TestOneParticleDensityOffdiagonals{A,T,C} 
    column::C
end
Base.eltype(::TestOneParticleDensityOffdiagonals{A,T}) where {A,T} = Tuple{A,T}
Base.IteratorSize(::TestOneParticleDensityOffdiagonals) = Base.SizeUnknown()
# Base.length(od::TestOneParticleDensityOffdiagonals) = num_offdiagonals(od.column)
function Base.iterate(od::TestOneParticleDensityOffdiagonals, state=(1, 1))
    c = od.column
    omm = c.omm
    n_modes = num_modes(c.address)
    i, j = state # i: mode number for creation, j: index in omm for annihilation
    #  ∑_{ij} v_i^* v_j â^†_{i} â_{j} |address⟩
    while j <= length(omm)
        src = omm[j]
        while i <= n_modes
            if i != src.mode # omit same mode as they contribute to diagonal
                # create new address with excitation
                dst = find_mode(c.address, i)
                address, value = excitation(c.address, (dst,), (src,))
                if !iszero(value)
                    value *= conj(c.operator.test_vector[i]) * c.operator.test_vector[src.mode]
                    # choose next state
                    if i + 1 <= n_modes
                        return (address, value), (i + 1, j)
                    else
                        return (address, value), (1, j + 1)
                    end
                end
            end
            i += 1
        end
        j += 1
        i = 1
    end
    return nothing
end

"""
    TestOneParticleDensityGradient(test_vector, jacobian=nothing; normalize=true,
        zeta=0) <: AbstractOperator{SVector{m,T}}
 
An expectation value with this operator yields a gradient `TestOneParticleDensity` 
for a given test vector. Here `test_vector` is the test vector and `T` is the eltype of 
`test_vector`. If `normalize` is true (default), the vector is normalized before use. 
`zeta` is a expectation value of `TestOneParticleDensity` for a given `test_vector`.

```math
    ζ = \\langle ∑_{i,j} ρ̂ {(1)}_{i,j} v_{i}^* v_{j} \\rangle
```

There are two cases involves:

- `jacobian <:Nothing`: gradient is calculated for each element of `vec`, 
and the operator is defined as below. A one particle operator constructed 
from a provided test vector. Also, `m` = # of sites.

```math
    ∂ρ̂ {(1)}/∂v_j = ∑_{i} v_{i}^* â^†_{i} â_{j} - ζ v_{j}^*
```

-`jacobian <:AbstractMatrix`: gradient is calculated with respect 
to the parameters of the given test_vector which has a fix functional form. 
Here, `jacobian` is a matrix of dimension (m × # of sites) representing 
transpose of the jacobian of test_vector with respect to its parameters 
`α₁,α₂,...,αₘ`.

```math
    ∂_α ρ̂ {(1)}= ∑_{ij} (v_{i}^* ∂v_j(α)/∂α + 
        ∂v_i^*(α)/∂α v_{j}) (â^†_{i} â_{j} - ζ δ_{i,j})
```
"""
struct TestOneParticleDensityGradient{T,Dim,Zeta,V<:SArray{<:Any,T},J} <: AbstractOperator{SVector{Dim,T}}
    test_vector::V
    jacobian::J
end
function TestOneParticleDensityGradient(test_vector, jacobian=nothing; zeta = 0, 
    normalize=true)

    if jacobian isa Nothing
        T = float(eltype(test_vector))
        dim = length(test_vector)
        if !(test_vector isa SVector)
            test_vector = SVector{dim,T}(test_vector)
        end
        if normalize
            test_vector = test_vector / norm(test_vector)
        end
    else
        if !(test_vector isa AbstractVector && jacobian isa AbstractMatrix)
            error("(test_vector, jacobian) must be (vector, matrix)")
        end
        T = float(eltype(test_vector))
        S = length(test_vector)
        dim = length(jacobian[:,1])
        if !(test_vector isa SVector && jacobian isa SMatrix)
            test_vector = SVector{S,T}(test_vector)
            jacobian = SMatrix{dim,S,T}(jacobian)
        end
        if normalize
            test_vector = test_vector/norm(test_vector)
            jacobian = jacobian/norm(test_vector)
        end
    end
    return TestOneParticleDensityGradient{T,dim,Float64(zeta),typeof(test_vector),typeof(jacobian)}(
            test_vector,jacobian)
end

function Base.show(io::IO, topd::TestOneParticleDensityGradient{<:Any,M}) where M
    print(io, "TestOneParticleDensityGradient(", topd.test_vector, topd.jacobian,)
    if !(norm(topd.test_vector) ≈ 1.0)
        print(io, "; normalize=false")
    end
    print(io, ")")
end

Interfaces.LOStructure(::Type{<:TestOneParticleDensityGradient}) = IsHermitian()
function Interfaces.allows_address_type(
    od::TestOneParticleDensityGradient{T}, ::Type{B}
) where {T,B}
    M = num_modes(B)
    B <: SingleComponentFockAddress && length(od.test_vector) == M
end

struct TestOneParticleDensityGradientColumn{O,A,T,OMM} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    omm::OMM
end
function Interfaces.operator_column(o::TestOneParticleDensityGradient, add::A) where {A}
    allows_address_type(o, A) || throw(ArgumentError("Address type not allowed for this operator"))
    omm = occupied_mode_map(add)
    return TestOneParticleDensityGradientColumn{typeof(o),A,eltype(o),typeof(omm)}(o, add, omm)
end
Interfaces.parent_operator(c::TestOneParticleDensityGradientColumn) = c.operator
Interfaces.starting_address(c::TestOneParticleDensityGradientColumn) = c.address
function Interfaces.diagonal_element(c::TestOneParticleDensityGradientColumn{
    <:TestOneParticleDensityGradient{<:Any,Dim,zeta},<:Any,T}) where {Dim,zeta,T}
    Onr = onr(c.address)
    M = num_modes(c.address)
    if Dim == M
        val = T(conj(c.operator.test_vector) .* (Onr .- zeta))
    else
        val = zero(T)
        @inbounds for i in 1:M
            val += (conj(c.operator.jacobian[:,i]) * c.operator.test_vector[i] .+ 
                conj.(c.operator.test_vector[i]) * c.operator.jacobian[:,i]) * (Onr[i] - zeta)
        end
    end
    return val
end
function Interfaces.num_offdiagonals(c::TestOneParticleDensityGradientColumn)
    return length(c.omm) * (num_modes(c.address) - 1)
end
function Interfaces.offdiagonals(c::TestOneParticleDensityGradientColumn{<:Any,A,T}) where {A,T}
    TestOneParticleDensityGradientOffdiagonals{A,T,typeof(c)}(c)
end
struct TestOneParticleDensityGradientOffdiagonals{A,T,C}
    column::C
end
Base.eltype(::TestOneParticleDensityGradientOffdiagonals{A,T}) where {A,T} = Tuple{A,T}
Base.IteratorSize(::TestOneParticleDensityGradientOffdiagonals) = Base.SizeUnknown()
# Base.length(od::TestOneParticleDensityGradientOffdiagonals) = num_offdiagonals(od.column)
function Base.iterate(od::TestOneParticleDensityGradientOffdiagonals{A,T}, state=(1, 1)) where {A,T}
    dim = size(od.column.operator.test_vector)[end]
    if length(T) == dim
        return fullvectorTestOneParticleDensityGradient(od, state)
    else
        return fixfunctionTestOneParticleDensityGradient(od, state)
    end
end

@inline function fixfunctionTestOneParticleDensityGradient(
    od::TestOneParticleDensityGradientOffdiagonals{<:Any,T}, state
    ) where {T}
    c = od.column
    omm = c.omm
    n_modes = num_modes(c.address)
    i, j = state # i: mode number for creation, j: index in omm for annihilation
    #  ∑_{ij} v_i^* v_j â^†_{i} â_{j} |address⟩
    while j <= length(omm)
        src = omm[j]
        while i <= n_modes
            if i != src.mode # omit same mode as they contribute to diagonal
                # create new address with excitation
                dst = find_mode(c.address, i)
                address, value = excitation(c.address, (dst,), (src,))
                if !iszero(value)
                    val = T((conj(c.operator.test_vector[i]) * 
                        c.operator.jacobian[:,src.mode] .+ 
                        conj.(c.operator.jacobian[:,i]) * 
                        c.operator.test_vector[src.mode]) * value)
                    # choose next state
                    if i + 1 <= n_modes
                        return (address, val), (i + 1, j)
                    else
                        return (address, val), (1, j + 1)
                    end
                end
            end
            i += 1
        end
        j += 1
        i = 1
    end
    return nothing
end

@inline function fullvectorTestOneParticleDensityGradient(
    od::TestOneParticleDensityGradientOffdiagonals{<:Any,T}, state
    ) where {T}
    c = od.column
    omm = c.omm
    n_modes = num_modes(c.address)
    i, j = state # i: mode number for creation, j: index in omm for annihilation
    #  ∑_{ij} v_i^* v_j â^†_{i} â_{j} |address⟩
    while j <= length(omm)
        src = omm[j]
        while i <= n_modes
            if i != src.mode # omit same mode as they contribute to diagonal
                # create new address with excitation
                dst = find_mode(c.address, i)
                address, value = excitation(c.address, (dst,), (src,))
                if !iszero(value)
                    val = zeros(eltype(T),n_modes)
                    val[src.mode] += conj(c.operator.test_vector[i]) * value
                    # choose next state
                    if i + 1 <= n_modes
                        return (address, T(val)), (i + 1, j)
                    else
                        return (address, T(val)), (1, j + 1)
                    end
                end
            end
            i += 1
        end
        j += 1
        i = 1
    end
    return nothing
end


"""
     TestTwoParticleDensity(v; normalize=true) <: AbstractOperator{eltype(v)}
A two particle operator constructed from a provided test vector `v`. An expectation value
with this operator yields a lower bound on the largest eigenvalue (and an upper bound on the
smallest eigenvalue) of the two-particle density matrix.
If `normalize` is true (default), the vector is normalized before use.

```math
    ρ̂ {(2)}= ∑_{ij,kl} v_{ij}^* v_{kl} â^†_{i} â^†_{j} â_{l} â_{k}
```
Also, in `vᵢⱼ`, i and j are site indices (with i < j). 
"""
struct TestTwoParticleDensity{T,V<:SVector{<:Any,T},Dim} <: AbstractOperator{T}
    test_vector::V
end
function TestTwoParticleDensity(v; normalize=true)
    if normalize
        v = v / norm(v)
    end
    T = float(eltype(v))
    dim = length(v)
    tv = SVector{dim,T}(v)
    return TestTwoParticleDensity{T,typeof(tv),dim}(tv)
end

function Base.show(io::IO, topd::TestTwoParticleDensity)
    print(io, "TestTwoParticleDensity(", topd.test_vector)
    if !(norm(topd.test_vector) ≈ 1.0)
        print(io, "; normalize=false")
    end
    print(io, ")")
end

Interfaces.LOStructure(::Type{<:TestTwoParticleDensity}) = IsHermitian()
function Interfaces.allows_address_type(
    ::TestTwoParticleDensity{T,A,Dim}, ::Type{B}
) where {T,A,Dim,B}
    M = num_modes(B)
    return B <: SingleComponentFockAddress && Dim == binomial(M,2)
end

struct TestTwoParticleDensityColumn{A,T,O,OMM} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    omm::OMM
end
function Interfaces.operator_column(o::TestTwoParticleDensity, add::A) where {A}
    allows_address_type(o, A) || throw(ArgumentError("Address type not allowed for this operator"))
    omm = occupied_mode_map(add)
    return TestTwoParticleDensityColumn{A,eltype(o),typeof(o),typeof(omm)}(o, add, omm)
end
Interfaces.parent_operator(c::TestTwoParticleDensityColumn) = c.operator
Interfaces.starting_address(c::TestTwoParticleDensityColumn) = c.address
function Interfaces.diagonal_element(c::TestTwoParticleDensityColumn{<:Any,T}) where {T}
    val = zero(T)
    omm = c.omm
    @inbounds for i in 1:length(omm), j in 1:i-1
        val +=  2*abs2(c.operator.test_vector[index((omm[i].mode,omm[j].mode))])
    end
    return val
end

function Interfaces.num_offdiagonals(c::TestTwoParticleDensityColumn)
    return binomial(length(c.omm),2) * (binomial(num_modes(c.address),2) - 1)
end
function Interfaces.offdiagonals(c::TestTwoParticleDensityColumn{A,T}) where {A,T}
    TestTwoParticleDensityOffdiagonals{A,T,typeof(c)}(c)
end
struct TestTwoParticleDensityOffdiagonals{A,T,C}
    column::C
end
Base.eltype(::TestTwoParticleDensityOffdiagonals{A,T}) where {A,T} = Tuple{A,T}
Base.IteratorSize(::TestTwoParticleDensityOffdiagonals) = Base.SizeUnknown()
Base.length(od::TestTwoParticleDensityOffdiagonals) = num_offdiagonals(od.column)
function Base.iterate(od::TestTwoParticleDensityOffdiagonals, state=(2, 1, 2, 1))
    c = od.column
    omm = c.omm
    n_modes = num_modes(c.address)
    i, j, k, l = state # i,j: mode number for creation, k,l: indices in omm for annihilation
    #  ∑_{ij.kl} v_{ij}^* v_{kl} â^†_{i} â^†_{j} â_{l} â_{k} |address⟩
    while l <= length(omm) - 1
        src2 = omm[l]
        while k <= length(omm)
            src1 = omm[k]
            while j<= n_modes - 1
                dst2 = find_mode(c.address, j)
                if (dst2.occnum == 0 || j == src2.mode || j == src1.mode)
                    while i <= n_modes
                        dst1 = find_mode(c.address, i)
                        if (dst1.occnum == 0 || i == src2.mode || i == src1.mode)
                            if !(i == src1.mode && j == src2.mode)# omit same mode as they contribute to diagonal
                                address, val = excitation(c.address, (dst1, dst2,), (src2, src1,))
                                if !iszero(val)
                                    val *= 2*conj(c.operator.test_vector[index((i,j))]) * 
                                        c.operator.test_vector[index((src1.mode,src2.mode))]
                                    # choose next state
                                    if i + 1 <= n_modes
                                        return (address, val), (i + 1, j, k, l)
                                    else
                                        return (address, val), (j + 2, j + 1, k, l)
                                    end
                                end
                            end
                        end
                        i += 1
                    end
                end
                j += 1
                i = j + 1
            end
            k += 1
            j = 1
            i = 2
        end
        l += 1
        k = l + 1
        j = 1
        i = 2
    end
    return nothing
end

"""
    TestTwoParticleDensityGradient(test_vector, jacobian=nothing; normalize=true,
        zeta=0) <: AbstractOperator{SVector{m,T}}
 
An expectation value with this operator yields a gradient `TestTwoParticleDensity` 
for a given test vector. Here `test_vector` is the test vector and `T` is the eltype of 
`test_vector`. If `normalize` is true (default), the vector is normalized before use. 
`zeta` is a expectation value of `TestTwoParticleDensity` for a given `test_vector`.

```math
    ζ = \\langle ∑_{ij,kl} ρ̂ {(2)}_{ij,kl} v_{ij}^* v_{kl} \\rangle
```

There are two cases involves:

- `jacobian <:Nothing`: gradient is calculated for each element of `vec`, 
and the operator is defined as below. A one particle operator constructed 
from a provided test vector. Also, `m` = binomial(# of sites, 2).


```math
    ∂ρ̂ {(2)}/∂v_{kl} = ∑_{ij} (v_{ij}^* (â^†_{i} â^†_{j} â_{l} â_{k} - 
        ζ (δ_{ik}δ_{jl} + δ_{il}δ_{jk}) v_{ij}^*)
```

-`jacobian <:AbstractMatrix`: gradient is calculated with respect 
to the parameters of the given test_vector which has a fix functional form. Here, 
`jacobian` is a matrix of dimension (m × binomial(# of sites, 2)) representing 
transpose of the jacobian of test_vector with respect to its parameters `α₁,α₂,...,αₘ`. 

```math
    ∂_α ρ̂ {(2)}= ∑_{ij, kl} (v_{ij}^* ∂v_{kl}(α)/∂α + 
        ∂v_{ij}^*(α)/∂α v_{kl}) (â^†_{i} â^†_{j} â_{l} â_{k} - 
        ζ (δ_{ik}δ_{jl} + δ_{il}δ_{jk}))
```
Also, in `vᵢⱼ`, i and j are site indices (with i < j). 
"""
struct TestTwoParticleDensityGradient{T,Dim,Zeta,V<:SVector{<:Any,T},J} <: AbstractOperator{SVector{Dim,T}}
    test_vector::V
    jacobian::J
end
function TestTwoParticleDensityGradient(test_vector, jacobian=nothing; zeta = 0, 
    normalize=true)
    
    if jacobian isa Nothing
        T = float(eltype(test_vector))
        dim = length(test_vector)
        if !(test_vector isa SVector)
            test_vector = SVector{dim,T}(test_vector)
        end
        if normalize
            test_vector = test_vector / norm(test_vector)
        end
    else
        if !(test_vector isa AbstractVector && jacobian isa AbstractMatrix)
            error("(test_vector, jacobian) must be (vector, matrix)")
        end
        T = float(eltype(test_vector))
        S = length(test_vector)
        dim = length(jacobian[:,1])
        if !(test_vector isa SVector && jacobian isa SMatrix)
            test_vector = SVector{S,T}(test_vector)
            jacobian = SMatrix{dim,S,T}(jacobian)
        end
        if normalize
            test_vector = test_vector/norm(test_vector)
            jacobian = jacobian/norm(test_vector)
        end
    end
    return TestTwoParticleDensityGradient{T,dim,Float64(zeta),typeof(test_vector),typeof(jacobian)}(
        test_vector,jacobian)
end

function Base.show(io::IO, topd::TestTwoParticleDensityGradient{<:Any,M}) where M
    print(io, "TestTwoParticleDensityGradient(", topd.test_vector, topd.jacobian,)
    if !(norm(topd.test_vector) ≈ 1.0)
        print(io, "; normalize=false")
    end
    print(io, ")")
end

Interfaces.LOStructure(::Type{<:TestTwoParticleDensityGradient}) = IsHermitian()
function Interfaces.allows_address_type(
    od::TestTwoParticleDensityGradient{T,Dim}, ::Type{B}
) where {T,Dim,B}
    M = num_modes(B)
    return B <: SingleComponentFockAddress && length(od.test_vector) == binomial(M,2)
end

struct TestTwoParticleDensityGradientColumn{O,A,T,OMM} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    omm::OMM
end
function Interfaces.operator_column(o::TestTwoParticleDensityGradient, add::A) where {A}
    allows_address_type(o, A) || throw(ArgumentError("Address type not allowed for this operator"))
    omm = occupied_mode_map(add)
    return TestTwoParticleDensityGradientColumn{typeof(o),A,eltype(o),typeof(omm)}(o, add, omm)
end
Interfaces.parent_operator(c::TestTwoParticleDensityGradientColumn) = c.operator
Interfaces.starting_address(c::TestTwoParticleDensityGradientColumn) = c.address
function Interfaces.diagonal_element(c::TestTwoParticleDensityGradientColumn{
    <:TestTwoParticleDensityGradient{<:Any,dim,zeta},<:Any,T}) where {dim,zeta,T}
    M = num_modes(c.address)
    Onr = onr(c.address)
    if dim == binomial(M,2)
        @inbounds val = -conj(c.operator.test_vector) * zeta
    else
        val = zero(T)
        @inbounds for i in 1:M
            for j in 1:i-1
                val += ((conj(c.operator.jacobian[:, index((i, j))]) * 
                    c.operator.test_vector[index((i, j))]) + 
                    (c.operator.jacobian[:, index((i, j))] * 
                    conj(c.operator.test_vector[index((i, j))]))) * 2 * 
                    (2*Onr[i]*Onr[j] - zeta)
            end
        end
    end
    return val::T
end
function Interfaces.num_offdiagonals(c::TestTwoParticleDensityGradientColumn)
    return binomial(length(c.omm),2) * (binomial(num_modes(c.address),2)) - 1
end
function Interfaces.offdiagonals(c::TestTwoParticleDensityGradientColumn{<:Any,A,T}) where {A,T}
    TestTwoParticleDensityGradientOffdiagonals{A,T,typeof(c)}(c)
end
struct TestTwoParticleDensityGradientOffdiagonals{A,T,C}
    column::C
end
Base.eltype(::TestTwoParticleDensityGradientOffdiagonals{A,T}) where {A,T} = Tuple{A,T}
Base.IteratorSize(::TestTwoParticleDensityGradientOffdiagonals) = Base.SizeUnknown()
# Base.length(od::TestTwoParticleDensityGradientOffdiagonals) = num_offdiagonals(od.column)
function Base.iterate(od::TestTwoParticleDensityGradientOffdiagonals{A,T}, state=(2, 1, 2, 1)) where {A,T}
    if od.column.operator.jacobian isa Nothing
        return fullvectorTestTwoParticleDensityGradient(od, state)
    else
        return fixfunctionTestTwoParticleDensityGradient(od, state)
    end
end

@inline function fixfunctionTestTwoParticleDensityGradient(od::TestTwoParticleDensityGradientOffdiagonals{<:Any,T}, state
    ) where {T}
    c = od.column
    omm = c.omm
    n_modes = num_modes(c.address)
    i, j, k, l = state # i,j: mode number for creation, k,l: indices in omm for annihilation
    #  ∑_{ij.kl} v_{ij}^* v_{kl} â^†_{i} â^†_{j} â_{l} â_{k} |address⟩
    while l <= length(omm) - 1
        src2 = omm[l]
        while k <= length(omm)
            src1 = omm[k]
            while j<= n_modes - 1
                dst2 = find_mode(c.address, j)
                if (dst2.occnum == 0 || j == src2.mode || j == src1.mode)
                    while i <= n_modes
                        dst1 = find_mode(c.address, i)
                        if (dst1.occnum == 0 || i == src2.mode || i == src1.mode)
                            if !(i == src1.mode && j == src2.mode)# omit same mode as they contribute to diagonal
                                address, val = excitation(c.address, (dst1, dst2,), (src2, src1,))
                                if !iszero(val)
                                    value = (conj(c.operator.jacobian[:,index((i,j))]) * 
                                        c.operator.test_vector[index((src1.mode,src2.mode))] + 
                                        c.operator.jacobian[:,index((src1.mode,src2.mode))] * 
                                        conj(c.operator.test_vector[index((i,j))])) * 4 * val
                                    # choose next state
                                    if i + 1 <= n_modes
                                        return (address, value::T), (i + 1, j, k, l)
                                    else
                                        return (address, value::T), (j + 2, j + 1, k, l)
                                    end
                                end
                            end
                        end
                        i += 1
                    end
                end
                j += 1
                i = j + 1
            end
            k += 1
            j = 1
            i = 1
        end
        l += 1
        k = l + 1
        j = 1
        i = 1
    end
    return nothing
end

@inline function fullvectorTestTwoParticleDensityGradient(
    od::TestTwoParticleDensityGradientOffdiagonals{<:Any,T}, state
    ) where {T}
    c = od.column
    omm = c.omm
    n_modes = num_modes(c.address)
    i, j, k, l = state # i,j: mode number for creation, k,l: indices in omm for annihilation
    #  ∑_{ij.kl} v_{ij}^* v_{kl} â^†_{i} â^†_{j} â_{l} â_{k} |address⟩
     while l <= length(omm) - 1
        src2 = omm[l]
        while k <= length(omm)
            src1 = omm[k]
            while j<= n_modes - 1
                while i <= n_modes
                    dst = find_mode(c.address, (i,j,))
                    address, val = excitation(c.address, dst, (src2, src1,))
                    if !iszero(val)
                        value = zeros(eltype(T), length(T))
                        @inbounds value[index((src1.mode,src2.mode))] += 2 * 
                                    conj(c.operator.test_vector[index((i,j))] ) .* val
                        # choose next state
                        if i + 1 <= n_modes
                            return (address, T(value)), (i + 1, j, k, l)
                        else
                            return (address, T(value)), (j + 2, j + 1, k, l)
                        end
                    end
                    i += 1
                end
                j += 1
                i = j + 1
            end
            k += 1
            j = 1
            i = 1
        end
        l += 1
        k = l + 1
        j = 1
        i = 1
    end
    return nothing
end
