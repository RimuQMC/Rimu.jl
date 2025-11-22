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
     TestOneParticleDensity(v; normalize=true) <: AbstractOperator{typeof(v)}
A one particle operator constructed from a provided test vector `v`. An expectation value
with this operator yields a lower bound on the largest eigenvalue (and an upper bound on the
smallest eigenvalue) of the one-particle density matrix.
If `normalize` is true (default), the vector is normalized before use.

```math
    ρ̂ = ∑_{ij} v_i^* v_j â^†_{i} â_{j}
```
"""
struct TestOneParticleDensity{T,V<:SVector{<:Any,T},M} <: AbstractObservable{T}
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
function Interfaces.offdiagonals(c::TestOneParticleDensityColumn)
    TestOneParticleDensityOffdiagonals(c)
end
struct TestOneParticleDensityOffdiagonals{C}
    column::C
end
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
