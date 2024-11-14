"""
    SingleParticleExcitation(i, j) <: AbstractHamiltonian

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
    TwoParticleExcitation(i, j, k, l) <: AbstractHamiltonian

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
    ReducedDensityMatrix(P; ele_type = Float64) <: AbstractOperator{Matrix{ele_type}}

Represent the P-particle reduced density matrix:

```math
\\hat{ρ}^{(n)}_{j_1,...,j_1,k_1,...,k_n} =  \\prod_{i}^{n} â^†_{j_i} \\prod_{l}^{n} â_{k_{n+1-l}}
```

The indices `j_i` and `k_i` (all `<: Int`) represent the single particle sites on a lattice.
These indices are chosen in a specific pattern to ensure that unique elements of the
reduced density matrix are calculated. This calculation will provide sufficient information
for interpreting the largest eigenvalue. Additionally, the indices follow specific patterns
as they run in the following manner:

```math
j_n > ... > j_{i+1} > j_{i} > ... > j_1 \\And k_n> ... > k_{i+1} > k_{i} > ... > k_1
```
This specific pattern has a drawback: for n > 1, addr cannot be of <:BoseFS.

# Examples

```jldoctest
julia> dvec_b = PDVec(BoseFS{2,2}(1,1)=>0.5, BoseFS{2,2}(2,0)=>0.5)
2-element PDVec: style = IsDeterministic{Float64}()
  fs"|2 0⟩" => 0.5
  fs"|1 1⟩" => 0.5

julia> Op1 = ReducedDensityMatrix(1)
ReducedDensityMatrix(1)

julia> dot(dvec_b,Op1,dvec_b)
2×2 Matrix{Float64}:
 0.75      0.353553
 0.353553  0.25

julia> Op2 = ReducedDensityMatrix(2)
ReducedDensityMatrix(2)

julia> dot(dvec_b,Op2,dvec_b)
ERROR: ArgumentError: ReducedDensityMatrix(<:BoseFS, P > 1) is not measurable

julia> dvec_f = PDVec(FermiFS{2,4}(1,1,0,0)=>0.5, FermiFS{2,4}(0,1,1,0)=>0.5)
2-element PDVec: style = IsDeterministic{Float64}()
  fs"|⋅↑↑⋅⟩" => 0.5
  fs"|↑↑⋅⋅⟩" => 0.5

julia> dot(dvec_f,Op2,dvec_f)
6×6 Matrix{Float64}:
 0.25  0.0  0.25  0.0  0.0  0.0
 0.0   0.0  0.0   0.0  0.0  0.0
 0.25  0.0  0.25  0.0  0.0  0.0
 0.0   0.0  0.0   0.0  0.0  0.0
 0.0   0.0  0.0   0.0  0.0  0.0
 0.0   0.0  0.0   0.0  0.0  0.0
```
# See also
* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`SingleParticleExcitation`](@ref)
* [`TwoParticleExcitation`](@ref)
"""
struct ReducedDensityMatrix{TT, P} <: AbstractOperator{Matrix{TT}} end
ReducedDensityMatrix(P::Int; ele_type = Float64) = ReducedDensityMatrix{ele_type, P}()
ReducedDensityMatrix(;P::Int = 1, ele_type = Float64) = ReducedDensityMatrix{ele_type, P}()
function Base.show(io::IO, op::ReducedDensityMatrix{<:Any, P}) where {P}
    print(io, "ReducedDensityMatrix($P)")
end

LOStructure(::Type{<:ReducedDensityMatrix}) = IsHermitian()

function Interfaces.dot_from_right(
    left::AbstractDVec, op::ReducedDensityMatrix{<:Any, P}, right::AbstractDVec
) where {P}
    if all((keytype(left) <: BoseFS, P > 1))
         throw(ArgumentError("ReducedDensityMatrix(<:BoseFS, P > 1) is not measurable"))
    end
    dim = binomial(num_modes(keytype(left)), P)
    T = promote_type(Float64, valtype(left), valtype(right))
    ρ = sum_mutating!(
        zeros(T, (dim, dim)),
        ReducedDensityMatrixCalculcator!{P}(left, dim),
        pairs(right)
    )
    return (ρ .+ ρ') ./ 2
end
# This struct used to calculate matrix elements of `ReducedDensityMatrix`
# It was introduced because passing a function to `sum` in `dot_from_right` was causing
# type instabilites.
"""
    calc! = ReducedDensityMatrixCalculator!{P}(left, dim)
Instantiate a `ReducedDensityMatrixCalculator!{P}` object to calculate matrix elements of
`ReducedDensityMatrix`.

    calc!(rdm, pair)

Add the contribution of `pair` to the reduced density matrix to `rdm`.
"""
struct ReducedDensityMatrixCalculcator!{P,D}
    left::D
    dim::Int

    ReducedDensityMatrixCalculcator!{P}(left, dim) where {P} = new{P,typeof(left)}(left, dim)
end

function (calc!::ReducedDensityMatrixCalculcator!{P})(result, pair) where {P}
    addr_right, val_right = pair
    left = calc!.left
    T = eltype(result)

    for j in axes(result, 2)
        dsts = find_mode(addr_right, vertices(j, Val(P)))
        for i in axes(result, 1)
            srcs = reverse(find_mode(addr_right, vertices(i, Val(P))))

            addr_left, elem = excitation(addr_right, dsts, srcs)
            @inbounds result[i, j] += T(conj(left[addr_left]) * elem * val_right)
        end
    end
    return result
end
