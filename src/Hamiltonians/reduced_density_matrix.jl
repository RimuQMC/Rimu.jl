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
    ReducedDensityMatrix(addr::SingleComponentFockAddress; n = 1, ele_type = Float64) <: AbstractOperator

Represent the n-particle reduced density matrix:

```math
ρ̂^{(n)}_{j_1,...,j_1,k_1,...,k_n} =  \\prod_{i}^{n} â^†_{j_i} \\prod_{i}^{n} â_{n+1-k}
```

Where `j_i` and `k_i` (all `<: Int`) specify the single particle sites on a lattice. Also, ``ele_type`` specifies
the type of each element in the reduced density matrix.
Additionally, the indices run in the following manners:

```math
j_n> ... > j_{i+1} > j_{i} > ... > j_1 and k_n> ... > k_{i+1} > k_{i} > ... > k_1
```

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`SingleParticleExcitation`](@ref)
* [`TwoParticleExcitation`](@ref)
"""
struct ReducedDensityMatrix{TT} <: AbstractOperator{TT}
    M::Int
    n::Int
end
function ReducedDensityMatrix(addr::SingleComponentFockAddress; n = 1, ele_type = Float64)
    M = num_modes(addr)
    return ReducedDensityMatrix{ele_type}(M,n)
end
function Base.show(io::IO, g2::ReducedDensityMatrix)
    print(io, "ReducedDensityMatrix(num_modes = $(g2.M), n=$(g2.n))")
end

LOStructure(::Type{<:ReducedDensityMatrix}) = IsHermitian()

function Interfaces.allows_address_type(
    g2::ReducedDensityMatrix, A::Type{<:AbstractDVec}
)
    result = g2.M == num_modes(A)
    return result
end

function Interfaces.dot_from_right(left::AbstractDVec, g2::ReducedDensityMatrix, right::AbstractDVec)
    M = num_modes(keytype(left))
    n = g2.n
    dim = binomial(M,n)
    ρ = zeros(valtype(right),(dim,dim))
    ρ .+= ele_ReducedDensityMatrix(ρ, left, right, M, Val(n))
    return (ρ.+ρ')./2
end

function ele_ReducedDensityMatrix(matrix_element, left, right, M, ::Val{n}) where {n}
    t1=0
    t2=0
    for ij in Iterators.product(ntuple(q1->(n-q1+1:M),Val(n))...)
        if all(ntuple(q1->ij[q1+1]<ij[q1],Val(n-1)))
            t1+=1
            t2=0
            for kl in Iterators.product(ntuple(q2->(n-q2+1:M),Val(n))...)
                if all(ntuple(q1->kl[q1+1]<kl[q1],Val(n-1)))
                    t2+=1
                    matrix_element[t1,t2] += sum(pairs(right)) do (k,v)
                        xs=find_mode(k,reverse(ij))
                        ys=find_mode(k,kl)
                        if  all(x -> x.occnum == 0, xs) ||  all(y -> y.occnum == 1, ys)
                            nv, α = excitation(k,xs,ys)
                            conj(left[nv]) * v * α
                        else
                            0.0
                        end 
                    end
                end
            end
        end
    end
    return matrix_element
end
