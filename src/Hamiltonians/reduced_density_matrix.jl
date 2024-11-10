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
    ReducedDensityMatrix(addr::SingleComponentFockAddress; n = 1) <: AbstractOperator

Represent the n-particle reduced density matrix:

```math
ρ̂^{(n)}_{j_1,...,j_1,k_1,...,k_n} =  \\prod_{i}^{n} â^†_{j_i} \\prod_{l}^{n} â_{k_{n+1-l}}
```

Where `j_i` and `k_i` (all `<: Int`) specify the single particle sites on a lattice.
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
struct ReducedDensityMatrix{N} <: AbstractOperator{Matrix{Float64}} end
ReducedDensityMatrix(N) = ReducedDensityMatrix{N}()
ReducedDensityMatrix(;n=1) = ReducedDensityMatrix{n}()
function Base.show(io::IO, op::ReducedDensityMatrix{N}) where {N}
    print(io, "ReducedDensityMatrix($N)")
end

LOStructure(::Type{<:ReducedDensityMatrix}) = IsHermitian()

function Interfaces.dot_from_right(
    left::AbstractDVec, op::ReducedDensityMatrix{N}, right::AbstractDVec
) where {N}
    dim = binomial(num_modes(keytype(left)), N)
    ρ = sum(ReducedDensityMatrixCalculcator{N}(left, dim), pairs(right))
    return (ρ .+ ρ') ./ 2

# This struct used to calculate matrix elements of `ReducedDensityMatrix`
# It was introduced because passing a function to `sum` in `dot_from_right` was causing
# type instabilites.
struct ReducedDensityMatrixCalculcator{N,D}
    left::D
    dim::Int

    ReducedDensityMatrixCalculcator{N}(left, dim) where {N} = new{N,typeof(left)}(left, dim)
end

function (calc::ReducedDensityMatrixCalculcator{N})(pair) where {N}
    addr_right, val_right = pair
    left = calc.left
    dim = calc.dim

    T = promote_type(Float64, valtype(left), typeof(val_right))
    result = zeros(T, (dim, dim))
    for j in axes(result, 2)
        dsts = find_mode(addr_right, vertices(j, Val(N)))
        for i in axes(result, 1)
            srcs = reverse(find_mode(addr_right, vertices(i, Val(N))))

            addr_left, elem = excitation(addr_right, dsts, srcs)
            @inbounds result[i, j] += T(conj(left[addr_left]) * elem * val_right)
        end
    end
    return result
end
