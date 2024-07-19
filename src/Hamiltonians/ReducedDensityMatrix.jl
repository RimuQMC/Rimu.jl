"""
    DensityMatrixDiagonal(mode; component=0) <: AbstractHamiltonian

Represent a diagonal element of the single-particle density:

```math
\\hat{n}_{i,σ} = \\hat a^†_{i,σ} \\hat a_{i,σ}
```

where ``i`` is the `mode` and ``σ`` is the `component`. If `component` is zero, the sum over
all components is computed.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
"""
struct DensityMatrixDiagonal{C} <: AbstractHamiltonian{Float64}
    mode::Int
end
DensityMatrixDiagonal(mode; component=0) = DensityMatrixDiagonal{component}(mode)

function diagonal_element(dmd::DensityMatrixDiagonal{1}, add::SingleComponentFockAddress)
    return float(find_mode(add, dmd.mode).occnum)
end
function diagonal_element(dmd::DensityMatrixDiagonal{0}, add::SingleComponentFockAddress)
    return float(find_mode(add, dmd.mode).occnum)
end

function diagonal_element(dmd::DensityMatrixDiagonal{0}, add::CompositeFS)
    return float(sum(a -> find_mode(a, dmd.mode).occnum, add.components))
end
function diagonal_element(dmd::DensityMatrixDiagonal{C}, add::CompositeFS) where {C}
    comp = add.components[C]
    return float(find_mode(comp, dmd.mode).occnum)
end

function diagonal_element(dmd::DensityMatrixDiagonal{0}, add::BoseFS2C)
    return float(find_mode(add.bsa, dmd.mode).occnum + find_mode(add.bsb, dmd.mode).occnum)
end
function diagonal_element(dmd::DensityMatrixDiagonal{1}, add::BoseFS2C)
    comp = add.bsa
    return float(find_mode(comp, dmd.mode).occnum)
end
function diagonal_element(dmd::DensityMatrixDiagonal{2}, add::BoseFS2C)
    comp = add.bsb
    return float(find_mode(comp, dmd.mode).occnum)
end

num_offdiagonals(dmd::DensityMatrixDiagonal, _) = 0
LOStructure(::Type{<:DensityMatrixDiagonal}) = IsDiagonal()

"""
    SingleParticleReducedDensityMatrix(i, j) <: AbstractHamiltonian

Represent a {i,j} element of the single-particle reduced density matrix:

```math
\\hat{ρ}^{(1)}_{i,j} = \\langle \\psi | \\hat a^†_{i} \\hat a_{j} | \\psi \\rangle
```

where ``i`` and ``j`` are the `mode` and ``| \\psi \\rangle`` is the `state-ket`.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`TwoParticleReducedDensityMatrix`](@ref)
"""

struct SingleParticleReducedDensityMatrix <: AbstractHamiltonian{Float64}
    I::Int
    J::Int
end

function Base.show(io::IO, spd::SingleParticleReducedDensityMatrix)
    print(io, "SingleParticleReducedDensityMatrix($(spd.I), $(spd.J))")
end

Rimu.LOStructure(::Type{T}) where T<:SingleParticleReducedDensityMatrix = AdjointUnknown()

function Rimu.diagonal_element(spd::SingleParticleReducedDensityMatrix, add::SingleComponentFockAddress)
    src = find_mode(add, spd.J)
    dst = find_mode(add,spd.I)
    address, value = excitation(add, (dst,), (src,))
    if spd.I == spd.J
        return value
    else
        return 0.0
    end
end

function Rimu.num_offdiagonals(spd::SingleParticleReducedDensityMatrix, address::SingleComponentFockAddress)
    if spd.I == spd.J
        return 0
    else
        return 1
    end
end

function Rimu.get_offdiagonal(spd::SingleParticleReducedDensityMatrix, add::SingleComponentFockAddress, chosen)
    src = find_mode(add, spd.J)
    dst = find_mode(add,spd.I)
    address, value = excitation(add, (dst,), (src,))
    return address, value
end

"""
    SingleParticleReducedDensityMatrix(i, j, k, l) <: AbstractHamiltonian

Represent a {ij, kl} element of the two-particle reduced density matrix:

```math
\\hat{ρ}^{(2)}_{ij, kl} = \\langle \\psi | \\hat a^†_{i} \\hat a^†_{j} \\hat a_{l} \\hat a_{k} | \\psi \\rangle
```

where ``i``, ``j``, ``k``, and ``l`` are the `mode` and ``| \\psi \\rangle`` is the `state-ket`.

# See also

* [`single_particle_density`](@ref)
* [`SingleParticleDensity`](@ref)
* [`SingleParticleReducedDensityMatrix`](@ref)
"""

struct TwoParticleReducedDensityMatrix <: AbstractHamiltonian{Float64}
    I::Int
    J::Int
    K::Int
    L::Int
end

function Base.show(io::IO, spd::TwoParticleReducedDensityMatrix)
    print(io, "TwoParticleReducedDensityMatrix($(spd.I), $(spd.J), $(spd.K), $(spd.L))")
end

Rimu.LOStructure(::Type{T}) where T<:TwoParticleReducedDensityMatrix = AdjointUnknown()

function Rimu.diagonal_element(spd::TwoParticleReducedDensityMatrix, add::SingleComponentFockAddress)
    src = find_mode(add, (spd.L, spd.K))
    dst = find_mode(add,(spd.I, spd.J))
    address, value = excitation(add, (dst...,), (src...,))
    if (spd.I, spd.J) == (spd.K, spd.L) || (spd.I, spd.J) == (spd.L, spd.K)
        return value
    else
        return 0.0
    end
end

function Rimu.num_offdiagonals(spd::TwoParticleReducedDensityMatrix, address::SingleComponentFockAddress)
    if (spd.I, spd.J) == (spd.K, spd.L) || (spd.I, spd.J) == (spd.L, spd.K)
        return 0
    else
        return 6
    end
end

function Rimu.get_offdiagonal(spd::TwoParticleReducedDensityMatrix, add::BoseFS, chosen)
    if chosen<=2
        src = find_mode(add, (spd.L, spd.K))
        dst = find_mode(add,(spd.I, spd.J))
    elseif chosen <=4 && chosen>2
        src = find_mode(add, (spd.K, spd.L))
        dst = find_mode(add,(spd.I, spd.J))
    else
        src = find_mode(add, (spd.L, spd.K))
        dst = find_mode(add,(spd.J, spd.I))
    end
    if chosen%2 == 0
        src, dst = dst, src
    end
    address, value = excitation(add, (dst...,), (src...,))
    return address, value/6
end

function Rimu.get_offdiagonal(spd::TwoParticleReducedDensityMatrix, add::FermiFS, chosen)
    if chosen <= 2
        src = find_mode(add, (spd.L, spd.K))
        dst = find_mode(add,(spd.I, spd.J))
    elseif chosen <= 4 && chosen > 2
        src = find_mode(add, (spd.K, spd.L))
        dst = find_mode(add,(spd.I, spd.J))
    else
        src = find_mode(add, (spd.L, spd.K))
        dst = find_mode(add,(spd.J, spd.I))
    end
    if chosen%2 == 0
        src, dst = dst, src
    end
    address, value = excitation(add, (dst...,), (src...,))
    if chosen <= 2
        return address, value/6
    else
        return address, -value/6
    end
end

