
import Rimu.Hamiltonians: num_offdiagonals, diagonal_element, get_offdiagonal

"""
    G2MomCorrelator(d::Int) <: AbstractOperator{ComplexF64}

Two-body correlation operator representing the density-density
correlation at distance `d`. It returns a `Complex` value.

Correlation within a single component:
```math
\\hat{G}^{(2)}(d) = \\frac{1}{M}\\sum_{spqr=1}^M e^{-id(p-q)2π/M} a^†_{s} a^†_{p}  a_q a_r δ_{s+p,q+r}
```

The diagonal element, where `(p-q)=0`, is
```math
\\frac{1}{M}\\sum_{k,p=1}^M a^†_{k} b^†_{p}  b_p a_k .
```

# Arguments
- `d::Integer`: the distance between two particles.


# See also

* [`Rimu.G2RealCorrelator`](@ref)
* [`Rimu.G2RealSpace`](@ref)
* [`Rimu.AbstractOperator`](@ref)
* [`Rimu.AllOverlaps`](@ref)
"""
struct G2MomCorrelator{C} <: AbstractOperator{ComplexF64}
    d::Int
end
# The type parameter `C` is not used here, but may be used for future extensions.
# It is kept here for consistency with `RimuLegacyHamiltonians.jl`.
function G2MomCorrelator(d::Int)
    return G2MomCorrelator{3}(d)
end

function Rimu.Interfaces.allows_address_type(g2m::G2MomCorrelator, ::Type{A}) where {A}
    return num_modes(A) > g2m.d && A <: SingleComponentFockAddress
end

function Base.show(io::IO, g::G2MomCorrelator{3})
    # 3 is the default value for the type parameter
    print(io, "G2MomCorrelator($(g.d))")
end

function num_offdiagonals(g::G2MomCorrelator, addr::SingleComponentFockAddress)
    m = num_modes(addr)
    singlies, doublies = num_singly_doubly_occupied_sites(addr)
    return singlies * (singlies - 1) * (m - 2) + doublies * (m - 1)
end

function diagonal_element(g::G2MomCorrelator, addr::SingleComponentFockAddress)
    M = num_modes(addr)
    onrep = onr(addr)
    gd = 0
    for p in 1:M
        iszero(onrep[p]) && continue
        for k in 1:M
            gd += onrep[k] * onrep[p] # a†_p a_p a†_k a_k
        end
    end
    return ComplexF64(gd / M)
end

function get_offdiagonal(
    g::G2MomCorrelator,
    addr::A,
    chosen,
)::Tuple{A,ComplexF64} where {A<:SingleComponentFockAddress}
    M = num_modes(addr)
    new_add, gamma, Δp = momentum_transfer_excitation(addr, chosen, OccupiedModeMap(addr))
    gd = exp(-im * g.d * Δp * 2π / M) * gamma
    return new_add, ComplexF64(gd / M)
end
