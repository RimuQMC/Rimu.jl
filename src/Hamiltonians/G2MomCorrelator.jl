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
    new_add, gamma, Δp = momentum_transfer_excitation(addr, chosen, occupied_mode_map(addr))
    gd = exp(-im * g.d * Δp * 2π / M) * gamma
    return new_add, ComplexF64(gd / M)
end

"""
    G1RealtoMomCorrelator{T=Float64}() <: AbstractObservable{Vector{T}}

One-body operator that calculates the particle density for the momentum ``k =
{-π + 2π/M, -π + 4π/M, ..., π - 2π/M, π}``. It returns a `Complex` value.

Correlation within a single component:
```math
\\hat{G}^{(1)}(k) = \\frac{1}{M}\\sum_{m,n=1}^M e^{-i k(m-n)} a^†_{m} a_n
```
See also [`single_particle_density`](@ref), [`SingleParticleDensity`](@ref),
[`SingleParticleExcitation`](@ref), [`TwoParticleExcitation`](@ref).
"""
struct G1RealtoMomCorrelator{T} <: AbstractObservable{Vector{T}} end

G1RealtoMomCorrelator() = G1RealtoMomCorrelator{ComplexF64}()

function Base.show(io::IO, ::G1RealtoMomCorrelator{T}) where {T}
    print(io, "G1RealtoMomCorrelator{$T}()")
end

LOStructure(::Type{<:G1RealtoMomCorrelator}) = IsHermitian()

function Interfaces.dot_from_right(
    left::AbstractDVec, op::G1RealtoMomCorrelator{TT}, right::AbstractDVec
) where {TT}
    M = num_modes(keytype(left))
    ρ = sum_mutating!(
        Vector{TT}([0.0 for _ in 1:M]),
        G1RealtoMomCorrelatorCalculcator!{TT}(left),
        pairs(right)
    )
    return ρ / M
end

# This struct is used to calculate matrix elements of `G1RealtoMomCorrelator`
# It was introduced because passing a function to `sum` in `dot_from_right` was causing
# type instabilites.
"""
    calc! = G1RealtoMomCorrelatorCalculator!{}(left)
Instantiate a `G1RealtoMomCorrelatorCalculator!{}` object to calculate vector elements of
`G1RealtoMomCorrelator`.

    calc!(G2, pair)

Add the contribution of `pair` to the G2 correlator to `G2`.
"""
struct G1RealtoMomCorrelatorCalculcator!{TT,D}
    left::D

    G1RealtoMomCorrelatorCalculcator!{TT}(left) where {TT} = new{TT,typeof(left)}(left)
end

function (calc!::G1RealtoMomCorrelatorCalculcator!{TT})(result, pair) where {TT}
    addr_right, val_right = pair
    occ = occupied_mode_map(addr_right)
    left = calc!.left
    M =length(result)
    x, y = 1,1
    while x < length(occ) + 1
        srcs = occ[x]
        while y < M + 1
            dsts = find_mode(addr_right, y)
            if dsts.occnum == 0 || y == srcs.mode
                addr_left, elem = excitation(addr_right, (dsts,), (srcs,))
                k = 1
                while k < M + 1
                    result[k] += TT(conj(left[addr_left]) * elem * val_right * 
                                    exp(-im * 2π * (k - M/2)*(y - srcs.mode)/M))
                    k += 1
                end
            end
            y += 1
        end
        x += 1
        y = 1
    end
    return result 
end

"""
    G2RealtoMomCorrelator{T=Float64}() <: AbstractObservable{Matrix{T}}

Two-body correlation operator that calculates the density-density 
correlation between the momentums given by the set ``k =
{-π + 2π/M, -π + 4π/M, ..., π - 2π/M, π}``. It returns a `Complex` value.

Correlation within a single component:
```math
\\hat{G}^{(2)}(k_1, k_2) = \\frac{1}{M}\\sum_{m,n,o,p=1}^M 
                    e^{-i [k_1(m-p) + k_2(n -o)} a^†_{m} a^†_{n} a_o a_p
```
See also [`single_particle_density`](@ref), [`SingleParticleDensity`](@ref),
[`SingleParticleExcitation`](@ref), [`TwoParticleExcitation`](@ref).
"""
struct G2RealtoMomCorrelator{T} <: AbstractObservable{Matrix{T}} end

G2RealtoMomCorrelator() = G2RealtoMomCorrelator{ComplexF64}()

function Base.show(io::IO, ::G2RealtoMomCorrelator{T}) where {T}
    print(io, "G2RealtoMomCorrelator{$T}()")
end

LOStructure(::Type{<:G2RealtoMomCorrelator}) = IsHermitian()

function Interfaces.dot_from_right(
    left::AbstractDVec, op::G2RealtoMomCorrelator{TT}, right::AbstractDVec
) where {TT}
    M = num_modes(keytype(left))
    k_vals = SVector{M,Float64}([i for i in -M+2:2:M]*(π/M))
    ρ = sum_mutating!(
        zeros(TT, (M,M)),
        G2RealtoMomCorrelatorCalculcator!{TT,M}(left, k_vals),
        pairs(right)
    )
    return ρ / M^2
end

# This struct is used to calculate matrix elements of `G2RealtoMomCorrelator`
# It was introduced because passing a function to `sum` in `dot_from_right` was causing
# type instabilites.
"""
    calc! = G2RealtoMomCorrelatorCalculator!{TT,M}(left, k_vals)
Instantiate a `G2RealtoMomCorrelatorCalculator!{}` object to calculate matrix elements of
`G2RealtoMomCorrelator`.

    calc!(G2, pair)

Add the contribution of `pair` to the G2 correlator to `G2`.
"""
struct G2RealtoMomCorrelatorCalculcator!{TT,M,D}
    left::D
    k_vals::SVector{M}

    G2RealtoMomCorrelatorCalculcator!{TT,M}(left, k_vals) where {TT,M} = new{TT,M,typeof(left)}(left, k_vals)
end

function (calc!::G2RealtoMomCorrelatorCalculcator!{TT,M})(result, pair) where {TT,M}
    addr_right, val_right = pair
    omm = occupied_mode_map(addr_right)
    left = calc!.left
    k_vals = calc!.k_vals
    i,j,k,l = 2,1,2,1
    while l <= length(omm) - 1
        src2 = omm[l]
        while k <= length(omm)
            src1 = omm[k]
            while j<= M - 1
                dst2 = find_mode(addr_right, j)
                if (dst2.occnum == 0 || j == src2.mode || j == src1.mode)
                    while i <= M
                        dst1 = find_mode(addr_right, i)
                        if (dst1.occnum == 0 || i == src2.mode || i == src1.mode)
                            addr_left, elem = excitation(addr_right, (dst1, dst2,), (src2, src1,))
                            if !iszero(elem)
                                in2 = 1
                                while in2 < M + 1
                                    in1 = 1
                                    while in1 < M + 1
                                        result[in1, in2] += TT(conj(left[addr_left]) * elem * val_right * 
                                                        (exp(-im * (k_vals[in1] * (i-src1.mode) + k_vals[in2] * (j-src2.mode))) -
                                                        exp(-im * (k_vals[in1] * (i-src2.mode) + k_vals[in2] * (j-src1.mode))) -
                                                        exp(-im * (k_vals[in1] * (j-src1.mode) + k_vals[in2] * (i-src2.mode))) + 
                                                        exp(-im * (k_vals[in2] * (i-src1.mode) + k_vals[in1] * (j-src2.mode)))))
                                        in1 += 1
                                    end
                                    in2 += 1
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
        
    return result 
end

