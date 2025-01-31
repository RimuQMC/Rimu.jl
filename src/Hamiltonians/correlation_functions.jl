"""
    circshift_dot(arr1, arr2, inds)

Fast, non-allocation version of

```julia
dot(arr1, circshift(arr2, inds))
```
"""
function circshift_dot(arr1, arr2, inds)
    _circshift_dot!(arr1, (), arr2, (), axes(arr2), Tuple(inds))
end

# The following is taken from Julia's implementation of circshift.
@inline function _circshift_dot!(
    dst, rdest, src, rsrc,
    inds::Tuple{AbstractUnitRange,Vararg{Any}},
    shiftamt::Tuple{Integer,Vararg{Any}}
)
    ind1, d = inds[1], shiftamt[1]
    s = mod(d, length(ind1))
    sf, sl = first(ind1)+s, last(ind1)-s
    r1, r2 = first(ind1):sf-1, sf:last(ind1)
    r3, r4 = first(ind1):sl, sl+1:last(ind1)
    tinds, tshiftamt = Base.tail(inds), Base.tail(shiftamt)

    return _circshift_dot!(dst, (rdest..., r1), src, (rsrc..., r4), tinds, tshiftamt) +
        _circshift_dot!(dst, (rdest..., r2), src, (rsrc..., r3), tinds, tshiftamt)
end
@inline function _circshift_dot!(dst, rdest, src, rsrc, inds, shiftamt)
    return dot(view(dst, rdest...), view(src, rsrc...))
end

"""
    G2RealCorrelator(d::Int) <: AbstractOperator{Float64}

Two-body operator for density-density correlation between sites separated by `d`
with `0 ≤ d < M`.
```math
    \\hat{G}^{(2)}(d) = \\frac{1}{M} \\sum_i^M \\hat{n}_i (\\hat{n}_{i+d} - \\delta_{0d}).
```
Assumes a one-dimensional lattice with periodic boundary conditions where
```math
    \\hat{G}^{(2)}(-M/2 \\leq d < 0) = \\hat{G}^{(2)}(|d|),
```
```math
    \\hat{G}^{(2)}(M/2 < d < M) = \\hat{G}^{(2)}(M - d),
```
and normalisation
```math
    \\sum_{d=0}^{M-1} \\langle \\hat{G}^{(2)}(d) \\rangle = \\frac{N (N-1)}{M}.
```

For multicomponent basis, calculates correlations between all particles equally,
equivalent to stacking all components into a single Fock state.

# Arguments
- `d::Integer`: distance between sites.

# See also

* [`HubbardReal1D`](@ref)
* [`G2RealSpace`](@ref)
* [`G2MomCorrelator`](@ref)
* [`AbstractOperator`](@ref)
* [`AllOverlaps`](@ref)
"""
struct G2RealCorrelator{D} <: AbstractOperator{Float64}
end

G2RealCorrelator(d::Int) = G2RealCorrelator{d}()

function Base.show(io::IO, ::G2RealCorrelator{D}) where {D}
    print(io, "G2RealCorrelator($D)")
end

LOStructure(::Type{<:G2RealCorrelator}) = IsDiagonal()
function Interfaces.allows_address_type(::G2RealCorrelator{D}, ::Type{A}) where {D,A}
    return num_modes(A) > D
end

function diagonal_element(::G2RealCorrelator{0}, addr::SingleComponentFockAddress)
    M = num_modes(addr)
    v = onr(addr)
    return dot(v, v .- 1) / M
end
function diagonal_element(::G2RealCorrelator{D}, addr::SingleComponentFockAddress) where {D}
    M = num_modes(addr)
    d = mod(D, M)
    v = onr(addr)
    return circshift_dot(v, v, (d,)) / M
end

function diagonal_element(::G2RealCorrelator{0}, addr::CompositeFS)
    M = num_modes(addr)
    v = sum(map(onr, addr.components))
    return dot(v, v .- 1) / M
end
function diagonal_element(::G2RealCorrelator{D}, addr::CompositeFS) where {D}
    M = num_modes(addr)
    d = mod(D, M)
    v = sum(map(onr, addr.components))
    return circshift_dot(v, v, (d,)) / M
end

num_offdiagonals(::G2RealCorrelator, ::SingleComponentFockAddress) = 0
num_offdiagonals(::G2RealCorrelator, ::CompositeFS) = 0

"""
    G2RealSpace(geometry::CubicGrid, σ=1, τ=1; sum_components=false) <: AbstractOperator{SArray}

Two-body operator for density-density correlation for all [`Displacements`](@ref) ``d⃗`` in the specified `geometry`.

```math
    \\hat{G}^{(2)}_{σ,τ}(d⃗) = \\frac{1}{M} ∑_{i⃗} n̂_{σ,i⃗} (n̂_{τ,i⃗+d⃗} - δ_{0⃗,d⃗}δ_{σ,τ}).
```

For multicomponent addresses, `σ` and `τ` control the components involved. Alternatively,
`sum_components` can be set to `true`, which treats all particles as belonging to the same
component.

# Examples

```jldoctest
julia> geom = CubicGrid(2, 2);

julia> g2 = G2RealSpace(geom)
G2RealSpace(CubicGrid((2, 2), (true, true)), 1,1)

julia> diagonal_element(g2, BoseFS(2,0,1,1))
2×2 StaticArraysCore.SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 0.5  1.0
 0.5  1.0

julia> g2_cross = G2RealSpace(geom, 1, 2)
G2RealSpace(CubicGrid((2, 2), (true, true)), 1,2)

julia> g2_sum = G2RealSpace(geom, sum_components=true)
G2RealSpace(CubicGrid((2, 2), (true, true)); sum_components=true)

julia> diagonal_element(g2, fs"|⇅⋅↓↑⟩")
2×2 StaticArraysCore.SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 0.0  0.0
 0.0  0.5

julia> diagonal_element(g2_cross, fs"|⇅⋅↓↑⟩")
2×2 StaticArraysCore.SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 0.25  0.25
 0.25  0.25

julia> diagonal_element(g2_sum, fs"|⇅⋅↓↑⟩")
2×2 StaticArraysCore.SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 0.5  1.0
 0.5  1.0
```

# See also

* [`CubicGrid`](@ref)
* [`HubbardRealSpace`](@ref)
* [`G2RealCorrelator`](@ref)
* [`G2MomCorrelator`](@ref)
* [`AbstractOperator`](@ref)
* [`AllOverlaps`](@ref)
"""
struct G2RealSpace{A,B,G<:CubicGrid,S} <: AbstractOperator{S}
    geometry::G
    init::S
end
function G2RealSpace(geometry::CubicGrid, σ::Int=1, τ::Int=σ; sum_components=false)
    if σ < 1 || τ < 1
        throw(ArgumentError("`σ` and `τ` must be positive integers"))
    end
    if sum_components
        if σ ≠ 1 || τ ≠ 1
            throw(ArgumentError("`σ` or `τ` can't be set if `sum_components=true`"))
        end
        σ = τ = 0
    end

    init = zeros(SArray{Tuple{size(geometry)...}})
    return G2RealSpace{σ,τ,typeof(geometry),typeof(init)}(geometry, init)
end

function Base.show(io::IO, g2::G2RealSpace{A,B}) where {A,B}
    print(io, "G2RealSpace($(g2.geometry), $A,$B)")
end
function Base.show(io::IO, g2::G2RealSpace{0,0})
    print(io, "G2RealSpace($(g2.geometry); sum_components=true)")
end

LOStructure(::Type{<:G2RealSpace}) = IsDiagonal()
VectorInterface.scalartype(::G2RealSpace) = Float64 # needed because eltype is a vector

function Interfaces.allows_address_type(
    g2::G2RealSpace{C1,C2}, A::Type{<:AbstractFockAddress}
) where {C1,C2}
    result = prod(size(g2.geometry)) == num_modes(A)
    return result && 0 ≤ C1 ≤ num_components(A) && 0 ≤ C2 ≤ num_components(A)
end

num_offdiagonals(g2::G2RealSpace, _) = 0

@inline function _g2_diagonal_element(
    g2::G2RealSpace{A,B}, onr1::SArray, onr2::SArray
) where {A, B}
    geo = g2.geometry
    result = g2.init

    @inbounds for i in eachindex(result)
        res_i = 0.0
        displacement = Displacements(geo)[i]

        # Case of n_i(n_i - 1) on the same component
        if A == B && iszero(displacement)
            onr1_minus_1 = max.(onr1 .- 1, 0)
            result = setindex(result, dot(onr2, onr1_minus_1), i)
        else
            result = setindex(result, circshift_dot(onr2, onr1, displacement), i)
        end
    end
    return result ./ length(geo)
end
function diagonal_element(g2::G2RealSpace{A,A}, addr::SingleComponentFockAddress) where {A}
    onr1 = onr(addr, g2.geometry)
    return _g2_diagonal_element(g2, onr1, onr1)
end
function diagonal_element(g2::G2RealSpace{A,B}, addr::CompositeFS) where {A,B}
    onr1 = onr(addr.components[A], g2.geometry)
    onr2 = onr(addr.components[B], g2.geometry)
    return _g2_diagonal_element(g2, onr1, onr2)
end
function diagonal_element(g2::G2RealSpace{0,0}, addr::CompositeFS)
    onr1 = sum(x -> onr(x, g2.geometry), addr.components)
    return _g2_diagonal_element(g2, onr1, onr1)
end

"""
    SuperfluidCorrelator(d::Int) <: AbstractOperator{Float64}

Operator for extracting superfluid correlation between sites separated by a distance `d`
with `0 ≤ d < M`:

```math
    \\hat{C}_{\\text{SF}}(d) = \\frac{1}{M} \\sum_{i}^{M} a_{i}^{\\dagger} a_{i + d}
```
Assumes a one-dimensional lattice with ``M`` sites and periodic boundary conditions. ``M``
is also the number of modes in the Fock state address.

# Usage
Superfluid correlations can be extracted from a Monte Carlo calculation by wrapping
`SuperfluidCorrelator` with [`AllOverlaps`](@ref) and passing into
[`ProjectorMonteCarloProblem`](@ref) with the `replica` keyword argument. For an example
with a similar use of [`G2RealCorrelator`](@ref) see
[G2 Correlator Example](https://RimuQMC.github.io/Rimu.jl/previews/PR227/generated/G2-example.html).


See also [`HubbardReal1D`](@ref), [`G2RealCorrelator`](@ref), [`AbstractOperator`](@ref),
and [`AllOverlaps`](@ref).
"""
struct SuperfluidCorrelator{D} <: AbstractOperator{Float64}
end

SuperfluidCorrelator(d::Int) = SuperfluidCorrelator{d}()

function Base.show(io::IO, ::SuperfluidCorrelator{D}) where {D}
    print(io, "SuperfluidCorrelator($D)")
end
function Interfaces.allows_address_type(::SuperfluidCorrelator{D}, ::Type{A}) where {D,A}
    return num_modes(A) > D
end

function num_offdiagonals(::SuperfluidCorrelator, addr::SingleComponentFockAddress)
    return num_occupied_modes(addr)
end

function get_offdiagonal(
    ::SuperfluidCorrelator{D}, addr::SingleComponentFockAddress, chosen
) where {D}
    src = find_occupied_mode(addr, chosen)
    dst = find_mode(addr, mod1(src.mode + D, num_modes(addr)))
    address, value = excitation(addr, (dst,), (src,))
    return address, value / num_modes(addr)
end

function diagonal_element(::SuperfluidCorrelator{0}, addr::SingleComponentFockAddress)
    return num_particles(addr) / num_modes(addr)
end
function diagonal_element(
    ::SuperfluidCorrelator{D}, addr::SingleComponentFockAddress
) where {D}
    return 0.0
end


"""
    StringCorrelator(d::Int; address=nothing, type=nothing) <: AbstractOperator{T}

Operator for extracting string correlation between lattice sites on a one-dimensional
Hubbard lattice separated by a distance `d` with `0 ≤ d < M`

```math
    Ĉ_{\\text{string}}(d) = \\frac{1}{M} \\sum_{j}^{M} δ n̂_j
                                         (e^{i π \\sum_{j ≤ k < j + d} δ n̂_k}) δ n̂_{j+d}
```
Here, ``δ n̂_j = n̂_j - n̄`` is the boson number deviation from the mean filling
number and ``n̄ = N/M`` is the mean filling number of lattice sites with ``N`` particles and
``M`` lattice sites (or modes).

Assumes a one-dimensional lattice with periodic boundary conditions. For usage
see [`SuperfluidCorrelator`](@ref) and [`AllOverlaps`](@ref).

The default element type `T` is `ComplexF64`. This can be overridden with the `type` keyword
argument. If an `address` is provided, then `T` is calculated from the address type.
It is set to `ComplexF64` for non-integer filling numbers, and to `Float64` for integer
filling numbers or if `d==0`.

See also [`HubbardReal1D`](@ref), [`G2RealCorrelator`](@ref), [`SuperfluidCorrelator`](@ref),
[`AbstractOperator`](@ref), and [`AllOverlaps`](@ref).
"""
struct StringCorrelator{D,T} <: AbstractOperator{T}
end

function StringCorrelator(d::Int; address=nothing, type=nothing)
    if type === nothing
        if iszero(d)
            type = Float64
        elseif address === nothing
            type = ComplexF64
        else
            M = num_modes(address)
            N = num_particles(address)
            if !ismissing(N) && iszero(N % M)
                type = Float64
            else
                type = ComplexF64
            end
        end
    end
    return StringCorrelator{d,type}()
end

function Base.show(io::IO, ::StringCorrelator{D,T}) where {D,T}
    print(io, "StringCorrelator($D; type=$T)")
end
function Interfaces.allows_address_type(::StringCorrelator{D}, ::Type{A}) where {D,A}
    return num_modes(A) > D && A <: SingleComponentFockAddress
end

LOStructure(::Type{<:StringCorrelator}) = IsDiagonal()

function diagonal_element(::StringCorrelator{0}, addr::SingleComponentFockAddress)
    M = num_modes(addr)
    N = num_particles(addr)
    n̄ = N/M
    v = onr(addr)

    result = 0.0
    for i in eachindex(v)
        result += (v[i] - n̄)^2
    end

    return result / M
end

num_offdiagonals(::StringCorrelator, ::SingleComponentFockAddress) = 0

function diagonal_element(
    ::StringCorrelator{D,T}, addr::SingleComponentFockAddress
) where {D,T}
    M = num_modes(addr)
    N = num_particles(addr)
    d = mod(D, M)

    if !ismissing(N) && iszero(N % M)
        return T(_string_diagonal_real(d, addr))
    else
        return T(_string_diagonal_complex(d, addr))
    end
end

function _string_diagonal_complex(d, addr)
    M = num_modes(addr)
    N = num_particles(addr)
    n̄ = N/M
    v = onr(addr)

    result = ComplexF64(0)
    for i in eachindex(v)
        phase_sum = sum((v[mod1(k, M)] - n̄) for k in i:1:(i+d-1))

        result += (v[i] - n̄) * exp(pi * im * phase_sum) * (v[mod1(i + d, M)] - n̄)
    end

    return result / M
end
function _string_diagonal_real(d, addr)
    M = num_modes(addr)
    N = num_particles(addr)
    n̄ = N ÷ M
    v = onr(addr)

    result = 0.0
    for i in eachindex(v)
        phase_sum = sum((v[mod1(k, M)] - n̄) for k in i:1:(i+d-1))

        result += (v[i] - n̄) * (-1)^phase_sum * (v[mod1(i + d, M)] - n̄)
    end

    return result / M
end
