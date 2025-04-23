"""
    shift_lattice(is)
Circular shift contiguous indices `is` in interval `[M÷2, M÷2)` such that set starts with 0,
where `M=length(is)`.

Inverse operation: [`shift_lattice_inv`](@ref). Used in [`HubbardReal1DEP`](@ref) and
[`HubbardMom1DEP`](@ref)
"""
shift_lattice(is) = circshift(is, cld(length(is),2))

"""
    shift_lattice_inv(js)
Circular shift indices starting with 0 into a contiguous set in interval `[M÷2, M÷2)`,
where `M=length(js)`.

Inverse operation of [`shift_lattice`](@ref). Used in [`HubbardReal1DEP`](@ref) and
[`HubbardMom1DEP`](@ref)
"""
shift_lattice_inv(js) = circshift(js, fld(length(js),2))

"""
    HubbardReal1DEP(address; u=1.0, t=1.0, v_ho=1.0)

Implements a one-dimensional Bose Hubbard chain in real space with external potential.

```math
\\hat{H} = - \\sum_i \\left(t a_i^† a_{i+1} + t^* a_{i+1}^† a_i \\right) + 
\\sum_i ϵ_i n_i + \\frac{u}{2}\\sum_i n_i (n_i-1)
```

# Arguments

* `address`: the starting address, defines number of particles and sites.
* `u`: the interaction parameter.
* `t`: the hopping strength.
* `v_ho`: strength of the external harmonic oscillator potential ``ϵ_i = v_{ho} i^2``.
The first index is `i=0` and the maximum of the potential occurs in the centre of the
lattice.

# See also

* [`HubbardReal1D`](@ref)
* [`HubbardMom1D`](@ref)
* [`ExtendedHubbardReal1D`](@ref)

"""
struct HubbardReal1DEP{TT,A<:AbstractFockAddress,U,T,M} <: AbstractHamiltonian{TT}
    address::A
    ep::SVector{M,TT}
end

function HubbardReal1DEP(address::SingleComponentFockAddress{<:Any,M}; u=1.0, t=1.0, v_ho=1.0) where M
    U, T, V = promote(float(u), float(t), float(v_ho))
    # js = range(1-cld(M,2); length=M)
    is = range(-fld(M,2); length=M) # [-M÷2, M÷2) including left boundary
    js = shift_lattice(is) # shifted such that js[1] = 0
    potential = SVector{M}(V*j^2 for j in js)
    return HubbardReal1DEP{typeof(U),typeof(address),U,T,M}(address, potential)
end

function Base.show(io::IO, h::HubbardReal1DEP)
    compact_addr = repr(h.address, context=:compact => true) # compact print address
    print(io, "HubbardReal1DEP($(compact_addr); u=$(h.u), t=$(h.t), v_ho=$(h.ep[2]))")
end

LOStructure(::Type{<:HubbardReal1DEP{<:Real}}) = IsHermitian()
function LOStructure(::Type{<:HubbardReal1DEP{<:Complex,<:Any,U,T}}) where {U,T}
    if iszero(imag(U))
        return IsHermitian() # still Hermitian with complex t
    else
        return AdjointKnown()
    end
end

function LinearAlgebra.adjoint(h::HubbardReal1DEP{TT,A,U,T,M}) where {TT<:Complex,A,U,T,M}
    return HubbardReal1DEP{TT,A,conj(U),T,M}(h.address, conj(h.ep))
end

Base.getproperty(h::HubbardReal1DEP, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::HubbardReal1DEP{<:Any,<:Any,U}, ::Val{:u}) where U = U
Base.getproperty(h::HubbardReal1DEP{<:Any,<:Any,<:Any,T}, ::Val{:t}) where T = T
Base.getproperty(h::HubbardReal1DEP, ::Val{:address}) = getfield(h, :address)
Base.getproperty(h::HubbardReal1DEP, ::Val{:ep}) = getfield(h, :ep)
Base.getproperty(h::HubbardReal1DEP, ::Val{:boundary_condition}) = :periodic

starting_address(h::HubbardReal1DEP) = h.address

dimension(::HubbardReal1DEP, address) = number_conserving_dimension(address)

function num_offdiagonals(::HubbardReal1DEP, address::SingleComponentFockAddress)
    return 2 * num_occupied_modes(address)
end

function diagonal_element(h::HubbardReal1DEP, address::SingleComponentFockAddress)
    sum(occupied_modes(address)) do index
        occnum, mode = index
        h.u * occnum * (occnum - 1) / 2 + h.ep[mode] * occnum
    end
end

function get_offdiagonal(h::HubbardReal1DEP, address::SingleComponentFockAddress, chosen)
    return _get_offdiagonal_hubbard_real_1D(h, address, chosen)
end
