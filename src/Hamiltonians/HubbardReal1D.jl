"""
    HubbardReal1D(address; u=1.0, t=1.0)

Implements a one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = - \\sum_i \\left(t a_i^† a_{i+1} + t^* a_{i+1}^† a_i \\right) + 
\\frac{u}{2}\\sum_i n_i (n_i-1)
```

# Arguments

* `address`: the starting address, defines number of particles and sites.
* `u`: the interaction parameter.
* `t`: the hopping strength.

# See also

* [`HubbardMom1D`](@ref)
* [`ExtendedHubbardReal1D`](@ref)

"""
struct HubbardReal1D{TT,A<:AbstractFockAddress,U,T} <: AbstractHamiltonian{TT}
    add::A
end

function HubbardReal1D(addr; u=1.0, t=1.0)
    U, T = promote(float(u), float(t))
    return HubbardReal1D{typeof(U),typeof(addr),U,T}(addr)
end

function Base.show(io::IO, h::HubbardReal1D)
    io = IOContext(io, :compact => true)
    print(io, "HubbardReal1D(")
    show(io, h.add)
    print(io, "; u=$(h.u), t=$(h.t))")
end

function starting_address(h::HubbardReal1D)
    return getfield(h, :add)
end

dimension(::HubbardReal1D, address) = number_conserving_dimension(address)

LOStructure(::Type{<:HubbardReal1D{<:Real}}) = IsHermitian()
function LOStructure(::Type{<:HubbardReal1D{<:Complex,<:Any,U,T}}) where {U,T}
    if iszero(imag(U))
        return IsHermitian() # still Hermitian with complex t
    else
        return AdjointKnown()
    end
end

function LinearAlgebra.adjoint(h::HubbardReal1D{TT,A,U,T}) where {TT<:Complex,A,U,T}
    return HubbardReal1D{TT,A,conj(U)+0im,T}(h.add)
end

Base.getproperty(h::HubbardReal1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::HubbardReal1D{<:Any,<:Any,U}, ::Val{:u}) where U = U
Base.getproperty(h::HubbardReal1D{<:Any,<:Any,<:Any,T}, ::Val{:t}) where T = T
Base.getproperty(h::HubbardReal1D, ::Val{:add}) = getfield(h, :add)
Base.getproperty(h::HubbardReal1D, ::Val{:boundary_condition}) = :periodic

function num_offdiagonals(::HubbardReal1D, address::SingleComponentFockAddress)
    return 2 * num_occupied_modes(address)
end

function diagonal_element(h::HubbardReal1D, address::SingleComponentFockAddress)
    h.u * bose_hubbard_interaction(address) / 2
end

function get_offdiagonal(h::HubbardReal1D, add::SingleComponentFockAddress, chosen)
    return _get_offdiagonal_hubbard_real_1D(h, add, chosen)
end

function _get_offdiagonal_hubbard_real_1D(h, add, chosen)
    naddress, onproduct = hopnextneighbour(add, chosen, h.boundary_condition)
    if h.t isa Complex && chosen % 2 != 0
        return naddress, - conj(h.t) * onproduct
    else
        return naddress, - h.t * onproduct
    end
end
