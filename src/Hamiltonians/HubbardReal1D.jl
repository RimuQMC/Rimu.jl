"""
    HubbardReal1D(;[u=1.0, t=1.0])

Implements a one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1)
```

# Parameters
- `u`: the interaction parameter
- `t`: the hopping strength

"""
struct HubbardReal1D{TT,A<:AbstractFockAddress,U,T} <: AbstractHamiltonian{TT}
    add::A
end

function HubbardReal1D(addr; u=1.0, t=1.0)
    U, T = promote(float(u), float(t))
    return HubbardReal1D{typeof(U),typeof(addr),U,T}(addr)
end

function Base.show(io::IO, h::HubbardReal1D)
    print(io, "HubbardReal1D($(h.add); u=$(h.u), t=$(h.t))")
end

function starting_address(h::HubbardReal1D)
    return getfield(h, :add)
end

LOStructure(::Type{<:HubbardReal1D{<:Real}}) = HermitianLO()

Base.getproperty(h::HubbardReal1D, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::HubbardReal1D{<:Any,<:Any,U}, ::Val{:u}) where U = U
Base.getproperty(h::HubbardReal1D{<:Any,<:Any,<:Any,T}, ::Val{:t}) where T = T
Base.getproperty(h::HubbardReal1D, ::Val{:add}) = getfield(h, :add)

function diagME(h::HubbardReal1D, address::BoseFS)
    h.u * bosehubbardinteraction(address) / 2
end

function numOfHops(::HubbardReal1D, address::BoseFS)
    return numberlinkedsites(address)
end

function hop(h::HubbardReal1D, add::BoseFS, chosen)
    naddress, onproduct = hopnextneighbour(add, chosen)
    return naddress, - h.t * sqrt(onproduct)
end
