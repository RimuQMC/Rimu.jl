###
### This file contains methods defined on `AbstractDVec`
### The type definition and relevant methods are found in the file "src/Interfaces/dictvectors.jl"
###
function Base.show(io::IO, dvec::AbstractDVec)
    summary(io, dvec)
    limit, _ = displaysize()
    for (i, p) in enumerate(pairs(dvec))
        if length(dvec) > i > limit - 4
            print(io, "\n  ⋮   => ⋮")
            break
        else
            print(io, "\n  ")
            show(IOContext(io, :compact => true), p[1])
            print(io, " => ")
            show(IOContext(io, :compact => true), p[2])
        end
    end
end

###
### Types
###
Base.keytype(::Type{<:AbstractDVec{K}}) where {K} = K
Base.keytype(dv::AbstractDVec) = keytype(typeof(dv))
Base.valtype(::Type{<:AbstractDVec{<:Any,V}}) where {V} = V
Base.valtype(dv::AbstractDVec) = valtype(typeof(dv))
Base.eltype(::Type{<:AbstractDVec{K,V}}) where {K,V} = Pair{K,V}
Base.eltype(dv::AbstractDVec) = eltype(typeof(dv))

Base.isreal(v::AbstractDVec) = valtype(v)<:Real
Base.ndims(::AbstractDVec) = 1

###
### copy*, zero*
###
zero!(v::AbstractDVec) = empty!(v)

Base.zero(dv::AbstractDVec) = empty(dv)

function Base.similar(dvec::AbstractDVec, args...; kwargs...)
    return sizehint!(empty(dvec, args...; kwargs...), length(dvec))
end

@inline function Base.copyto!(w::AbstractDVec, v)
    sizehint!(w, length(v))
    for (key, val) in pairs(v)
        w[key] = val
    end
    return w
end
@inline function Base.copy!(w::AbstractDVec, v)
    empty!(w)
    return copyto!(w, v)
end
Base.copy(v::AbstractDVec) = copyto!(empty(v), v)

###
### Linear algebra
###
function Base.sum(f, x::AbstractDVec)
    return sum(f, values(x))
end

function LinearAlgebra.norm(x::AbstractDVec, p::Real=2)
    if p === 1
        return float(sum(abs, values(x)))
    elseif p === 2
        return sqrt(sum(abs2, values(x)))
    elseif p === Inf
        return float(mapreduce(abs, max, values(x), init=real(zero(valtype(x)))))
    else
        error("$p-norm of $(typeof(x)) is not implemented.")
    end
end

@inline function LinearAlgebra.mul!(w::AbstractDVec, v::AbstractDVec, α)
    empty!(w)
    sizehint!(w, length(v))
    for (key, val) in pairs(v)
        w[key] = val * α
    end
    return w
end

# copying multiplication with scalar
function Base.:*(α::T, x::AbstractDVec{<:Any,V}) where {T,V}
    return mul!(similar(x, promote_type(T, V)), x, α)
end
Base.:*(x::AbstractDVec, α) = α * x

"""
    add!(x::AbstractDVec,y::AbstactDVec)

Inplace add `x+y` and store result in `x`.
"""
@inline function add!(x::AbstractDVec{K}, y::AbstractDVec{K}) where {K}
    for (k, v) in pairs(y)
        x[k] += v
    end
    return x
end
add!(x::AbstractVector, y) = x .+= values(y)

@inline function LinearAlgebra.axpy!(α, x::AbstractDVec, y::AbstractDVec)
    for (k, v) in pairs(x)
        y[k] += α * v
    end
    return y
end

function LinearAlgebra.rmul!(x::AbstractDVec, α)
    for (k, v) in pairs(x)
        x[k] = v * α
    end
    return x
end

# BLAS-like function: y = α*x + β*y
function LinearAlgebra.axpby!(α, x::AbstractDVec, β, y::AbstractDVec)
    rmul!(y, β) # multiply every non-zero element
    axpy!(α, x, y)
    return y
end

function LinearAlgebra.dot(x::AbstractDVec, y::AbstractDVec)
    # try to save time by looking for the smaller vec
    if isempty(x) || isempty(y)
        return zero(promote_type(valtype(x), valtype(y)))
    elseif length(x) < length(y)
        result = sum(pairs(x)) do (key, val)
            conj(val) * y[key]
        end
    else
        result = sum(pairs(y)) do (key, val)
            conj(x[key]) * val
        end
    end
    return result # the type is promote_type(T1,T2) - could be complex!
end

Base.isequal(x::AbstractDVec{K1}, y::AbstractDVec{K2}) where {K1,K2} = false
function Base.isequal(x::AbstractDVec{K}, y::AbstractDVec{K}) where {K}
    x === y && return true
    length(x) != length(y) && return false
    all(pairs(x)) do (k, v)
        isequal(y[k], v)
    end
    return true
end

Base.:(==)(x::AbstractDVec, y::AbstractDVec) = isequal(x, y)

"""
    walkernumber(w)

Compute the number of walkers in `w`. It is used for updating the shift. Overload this
function for modifying population control.

In most cases `walkernumber(w)` is identical to `norm(w,1)`. For `AbstractDVec`s with
complex coefficients it reports the one norm separately for the real and the imaginary part
as a `ComplexF64`. See [`Norm1ProjectorPPop`](@ref).
"""
walkernumber(w) = walkernumber(StochasticStyle(w), w)
# use StochasticStyle trait for dispatch
walkernumber(::StochasticStyle, w) = dot(Norm1ProjectorPPop(), w)
# complex walkers as two populations
# the following default is fast and generic enough to be good for real walkers and

function Base.:*(h::AbstractHamiltonian{E}, v::AbstractDVec{K,V}) where {E, K, V}
    T = promote_type(E, V) # allow for type promotion
    w = empty(v, T) # allocate new vector; non-mutating version
    for (key, val) in pairs(v)
        w[key] += diagonal_element(h, key)*val
        off = offdiagonals(h, key)
        for (add, elem) in off
            w[add] += elem*val
        end
    end
    return w
end

# three argument version
function LinearAlgebra.mul!(w::AbstractDVec, h::AbstractHamiltonian, v::AbstractDVec)
    empty!(w)
    for (key,val) in pairs(v)
        w[key] += diagonal_element(h, key)*val
        for (add,elem) in offdiagonals(h, key)
            w[add] += elem*val
        end
    end
    return w
end

"""
    dot(x, H::AbstractHamiltonian, v)

Evaluate `x⋅H(v)` minimizing memory allocations.
"""
function LinearAlgebra.dot(x::AbstractDVec, LO::AbstractHamiltonian, v::AbstractDVec)
    return dot(LOStructure(LO), x, LO, v)
end

LinearAlgebra.dot(::AdjointUnknown, x, LO::AbstractHamiltonian, v) = dot_from_right(x,LO,v)
# default for LOs without special structure: keep order

function LinearAlgebra.dot(::LOStructure, x, LO::AbstractHamiltonian, v)
    if length(x) < length(v)
        return conj(dot_from_right(v, LO', x)) # turn args around to execute faster
    else
        return dot_from_right(x,LO,v) # original order
    end
end

"""
    Hamiltonians.dot_from_right(x, LO, v)
Internal function evaluates the 3-argument `dot()` function in order from right
to left.
"""
function dot_from_right(
    x::AbstractDVec{K,T}, LO::AbstractHamiltonian{U}, v::AbstractDVec{K,V}
) where {K,T,U,V}
    result = zero(promote_type(T, U, V))
    for (key, val) in pairs(v)
        result += conj(x[key]) * diagonal_element(LO, key) * val
        for (add, elem) in offdiagonals(LO, key)
            result += conj(x[add]) * elem * val
        end
    end
    return result
end
