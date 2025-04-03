"""
    TimeEvolutionStrategy
Abstract type for strategies used with [`TimeEvolution`](@ref)

## Implemented strategies

* [`LinearTimeEvolution`](@ref): evolve with a linear approximation ``\exp(-iHdt) \approx 1 - iHdt``.
* [`SemilinearTimeEvolution`](@ref): evolve with a linear approximation except for diagonal
elements, which are calculated exactly as exponentials.
* [`QuadraticTimeEvolution`](@ref): evolve with a second order approximation
``\exp(-iHdt) \approx 1 - iHdt - \frac{1}{2}H^2dt^2``.
"""
abstract type TimeEvolutionStrategy end

struct LinearTimeEvolution <: TimeEvolutionStrategy end

struct SemilinearTimeEvolution <: TimeEvolutionStrategy end

struct QuadraticTimeEvolution <: TimeEvolutionStrategy end

"""
    TimeEvolution(h::AbstractHamiltonian,s<:TimeEvolutionStrategy,dt) <: AbstractOperator{ComplexF64}

Time evolution operator that evolves a state forward in time by `dt` using the Hamiltonian
`H` and the specified [`TimeEvolutionStrategy`](@ref).
"""
struct TimeEvolution{H<:AbstractHamiltonian,S<:TimeEvolutionStrategy} <: AbstractOperator{ComplexF64}
    hamiltonian::H
    strategy::S
    dt::Float64
end

TimeEvolution(h, s, dt) = TimeEvolution{typeof{h}, typeof{s}}(h, s, dt)

allows_address_type(U::TimeEvolution, add) = allows_address_type(U.hamiltonian, add)

dimension(U::TimeEvolution, addr) = dimension(U.hamiltonian, addr)

function diagonal_element(U::TimeEvolution{<:Any, <:LinearTimeEvolution}, add)
    return 1 - im*U.dt*diagonal_element(U.hamiltonian, add)
end

function diagonal_element(U::TimeEvolution{<:Any, <:SemilinearTimeEvolution}, add)
    return exp(-im*U.dt*diagonal_element(U.hamiltonian, add))
end

function diagonal_element(U::TimeEvolution{<:Any, <:QuadraticTimeEvolution}, add)
    return 1 - im*U.dt*diagonal_element(U.hamiltonian, add) - 0.5*diagonal_element(Product(U.hamiltonian, U.hamiltonian), add)*(U.dt)^2
end

num_offdiagonals(U::TimeEvolution{<:Any, <:LinearTimeEvolution}, add) = num_offdiagonals(U.hamiltonian, add)

num_offdiagonals(U::TimeEvolution{<:Any, <:SemilinearTimeEvolution}, add) = num_offdiagonals(U.hamiltonian, add)

num_offdiagonals(U::TimeEvolution{<:Any, <:QuadraticTimeEvolution}, add) = num_offdiagonals(Product(U.hamiltonian, U.hamiltonian), add)

function get_offdiagonal(U::TimeEvolution{<:Any, <:LinearTimeEvolution}, add, chosen)
    newadd, val = get_offdiagonal(U.hamiltonian, add, chosen)
    val *= -im*U.dt
    return newadd, val
end

function get_offdiagonal(U::TimeEvolution{<:Any, <:SemilinearTimeEvolution}, add, chosen)
    newadd, val = get_offdiagonal(U.hamiltonian, add, chosen)
    val *= -im*U.dt
    return newadd, val
end

function get_offdiagonal(U::TimeEvolution{<:Any, <:QuadraticTimeEvolution}, add, chosen)
    new_add, val_1 = get_offdiagonal(Product(U.hamiltonian, U.hamiltonian), add, chosen)
    val = -im*U.dt*dot(DVec(new_add => 1.0), U.hamiltonian, DVec(add => 1.0)) - 0.5*(U.dt^2)*val_1
    return new_add, val
end


