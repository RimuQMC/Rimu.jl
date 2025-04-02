"""
    Product(H1::AbstractHamiltonian, H2::AbstractHamiltonian) <: AbstractHamiltonian

Product of two Hamiltonians.
"""
struct Product{T,H1<:AbstractHamiltonian,H2<:AbstractHamiltonian} <: AbstractHamiltonian{T}
    hamiltonian_1::H1
    hamiltonian_2::H2
end

Product(h1::AbstractHamiltonian{T1},h2::AbstractHamiltonian{T2}) where {T1, T2} = Product{promote_type(T1,T2), typeof(h1), typeof(h2)}(h1,h2)

allows_address_type(H::Product, add) = allows_address_type(H.hamiltonian_1, add) && allows_address_type(H.hamiltonian_2, add)

dimension(H::Product, add) = max(dimension(H.hamiltonian_1, add), dimension(H.hamiltonian_2, add))

function diagonal_element(H::Product, add)
    total = diagonal_element(H.hamiltonian_1, add)*diagonal_element(H.hamiltonian_2, add)
    for (add_1, val_1) in offdiagonals(H.hamiltonian_2, add)
        for (add_2, val_2) in offdiagonals(H.hamiltonian_1, add_1)
            if add_2 == add
                total += val_1*val_2
            end
        end
    end
    return total
end

function num_offdiagonals(H::Product, add::T) where {T}
    ods = T[]
    for (add_1, val_1) in offdiagonals(H.hamiltonian_2, add)
        if !(add_1 in ods) && diagonal_element(H.hamiltonian_1, add) != 0
            push!(ods, add_1)
        end
        for (add_2, val_2) in offdiagonals(H.hamiltonian_1, add_1)
            if !(add_2 in ods) && add_2 != add
                push!(ods, add_2)
            end
        end
    end
    for (add_2, val_2) in offdiagonals(H.hamiltonian_1, add)
        if !(add_2 in ods) && diagonal_element(H.hamiltonian_2, add) != 0
            push!(ods, add_2)
        end
    end
    return length(ods)
end

function get_offdiagonal(H::Product{T}, add::A, chosen) where {T,A}
    total = zero(T)
    ods = A[]
    found = false
    new_add = add
    for (add_1, val_1) in offdiagonals(H.hamiltonian_2, add)
        if !(add_1 in ods) && diagonal_element(H.hamiltonian_1, add_1) != 0
            push!(ods, add_1)
            if !found && length(ods) == chosen
                found = true
                new_add = add_1
                total += val_1*diagonal_element(H.hamiltonian_1, add_1)
            end
        end
        for (add_2, val_2) in offdiagonals(H.hamiltonian_1, add_1)
            if !(add_2 in ods) && add_2 != add
                push!(ods, add_2)
                if !found && length(ods) == chosen
                    found = true
                    new_add = add_2
                end
            end
            if found && add_2 == new_add
                total += val_1*val_2
            end
        end
    end
    for (add_2, val_2) in offdiagonals(H.hamiltonian_1, add)
        if diagonal_element(H.hamiltonian_2, add) == 0
            continue
        end
        if !(add_2 in ods)
            push!(ods, add_2)
            if !found && length(ods) == chosen
                found = true
                new_add = add_2
            end
        end
        if found && add_2 == new_add
            total += val_2*diagonal_element(H.hamiltonian_2, add)
        end
    end
    return new_add, total
end