"""
    Product(o1::AbstractOperator, o2::AbstractOperator) <: AbstractOperator

Product of two operators.
"""
struct Product{T,O1<:AbstractOperator,O2<:AbstractOperator} <: AbstractOperator{T}
    op_1::O1
    op_2::O2
end

Product(o1::AbstractOperator{T1},o2::AbstractOperator{T2}) where {T1, T2} = Product{promote_type(T1,T2), typeof(o1), typeof(o2)}(o1,o2)

allows_address_type(O::Product, add) = allows_address_type(O.op_1, add) && allows_address_type(O.op_2, add)

dimension(O::Product, add) = max(dimension(O.op_1, add), dimension(O.op_2, add))

function diagonal_element(O::Product, add)
    total = diagonal_element(O.op_1, add)*diagonal_element(O.op_2, add)
    for (add_1, val_1) in offdiagonals(O.op_2, add)
        for (add_2, val_2) in offdiagonals(O.op_1, add_1)
            if add_2 == add
                total += val_1*val_2
            end
        end
    end
    return total
end

function num_offdiagonals(O::Product, add::T) where {T}
    ods = T[]
    for (add_1, val_1) in offdiagonals(O.op_2, add)
        if !(add_1 in ods) && diagonal_element(O.op_1, add) != 0
            push!(ods, add_1)
        end
        for (add_2, val_2) in offdiagonals(O.op_1, add_1)
            if !(add_2 in ods) && add_2 != add
                push!(ods, add_2)
            end
        end
    end
    for (add_2, val_2) in offdiagonals(O.op_1, add)
        if !(add_2 in ods) && diagonal_element(O.op_2, add) != 0
            push!(ods, add_2)
        end
    end
    return length(ods)
end

function get_offdiagonal(O::Product{T}, add::A, chosen) where {T,A}
    total = zero(T)
    ods = A[]
    found = false
    new_add = add
    for (add_1, val_1) in offdiagonals(O.op_2, add)
        if !(add_1 in ods) && diagonal_element(O.op_1, add_1) != 0
            push!(ods, add_1)
            if !found && length(ods) == chosen
                found = true
                new_add = add_1
                total += val_1*diagonal_element(O.op_1, add_1)
            end
        end
        for (add_2, val_2) in offdiagonals(O.op_1, add_1)
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
    for (add_2, val_2) in offdiagonals(O.op_1, add)
        if diagonal_element(O.op_2, add) == 0
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
            total += val_2*diagonal_element(O.op_2, add)
        end
    end
    return new_add, total
end