# Based on Public.jl
# Based on Compat.jl
# License: MIT
# https://github.com/JuliaLang/Compat.jl
# https://github.com/JuliaLang/Compat.jl/blob/master/src/compatmacro.jl
# https://github.com/JuliaLang/Compat.jl/blob/a62d95906f3c16c9fa0fe8369a6613dcfd2a4659/src/compatmacro.jl

# https://github.com/JuliaLang/julia/pull/50105
@static if Base.VERSION >= v"1.11.0-DEV.469"
    macro public(symbols_expr::Union{Symbol,Expr})
        symbols = _get_symbols(symbols_expr)
        return esc(Expr(:public, symbols...))
    end
else
    macro public(symbols_expr::Union{Symbol,Expr})
        return nothing
    end
end

_get_symbols(sym::Symbol) = [sym]

function _get_symbols(expr::Expr)
    # `expr` must either be a "valid macro expression" or a tuple expression.

    # If `expr` is a "valid macro expression", then we simply return it:
    if _is_valid_macro_expr(expr::Expr)
        return [expr.args[1]]
    end

    # If `expr` is not a "valid macro expression", then we check to make sure
    # that it is a tuple expression:
    if expr.head != :tuple
        msg = """
            Invalid expression head `$(expr.head)` in expression `$(expr)`.
            Try `@public foo, bar, @hello, @world`
        """
        throw(ArgumentError(msg))
    end

    # Now that we know that `expr` is a tuple expression, we iterate over
    # each element of the tuple
    num_symbols = length(expr.args)
    symbols = Vector{Symbol}(undef, num_symbols)
    for (i, arg) in enumerate(expr.args)
        if arg isa Symbol
            symbols[i] = arg
        elseif _valid_macro(arg)
            symbols[i] = arg.args[1]
        else
            throw(ArgumentError("cannot mark `$arg` as public. Try `@compat public foo, bar`."))
        end
    end
    return symbols
end

# Return true if and only if `expr` is a valid macro call with no arguments.
#
# Example of "good" input: `@foo`
function _is_valid_macro_expr(expr::Expr)
    # `expr` must be a `:macrocall` expression:
    Meta.isexpr(expr, :macrocall) || return false

    # `expr` must have exactly two arguments:
    (length(expr.args) == 2) || return false

    # The first argument must be a Symbol:
    (expr.args[1] isa Symbol) || return false

    # The first argument must begin with `@`
    arg1_str = string(expr.args[1])
    (arg1_str[1] == '@') || return false

    # The first argument must have length >= 2
    # (because otherwise the first argument would just be `@`, which doesn't
    # make sense)
    (length(arg1_str) >= 2) || return false

    # The second argument must be a `LineNumberNode`
    (expr.args[2] isa LineNumberNode) || return false

    return true
end
