export BinaryOuter

"""
    BinaryOuter{S,B<:BinaryComponent}

The outer orbit of a hierarchical triple system.

Wraps an ordinary binary component `inner` whose parameters appear in the
parameter tuple with a suffix `S` (e.g. `Symbol("_2")`, giving `PB_2`, `A1_2`,
`T0_2`, ...). On correction, the suffix is stripped so the wrapped component sees
the canonical parameter names it expects (`PB`, `A1`, `T0`, ...).

This corresponds to `BinaryDD2` in `PINT`. The outer orbit must be placed
*before* the inner binary in the component list so that its delay is accumulated
first and propagated into the epoch at which the inner orbit is evaluated. This
reproduces the physical coupling of a hierarchical triple rather than naively
adding two independent binary delays.
"""
struct BinaryOuter{S,B<:BinaryComponent} <: BinaryComponent
    inner::B
end

BinaryOuter(suffix::Symbol, inner::B) where {B<:BinaryComponent} =
    BinaryOuter{suffix,B}(inner)

"""
    strip_param_suffix(params::NamedTuple, ::Val{S})

Build a new `NamedTuple` from `params`, keeping only the entries whose names end
with the suffix `S` and removing that suffix from their names. This lets an
outer-orbit binary read its `_2`-suffixed parameters using the canonical names.

Implemented as a `@generated` function so the renaming happens at compile time
and the result is allocation-free.
"""
@generated function strip_param_suffix(params::NamedTuple{names}, ::Val{S}) where {names,S}
    sfx = string(S)
    newnames = Symbol[]
    exprs = Expr[]
    for n in names
        s = string(n)
        if endswith(s, sfx) && length(s) > length(sfx)
            push!(newnames, Symbol(s[1:(end-length(sfx))]))
            push!(exprs, :(getfield(params, $(QuoteNode(n)))))
        end
    end
    return :(NamedTuple{$(Tuple(newnames))}(($(exprs...),)))
end

function correct_toa(
    binary::BinaryOuter{S},
    toa::TOA,
    toacorr::TOACorrection,
    params::NamedTuple,
) where {S}
    return correct_toa(binary.inner, toa, toacorr, strip_param_suffix(params, Val(S)))
end

function show(io::IO, binary::BinaryOuter{S}) where {S}
    print(io, "BinaryOuter{$(string(S))}($(repr(binary.inner)))")
end
