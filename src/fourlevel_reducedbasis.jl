module fourlevel_reducedbasis

using QuantumOptics, Base.Cartesian
using Combinatorics: combinations

export ReducedAtomBasis, reducedatomstate, reducedsigmaplus,
    reducedsigmaminus, reducedsigmaplussigmaminus, reducedsigmapm

import Base: ==

using ..interaction, ..AtomicArrays

const Excitation = Pair{Int,Int}

"""
    ReducedAtomBasis(N::Int, M::Int; MS::Int=0, T::Type{<:Int}=Int)

Basis for N four-level atoms restricted to the subspace with
`MS ≤ excitation number ≤ M`.
"""
struct ReducedAtomBasis{N,M,MS,T<:Int} <: Basis
    shape::Vector{T}              # length-1 vector holding dimension
    indexMapper::Vector{Pair{Vector{Excitation},T}}

    function ReducedAtomBasis(N::T, M::T; MS::T=zero(T)) where T<:Int
        @assert 1 ≤ N
        @assert 0 ≤ MS ≤ M               "MS ≤ M must hold"
        dim   = sum(binomial(N,k)*3^k for k in MS:M)
        states = Vector{Excitation}[]
        for k in MS:M
            for atoms in combinations(1:N,k)
                for ms in Iterators.product(ntuple(_->(-1:1),k)...)
                    push!(states, [Excitation(a, m) for (a,m) in zip(atoms,ms)])
                end
            end
        end
        indexMapper = states .=> collect(1:dim)
        new{N,M,MS,T}([dim], indexMapper)
    end
end

ReducedAtomBasis(N::Int,M::Int,MS::Int) = ReducedAtomBasis{Int}(N,M;MS)

function Base.getproperty(b::ReducedAtomBasis{N,M,MS}, s::Symbol) where {N,M,MS}
    if s === :N
        N
    elseif s === :M
        M
    elseif s === :MS
        MS
    else
        getfield(b, s)
    end
end

==(b1::T, b2::T) where T<:ReducedAtomBasis = true

# Base.show(io, b::ReducedAtomBasis) = print(io,
#     "ReducedAtom(N=$(b.N), M=$(b.M), MS=$(b.MS))")
function Base.show(stream::IO, b::ReducedAtomBasis)
    write(stream, "ReducedAtom(N=$(b.N), M=$(b.M), MS=$(b.MS))")
end


index(b::ReducedAtomBasis, ex::Vector{Excitation}) = begin
    # canonical sorting to avoid duplicates like [(2,0),(1,-1)]
    key = sort!(copy(ex); by=x->x.first)
    i   = findfirst(p->p.first==key, b.indexMapper)
    i === nothing && throw(BoundsError(b,key))
    b.indexMapper[i].second
end

reducedatomstate(b, ex::Vector{Excitation}) =
    basisstate(b, index(b, ex))

# Operators

"""
    reducedatomtransition(b::ReducedAtomBasis, to::Vector{Int}, from::Vector{Int})
Transition operator ``|\\mathrm{to}⟩⟨\\mathrm{from}|``, where to and from are given as vectors denoting the excitations.
"""
function reducedatomtransition(b::ReducedAtomBasis, to::Vector{Excitation}, from::Vector{Excitation})
    op = SparseOperator(b)
    op.data[index(b, to), index(b, from)] = 1.
    op
end
# reducedatomtransition(b::ReducedAtomBasis, to, from) = reducedatomtransition(b, convert(Vector{Int}, to), convert(Vector{Int}, from))
# reducedatomtransition(b::ReducedAtomBasis, to::Int, from::Int) = reducedatomtransition(b, [to], [from])

function reducedsigmaplus(b::ReducedAtomBasis, j::Int, m::Int)  # m ∈ {-1,0,1}
    @assert 1 ≤ j ≤ b.N && m ∈ (-1:1)
    op = SparseOperator(b)
    # loop over m-1 → m excitation manifolds
    for k in b.MS+1:b.M
        to, from = transition_idx(b, j, m, k)
        @inbounds for n in eachindex(to)
            op.data[to[n], from[n]] = 1.0
        end
    end
    op
end

reducedsigmaminus(b,j,m) = dagger(reducedsigmaplus(b,j,m))
reducedsigmapm(b,j,m)      = reducedsigmaplus(b,j,m) * reducedsigmaminus(b,j,m)

function reducedsigmaplussigmaminus(b,i,m; j=i, mp=m)
    op = SparseOperator(b)
    for s1 in b.indexMapper
        ((j,mp) in s1.first) || continue
        for s2 in b.indexMapper
            ((i,m) in s2.first) || continue
            op.data[s2.second, s1.second] = 1.0
        end
    end
    op
end

# ────────────────────────────────────────────────────────────────────────────
# Utility: normalise any user input into Vector{Excitation}
# Accepts vectors of tuples, pairs, or Excitation as well as single instances.
# ────────────────────────────────────────────────────────────────────────────
function _to_vec_exc(ex)
    # ground state
    isempty(ex)              && return Excitation[]
    ex isa Excitation        && return Excitation[ex]
    ex isa Tuple{Int,Int}    && return Excitation[(Excitation(ex[1], ex[2]))]
    ex isa Pair{<:Integer,<:Integer} &&
        return Excitation[(Excitation(first(ex), last(ex)))]
    ex isa AbstractVector    && return Excitation[_to_vec_exc(e)[1] for e in ex]
    throw(ArgumentError("Cannot convert $ex to a Vector{Excitation}."))
end

# ────────────────────────────────────────────────────────────────────────────
# index(basis, excitations) → 1-based linear index into Hilbert space
# ────────────────────────────────────────────────────────────────────────────
"""
    index(b::ReducedAtomBasis, excitations)

Return the position of the basis state **defined by `excitations`** in the
internal ordering of `b`.

* `excitations` may be  
  – a `Vector{Excitation}`                       – preferred  
  – a `Vector` of `(atom, m)` tuples or pairs    – auto-converted  
  – a single `Excitation`, tuple, or pair        – for one excited atom  
  – `[]` or `()`                                 – the ground state.

`BoundsError` is thrown when the requested state is absent (e.g. too many
excitations for this `M`,`MS`).
"""
function index(b::ReducedAtomBasis, excitations)
    vec = _to_vec_exc(excitations)                     # canonical input
    k   = length(vec)
    @assert b.MS ≤ k ≤ b.M "Excitation count $k not allowed by this basis"

    # Sort by atom index to ensure unique keys irrespective of input order
    key = sort(vec; by = first)

    pos = findfirst(p -> p.first == key, b.indexMapper)
    pos === nothing && throw(BoundsError(b, key))
    return b.indexMapper[pos].second                  # 1-based linear index
end

# Short aliases for common call patterns
index(b::ReducedAtomBasis, ex::Excitation)           = index(b, [ex])
index(b::ReducedAtomBasis, ex::Tuple{Int,Int})       = index(b, [ex])
index(b::ReducedAtomBasis, ex::Pair{<:Integer,<:Integer}) = index(b, [ex])

"""
    transition_idx(b, j, m, k)

Return two vectors `(to, from)` with the linear indices of all basis states

* `to`   : states in the **k-excitation** manifold that **contain atom `j`
           in sub-level `m`**;
* `from` : states in the **(k − 1)-excitation** manifold that **do *not***
           contain any excitation on atom `j`.

Used by σ⁺ to promote atom `j` from ground → |1,m⟩ while the total
excitation count grows from `k-1` to `k`.
"""
function transition_idx(b::ReducedAtomBasis, j::Int, m::Int, k::Int)
    @assert 1 ≤ j ≤ b.N
    @assert m ∈ (-1:1)
    @assert b.MS + 1 ≤ k ≤ b.M

    target = Excitation(j, m)                          # (atom, m) as Pair

    # 1)  k–excitation states *with* (j,m)
    to = [p.second for p in b.indexMapper
                 if length(p.first) == k &&
                    any(e -> e == target, p.first)]

    # 2) (k−1)–excitation states *without* atom j at all
    from = [p.second for p in b.indexMapper
                    if length(p.first) == k-1 &&
                       all(e -> first(e) != j, p.first)]

    return to, from
end

end #  module