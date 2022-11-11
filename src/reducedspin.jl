module reducedspin

using QuantumOptics, Base.Cartesian
using Combinatorics: combinations

export ReducedSpinBasis, reducedspintransition, reducedspinstate, reducedsigmap,
    reducedsigmam, reducedsigmax, reducedsigmay, reducedsigmaz, reducedsigmapsigmam

import Base: ==

using ..interaction, ..AtomicArrays

"""
    ReducedSpinBasis(N, M)
Basis for a system of N spin 1/2 systems, up to the M'th excitation.
"""
struct ReducedSpinBasis{N, M, MS, T<:Int} <: Basis
    shape::Vector{T}
    indexMapper::Vector{Pair{Vector{T},T}}

    function ReducedSpinBasis(N::T, M::T, MS::T) where T<:Int
        if N < 1
            throw(DimensionMismatch())
        end
        if M < 1
            throw(DimensionMissmatch())
        end
        if M < MS
            throw(DimensionMissmatch())
        end

        dim = sum(binomial(N, k) for k=MS:M)
        inds = Vector{T}[]
        for k=MS:M
            append!(inds, collect(combinations(1:N,k)))
        end
        @assert length(inds)==dim
        indexMapper = (inds .=> [1:dim;])
        new{N, M, MS, T}([dim], indexMapper)
    end
end
ReducedSpinBasis(N::Int, M::Int) = ReducedSpinBasis(N, M, 0)
function Base.show(stream::IO, b::ReducedSpinBasis)
    write(stream, "ReducedSpin(N=$(b.N), M=$(b.M), MS=$(b.MS))")
end
function Base.getproperty(b::ReducedSpinBasis{N,M,MS}, s::Symbol) where {N,M,MS}
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

==(b1::T, b2::T) where T<:ReducedSpinBasis = true

"""
    index(b::ReducedSpinBasis, x:Vector{Int})
    Get the state index given excitation's positions.
"""
function index(b::ReducedSpinBasis, x::Vector{Int})
    @assert length(x) <= b.M
    @assert length(x) >= b.MS
    x_ = sort(x)
    i = findfirst(y->y[1]==x_, b.indexMapper)
    isa(i, Nothing) && throw(BoundsError(b, x_))
    return b.indexMapper[i][2]
end
index(b::ReducedSpinBasis, x::Int) = index(b, [x])
index(b::ReducedSpinBasis, x) = index(b, convert(Vector{Int}, x))

"""
    reducedspintransition(b::ReducedSpinBasis, to::Vector{Int}, from::Vector{Int})
Transition operator ``|\\mathrm{to}⟩⟨\\mathrm{from}|``, where to and from are given as vectors denoting the excitations.
"""
function reducedspintransition(b::ReducedSpinBasis, to::Vector{Int}, from::Vector{Int})
    op = SparseOperator(b)
    op.data[index(b, to), index(b, from)] = 1.
    op
end
reducedspintransition(b::ReducedSpinBasis, to, from) = reducedspintransition(b, convert(Vector{Int}, to), convert(Vector{Int}, from))
reducedspintransition(b::ReducedSpinBasis, to::Int, from::Int) = reducedspintransition(b, [to], [from])

"""
    reducedsigmap(b::ReducedSpinBasis, j::Int)
Sigma Plus Operator for the j-th particle.
"""
function reducedsigmap(b::ReducedSpinBasis, j::Int)
    N = b.N
    M = b.M
    MS = b.MS
    @assert MS!=M

    op = SparseOperator(b)
    for m=MS+1:M
        to, from = transition_idx(b, j, m)
        for i=1:length(to)
            op.data[to[i],from[i]] = 1.0
        end
    end
    return op
end

"""
    transition_idx(b, j, m)
Find the indices of all basis states where a transition from the `m-1` excitation
manifold into the `m` excitation manifold can occur by raising atom `j`
"""
function transition_idx(b, j, m)
    # Find all states with correct number of excitations
    to_ = filter(x->length(x[1])==m, b.indexMapper)
    from_ = filter(x->length(x[1])==m-1, b.indexMapper)

    # to all states that involve j but only from states that do not involve j
    filter!(x->j∈x[1],to_)
    filter!(x->!(j∈x[1]),from_)

    # Get and return the corresponding indices
    to = [t[2] for t=to_]
    from = [f[2] for f=from_]
    return to, from
end

"""
    projector_idx(b, j, m)
Find the indices of all basis states in the `m` excitation manifold involving
the atom `j` in order to build a projector onto the upper state of this atom.
"""
function projector_idx(b,j,m)
    i_ = filter(x->length(x[1])==m, b.indexMapper)
    filter!(x->j∈x[1],i_)
    return getindex.(i_,2)
end

"""
    reducedsigmam(b::ReducedSpinBasis, j::Int)
Sigma Minus Operator for the j-th particle.
"""
function reducedsigmam(b::ReducedSpinBasis, j::Int)
    return dagger(reducedsigmap(b, j))
end

"""
    reducedsigmax(b::ReducedSpinBasis, j::Int)
Sigma-X Operator for the j-th particle.
"""
function reducedsigmax(b::ReducedSpinBasis, j::Int)
    return reducedsigmap(b, j) + reducedsigmam(b, j)
end

"""
    reducedsigmay(b::ReducedSpinBasis, j::Int)
Sigma-Y Operator for the j-th particle.
"""
function reducedsigmay(b::ReducedSpinBasis, j::Int)
    return im*(-reducedsigmap(b, j) + reducedsigmam(b, j))
end

"""
    reducedsigmaz(b::ReducedSpinBasis, j::Int)
Sigma-Z Operator for the j-th particle.
"""
function reducedsigmaz(b::ReducedSpinBasis, j::Int)
    proj = SparseOperator(b)
    for m=b.MS:b.M
        i = projector_idx(b,j,m)
        for i_=i
            proj.data[i_,i_] += 1.0
        end
    end
    return 2*proj - one(b)
end

"""
    reducedspinstate(b::ReducedSpinBasis, inds::Vector{Int})
State where the excitations are placed in the atoms given by `inds`. Note, that
`b.MS <= length(inds) <= b.M` must be satisfied.
# Examples
```julia-repl
julia> b = AtomicArrays.ReducedSpinBasis(4,2)
ReducedSpin(N=4, M=2, MS=0)
julia> GS = AtomicArrays.reducedspinstate(b,[]) # get the ground state
Ket(dim=11)
  basis: ReducedSpin(N=4, M=2, MS=0)
 1.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
julia> ψ2 = AtomicArrays.reducedspinstate(b,[1,2]) # First and second atom excited
Ket(dim=11)
  basis: ReducedSpin(N=4, M=2, MS=0)
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 1.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
```
"""
function reducedspinstate(b::ReducedSpinBasis, inds::Vector{Int})
    basisstate(b, index(b, inds))
end
reducedspinstate(b::ReducedSpinBasis, n) = reducedspinstate(b, convert(Vector{Int}, n))
reducedspinstate(b::ReducedSpinBasis, n::Int) = reducedspinstate(b, [n])

"""
    reducedsigmapsigmam(b::ReducedSpinBasis, i, j)
Create the operator product σᵢ⁺σⱼ⁻ directly on a [`ReducedSpinBasis`](@ref).
Useful especially for a basis where only one excitation is included, since then the
single operators are zero (do not conserve excitation number), but the product is not.
"""
function reducedsigmapsigmam(b::ReducedSpinBasis, i, j)
    op = SparseOperator(b)
    for idx1 in b.indexMapper
        if j in idx1[1]
            for idx2 in b.indexMapper
                i in idx2[1] && (op.data[idx2[2], idx1[2]] = 1)
            end
        end
    end
    return op
end

"""
    Hamiltonian(S::SpinCollection, M::Int=1)
Builds the dipole-dipole Hamiltonian.
# Arguments
* S: SpinCollection
* M: Number of excitations.
"""
function Hamiltonian(S::SpinCollection, M::Int=1)
    N = length(S.spins)
    b = ReducedSpinBasis(N, M)
    OmegaM = interaction.OmegaMatrix(S)

    H = SparseOperator(b)
    for i=1:N
        if S.spins[i].delta != 0.
            H += 0.5*S.spins[i].delta*reducedsigmaz(b,i)
        end
    end
    for i=1:N, j=1:N
        if i != j
            H += OmegaM[i, j]*reducedsigmapsigmam(b,i,j)
        end
    end

    return H
end

"""
    JumpOperators(S::SpinCollectino, M::Int=1)
Gamma Matix and Jump Operators for dipole-dipole interactino.
# Arguments
* S: Spin collection.
* M: Number of excitations.
"""
function JumpOperators(S::SpinCollection, M::Int=1)
        N = length(S.spins)
        b = ReducedSpinBasis(N, M)

        sm(j) = reducedsigmam(b, j)
        Jumps = [ sm(j) for j=1:N]
        GammaM = interaction.GammaMatrix(S)

        return GammaM, Jumps
    end

"""
    timeevolution(T, S::SpinCollection, psi0; kwargs...)
"""
function timeevolution(T, S::SpinCollection, psi0::Union{Ket{B}, DenseOpType{B, B}}; kwargs...) where B <: ReducedSpinBasis
    M = isa(psi0, Ket) ? psi0.basis.M : psi0.basis_l.M
    H = Hamiltonian(S, M)
    GammaM, J = JumpOperators(S, M)
    return QuantumOptics.timeevolution.master(T, psi0, H, J; rates=GammaM, kwargs...)
end

"""
    Hamiltonian_nh(S::SpinCollection,  M::Int=1)
Non-Hermitian Hamiltonian
# Argument
* S: SpinCollection
* M: Number of excitations.
"""
function Hamiltonian_nh(S::SpinCollection, M::Int=1, MS::Int=1)
    N = length(S.spins)
    b = ReducedSpinBasis(N, M, MS)
    OmegaM = interaction.OmegaMatrix(S)
    GammaM = interaction.GammaMatrix(S)

    H = SparseOperator(b)
    for i=1:N
        if S.spins[i].delta != 0.
            H += 0.5*S.spins[i].delta * reducedsigmaz(b,i)
        end
    end
    for i=1:N, j=1:N
            H += (OmegaM[i, j] - 0.5im*GammaM[i, j])*reducedsigmapsigmam(b,i,j)
    end

    return H
end

"""
    schroedinger_nh(T, S::SpinCollection, psi0; kwargs...)
    Time evolution of the non-Hermitian Hamiltonian in the single excitation manifold (M=1) via the Schrödinger equation.
"""
function schroedinger_nh(T, S::SpinCollection, psi0::Ket{B}; kwargs...) where B <: ReducedSpinBasis
    @assert psi0.basis.M == psi0.basis.MS == 1
    H = Hamiltonian_nh(S, 1, 1)

    return QuantumOptics.timeevolution.schroedinger(T, psi0, H; kwargs...)
end

end # module