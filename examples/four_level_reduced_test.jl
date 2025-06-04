begin
    if pwd()[end-14:end] == "AtomicArrays.jl"
        PATH_ENV = "."
    else
        PATH_ENV = "../"
    end

    using Pkg
    Pkg.activate(PATH_ENV)
end

using QuantumOptics
using AtomicArrays

# test 1: "Basis construction and indexing"
N, M = 2, 1
b = AtomicArrays.ReducedAtomBasis(N, M)            # MS defaults to 0

# (1) Dimension check:  Σ_{k=0}^{1} C(N,k)*3^k = 1 + 2*3 = 7
b.shape[1] == 7

# (2) Ground state index must be 1
AtomicArrays.fourlevel_reducedbasis.index(b, Excitation[]) == 1

# (3) Single-atom excitation |1, m=−1⟩ on atom 2 exists and is normalized
idx_exc = AtomicArrays.fourlevel_reducedbasis.index(b, [Excitation(2, -1)])
ψ_exc   = AtomicArrays.reducedatomstate(b, [Excitation(2, -1)])
ψ_exc[idx_exc] ≈ 1.0
isapprox(norm(ψ_exc), 1.0)



# test 2: "Single–atom ladder operators" 
b   = AtomicArrays.ReducedAtomBasis(2, 1)          # same basis as above
gs  = AtomicArrays.reducedatomstate(b, Excitation[])        # |0,0⟩⊗|0,0⟩
ex  = AtomicArrays.reducedatomstate(b, [Excitation(2,1)])        # |0,0⟩⊗|0,0⟩
tensor(ex, dagger(gs))
sp  = AtomicArrays.reducedsigmaplus(b, 2, +1)                 # raises atom 1 to |1, m=+1⟩
sm  = AtomicArrays.reducedsigmaminus(b, 2, +1)                # hermitian adjoint
op_ex_g = AtomicArrays.fourlevel_reducedbasis.reducedatomtransition(b, [Excitation(2, +1)], Excitation[])
op_ex_g * gs

# (1) sp acting on ground → correct excited ket
ψ1 = sp * gs
ψ1 ≈ AtomicArrays.fourlevel_reducedbasis.reducedatomstate(b, [Excitation(2, +1)])

# (2) sm ∘ sp acts as projector onto |1,+1⟩ of atom 1
proj = sp * sm
ψ2   = proj * ψ1
ψ2 ≈ ψ1                      # idempotent on that state
proj * gs ≈ zero(gs)         # annihilates ground state

# (3) Built-in projector Π matches sm*sp
Π11 = AtomicArrays.fourlevel_reducedbasis.reducedsigmapm(b, 2, +1)
Π11.data == proj.data
