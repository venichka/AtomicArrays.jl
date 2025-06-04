###############################################################################
# test/runtests.jl
#
# Run with:  julia --project -e 'using Pkg; Pkg.test()'
###############################################################################

using Test
using QuantumOptics
using AtomicArrays

const FRB        = AtomicArrays.fourlevel_reducedbasis
const Excitation = FRB.Excitation      # (atom, m) pair alias

################################################################################
@testset "ReducedAtomBasis – construction & indexing" begin
    N, M = 2, 1
    b = AtomicArrays.ReducedAtomBasis(N, M)        # MS defaults to 0

    # (1) Hilbert-space dimension:  Σ_{k=0}^{1} C(2,k)·3^k = 1 + 2·3 = 7
    @test b.shape[1] == 7

    # (2) Ground state index must be 1
    @test FRB.index(b, Excitation[]) == 1

    # (3) Single-atom excitation |1, m = −1⟩ on atom 2
    idx_exc = FRB.index(b, [Excitation(2, -1)])
    ψ_exc   = FRB.reducedatomstate(b, [Excitation(2, -1)])

    @test ψ_exc[idx_exc] ≈ 1.0           # right slot populated
    @test isapprox(norm(ψ_exc), 1.0)     # state is normalised
end
################################################################################

@testset "Single-atom ladder operators" begin
    b   = AtomicArrays.ReducedAtomBasis(2, 1)
    gs  = FRB.reducedatomstate(b, Excitation[])         # |g⟩⊗|g⟩

    sp  = FRB.reducedsigmaplus(b, 2, +1)                           # raise atom 2 to m = +1
    sm  = FRB.reducedsigmaminus(b, 2, +1)                          # adjoint

    # (1) σ⁺ acting on ground → correct excited ket
    ψ1 = sp * gs
    @test ψ1 ≈ FRB.reducedatomstate(b, [Excitation(2, +1)])

    # (2) Projector property: σ⁺σ⁻ acts as |1,+1⟩⟨1,+1|
    proj = sp * sm
    @test proj * ψ1 ≈ ψ1                       # idempotent
    @test proj * gs  ≈ zero(gs)                # kills ground state

    # (3) Built-in Π matches σ⁺σ⁻
    Π22 = FRB.reducedsigmapm(b, 2, +1)
    @test Π22.data == proj.data
end
################################################################################
