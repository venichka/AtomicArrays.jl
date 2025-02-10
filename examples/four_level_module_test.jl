if pwd()[end-14:end] == "AtomicArrays.jl"
  PATH_ENV = "."
else
  PATH_ENV = "../"
end

using Pkg
Pkg.activate(PATH_ENV)

# using QuantumOptics, LinearAlgebra
# using BenchmarkTools
# using Plots

using Revise
using AtomicArrays

using LinearAlgebra

# Check FourLevelAtom struct
let
    # Create a 4-level atom at the origin, default Eg=0, Ee=1, and zero deltas.
    atom1 = AtomicArrays.FourLevelAtom([0.0, 0.0, 0.0])
    println("atom1 = ", atom1)
end

let 
    # Create an atom at [1,0,0] with Eg=0.2, Ee=1.5, and deltas=[0.1,0.2,0.3].
    atom2 = AtomicArrays.FourLevelAtom([1.0, 0.0, 0.0]; Eg=0.2, Ee=1.5, deltas=[0.1, 0.2, 0.3])
    println("atom2 = ", atom2)
end


# Check FourLevelAtomCollection struct
let 
    # Suppose we have 2 atoms at different positions
    positions = [[0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]]

    # Build a FourLevelAtomCollection with default polarizations & gammas
    coll1 = AtomicArrays.FourLevelAtomCollection(positions; Eg=0.0, Ee=1.0, deltas=[[0.1,0.2,0.3], [1, 1, 1]])
    println("coll1 = ", coll1)
    coll1.atoms
end

begin 
    positions = [[0.0,0.0,0.0],
                [0.0,1.0,0.0],
                [1.0,1.0,0.0]]  # 3 atoms
    N = length(positions)

    # Build a custom complex polarizations array of shape (3,3,3).
    polarizations = Array{ComplexF64,3}(undef, 3, 3, N)
    for n in 1:N
        for μ in 1:3
            for c in 1:3
                polarizations[μ, c, n] = rand() + im*rand()
            end
        end
    end

    # Build a gammas array of shape (3,3).
    gammas = [0.1 0.11 0.12;
              0.2 0.21 0.22;
              0.3 0.31 0.32]

    coll2 = AtomicArrays.FourLevelAtomCollection(
        positions;
        Eg=0.0,
        Ee=1.0,
        deltas=[[0.1,0.2,0.3]],
        polarizations=polarizations,
        gammas=gammas
    )

    println("coll2 = ", coll2)
    coll2.gammas
end

# Check CavityFourLevelAtomCollection
let 
    # 1) Build or reuse a CavityMode
    cavity = CavityMode(10; delta=0.0, eta=1.0, kappa=0.05)

    # 2) Reuse the coll2 from above (3 atoms)
    # coll2 is a FourLevelAtomCollection with 3 atoms

    # 3) Provide a single coupling constant, say 0.02
    cavity_sys1 = AtomicArrays.CavityFourLevelAtomCollection(cavity, coll2, [0.1 0.1 0.1; 0.1 0.1 0.1; 0.1 0.1 0.1])
    # cavity_sys1 = AtomicArrays.CavityFourLevelAtomCollection(cavity, coll2, 0.1)
    println("cavity_sys1 = ", cavity_sys1)
end

norm((coll2.polarizations)[1,:,2])
sqrt(sum((coll2.polarizations)[1,k,2]*conj((coll2.polarizations)[1,k,2]) for k = 1:3))

let
    ri = [1, 0, 0]
    rj = [1.0, 0, 0]
    # mui = [0,0,1]
    mui =  [- 1,1im,0] / sqrt(2)
    muj = [1,1im,0] / sqrt(2)
    gi = 0.1
    gj = 0.1
    AtomicArrays.interaction.Gamma(ri, rj, mui, muj, gi, gj)
end