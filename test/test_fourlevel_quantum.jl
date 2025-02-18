#!/usr/bin/env julia
# test_fourlevel_quantum.jl
# Example test file for "fourlevel_quantum" and "interaction" modules,
# updated to match the new system.jl definitions.

using Test
using LinearAlgebra
using QuantumOptics

using AtomicArrays

# Adjust these imports to match how you load your modules in your package.
# For demonstration, we show relative imports:
# using ..system            # <--- your updated system.jl
# using ..fourlevel_quantum
# using ..interaction

@testset "FourLevelQuantum Tests" begin

    # ----------------------------------------------------------------------------
    # 1) Test that we can build a FourLevelAtomCollection with the new definitions
    # ----------------------------------------------------------------------------
    @testset "FourLevelAtomCollection creation" begin
        # Example: 2 atoms at different positions, each with a different 'delta' value
        positions = [[0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0]]
        deltas = [0.1, 0.2]

        # We'll create random complex polarizations of shape (3,3,N)
        N = length(positions)
        pols = Array{ComplexF64,3}(undef, 3, 3, N)
        for n in 1:N
            for μ in 1:3
                for c in 1:3
                    pols[μ, c, n] = rand() + im*rand()
                end
            end
        end

        # We'll use uniform decay rates of shape (3,N)
        gam = fill(0.05, (3, N))  # e.g. 0.05 for each transition

        # 1A) Construct by positions + deltas
        coll1 = AtomicArrays.FourLevelAtomCollection(positions; 
            deltas=deltas,
            polarizations=pols,
            gammas=gam
        )
        @test length(coll1.atoms) == 2
        @test size(coll1.polarizations) == (3,3,2)
        @test size(coll1.gammas) == (3,2)

        # Check that the first atom has the correct delta
        @test coll1.atoms[1].delta == 0.1
        # The second atom at position [1,0,0] has delta = 0.2
        @test coll1.atoms[2].delta == 0.2

        # 1B) Construct by explicit "atoms" + pol/gam
        atoms = [
            AtomicArrays.FourLevelAtom([0.,0.,0.]; delta=0.3),
            AtomicArrays.FourLevelAtom([0.,1.,0.]; delta=0.4)
        ]
        pols2 = rand(ComplexF64, 3,3,2)
        gam2  = fill(0.1, (3,2))

        coll2 = AtomicArrays.FourLevelAtomCollection(
            atoms; 
            polarizations=pols2,
            gammas=gam2
        )
        @test length(coll2.atoms) == 2
        @test coll2.atoms[1].delta == 0.3
        @test coll2.atoms[2].delta == 0.4
        @test size(coll2.polarizations) == (3,3,2)
        @test size(coll2.gammas) == (3,2)
    end

    # ----------------------------------------------------------------------------
    # 2) Test building a Hamiltonian with 'fourlevel_quantum.Hamiltonian'
    # ----------------------------------------------------------------------------
    @testset "Hamiltonian tests" begin
        positions = [[0.,0.,0.],
                     [1.,0.,0.]]
        deltas = [0.1, 0.2]
        pols = rand(ComplexF64, 3,3,2)
        gam  = fill(0.05, (3,2))

        coll = AtomicArrays.FourLevelAtomCollection(positions; 
            deltas=deltas,
            polarizations=pols,
            gammas=gam
        )

        # Minimal test: no magnetic field, no external drive, no dipole-dipole
        H = AtomicArrays.fourlevel_quantum.Hamiltonian(coll; 
            magnetic_field=nothing,
            external_drive=nothing,
            dipole_dipole=false
        )

        # The dimension of basis should be 4^N = 16 for N=2
        b = AtomicArrays.fourlevel_quantum.basis(coll)
        dim_b = prod(b.shape)
        @test size(H.data) == (dim_b, dim_b)

        # For real arguments (no drive phases), the Hamiltonian should be Hermitian
        @test ishermitian(Matrix(H.data))

        # Now add an external drive (some random 3×N matrix).
        # The code expects 'external_drive' to be shape (3,N), so let's do that:
        drive = rand(ComplexF64, 3,2)  # can be complex
        H_drive = AtomicArrays.fourlevel_quantum.Hamiltonian(coll; 
            magnetic_field=nothing,
            external_drive=drive,
            dipole_dipole=false
        )
        # Possibly not purely real => still can check shape
        @test size(H_drive.data) == (dim_b, dim_b)

        # Enable dipole_dipole
        H_dd = AtomicArrays.fourlevel_quantum.Hamiltonian(coll;
            magnetic_field=nothing,
            external_drive=nothing,
            dipole_dipole=true
        )
        @test size(H_dd.data) == (dim_b, dim_b)

        # Non-Hermitian Hamiltonian test
        H_nh = AtomicArrays.fourlevel_quantum.Hamiltonian_nh(coll)
        @test size(H_nh.data) == (dim_b, dim_b)
        # The presence of -0.5i * Gamma => not Hermitian
        @test !ishermitian(Matrix(H_nh.data))
    end

    # ----------------------------------------------------------------------------
    # 3) Test the JumpOperators
    # ----------------------------------------------------------------------------
    @testset "Jump operator tests" begin
        # Reuse a small collection
        positions = [[0.,0.,0.],
                     [1.,0.,0.]]
        deltas = [0.1, 0.2]
        pols = rand(ComplexF64, 3,3,2)
        gam  = fill(0.1, (3,2))

        coll = AtomicArrays.FourLevelAtomCollection(positions; 
            deltas=deltas,
            polarizations=pols,
            gammas=gam
        )

        Γ, J = AtomicArrays.fourlevel_quantum.JumpOperators(coll)
        # J should be a vector of length 3*N = 3*2=6
        @test length(J) == 6

        # Each jump operator is dimension (16×16) for 2 atoms
        for op in J
            @test size(op.data) == (16,16)
        end

        # Γ is of size= (N*N*3*3)= 2*2*3*3=36
        @test size(Γ) == (2, 2, 3, 3)
    end

    # ----------------------------------------------------------------------------
    # 4) Test interaction functions
    # ----------------------------------------------------------------------------
    @testset "interaction function checks" begin
        # We'll do some small checks on e.g. F, G, Omega, Gamma
        r1 = [0.,0.,0.]
        r2 = [0.,0.,0.]
        μ1 = [1.,0.,0.]
        μ2 = [0.,1.,0.]

        # If r1 == r2 => norm=0 => F => 2/3, G => 0 in your definitions
        fval = AtomicArrays.interaction.F(r1, r2, μ1, μ2, 2π, 2π)
        @test isapprox(fval, 2/3; atol=1e-7)

        gval = AtomicArrays.interaction.G(r1, r2, μ1, μ2, 2π, 2π)
        @test isapprox(gval, 0; atol=1e-7)

        # Check Omega, Gamma
        om = AtomicArrays.interaction.Omega(r1, r2, μ1, μ2, 1.0, 1.0, 2π, 2π)
        ga = AtomicArrays.interaction.Gamma(r1, r2, μ1, μ2, 1.0, 1.0, 2π, 2π)
        @test isfinite(om)
        @test isfinite(ga)

        # GreenTensor => for r=0 => might be large or singular, let's do r != 0
        r2 = [0.,0.,0.1]
        Gt = AtomicArrays.interaction.GreenTensor(r2, 2π)
        @test size(Gt) == (3,3)
    end

end
