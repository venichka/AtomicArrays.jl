module fourlevel_quantum

using ..AtomicArrays
#import ..AtomicArrays: FourLevelAtom, FourLevelAtomCollection
using QuantumOptics, LinearAlgebra

using ..interaction

# TODO 
# export 

const D_LEVEL = 4  # 4-level basis dimension per atom
const atombasis = NLevelBasis(D_LEVEL)
# Indices of atom levels in spherical coordinates
const idx_g       = 1  # ground
const idx_e_minus = 2  # excited sublevel m=-1
const idx_e_0     = 3  # m=0
const idx_e_plus  = 4  # m=+1
# Indices of atom levels in cartesian coordinates
const idx_x = 2  # excited sublevel x
const idx_y = 3  # y
const idx_z = 4  # z
# Define projection operators in spherical basis
const sigma_ge_minus_ = transition(atombasis, idx_g, idx_e_minus)
const sigma_ge_0_ = transition(atombasis, idx_g, idx_e_0)
const sigma_ge_plus_ = transition(atombasis, idx_g, idx_e_plus)
const sigma_gg_ = transition(atombasis, idx_g, idx_g)
const sigma_e_minus_e_minus_ = transition(atombasis, idx_e_minus, idx_e_minus)
const sigma_e_0_e_0_ = transition(atombasis, idx_e_0, idx_e_0)
const sigma_e_plus_e_plus_ = transition(atombasis, idx_e_plus, idx_e_plus)
sigmas_ge_ = [sigma_ge_minus_, sigma_ge_0_, sigma_ge_plus_]
sigmas_eg_ = dagger.([sigma_ge_minus_, sigma_ge_0_, sigma_ge_plus_])
sigmas_ee_ = [sigma_e_minus_e_minus_, sigma_e_0_e_0_, sigma_e_plus_e_plus_]
# Projection operators in cartesian basis
const sigma_gx_ = sigma_ge_minus_
const sigma_gy_ = sigma_ge_0_
const sigma_gz_ = sigma_ge_plus_
const I_atom = identityoperator(atombasis)

"""
    fourlevel_quantum.basis(x)
Get basis of the given System (spherical or cartesian)
"""
basis(x::FourLevelAtom) = atombasis
basis(x::FourLevelAtomCollection) = CompositeBasis([basis(s) for s=x.atoms]...)
basis(N::Int) = CompositeBasis([atombasis for s=1:N]...)
basis(x::CavityMode) = FockBasis(x.cutoff)
basis(x::CavityFourLevelAtomCollection) = CompositeBasis(basis(x.cavity), basis(x.atomcollection))

"""
    fourlevel_quantum.dim(state)
Number of atoms described by this state.
"""
function dim(ρ::AbstractOperator)
    return ρ.basis_l.N
end

"""
    fourlevel_quantum.Hamiltonian(A; magnetic_field=nothing, 
                                  external_drive=nothing,
                                  dipole_dipole=true)
System Hamitonian in spherical basis.

- magnetic field = g_J μ_B B
- external drive = Rabi constant (precomputed)
- dipole-dipole = bool variable
"""
function Hamiltonian(A::FourLevelAtomCollection; magnetic_field=nothing,
                     external_drive=nothing, dipole_dipole=true)
    atoms = A.atoms
    N = length(atoms)
    N_sublevels = size(A.polarizations, 1)
    sublevels_m = [-1.0, 0.0, 1.0]
    b = basis(A)
    H = SparseOperator(b)
    # atom Hamiltonian
    for i=1:N
        for m = 1:N_sublevels
            if atoms[i].delta != 0.0
                H += atoms[i].delta*embed(b,i,sigmas_ee_[m])
            end
            if magnetic_field  !== nothing && m != 2
                B = magnetic_field
                H += sublevels_m[m] * B *embed(b,i,sigmas_ee_[m])
            end
        end
    end
    # external drive
    if !isnothing(external_drive)
        Ω_R = external_drive
        for i = 1:N 
            for m = 1:N_sublevels
                H += - (Ω_R[m,i] * embed(b,i,sigmas_ge_[m]) + 
                        conj(Ω_R[m,i]) * embed(b,i,sigmas_eg_[m]))
            end
        end
    end
    # dipole-dipole
    if dipole_dipole
        mu = A.polarizations
        gamma = A.gammas
        for i = 1:N
            for j = 1:N
                if i == j
                    continue
                end
                for k = 1:N_sublevels
                    for m = 1:N_sublevels
                        sigmap_ik = embed(b, i, sigmas_eg_[k])
                        sigmam_jm = embed(b, j, sigmas_ge_[m])
                        H += interaction.Omega(atoms[i].position,
                                    atoms[j].position, 
                                    mu[k,:,i], mu[m,:,j], 
                                    gamma[k, i], gamma[m, j],
                                    atoms[i].delta+2π,
                                    atoms[j].delta+2π) * sigmap_ik * sigmam_jm
                    end
                end
            end
        end
    end
    return H
end

"""
    fourlevel_quantum.Hamiltonian_nh(A; magnetic_field=nothing, 
                                     external_drive=nothing)
System non-Hermitian Hamitonian in spherical basis (Omega_ij - i/2*Gamma_ij).

- magnetic field = g_J μ_B B
- external drive = Rabi constant (precomputed)
"""
function Hamiltonian_nh(A::FourLevelAtomCollection; magnetic_field=nothing,
                        external_drive=nothing)
    atoms = A.atoms
    mu = A.polarizations
    gamma = A.gammas
    N = length(atoms)
    N_sublevels = size(mu, 1)
    sublevels_m = [-1.0, 0.0, 1.0]
    b = basis(A)
    H = SparseOperator(b)
    # atom Hamiltonian
    for i=1:N
        for m = 1:N_sublevels
            if atoms[i].delta != 0.0
                H += atoms[i].delta*embed(b,i,sigmas_ee_[m])
            end
            if magnetic_field  !== nothing && m != 2
                B = magnetic_field
                H += sublevels_m[m] * B *embed(b,i,sigmas_ee_[m])
            end
        end
    end
    # external drive
    if !isnothing(external_drive)
        Ω_R = external_drive
        for i = 1:N 
            for m = 1:N_sublevels
                H += - (Ω_R[m,i] * embed(b,i,sigmas_ge_[m]) + 
                        conj(Ω_R[m,i]) * embed(b,i,sigmas_eg_[m]))
            end
        end
    end
    # dipole-dipole + non-Hermitian part
    for i = 1:N
        for j = 1:N
            if i == j
                continue
            end
            for k = 1:N_sublevels
                for m = 1:N_sublevels
                    sigmap_ik = embed(b, i, sigmas_eg_[k])
                    sigmam_jm = embed(b, j, sigmas_ge_[m])
                    args = [atoms[i].position, atoms[j].position, 
                            mu[k,:,i], mu[m,:,j], 
                            gamma[k, i], gamma[m, j],
                            atoms[i].delta+2π, atoms[j].delta+2π]
                    H += (interaction.Omega(args...) - 
                         0.5im*interaction.Gamma(args...))*sigmap_ik*sigmam_jm
                end
            end
        end
    end
    return H
end

"""
    fourlevel_quantum.JumpOperators(A)
Jump operators of the given system.
- J_mi has dimensions 3xN, where first is the number of sublevel transitions related to m = [-1, 0, +1]
"""
function JumpOperators(A::FourLevelAtomCollection; flatten=false)
    J = SparseOpType[embed(basis(A), j, sigmas_ge_[m]) 
                    for m=1:size(A.polarizations,1), j=1:length(A.atoms)]
    if flatten
        J = vcat(J...)
        Γ = interaction.GammaMatrix_4level(A)
    else
        Γ = interaction.GammaTensor_4level(A)
    end
    return Γ, J
end





################################################################################
# (1) Build Single-Atom Hamiltonian
################################################################################
"""
    build_single_atom_hamiltonian(atom::FourLevelAtom, pol::Matrix{<:Complex}, drive_params, B)

Builds the single-atom Hamiltonian for one 4-level atom.

- `atom`:  a `FourLevelAtom` with fields:
   - `Eg`, `Ee`, `deltas` (3 sublevels).
- `pol`:   a (3×3) complex matrix for transition polarizations (one row per sublevel, columns x,y,z).
- `drive_params`: user-defined object (e.g. has `Ω` field) for external Rabi drives.
- `B`: Magnetic field vector (e.g. `[Bx, By, Bz]`).

Returns an `Operator` on a single 4-level space (`NLevelBasis(4)`).

You can refine how you add Zeeman or external drive terms, 
depending on the physics of your system.
"""
function build_single_atom_hamiltonian(
    atom::FourLevelAtom,
    pol::Matrix{<:Complex},
    drive_params,
    B::Vector{<:Real}
)
    b_single = NLevelBasis(D_LEVEL)

    # Indices for clarity:
    idx_g       = 1  # ground
    idx_e_minus = 2  # excited sublevel m=-1
    idx_e_0     = 3  # m=0
    idx_e_plus  = 4  # m=+1

    # For simpler notation:
    Eg = atom.Eg
    Ee = atom.Ee
    δ  = atom.deltas  # length=3 => [δ_{-1}, δ_{0}, δ_{+1}]

    # 1) Diagonal part: Eg for ground, (Ee + δ[m]) for each excited
    H_diag = Eg * transition(b_single, idx_g, idx_g) +
             (Ee + δ[1]) * transition(b_single, idx_e_minus, idx_e_minus) +
             (Ee + δ[2]) * transition(b_single, idx_e_0,     idx_e_0) +
             (Ee + δ[3]) * transition(b_single, idx_e_plus,  idx_e_plus)

    # 2) Possibly incorporate a Zeeman shift from B?
    #    We assume it's already folded into `δ`.

    # 3) External drive couplings
    #    If drive_params is non-nothing, we add off-diagonal Rabi terms.
    H_drive = 0.0 .* identityoperator(b_single)
    if drive_params !== nothing
        # Suppose drive_params has a field `Ω::Vector{<:Real}` of length 3:
        Ω = drive_params.Ω  # e.g. [Ω_-1, Ω_0, Ω_+]
        # Add ground<->excited couplings
        H_drive += Ω[1]*(transition(b_single, idx_e_minus, idx_g) + transition(b_single, idx_g, idx_e_minus))
        H_drive += Ω[2]*(transition(b_single, idx_e_0,     idx_g) + transition(b_single, idx_g, idx_e_0))
        H_drive += Ω[3]*(transition(b_single, idx_e_plus,  idx_g) + transition(b_single, idx_g, idx_e_plus))
    end

    return H_diag + H_drive
end


################################################################################
# (2) Build the total Hilbert space from a 4-level atom collection
################################################################################
"""
    build_total_basis(coll::FourLevelAtomCollection)

Create an `NLevelBasis(4^N)` that is the tensor product of N single-atom
4-level bases, where N = length(coll.atoms).
"""
function build_total_basis(coll::FourLevelAtomCollection)
    N = length(coll.atoms)
    b_single = NLevelBasis(D_LEVEL)
    return tensorbasis(fill(b_single, N)...)
end

################################################################################
# (3) Build the total Hamiltonian for the entire collection
################################################################################
"""
    build_total_hamiltonian(coll::FourLevelAtomCollection;
                            B=[0,0,0], drive_params=nothing, dipole_coupling=nothing)

Constructs the total Hamiltonian for N 4-level atoms, including:
- Single-atom Hamiltonians for each atom (with possible external drive, B-field)
- Pairwise dipole-dipole interactions if `dipole_coupling` is given

Returns `(b_total, H_total)`.

- `B`: a magnetic field vector
- `drive_params`: object that might have `.Ω` for the external drive
- `dipole_coupling`: either `nothing` (no dipole–dipole) or some data structure 
   (matrix or function) that we pass to `build_dipole_dipole` (see below).
"""
function build_total_hamiltonian(
    coll::FourLevelAtomCollection;
    B::Vector{<:Real} = [0,0,0],
    drive_params=nothing,
    dipole_coupling=nothing
)
    N = length(coll.atoms)
    b_total = build_total_basis(coll)

    H_total = 0.0 .* identityoperator(b_total)
    # (a) Sum of single-atom Hamiltonians
    for n in 1:N
        atom = coll.atoms[n]
        pol_n = coll.polarizations[:, :, n]  # shape (3,3) for transitions
        H_atom = build_single_atom_hamiltonian(atom, pol_n, drive_params, B)
        H_total += expand(H_atom, N, n)
    end

    # (b) Add dipole-dipole (or other) interactions if requested
    if dipole_coupling !== nothing
        H_dd = build_dipole_dipole(coll, b_total, dipole_coupling)
        H_total += H_dd
    end

    return b_total, H_total
end


################################################################################
# (4) Build dipole-dipole interaction
################################################################################
"""
    build_dipole_dipole(coll::FourLevelAtomCollection, b_total, coupling_data)

Build pairwise dipole-dipole terms for the collection.

Here, you can implement your own flip-flop or full dipole form. 
Below is a simplified "exchange" example:
    H_dd = Σ_{i<j} cᵢⱼ ( σg eᵢ σe gⱼ + h.c. )

- `coupling_data` can be a matrix c[i,j] or a function c(i,j).
- This is an example skeleton; adapt to your actual dipole formula.

Returns an `Operator` on `b_total`.
"""
function build_dipole_dipole(coll::FourLevelAtomCollection, b_total, coupling_data)
    N = length(coll.atoms)
    b_single = NLevelBasis(D_LEVEL)
    idx_g = 1
    idx_es = [2,3,4]  # excited sublevels

    H_dd = 0.0 .* identityoperator(b_total)

    for i in 1:(N-1)
        for j in (i+1):N
            # fetch coupling
            val = (typeof(coupling_data) <: Function) ? coupling_data(i,j) : coupling_data[i,j]

            # For each excited sublevel
            for e in idx_es
                op_ge = transition(b_single, idx_g, e)  # |g><e|
                op_eg = transition(b_single, e, idx_g)  # |e><g|

                # expand to i-th, j-th
                op_ge_i = expand(op_ge, N, i)
                op_eg_j = expand(op_eg, N, j)

                op_ge_j = expand(op_ge, N, j)
                op_eg_i = expand(op_eg, N, i)

                # add cᵢⱼ * (op_ge_i * op_eg_j + op_ge_j * op_eg_i)
                H_dd += val * (op_ge_i*op_eg_j + op_ge_j*op_eg_i)
            end
        end
    end
    return H_dd
end


################################################################################
# (5) Build jump (projection) operators for the entire collection
################################################################################
"""
    build_jump_operators(coll::FourLevelAtomCollection, b_total)

Creates the Lindblad collapse operators for each atom's 3 excited sublevels 
to ground, using the per-transition gamma rates in `coll.gammas[μ, n]`.

Returns a vector of operators on `b_total`.
"""
function build_jump_operators(coll::FourLevelAtomCollection, b_total)
    N = length(coll.atoms)
    b_single = NLevelBasis(D_LEVEL)
    idx_g = 1

    # The 3 transitions => excited sublevels = 2,3,4 => μ=1..3
    c_ops = []
    for n in 1:N
        for μ in 1:3
            γ = coll.gammas[μ, n]
            e_ind = μ + 1  # if μ=1 => 2; μ=2 =>3; μ=3 =>4
            c_single = sqrt(γ)*transition(b_single, idx_g, e_ind)
            push!(c_ops, expand(c_single, N, n))
        end
    end
    return c_ops
end


################################################################################
# (6) One-stop function: build both H_total and jump operators
################################################################################
"""
    build_ensemble_operators(coll; B, drive_params, dipole_coupling)

Returns `(b_total, H_total, c_ops)` where:
- `b_total`: the combined N-atom basis
- `H_total`: total Hamiltonian
- `c_ops`: array of Lindblad collapse operators

This gives you everything in one shot.
"""
function build_ensemble_operators(
    coll::FourLevelAtomCollection;
    B=[0.0,0.0,0.0],
    drive_params=nothing,
    dipole_coupling=nothing
)
    b_total, H_total = build_total_hamiltonian(
        coll; B=B, drive_params=drive_params, dipole_coupling=dipole_coupling
    )
    c_ops = build_jump_operators(coll, b_total)
    return b_total, H_total, c_ops
end

################################################################################
# (7) Example high-level simulation function
################################################################################
"""
    simulate_ensemble(coll; B=[0,0,0], drive_params=nothing, dipole_coupling=nothing,
                      tspan=(0.0,10.0), ρ0=nothing)

Builds the total Hamiltonian and jump operators from the entire `coll`, 
defines an initial state if none is given (all ground), then calls 
`timeevolution.master_dynamic`.

Returns `(tout, ρt, b_total)`.
"""
function simulate_ensemble(
    coll::FourLevelAtomCollection;
    B=[0.0,0.0,0.0],
    drive_params=nothing,
    dipole_coupling=nothing,
    tspan=(0.0, 10.0),
    ρ0=nothing
)
    # 1) Build Hamiltonian & jump ops
    b_total, H_total, c_ops = build_ensemble_operators(
        coll; B=B, drive_params=drive_params, dipole_coupling=dipole_coupling
    )

    # 2) Define initial state if none given
    N = length(coll.atoms)
    if ρ0 === nothing
        b_single = NLevelBasis(D_LEVEL)
        ψ_g = basisstate(b_single, 1)  # ground
        ψ0 = tensorstate(fill(ψ_g, N)...)
        ρ0 = densitymatrix(ψ0)
    end

    # 3) Solve master eq
    tout, ρt = timeevolution.master_dynamic(H_total, ρ0, tspan, c_ops)

    return tout, ρt, b_total
end

end  # module fourlevel_quantum