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
using QuantumOptics


# 1) Define the positions for 4 four-level atoms along the x-axis
positions = [
    [0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0],
    [0.4, 0.0, 0.0],
    [0.6, 0.0, 0.0]
]
N = length(positions)

pols = AtomicArrays.polarizations_spherical(N)

AtomicArrays.polarizations_spherical()
AtomicArrays.polarizations_cartesian()
AtomicArrays.U_sph_to_cart()*AtomicArrays.polarizations_spherical()[:,3]
AtomicArrays.U_cart_to_sph()*AtomicArrays.polarizations_cartesian() 

# 3) We'll define uniform decay rates for each transition, e.g. 0.05
gam = [AtomicArrays.gammas(0.15)[m] for m=1:3, j=1:N]

# 4) We'll define the "delta" for each atom (all zero for demonstration)
deltas = [0.0, 0.0, 0.0, 0.0]

# 5) Build the FourLevelAtomCollection
#    This uses the updated system definitions where each FourLevelAtom has a position + delta
coll = AtomicArrays.FourLevelAtomCollection(positions;
    deltas = deltas,
    polarizations = pols,
    gammas = gam
)

println("Constructed FourLevelAtomCollection with realistic sublevel polarizations.")
println("pols size = ", size(coll.polarizations))

# 6) Define a plane wave field in +y direction:
#    amplitude=1.0, wavevector magnitude=2π => (0,2π,0)
#    polarization => e.g. linear along x
amplitude = 0.2
k_mod = 2π
angle_k = [0.0, π/2]  # => +y direction
polarisation = [1.0, 0.0, 0.0]
pos_0 = [0.0, 0.0, 0.0]

field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0)

# 7) Evaluate plane-wave at each atom, build external_drive of shape (3,N)
E_vec = [AtomicArrays.field.plane(positions[j], field) for j = 1:N]
external_drive = AtomicArrays.field.rabi(E_vec, coll)
rabi_array = AtomicArrays.field.rabi(field, AtomicArrays.field.plane, coll)

println("Computed external_drive = ", external_drive)

# 8) Build the Hamiltonian and jump operators
H = AtomicArrays.fourlevel_quantum.Hamiltonian(coll; magnetic_field=nothing,
                external_drive=external_drive,
                dipole_dipole=false)

Γ, J_ops = AtomicArrays.fourlevel_quantum.JumpOperators(coll)

# Print summary
println("Hamiltonian dimension = ", size(H.data))
println("Number of jump ops = ", length(J_ops), "  => each is dimension ", size(J_ops[1].data))

# (Optional) If you want to do time evolution:
# using QuantumOptics
# b = basis(coll)
# dim_total = dimension(b) # 4^4 = 256
# # initial state => all ground
# using .fourlevel_quantum: idx_g
# b_single = NLevelBasis(4)
# ψ_g = basisstate(b_single, idx_g)
# ψ0 = tensorstate(fill(ψ_g, N)...)
# ρ0 = densitymatrix(ψ0)
# tspan = (0.0, 10.0)
# c_ops = [] # if you'd also want to add collapse ops from Γ, J_ops, you'd do e.g. 
# # for i in 1:length(J_ops)
# #     c_ops_i = sqrt(rate)*J_ops[i] # or something like that
# # end
# # or consider a standard Lindblad approach with JumpOperators = ...
# # For demonstration, we won't finalize that here.

# end script
println("Done.")
