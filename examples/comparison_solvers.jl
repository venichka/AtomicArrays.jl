using QuantumOptics, CollectiveSpins
const cs = CollectiveSpins
using PyPlot
using LinearAlgebra, DifferentialEquations
using BenchmarkTools

using AtomicArrays
const EMField = AtomicArrays.field.EMField

# System parameters
const a = 0.18
const γ = 1.
const e_dipole = [0,0,1.]
const T = [0:0.05:500;]
const N = 5
const Ncenter = 3
const NMAX = 100

const pos = cs.geometry.chain(a, N)
const Delt_S = [(i < N) ? 0.0 : 0.5 for i = 1:N]
const S = SpinCollection(pos, e_dipole; gammas=γ, deltas=Delt_S)
const em_inc_function = AtomicArrays.field.plane

# Define Spin 1/2 operators
spinbasis = SpinBasis(1//2)
sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)
I_spin = identityoperator(spinbasis)

# Incident field
E_ampl = 0.2 + 0im
E_kvec = 2π
E_pos0 = [0.0, 0.0, 0.0]
E_polar = [1.0, 0im, 0.0]
E_angle = [-π / 2, 0.0]

E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
    position_0=E_pos0, waist_radius=0.1)

"""Impinging field"""

x = range(-3.0, 3.0, NMAX)
y = 0.0
z = range(-5.0, 5.0, NMAX)
e_field = Matrix{ComplexF64}(undef, length(x), length(z))
for i = 1:length(x)
    for j = 1:length(z)
        e_field[i, j] = em_inc_function([x[i], y, z[j]], E_inc)[3]
    end
end

#fig_0 = PyPlot.figure(figsize=(7, 4))
#PyPlot.contourf(x, z, real(e_field)', 30)
#for p in pos
#    PyPlot.plot(p[1], p[3], "o", color="w", ms=2)
#end
#PyPlot.xlabel("x")
#PyPlot.ylabel("z")
#PyPlot.colorbar(label="Amplitude")
#display(fig_0)


E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:N]
Om_R = AtomicArrays.field.rabi(E_vec, S.polarizations)

#fig_1, axs = PyPlot.subplots(ncols=1, nrows=2, figsize=(5.7, 3),
#    constrained_layout=true)
#axs[1].plot(abs.(Om_R))
#axs[1].plot(abs.(Om_R), "o")
#axs[1].set_title(L"|\Omega_R|")
#axs[2].plot(real(Om_R), "-o")
#axs[2].plot(imag(Om_R), "-o")
#axs[2].set_title(L"\Re(\Omega_R), \Im(\Omega_R)")
#display(fig_1)

"""System Hamiltonian"""

#Γ, J = CollectiveSpins.quantum.JumpOperators(S)
#Jdagger = [dagger(j) for j=J]
#Ω = CollectiveSpins.interaction.OmegaMatrix(S)
#H = CollectiveSpins.quantum.Hamiltonian(S) - sum(Om_R[j]*J[j]+
#                                                       conj(Om_R[j])*Jdagger[j]
#                                                       for j=1:N)
Γ, J = AtomicArrays.quantum.JumpOperators(S)
Jdagger = [dagger(j) for j = J]
Ω = AtomicArrays.interaction.OmegaMatrix(S)
H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
                                                          conj(Om_R[j]) * Jdagger[j]
                                                          for j = 1:N)

H.data

"""Dynamics"""

# Initial state (Bloch state)
const phi = 0.
const theta = pi/2.

# Time evolution

# Independent
state0 = cs.independent.blochstate(phi, theta, N)
tout, state_ind_t = cs.independent.timeevolution(T, S, state0)

# Meanfield
state0 = cs.meanfield.blochstate(phi, theta, N)
#tout, state_mf_t = cs.meanfield.timeevolution(T, S, state0)
tout, state_mf_t = AtomicArrays.meanfield.timeevolution_field(T, S, Om_R, state0)
#@benchmark AtomicArrays.meanfield.timeevolution_field(T, S, Om_R, state0)


# Meanfield + Correlations
state0 = cs.mpc.blochstate(phi, theta, N)
#tout, state_mpc_t = cs.mpc.timeevolution(T, S, state0)
tout, state_mpc_t = AtomicArrays.mpc.timeevolution_field(T, S, Om_R, state0)
#@benchmark AtomicArrays.mpc.timeevolution_field(T, S, Om_R, state0)

# Quantum: master equation
sx_master = Float64[]
sy_master = Float64[]
sz_master = Float64[]

Cxx_master = Float64[]
Cxy_master = Float64[]
Cyz_master = Float64[]

td_ind = Float64[]
td_mf  = Float64[]
td_mpc = Float64[]

embed(op::Operator) = QuantumOptics.embed(cs.quantum.basis(S), Ncenter, op)

function fout(t, rho)
    i = findfirst(isequal(t), T)
    rho_ind = cs.independent.densityoperator(state_ind_t[i])
    rho_mf  = cs.meanfield.densityoperator(state_mf_t[i])
    rho_mpc = cs.mpc.densityoperator(state_mpc_t[i])
    push!(td_ind, tracedistance(rho, rho_ind))
    push!(td_mf,  tracedistance(rho, rho_mf))
    push!(td_mpc, tracedistance(rho, rho_mpc))
    push!(sx_master, real(expect(embed(sx), rho)))
    push!(sy_master, real(expect(embed(sy), rho)))
    push!(sz_master, real(expect(embed(sz), rho)))
    push!(Cxx_master, real(expect((J[2]+Jdagger[2])*(J[5]+Jdagger[5]), rho)))
    push!(Cxy_master, real(expect((J[2]+Jdagger[2])*(im*(J[5]-Jdagger[5])), rho)))
    push!(Cyz_master, real(expect((im*(J[2]-Jdagger[2]))*(Jdagger[5]*J[5]-J[5]*Jdagger[5]), rho)))
    return nothing
end

Ψ₀ = cs.quantum.blochstate(phi,theta,N)
ρ₀ = Ψ₀⊗dagger(Ψ₀)
QuantumOptics.timeevolution.master_h(T, ρ₀, H, J; fout=fout, rates=Γ)
#cs.quantum.timeevolution(T, S, ρ₀, fout=fout)

# Expectation values
mapexpect(op, states) = map(s->(op(s)[Ncenter]), states)
mapexpect_corr(op, states) = map(s->(op(s)[2,5]), states)

sx_ind = mapexpect(cs.independent.sx, state_ind_t)
sy_ind = mapexpect(cs.independent.sy, state_ind_t)
sz_ind = mapexpect(cs.independent.sz, state_ind_t)

sx_mf = mapexpect(cs.meanfield.sx, state_mf_t)
sy_mf = mapexpect(cs.meanfield.sy, state_mf_t)
sz_mf = mapexpect(cs.meanfield.sz, state_mf_t)

sx_mpc = mapexpect(cs.mpc.sx, state_mpc_t)
sy_mpc = mapexpect(cs.mpc.sy, state_mpc_t)
sz_mpc = mapexpect(cs.mpc.sz, state_mpc_t)
Cxx_mpc = mapexpect_corr(cs.mpc.Cxx, state_mpc_t)
Cxy_mpc = mapexpect_corr(cs.mpc.Cxy, state_mpc_t)
Cyz_mpc = mapexpect_corr(cs.mpc.Cyz, state_mpc_t)

# Plots
PyPlot.figure(figsize=(5, 5))
PyPlot.plot(T, td_mf)
PyPlot.plot(T, td_mpc)
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\frac{1}{2} Tr\left[ \sqrt{(\rho_a - \rho_q)^\dagger (\rho_a - \rho_q)} \right]")
PyPlot.title("Projection of approximate solutions on quantum")
display(gcf())

fig = PyPlot.figure(figsize=(8, 12))
PyPlot.subplot(311)
PyPlot.plot(T, sx_master, label="master")
PyPlot.plot(T, sx_mf, label="mean field")
PyPlot.plot(T, sx_mpc, label="mpc")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_x \rangle")
PyPlot.legend()

PyPlot.subplot(312)
PyPlot.plot(T, sy_master, label="master")
PyPlot.plot(T, sy_mf, label="mean field")
PyPlot.plot(T, sy_mpc, label="mpc")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_y \rangle")
PyPlot.legend()

PyPlot.subplot(313)
PyPlot.plot(T, sz_master, label="master")
PyPlot.plot(T, sz_mf, label="mean field")
PyPlot.plot(T, sz_mpc, label="mpc")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_z \rangle")
PyPlot.legend()
display(fig)

fig_1 = PyPlot.figure(figsize=(8, 12))
PyPlot.subplot(311)
PyPlot.plot(T, Cxx_master, label="master")
PyPlot.plot(T, Cxx_mpc, label="mpc")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_4^x \sigma_5^x \rangle")
PyPlot.legend()

PyPlot.subplot(312)
PyPlot.plot(T, Cxy_master, label="master")
PyPlot.plot(T, Cxy_mpc, label="mpc")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_4^x \sigma_5^y \rangle")
PyPlot.legend()

PyPlot.subplot(313)
PyPlot.plot(T, Cyz_master, label="master")
PyPlot.plot(T, Cyz_mpc, label="mpc")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_4^y \sigma_5^z \rangle")
PyPlot.legend()
display(fig_1)