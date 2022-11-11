using QuantumOptics
using CollectiveSpins
using PyPlot

using BenchmarkTools

using AtomicArrays
const EMField = AtomicArrays.field.EMField


# Expectation values
mapexpect(op, states) = map(s->(op(s)[3]), states)

dag(x) = conj(transpose(x))
embed(op::Operator, i) = QuantumOptics.embed(CollectiveSpins.quantum.basis(S),
    i, op)

const PATH_FIG = "/Users/jimi/Google Drive/Work/In process/Projects/\
                  Collective_effects_QMS/Figures/two_arrays"

#em_inc_function = AtomicArrays.field.gauss
em_inc_function = AtomicArrays.field.plane
const NMAX = 100

edipole = [0, 0, 1]
γ = 1
N = 5
T = [0:0.1:50;]
d = 0.35

spinbasis = SpinBasis(1 // 2)
sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

# System geometry
systemgeometry = CollectiveSpins.geometry.chain(d, N)
pos = systemgeometry
Delt_S = [(i < N) ? 0.0 : 0.0 for i = 1:N]
S = CollectiveSpins.SpinCollection(systemgeometry, edipole; gammas=γ,
    deltas=Delt_S)
basis = CollectiveSpins.quantum.basis(S)
I = identityoperator(spinbasis)

# Incident field
E_ampl = 0.1 + 0im
E_kvec = 2π
E_pos0 = [0.0 * d, 0.0, 0.0]
E_polar = [1.0, 0im, 0.0]
E_angle = [-π / 2, 0.0]

E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
    position_0=E_pos0, waist_radius=0.1)

"""Impinging field"""

x = range(-3.0, 3.0, NMAX)
y = 0.5 * (pos[1][2] + pos[N][2])
z = range(-5.0, 5.0, NMAX)
e_field = Matrix{ComplexF64}(undef, length(x), length(z))
for i = 1:length(x)
    for j = 1:length(z)
        e_field[i, j] = em_inc_function([x[i], y, z[j]], E_inc)[3]
    end
end

fig_0 = PyPlot.figure(figsize=(7, 4))
PyPlot.contourf(x, z, real(e_field)', 30)
for p in systemgeometry
    PyPlot.plot(p[1], p[3], "o", color="w", ms=2)
end
PyPlot.xlabel("x")
PyPlot.ylabel("z")
PyPlot.colorbar(label="Amplitude")
display(fig_0)


E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:N]

Om_R = AtomicArrays.field.rabi(E_vec, S.polarizations)

fig_1, axs = PyPlot.subplots(ncols=1, nrows=2, figsize=(5.7, 3),
    constrained_layout=true)
axs[1].plot(abs.(Om_R))
axs[1].plot(abs.(Om_R), "o")
axs[1].set_title(L"|\Omega_R|")
axs[2].plot(real(Om_R), "-o")
axs[2].plot(imag(Om_R), "-o")
axs[2].set_title(L"\Re(\Omega_R), \Im(\Omega_R)")
display(fig_1)

"""System Hamiltonian"""

Γ, J = CollectiveSpins.quantum.JumpOperators(S)
Jdagger = [dagger(j) for j=J]
Ω = CollectiveSpins.interaction.OmegaMatrix(S)
H = CollectiveSpins.quantum.Hamiltonian(S) - sum(Om_R[j]*J[j]+
                                                       conj(Om_R[j])*Jdagger[j]
                                                       for j=1:N)
#Γ, J = AtomicArrays.quantum.JumpOperators(S)
#Jdagger = [dagger(j) for j = J]
#Ω = AtomicArrays.interaction.OmegaMatrix(S)
#H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
#                                                          conj(Om_R[j]) * Jdagger[j]
#                                                          for j = 1:N)

H.data

"""Dynamics"""

# Initial state (Bloch state)
const phi = 0.
const theta = pi/2.

# Meanfield evolution
state0 = CollectiveSpins.meanfield.blochstate(phi, theta, N)
tout, state_mf_t = AtomicArrays.meanfield.timeevolution_field(T, S,
    Om_R, state0)
#@benchmark AtomicArrays.meanfield.timeevolution_field(T, S,
#    Om_R, state0)


    # Meanfield + Correlations
state0 = CollectiveSpins.mpc.blochstate(phi, theta, N)
tout, state_mpc_t = AtomicArrays.mpc.timeevolution_field(T, S, 
    Om_R, state0);
#@benchmark AtomicArrays.mpc.timeevolution_field(T, S, 
#    Om_R, state0)

# Quantum: master equation
sx_master = Float64[]
sy_master = Float64[]
sz_master = Float64[]

td_mf = Float64[]
td_mpc = Float64[]


function fout(t, rho)
    i = findfirst(isequal(t), T)
    rho_mf = CollectiveSpins.meanfield.densityoperator(state_mf_t[i])
    rho_mpc = CollectiveSpins.mpc.densityoperator(state_mpc_t[i])
    push!(td_mf, tracedistance(rho, rho_mf))
    push!(td_mpc, tracedistance(rho, rho_mpc))
    push!(sx_master, real(expect(embed(sx, 1), rho)))
    push!(sy_master, real(expect(embed(sy, 1), rho)))
    push!(sz_master, real(expect(embed(sz, 1), rho)))
    #return nothing
end

Ψ₀ = CollectiveSpins.quantum.blochstate(0.0, pi / 2.0, N)
ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)

# Lindblad evolution
QuantumOptics.timeevolution.master_h(T, ρ₀, H, J; fout=fout, rates=Γ)

sx_mf = mapexpect(CollectiveSpins.meanfield.sx, state_mf_t)
sy_mf = mapexpect(CollectiveSpins.meanfield.sy, state_mf_t)
sz_mf = mapexpect(CollectiveSpins.meanfield.sz, state_mf_t)

sx_mpc = mapexpect(CollectiveSpins.mpc.sx, state_mpc_t)
sy_mpc = mapexpect(CollectiveSpins.mpc.sy, state_mpc_t)
sz_mpc = mapexpect(CollectiveSpins.mpc.sz, state_mpc_t)


# Plots
PyPlot.figure(figsize=(5, 5))
PyPlot.plot(T, td_mf)
PyPlot.plot(T, td_mpc)
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\frac{1}{2} Tr\left[ \sqrt{(\rho_a - \rho_q)^\dagger (\rho_a - \rho_q)} \right]")
PyPlot.title("Projection of approximate solutions on quantum")
display(gcf())

fig = PyPlot.figure(figsize=(5, 5))
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
