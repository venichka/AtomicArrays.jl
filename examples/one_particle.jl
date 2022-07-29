# Study of the scattering from 1 particle for Heedong's project

using QuantumOptics
using CollectiveSpins
using PyPlot

using AtomicArrays
const EMField = AtomicArrays.field_module.EMField
const sigma_matrices = AtomicArrays.meanfield_module.sigma_matrices
const mapexpect = AtomicArrays.meanfield_module.mapexpect


dag(x) = conj(transpose(x))
embed(op::Operator, i) = QuantumOptics.embed(CollectiveSpins.quantum.basis(system),
                                             i, op)

const PATH_FIG = "/Users/jimi/Google Drive/Work/In process/Projects/\
                  Collective_effects_QMS/Figures/one_particle"


#em_inc_function = AtomicArrays.field_module.gauss
em_inc_function = AtomicArrays.field_module.plane
const NMAX = 100

edipole = [0, 0, 1]
γ = 0.01
N = 1
T = [0:10.:10000;]
d = 0.3

spinbasis = SpinBasis(1//2)
sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

# System geometry
systemgeometry = CollectiveSpins.geometry.chain(d, N)
pos = systemgeometry
system = CollectiveSpins.SpinCollection(systemgeometry, edipole; gammas=γ, deltas=0.1)
basis = CollectiveSpins.quantum.basis(system)
I = identityoperator(spinbasis)

# Incident field
E_ampl = 1.0+ 0im
E_kvec = 2π
E_pos0 = [0.0*d,0.0,0.0]
E_polar = [0.0, 0im, 1.0]
E_angle = [-π/2, 0.0]

E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                             position_0 = E_pos0, waist_radius = 0.1)

"""Impinging field"""

x = range(-3., 3., NMAX)
y = 0.5*(pos[1][2] + pos[N][2])
z = range(-5., 5., NMAX)
e_field = Matrix{ComplexF64}(undef, length(x), length(z))
for i = 1:length(x)
    for j = 1:length(z)
        e_field[i,j] = em_inc_function([x[i],y,z[j]], E_inc)[3]
    end
end

fig_0 = PyPlot.figure(figsize=(7,4))
PyPlot.contourf(x, z, real(e_field)', 30)
for p in systemgeometry
    PyPlot.plot(p[1],p[3],"o",color="w",ms=2)
end
PyPlot.xlabel("x")
PyPlot.ylabel("z")
PyPlot.colorbar(label="Amplitude")


E_vec = [em_inc_function(system.spins[k].position, E_inc) for k = 1:N]

Om_R = AtomicArrays.field_module.rabi(E_vec, system.polarizations)

fig_1, axs = PyPlot.subplots(ncols=1, nrows=2, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].plot(abs.(Om_R))
axs[1].plot(abs.(Om_R), "o")
axs[1].set_title(L"|\Omega_R|")
axs[2].plot(real(Om_R),"-o")
axs[2].plot(imag(Om_R), "-o")
axs[2].set_title(L"\Re(\Omega_R), \Im(\Omega_R)")

"""System Hamiltonian"""

Γ, J = CollectiveSpins.quantum.JumpOperators(system)
Jdagger = [dagger(j) for j=J]
Ω = CollectiveSpins.interaction.OmegaMatrix(system)
H = -system.spins[1].delta*sp*sm - sum(Om_R[j]*sm + conj(Om_R[j])*sp
                                      for j=1:N)

H.data


# Meanfield evolution
state0 = CollectiveSpins.meanfield.blochstate(0., pi, N)
tout, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, system,
                                                                     Om_R, state0)

# Quantum: master equation
sx_master_1 = Float64[]
sy_master_1 = Float64[]
sz_master_1 = Float64[]
sm_master_1 = ComplexF64[]

td_mf  = Float64[]


function fout(t, rho)
    i = findfirst(isequal(t), T)
    rho_mf = CollectiveSpins.meanfield.densityoperator(state_mf_t[i])
    push!(td_mf,  tracedistance(rho, rho_mf))
    push!(sx_master_1, real(expect(sx, rho)))
    push!(sy_master_1, real(expect(sy, rho)))
    push!(sz_master_1, real(expect(sz, rho)))
    push!(sm_master_1, expect(sm, rho))
    #return nothing
end

Ψ₀ = CollectiveSpins.quantum.blochstate(0., pi, N)
ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)

# Lindblad evolution
QuantumOptics.timeevolution.master_h(T, ρ₀, H, [sm]; fout=fout, rates=Γ)

sm_steady_master = sm_master_1[end]


sx_mf_1 = mapexpect(CollectiveSpins.meanfield.sx, state_mf_t, 1)
sy_mf_1 = mapexpect(CollectiveSpins.meanfield.sy, state_mf_t, 1)
sz_mf_1 = mapexpect(CollectiveSpins.meanfield.sz, state_mf_t, 1)


"""Forward scattering"""

r_lim = 1000.
r_vec = r_lim*[sin(E_angle[1]),0.0,cos(E_angle[1])]
E_out = (AtomicArrays.field_module.total_field(em_inc_function,
                                               r_vec,
                                               E_inc,
                                               system, [sm_steady_master]))
E_in = (em_inc_function(r_vec, E_inc))

forward_scattering = 4.0*π/E_kvec * imag(r_lim / exp(im*E_kvec*r_lim) *
                                        E_polar' * (E_out .- E_in))








# Plots
PyPlot.figure(figsize=(5,5))
PyPlot.plot(T, td_mf)
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\frac{1}{2} Tr\left[ \sqrt{(\rho_a - \rho_q)^\dagger (\rho_a - \rho_q)} \right]")
PyPlot.title("Projection of approximate solutions on quantum")

fig = PyPlot.figure(figsize=(5,5))
PyPlot.subplot(311)
PyPlot.plot(T, sx_master_1, label="master_1")
PyPlot.plot(T, sx_mf_1, label="mean field_1")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_x \rangle")
PyPlot.legend()

PyPlot.subplot(312)
PyPlot.plot(T, sy_master_1, label="master_1")
PyPlot.plot(T, sy_mf_1, label="mean field_1")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_y \rangle")
PyPlot.legend()

PyPlot.subplot(313)
PyPlot.plot(T, sz_master_1, label="master_1")
PyPlot.plot(T, sz_mf_1, label="mean field_1")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_z \rangle")
PyPlot.legend()
