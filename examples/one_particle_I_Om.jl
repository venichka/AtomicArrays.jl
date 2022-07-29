# Study of the scattering from 1 particle for Heedong's project
# Pump intensity and frequency dependence

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

# Varying parameters
E_iter = range(1e-4, 2e-2, NMAX)
delta_iter = range(-1e-1, 1e-1, NMAX)

# Result
forward_scattering = zeros(NMAX, NMAX)

λ_0 = 1.0
c_light = 1.0
om_0 = 2π * c_light / λ_0
k_0 = om_0 / c_light
edipole = [0, 0, 1]
γ = 0.01
N = 1
T = [0:10.0:10000;]
d = 0.3

spinbasis = SpinBasis(1//2)
sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

# Field
E_pos0 = [0.0*d,0.0,0.0]
E_polar = [0.0, 0im, 1.0]
E_angle = [-π/2, 0.0]

Threads.@threads for iijj in CartesianIndices((NMAX, NMAX))
    (ii, jj) = Tuple(iijj)[1], Tuple(iijj)[2]
    # Varying parameters
    δ = delta_iter[ii]
    E_kvec = 2π + δ
    E_ampl = E_iter[jj]

    # System geometry
    systemgeometry = CollectiveSpins.geometry.chain(d, N)
    pos = systemgeometry
    system = CollectiveSpins.SpinCollection(systemgeometry, edipole; gammas=γ, deltas=δ)
    basis = CollectiveSpins.quantum.basis(system)
    I = identityoperator(spinbasis)

    # Incident field
    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                    position_0 = E_pos0, waist_radius = 0.1)
    # Rabi
    E_vec = [em_inc_function(system.spins[k].position, E_inc) for k = 1:N]
    Om_R = AtomicArrays.field_module.rabi(E_vec, system.polarizations)


    """System Hamiltonian"""

    Γ, J = CollectiveSpins.quantum.JumpOperators(system)
    H = -system.spins[1].delta*sp*sm - sum(Om_R[j]*sm + conj(Om_R[j])*sp
                                          for j=1:N)

    # Quantum: master equation
    sm_master_1 = ComplexF64[]

    function fout(t, rho)
        push!(sm_master_1, expect(sm, rho))
        #return nothing
    end

    Ψ₀ = CollectiveSpins.quantum.blochstate(0., pi, N)
    ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)

    # Lindblad evolution
    QuantumOptics.timeevolution.master_h(T, ρ₀, H, [sm]; fout=fout, rates=Γ)
    sm_steady_master = sm_master_1[end]

    """Forward scattering"""

    r_lim = 1000.
    r_vec = r_lim*[sin(E_angle[1]),
                   cos(E_angle[1])*sin(E_angle[2]),
                   cos(E_angle[1])*cos(E_angle[2])]
    E_out = (AtomicArrays.field_module.total_field(em_inc_function,
                                                   r_vec,
                                                   E_inc,
                                                   system, [sm_steady_master],
                                                   E_kvec))
    E_in = (em_inc_function(r_vec, E_inc))

    forward_scattering[ii,jj] = 4π/E_kvec * imag(r_lim / exp(im*E_kvec*r_lim) *
        E_polar' * (E_out .- E_in))

    print("$ii -- $jj \n")

end


NUM_LINES = 5
fig_1, axs = plt.subplots(ncols=1, nrows=2, figsize=(4, 8),
                        constrained_layout=true)
axs[1].contourf(delta_iter, E_iter , forward_scattering',30)
axs[1].set_yscale("log")
axs[1].set_xlabel(L"\delta = \omega_L - \omega_0")
axs[1].set_ylabel(L"E_{in}")
for i = 1:NUM_LINES
    number_lines = range(1, NMAX, NUM_LINES)
    axs[2].plot(delta_iter, forward_scattering[1:end, floor(Int, number_lines[i])],
                alpha=(1 - 0.8*number_lines[i]/NMAX), color="red")
end
axs[2].set_xlabel(L"\delta = \omega_L - \omega_0")
axs[2].set_ylabel(L"\sigma_{tot}")

# Plots
