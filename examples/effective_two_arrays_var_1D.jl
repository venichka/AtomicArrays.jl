# Two arrays: varying the frequency of the second array and other parameters

using CollectiveSpins
using QuantumOptics
using PyPlot
using LinearAlgebra
PyPlot.svg(true)

using AtomicArrays
const EMField = AtomicArrays.field.EMField
const collective_shift_1array = AtomicArrays.effective_interaction.collective_shift_1array
const sigma_matrices = AtomicArrays.meanfield.sigma_matrices
const mapexpect = AtomicArrays.meanfield.mapexpect


dag(x) = conj(transpose(x))


const PATH_FIG = "/Users/jimi/Google Drive/Work/In process/Projects/\
                  Collective_effects_QMS/Figures/two_arrays"

#em_inc_function = AtomicArrays.field.gauss
em_inc_function = AtomicArrays.field.plane
const NMAX = 100
const NMAX_T = 41
σ_tot_e = zeros(NMAX,2)
t_tot = zeros(NMAX,2)
delt_iter = range(-0.1, 0.1, NMAX)
#E_iter = range(3e-3, 3e-2, NMAX)
E_iter = 10.0.^range(-3, -1, NMAX)
pnts = Vector{Vector{Float64}}(undef, 400)

"""Parameters"""
c_light = 1.0
lam_0 = 1.0
k_0 = 2*π / lam_0
om_0 = 2.0*pi*c_light / lam_0

Nx = 100
Ny = 100
Nz = 2  # number of arrays
M = 1 # Number of excitations

d = 0.9#0.147
delt = 0.0778#0.147
Delt = 0.0
L = 0.7#0.7158

d_1 = d
d_2 = d + delt
γ_e = [1e-2 for i = 1:Nx*Ny*Nz]
μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:2]

# Incident field parameters
E_kvec = 1.0 * k_0
E_width = 0.3*d*sqrt(Nx*Ny)
E_pos0 = [0.0,0.0,0.0]
E_polar = [-1.0,0.0im,0.0]
E_polar = E_polar / norm(E_polar)
E_angle = [π,0.0]
E_ampl = 1.0 + im
E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                         position_0 = E_pos0, waist_radius = E_width)


"""Calculate the collective shift depending on the lattice constant"""

Omega_1, Gamma_1 = AtomicArrays.effective_interaction.effective_constants(d_1, Delt, γ_e[1], Nx)
Omega_2, Gamma_2 = AtomicArrays.effective_interaction.effective_constants(d_2, Delt, γ_e[1], Nx)

pos = [[0,0,-L/2], [0,0,L/2]]
S_1 = Spin(pos[1], delta=Omega_1)
S_2 = Spin(pos[2], delta=Omega_2 + Delt)
S = SpinCollection([S_1, S_2], [[1,0,0],[1,0,0]], 
                    gammas=[γ_e[1]+Gamma_1, γ_e[1]+Gamma_2])

Threads.@threads for kkii in CartesianIndices((2, NMAX))
    (kk, ii) = Tuple(kkii)[1], Tuple(kkii)[2]

    #delt = delt_iter[ii]
    E_ampl = E_iter[ii] + 0.0im
    #δ_S = [(i < Nx*Ny + 1) ? 0.0 : Delt for i = 1:Nx*Ny*Nz]

    Omega_1, Gamma_1 = AtomicArrays.effective_interaction.effective_constants(d_1, Delt, γ_e[1], Nx)
    Omega_2, Gamma_2 = AtomicArrays.effective_interaction.effective_constants(d_2, Delt, γ_e[1], Nx)

    pos = [[0,0,-L/2], [0,0,L/2]]
    S_1 = Spin(pos[1], delta=Omega_1)
    S_2 = Spin(pos[2], delta=Omega_2 + Delt)
    S = SpinCollection([S_1, S_2], [[1,0,0],[1,0,0]], 
                        gammas=[0.5*(1*γ_e[1]+Gamma_1), 0.5*(1*γ_e[1]+Gamma_2)])

    if (kk == 1)
        E_pos0 = [0.0,0.0,0.0]
        E_polar = [1.0,0.0im,0.0]
        E_polar = E_polar / norm(E_polar)
        E_angle = [0.0,0.0]
    elseif (kk == 2)
        E_pos0 = [0.0,0.0,0.0]
        E_polar = [-1.0,0.0im,0.0]
        E_polar = E_polar / norm(E_polar)
        E_angle = [π,0.0]
    end
    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                             position_0 = E_pos0, waist_radius = E_width)

    """Dynamics: meanfield"""

    # E_field vector for Rabi constant computation
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:length(S.spins)]
    Om_R = AtomicArrays.field.rabi(E_vec, μ)

    T = [0:250.:50000;]
    # Initial state (Bloch state)
    phi = 0.
    theta = pi/1.
    # Meanfield
    state0 = CollectiveSpins.meanfield.blochstate(phi, theta, length(S.spins))
    tout, state_mf_t = AtomicArrays.meanfield.timeevolution_field(T, S,
                                                                         Om_R,
                                                                         state0)

    t_ind = length(T)
    sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices(state_mf_t, t_ind)


    """Forward scattering"""

    r_lim = 1000.0
    σ_tot_e[ii,kk] = AtomicArrays.field.forward_scattering(r_lim, E_inc,
                                                                S, sm_mat);
    zlim = 500#0.7*(d+delt)*(Nx)
    n_samp = 400
    #t_tot[ii,kk], pnts = AtomicArrays.field.transmission_reg(
    #                                        E_inc, em_inc_function,
    #                                        S, sm_mat; samples=n_samp, 
    #                                        zlim=zlim, angle=[π, π])
    t_tot[ii,kk], pnts = AtomicArrays.field.transmission_plane(
                                            E_inc, em_inc_function,
                                            S, sm_mat; samples=n_samp, 
                                            zlim=zlim, size=[5, 5])

    println("$kk - $ii")
end

obj = AtomicArrays.field.objective(σ_tot_e[:,1], σ_tot_e[:,2])
efficiency = AtomicArrays.field.objective(t_tot[:,1], t_tot[:,2])

"""Plots"""

params_text = (
        L"d = " * string(d) * "\n" *
        L"\delta = " * string(delt) * "\n" *
        L"\Delta = " * string(Delt) * "\n" *
        #L"E_0 = " * string(round(E_ampl; digits=3)) * "\n" *
        L"L = " * string(L) * "\n"
        )

fig_1, axs = PyPlot.subplots(ncols=1, nrows=4, figsize=(6, 9),
                        constrained_layout=true)
#fig_1.tight_layout()
axs[1].plot(E_iter, σ_tot_e[:,1], label=L"0")
axs[1].plot(E_iter, σ_tot_e[:,2], label=L"\pi")
axs[1].set_xscale("log")
axs[1].set_xlabel(L"E_0")
axs[1].set_title("Scattering: 0, π")
axs[1].legend()
axs[1].text(E_iter[NMAX÷6], maximum(σ_tot_e)/2, params_text, fontsize=12, va="center")
axs[2].plot(E_iter, obj, label=L"M")
axs[2].set_xlabel(L"E_0")
axs[2].set_xscale("log")
axs[2].set_title("Objective")
axs[2].legend()
axs[3].plot(E_iter, t_tot[:,1], label=L"0")
axs[3].plot(E_iter, t_tot[:,2], label=L"\pi")
axs[3].set_xscale("log")
axs[3].set_xlabel(L"E_0")
axs[3].set_title("Transmission: 0, π")
axs[3].legend()
axs[4].plot(E_iter, efficiency, label=L"M")
axs[4].set_xscale("log")
axs[4].set_xlabel(L"E_0")
axs[4].set_title("Diode efficiency")
axs[4].legend()

display(fig_1)

#write("../Data/test.bin", I)

fig_2, axs = PyPlot.subplots(ncols=1, nrows=1, figsize=(6, 4),
                        constrained_layout=true)
#fig_1.tight_layout()
axs.plot(E_iter, σ_tot_e[:,1], label=L"0")
axs.plot(E_iter, σ_tot_e[:,2], label=L"\pi")
axs.set_xscale("log")
axs.set_xlabel(L"E_0")
axs.set_title("Scattering: 0, π")
axs.legend()
axs.text(E_iter[NMAX÷6], maximum(σ_tot_e)/2, params_text, fontsize=12, va="center")
display(fig_2)

fig_2.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/Evar_N10_RL_freq_opt_0.pdf", dpi=300)