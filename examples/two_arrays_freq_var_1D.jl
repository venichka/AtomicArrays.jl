# Two arrays: varying the frequency of the second array and other parameters

using CollectiveSpins
using QuantumOptics
using PyPlot
using LinearAlgebra
PyPlot.svg(true)

using AtomicArrays
const EMField = AtomicArrays.field_module.EMField
const sigma_matrices = AtomicArrays.meanfield_module.sigma_matrices


dag(x) = conj(transpose(x))


const PATH_FIG = "/Users/jimi/Google Drive/Work/In process/Projects/\
                  Collective_effects_QMS/Figures/two_arrays"

#em_inc_function = AtomicArrays.field_module.gauss
em_inc_function = AtomicArrays.field_module.plane
const NMAX = 20
const NMAX_T = 41
σ_tot = zeros(NMAX,2)
t_tot = zeros(NMAX,2)
Delt_iter = range(-0.1, 0.1, NMAX)
#E_iter = range(3e-3, 3e-2, NMAX)
E_iter = 10.0.^range(-3, -1, NMAX)
pnts = Vector{Vector{Float64}}(undef, 400)

"""Parameters"""
c_light = 1.0
lam_0 = 1.0
k_0 = 2*π / lam_0
om_0 = 2.0*pi*c_light / lam_0

Nx = 10
Ny = 10
Nz = 2  # number of arrays
M = 1 # Number of excitations
#d = 0.1888
#delt = 0.055
#L = 0.7#0.335678
d = 0.2
Delt = 0.039
d_1 = d; d_2 = d;
L = 0.189
pos_1 = AtomicArrays.geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_1/2, 
                                                           -(Ny-1)*d_1/2,
                                                           -L/2])
pos_2 = AtomicArrays.geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_2/2, 
                                                           -(Ny-1)*d_2/2,
                                                           L/2])
pos = vcat(pos_1, pos_2)
μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny*Nz]
γ_e = [1e-2 for i = 1:Nx*Ny*Nz]

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


Threads.@threads for kkii in CartesianIndices((2, NMAX))
    (kk, ii) = Tuple(kkii)[1], Tuple(kkii)[2]

    #Delt = Delt_iter[ii]
    E_ampl = E_iter[ii] + 0.0im
    δ_S = [(i < Nx*Ny + 1) ? 0.0 : Delt for i = 1:Nx*Ny*Nz]
    S = SpinCollection(pos,μ; gammas=γ_e, deltas=δ_S)

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
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:Nx*Ny*Nz]
    Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

    T = [0:250.:50000;]
    # Initial state (Bloch state)
    phi = 0.
    theta = pi/1.
    # Meanfield
    state0 = CollectiveSpins.meanfield.blochstate(phi, theta, Nx*Ny*Nz)
    tout, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S,
                                                                         Om_R,
                                                                         state0)

    t_ind = length(T)
    sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices(state_mf_t, t_ind)


    """Forward scattering"""

    r_lim = 1000.0
    σ_tot[ii,kk] = AtomicArrays.field_module.forward_scattering(r_lim, E_inc,
                                                                S, sm_mat);
    zlim = 500#0.7*(d+delt)*(Nx)
    n_samp = 400
    #t_tot[ii,kk], pnts = AtomicArrays.field_module.transmission_reg(
    #                                        E_inc, em_inc_function,
    #                                        S, sm_mat; samples=n_samp, 
    #                                        zlim=zlim, angle=[π, π])
    t_tot[ii,kk], pnts = AtomicArrays.field_module.transmission_plane(
                                            E_inc, em_inc_function,
                                            S, sm_mat; samples=n_samp, 
                                            zlim=zlim, size=[5, 5])

    println("$kk - $ii")
end

obj = AtomicArrays.field_module.objective(σ_tot[:,1], σ_tot[:,2])
efficiency = AtomicArrays.field_module.objective(t_tot[:,1], t_tot[:,2])

"""Plots"""

params_text = (
        L"d = " * string(d) * "\n" *
        L"\Delta = " * string(Delt) * "\n" *
        #L"E_0 = " * string(round(E_ampl; digits=3)) * "\n" *
        L"L = " * string(L) * "\n"
        )

fig_1, axs = PyPlot.subplots(ncols=1, nrows=4, figsize=(6, 9),
                        constrained_layout=true)
#fig_1.tight_layout()
axs[1].plot(E_iter, σ_tot[:,1], label=L"0")
axs[1].plot(E_iter, σ_tot[:,2], label=L"\pi")
axs[1].set_xscale("log")
axs[1].set_xlabel(L"E_0")
axs[1].set_title("Scattering: 0, π")
axs[1].legend()
axs[1].text(E_iter[NMAX÷6], maximum(σ_tot)/2, params_text, fontsize=12, va="center")
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
axs.plot(E_iter, σ_tot[:,1], label=L"0")
axs.plot(E_iter, σ_tot[:,2], label=L"\pi")
axs.set_xscale("log")
axs.set_xlabel(L"E_0")
axs.set_title("Scattering: 0, π")
axs.legend()
axs.text(E_iter[NMAX÷6], maximum(σ_tot)/2, params_text, fontsize=12, va="center")
display(fig_2)

fig_2.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/Evar_N10_RL_freq_opt_0.pdf", dpi=300)


fig_3f, axs = PyPlot.subplots(ncols=1, nrows=1, figsize=(6, 4),
                        constrained_layout=true)
#fig_1.tight_layout()
axs.plot(E_iter, σ_tot[:,1], label=L"0", color="r")
axs.plot(E_iter, σ_tot[:,2], label=L"\pi", color="b")
axs.plot(10.0.^range(-3, -1, 100), 30*freq_eff[:,1], "--", label=L"0, eff", color="r")
axs.plot(10.0.^range(-3, -1, 100), 30*freq_eff[:,2], "--", label=L"\pi, eff", color="b")
axs.set_xscale("log")
axs.set_xlabel(L"E_0")
axs.set_ylabel(L"\sigma_{tot}")
axs.set_title("Scattering: 0, π")
axs.legend()
#axs.text(E_iter[NMAX÷4], maximum(σ_tot)/3, params_text, fontsize=12, va="center")
display(fig_3f)


fig_3f.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/comp_Evar_freq.pdf", dpi=300)