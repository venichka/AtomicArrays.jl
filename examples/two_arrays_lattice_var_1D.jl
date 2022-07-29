# Two arrays: nonreciprocity

using CollectiveSpins
using QuantumOptics
using PyPlot
using LinearAlgebra
PyPlot.svg(true)

using Revise
using AtomicArrays
const EMField = AtomicArrays.field_module.EMField
const sigma_matrices = AtomicArrays.meanfield_module.sigma_matrices
const sigma_matrices_mpc = AtomicArrays.mpc_module.sigma_matrices


dag(x) = conj(transpose(x))


const PATH_FIG = "/Users/jimi/Google Drive/Work/In process/Projects/\
                  Collective_effects_QMS/Figures/two_arrays"

#em_inc_function = AtomicArrays.field_module.gauss
em_inc_function = AtomicArrays.field_module.plane
const NMAX = 100
const NMAX_T = 41
σ_tot = zeros(NMAX,2)
t_tot = zeros(NMAX,2)
delt_iter = range(-0.1, 0.1, NMAX)
E_iter = (10.0).^(range(-3.,-1.,NMAX))

"""Parameters"""
c_light = 1.0
lam_0 = 1.0
k_0 = 2*π / lam_0
om_0 = 2.0*pi*c_light / lam_0

Nx = 4
Ny = 4
Nz = 2  # number of arrays
M = 1 # Number of excitations
d = 0.147#0.48
delt = 0.147#0.256#0.0871859
d_1 = d
d_2 = d + delt
L = 0.7158#0.338#0.335678

pos_1 = AtomicArrays.geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_1/2, 
                                                           -(Ny-1)*d_1/2,
                                                           -L/2])
pos_2 = AtomicArrays.geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_2/2, 
                                                           -(Ny-1)*d_2/2,
                                                           L/2])
pos = vcat(pos_1, pos_2)


fig_1 = PyPlot.figure(figsize=(5,5))
x_at = [vec[1] for vec = pos]
y_at = [vec[2] for vec = pos]
z_at = [vec[3] for vec = pos]
PyPlot.scatter3D(x_at, y_at, z_at)
display(fig_1)

μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny*Nz]
γ_e = [1e-2 for i = 1:Nx*Ny*Nz]
S = SpinCollection(pos,μ; gammas=γ_e)
# Collective effects
Ωmat = OmegaMatrix(S)
Γmat = GammaMatrix(S)

Threads.@threads for kkii in CartesianIndices((2, NMAX))
    (kk, ii) = Tuple(kkii)[1], Tuple(kkii)[2]

    d_1 = d
    d_2 = d+delt
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
    S = SpinCollection(pos,μ; gammas=γ_e)
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
    # Incident field parameters
    E_ampl = E_iter[ii] + 0.0im
    E_kvec = 1.0 * k_0
    E_width = 0.3*d*sqrt(Nx*Ny)

    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                             position_0 = E_pos0, waist_radius = E_width)

    """Dynamics: meanfield"""

    # E_field vector for Rabi constant computation
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:Nx*Ny*Nz]
    Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

    T = [0:25000.:50000;]
    # Initial state (Bloch state)
    phi = 0.
    theta = pi/1.
    # Meanfield
    #state0 = CollectiveSpins.meanfield.blochstate(phi, theta, Nx*Ny*Nz)
    #tout, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S,
    #                                                                     Om_R,
    #                                                                     state0)
    #t_ind = length(T)
    #sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices(state_mf_t, t_ind)

    # MPC
    state0 = CollectiveSpins.mpc.blochstate(phi, theta, Nx*Ny*Nz)
    tout, state_mpc_t = AtomicArrays.mpc_module.timeevolution_field(T, S,
                                                                   Om_R, state0)
    t_ind = length(T)
    sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices_mpc(state_mpc_t, t_ind)


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
        L"d_1 = " * string(d_1) * "\n" *
        L"d_2 = " * string(round(d_2; digits=3)) * "\n" *
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

fig_2.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/Evar_N10_RL_lat.pdf", dpi=300)



fig_3, axs = PyPlot.subplots(ncols=1, nrows=1, figsize=(6, 4),
                        constrained_layout=true)
axs.plot(E_iter, σ_tot[:,1], label=L"0", color="r")
axs.plot(E_iter, σ_tot[:,2], label=L"\pi", color="b")
axs.plot(10.0.^range(-3, -1, 100), 
         maximum(σ_tot)/maximum(σ_tot_e)*σ_tot_e[:,1], 
         "--", label=L"0, eff", color="r")
axs.plot(10.0.^range(-3, -1, 100), 
         maximum(σ_tot)/maximum(σ_tot_e)*σ_tot_e[:,2], 
         "--", label=L"\pi, eff", color="b")
axs.set_xscale("log")
axs.set_xlabel(L"E_0")
axs.set_title("Scattering: 0, π")
axs.legend()
axs.text(E_iter[NMAX÷4], maximum(σ_tot)/3, params_text, fontsize=12, va="center")
display(fig_3)

fig_3.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/comp_Evar_lat.pdf", dpi=300)