# Two arrays: varying the frequency of the second array and other parameters
using Distributed
addprocs(20)

@everywhere begin
    if pwd()[end-14:end] == "AtomicArrays.jl"
        PATH_ENV = "."
    else
        PATH_ENV = "../"
    end
end

using SharedArrays
using DelimitedFiles

@everywhere using Pkg
@everywhere Pkg.activate(PATH_ENV)
@everywhere begin
    using ProgressMeter
    using CollectiveSpins
    using QuantumOptics
    using PyPlot
    using LinearAlgebra

    using AtomicArrays
    const EMField = AtomicArrays.field_module.EMField
    const sigma_matrices = AtomicArrays.meanfield_module.sigma_matrices


    dag(x) = conj(transpose(x))

    #em_inc_function = AtomicArrays.field_module.gauss
    em_inc_function = AtomicArrays.field_module.plane
    NMAX = 20
    NMAX_T = 41
    Delt_iter = range(0.03, 0.08, NMAX)
    E_iter = range(1e-2, 2e-2, NMAX)
    L_iter = range(6e-1, 8e-1, NMAX)
    d_iter = range(1.7e-1, 2.5e-1, NMAX)

    """Parameters"""
    c_light = 1.0
    lam_0 = 1.0
    k_0 = 2*π / lam_0
    om_0 = 2.0*pi*c_light / lam_0

    Nx = 10
    Ny = 10
    Nz = 2  # number of arrays
    M = 1 # Number of excitations
    #delt = 0.0035897435897435897 
    #d = 0.4666666
    #L = 0.405
    μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny*Nz]
    γ_e = [1e-2 for i = 1:Nx*Ny*Nz]

    # Incident field parameters
    #E_ampl = 4.5e-3 + 0.0im
    E_kvec = 1.0 * k_0
end

σ_tot = SharedArray{Float64,5}((NMAX,NMAX,NMAX,NMAX,2))
t_tot = SharedArray{Float64,5}((NMAX,NMAX,NMAX,NMAX,2))


@sync @showprogress @distributed for kkiijjmmnn in CartesianIndices((2, NMAX, NMAX, NMAX, NMAX))
    (kk, ii, jj, mm, nn) = Tuple(kkiijjmmnn)[1], Tuple(kkiijjmmnn)[2], Tuple(kkiijjmmnn)[3], Tuple(kkiijjmmnn)[4], Tuple(kkiijjmmnn)[5] 

    Delt = Delt_iter[ii]
    E_ampl = E_iter[jj] + 0.0im
    d = d_iter[mm]
    d_1 = d; d_2 = d;
    L = L_iter[nn]

    pos_1 = AtomicArrays.geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                                   position_0=[-(Nx-1)*d_1/2, 
                                                               -(Ny-1)*d_1/2,
                                                               -L/2])
    pos_2 = AtomicArrays.geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                                   position_0=[-(Nx-1)*d_2/2, 
                                                               -(Ny-1)*d_2/2,
                                                               L/2])
    pos = vcat(pos_1, pos_2)
    δ_S = [(i < Nx*Ny + 1) ? -0.5*Delt : 0.5*Delt for i = 1:Nx*Ny*Nz]
    γ_e = [(i < Nx * Ny + 1) ? 
            1e-2*(1.0 - 0.5*Delt/om_0)^3 : 1e-2*(1.0 + 0.5*Delt/om_0)^3 
            for i = 1:Nx*Ny*Nz]
    S = SpinCollection(pos,μ; gammas=γ_e, deltas=δ_S)

    if (kk == 1)
        E_pos0 = [0.0,0.0,0.0]
        E_polar = [1.0,0.0im,0.0]
        E_polar = E_polar / norm(E_polar)
        E_angle = [0.0,0.0]
        E_width = 0.3*d*sqrt(Nx*Ny)
    elseif (kk == 2)
        E_pos0 = [0.0,0.0,L]
        E_polar = [-1.0,0.0im,0.0]
        E_polar = E_polar / norm(E_polar)
        E_angle = [π,0.0]
        E_width = 0.3*d*sqrt(Nx*Ny)
    end
    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                             position_0 = E_pos0, waist_radius = E_width)

    """Dynamics: meanfield"""

    # E_field vector for Rabi constant computation
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:Nx*Ny*Nz]
    Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

    T = [0:1000.:50000;]
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
    σ_tot[ii,jj,mm,nn,kk] = AtomicArrays.field_module.forward_scattering(r_lim, E_inc,
                                                                S, sm_mat);

    """Transmission"""

    zlim = 500
    n_samp = 400
    #t_tot[ii,jj,mm,nn,kk], _ = AtomicArrays.field_module.transmission_reg(
    #                                        E_inc, em_inc_function,
    #                                        S, sm_mat; samples=n_samp, 
    #                                        zlim=zlim, angle=[π, π]);
    t_tot[ii,jj,mm,nn,kk], _ = AtomicArrays.field_module.transmission_plane(
                                            E_inc, em_inc_function,
                                            S, sm_mat; samples=n_samp, 
                                            zlim=zlim, size=[5, 5]);

end

obj_fun = AtomicArrays.field_module.objective(σ_tot[:,:,:,:,1], σ_tot[:,:,:,:,2])
opt_idx = indexin(maximum(obj_fun), obj_fun)[1]
diode_eff = AtomicArrays.field_module.objective(t_tot[:,:,:,:,1], t_tot[:,:,:,:,2])
opt_idx_t = indexin(maximum(diode_eff), diode_eff)[1]

"""Plots"""

#pygui(true)
gcf()
fig_1, axs = PyPlot.subplots(ncols=3, nrows=3, figsize=(12, 9),
                        constrained_layout=true)
c11 = axs[1,1].contourf(Delt_iter, E_iter, σ_tot[:,:,opt_idx[3],opt_idx[4],1]', 30, cmap="bwr")
axs[1,1].set_xlabel(L"\Delta / \omega_0")
axs[1,1].set_ylabel(L"E_0")
axs[1,1].set_title("Scattering: 0")
#fig_1.colorbar(c11,label=L"\sigma_0")

c12 = axs[1,2].contourf(Delt_iter, E_iter, σ_tot[:,:,opt_idx[3],opt_idx[4],2]', 30, cmap="bwr")
axs[1,2].set_xlabel(L"\Delta / \omega_0")
axs[1,2].set_ylabel(L"E_0")
axs[1,2].set_title("Scattering: π")
#fig_1.colorbar(c12,label=L"\sigma_\pi")

c13 = axs[1,3].contourf(Delt_iter, E_iter, obj_fun[:,:,opt_idx[3],opt_idx[4]]', 30, cmap="bwr")
axs[1,3].set_xlabel(L"\Delta / \omega_0")
axs[1,3].set_ylabel(L"E_0")
axs[1,3].set_title("Objective (larger better)")
#fig_1.colorbar(c13,label=L"\mathcal{M}")

c21 = axs[2,1].contourf(Delt_iter, d_iter, σ_tot[:,opt_idx[2],:,opt_idx[4],1]', 30, cmap="bwr")
axs[2,1].set_xlabel(L"\Delta / \omega_0")
axs[2,1].set_ylabel(L"d/\lambda_0")
axs[2,1].set_title("Scattering: 0")
#fig_1.colorbar(c21,label=L"\sigma_0")

c22 = axs[2,2].contourf(Delt_iter, d_iter, σ_tot[:,opt_idx[2],:,opt_idx[4],2]', 30, cmap="bwr")
axs[2,2].set_xlabel(L"\Delta / \omega_0")
axs[2,2].set_ylabel(L"d/\lambda_0")
axs[2,2].set_title("Scattering: π")
#fig_1.colorbar(c22,label=L"\sigma_\pi")

c23 = axs[2,3].contourf(Delt_iter, d_iter, obj_fun[:,opt_idx[2],:,opt_idx[4]]', 30, cmap="bwr")
axs[2,3].set_xlabel(L"\Delta / \omega_0")
axs[2,3].set_ylabel(L"d/\lambda_0")
axs[2,3].set_title("Objective (larger better)")
#fig_1.colorbar(c23,label=L"\mathcal{M}")

c31 = axs[3,1].contourf(E_iter, d_iter, σ_tot[opt_idx[1],:,:,opt_idx[4],1]', 30, cmap="bwr")
axs[3,1].set_xlabel(L"E_0")
axs[3,1].set_ylabel(L"d/\lambda_0")
axs[3,1].set_title("Scattering: 0")
#fig_1.colorbar(c31,label=L"\sigma_0")

c32 = axs[3,2].contourf(E_iter, d_iter, σ_tot[opt_idx[1],:,:,opt_idx[4],2]', 30, cmap="bwr")
axs[3,2].set_xlabel(L"E_0")
axs[3,2].set_ylabel(L"d/\lambda_0")
axs[3,2].set_title("Scattering: π")
#fig_1.colorbar(c32,label=L"\sigma_\pi")

c33 = axs[3,3].contourf(E_iter, d_iter, obj_fun[opt_idx[1],:,:,opt_idx[4]]', 30, cmap="bwr")
axs[3,3].set_xlabel(L"E_0")
axs[3,3].set_ylabel(L"d/\lambda_0")
axs[3,3].set_title("Objective (larger better)")
#fig_1.colorbar(c33,label=L"\mathcal{M}")

PyPlot.svg(true)
display(fig_1)


fig_1_1, axs = PyPlot.subplots(ncols=3, nrows=3, figsize=(12, 9),
                        constrained_layout=true)
c11 = axs[1,1].contourf(Delt_iter, L_iter, σ_tot[:,opt_idx[2],opt_idx[3],:,1]', 30, cmap="bwr")
axs[1,1].set_xlabel(L"\Delta / \omega_0")
axs[1,1].set_ylabel(L"L/\lambda_0")
axs[1,1].set_title("Scattering: 0")

c12 = axs[1,2].contourf(Delt_iter, L_iter, σ_tot[:,opt_idx[2],opt_idx[3],:,2]', 30, cmap="bwr")
axs[1,2].set_xlabel(L"\Delta / \omega_0")
axs[1,2].set_ylabel(L"L/\lambda_0")
axs[1,2].set_title("Scattering: π")

c13 = axs[1,3].contourf(Delt_iter, L_iter, obj_fun[:,opt_idx[2],opt_idx[3],:]', 30, cmap="bwr")
axs[1,3].set_xlabel(L"\Delta / \omega_0")
axs[1,3].set_ylabel(L"L/\lambda_0")
axs[1,3].set_title("Objective (larger better)")

c21 = axs[2,1].contourf(d_iter, L_iter, σ_tot[opt_idx[1],opt_idx[2],:,:,1]', 30, cmap="bwr")
axs[2,1].set_ylabel(L"d/\lambda_0")
axs[2,1].set_ylabel(L"L/\lambda_0")
axs[2,1].set_title("Scattering: 0")

c22 = axs[2,2].contourf(d_iter, L_iter, σ_tot[opt_idx[1],opt_idx[2],:,:,2]', 30, cmap="bwr")
axs[2,2].set_ylabel(L"d/\lambda_0")
axs[2,2].set_ylabel(L"L/\lambda_0")
axs[2,2].set_title("Scattering: π")

c23 = axs[2,3].contourf(d_iter, L_iter, obj_fun[opt_idx[1],opt_idx[2],:,:]', 30, cmap="bwr")
axs[2,3].set_ylabel(L"d/\lambda_0")
axs[2,3].set_ylabel(L"L/\lambda_0")
axs[2,3].set_title("Objective (larger better)")

c31 = axs[3,1].contourf(E_iter, L_iter, σ_tot[opt_idx[1],:,opt_idx[3],:,1]', 30, cmap="bwr")
axs[3,1].set_xlabel(L"E_0")
axs[3,1].set_ylabel(L"L/\lambda_0")
axs[3,1].set_title("Scattering: 0")

c32 = axs[3,2].contourf(E_iter, L_iter, σ_tot[opt_idx[1],:,opt_idx[3],:,2]', 30, cmap="bwr")
axs[3,2].set_xlabel(L"E_0")
axs[3,2].set_ylabel(L"L/\lambda_0")
axs[3,2].set_title("Scattering: π")

c33 = axs[3,3].contourf(E_iter, L_iter, obj_fun[opt_idx[1],:,opt_idx[3],:]', 30, cmap="bwr")
axs[3,3].set_xlabel(L"E_0")
axs[3,3].set_ylabel(L"L/\lambda_0")
axs[3,3].set_title("Objective (larger better)")

display(fig_1_1)


fig_1.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/fs4D_freq_1.png", dpi=300)
fig_2.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/tr4D_freq_1.png", dpi=300)
write("/home/nikita/Documents/Work/Projects/two_arrays/Data/fs4D_freq_1.bin", σ_tot)
write("/home/nikita/Documents/Work/Projects/two_arrays/Data/obj4D_freq_1.bin", obj_fun)
write("/home/nikita/Documents/Work/Projects/two_arrays/Data/tr4D_freq_1.bin", t_tot)
write("/home/nikita/Documents/Work/Projects/two_arrays/Data/eff4D_freq_1.bin", diode_eff)

fs_0 = σ_tot[opt_idx[1],opt_idx[2],opt_idx[3],opt_idx[4], 1]
fs_π = σ_tot[opt_idx[1],opt_idx[2],opt_idx[3],opt_idx[4], 2]
obj_max = maximum(obj_fun)

Delt_iter[opt_idx[1]]
E_iter[opt_idx[2]]
d_iter[opt_idx[3]]
L_iter[opt_idx[4]]

tr_0 = t_tot[opt_idx_t[1],opt_idx_t[2],opt_idx_t[3],opt_idx_t[4], 1]
tr_π = t_tot[opt_idx_t[1],opt_idx_t[2],opt_idx_t[3],opt_idx_t[4], 2]
eff_max = maximum(diode_eff)

Delt_iter[opt_idx_t[1]]
E_iter[opt_idx_t[2]]
d_iter[opt_idx_t[3]]
L_iter[opt_idx_t[4]]

#write("../Data/test.bin", I)

fig_2, axs = PyPlot.subplots(ncols=3, nrows=3, figsize=(12, 9),
                        constrained_layout=true)
c11 = axs[1,1].contourf(Delt_iter, E_iter, t_tot[:,:,opt_idx_t[3],opt_idx_t[4],1]', 30, cmap="bwr")
axs[1,1].set_xlabel(L"\Delta / \omega_0")
axs[1,1].set_ylabel(L"E_0")
axs[1,1].set_title("Scattering: 0")
#fig_1.colorbar(c11,label=L"\sigma_0")

c12 = axs[1,2].contourf(Delt_iter, E_iter, t_tot[:,:,opt_idx_t[3],opt_idx_t[4],2]', 30, cmap="bwr")
axs[1,2].set_xlabel(L"\Delta / \omega_0")
axs[1,2].set_ylabel(L"E_0")
axs[1,2].set_title("Scattering: π")
#fig_1.colorbar(c12,label=L"\sigma_\pi")

c13 = axs[1,3].contourf(Delt_iter, E_iter, diode_eff[:,:,opt_idx_t[3],opt_idx_t[4]]', 30, cmap="bwr")
axs[1,3].set_xlabel(L"\Delta / \omega_0")
axs[1,3].set_ylabel(L"E_0")
axs[1,3].set_title("Objective (larger better)")
#fig_1.colorbar(c13,label=L"\mathcal{M}")

c21 = axs[2,1].contourf(Delt_iter, d_iter, t_tot[:,opt_idx_t[2],:,opt_idx_t[4],1]', 30, cmap="bwr")
axs[2,1].set_xlabel(L"\Delta / \omega_0")
axs[2,1].set_ylabel(L"d/\lambda_0")
axs[2,1].set_title("Scattering: 0")
#fig_1.colorbar(c21,label=L"\sigma_0")

c22 = axs[2,2].contourf(Delt_iter, d_iter, t_tot[:,opt_idx_t[2],:,opt_idx_t[4],2]', 30, cmap="bwr")
axs[2,2].set_xlabel(L"\Delta / \omega_0")
axs[2,2].set_ylabel(L"d/\lambda_0")
axs[2,2].set_title("Scattering: π")
#fig_1.colorbar(c22,label=L"\sigma_\pi")

c23 = axs[2,3].contourf(Delt_iter, d_iter, diode_eff[:,opt_idx_t[2],:,opt_idx_t[4]]', 30, cmap="bwr")
axs[2,3].set_xlabel(L"\Delta / \omega_0")
axs[2,3].set_ylabel(L"d/\lambda_0")
axs[2,3].set_title("Objective (larger better)")
#fig_1.colorbar(c23,label=L"\mathcal{M}")

c31 = axs[3,1].contourf(E_iter, d_iter, t_tot[opt_idx_t[1],:,:,opt_idx_t[4],1]', 30, cmap="bwr")
axs[3,1].set_xlabel(L"E_0")
axs[3,1].set_ylabel(L"d/\lambda_0")
axs[3,1].set_title("Scattering: 0")
#fig_1.colorbar(c31,label=L"\sigma_0")

c32 = axs[3,2].contourf(E_iter, d_iter, t_tot[opt_idx_t[1],:,:,opt_idx_t[4],2]', 30, cmap="bwr")
axs[3,2].set_xlabel(L"E_0")
axs[3,2].set_ylabel(L"d/\lambda_0")
axs[3,2].set_title("Scattering: π")
#fig_1.colorbar(c32,label=L"\sigma_\pi")

c33 = axs[3,3].contourf(E_iter, d_iter, diode_eff[opt_idx_t[1],:,:,opt_idx_t[4]]', 30, cmap="bwr")
axs[3,3].set_xlabel(L"E_0")
axs[3,3].set_ylabel(L"d/\lambda_0")
axs[3,3].set_title("Objective (larger better)")
#fig_1.colorbar(c33,label=L"\mathcal{M}")

PyPlot.svg(true)
display(fig_2)