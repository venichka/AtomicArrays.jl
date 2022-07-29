# Two arrays: varying the frequency of the second array and other parameters
using Distributed
addprocs(6)

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

    using Revise
    using AtomicArrays
    const EMField = AtomicArrays.field_module.EMField
    const sigma_matrices_mf = AtomicArrays.meanfield_module.sigma_matrices
    const sigma_matrices_mpc = AtomicArrays.mpc_module.sigma_matrices


    dag(x) = conj(transpose(x))

    #em_inc_function = AtomicArrays.field_module.gauss
    em_inc_function = AtomicArrays.field_module.plane
    NMAX = 10
    NMAX_T = 41
    delt_iter = range(0.0, 0.7, NMAX)
    E_iter = range(1e-3, 1.5e-2, NMAX)
    L_iter = range(1e-1, 10e-1, NMAX)
    d_iter = range(1e-1, 10e-1, NMAX)

    """Parameters"""
    c_light = 1.0
    lam_0 = 1.0
    k_0 = 2*π / lam_0
    om_0 = 2.0*pi*c_light / lam_0

    Nx = 4
    Ny = 4
    Nz = 2  # number of arrays
    M = 1 # Number of excitations
    #delt = 0.0035897435897435897 
    #d = 0.4666666
    #L = 0.81
    μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny*Nz]
    γ_e = [1e-2 for i = 1:Nx*Ny*Nz]

    # Incident field parameters
    #E_ampl = 4.5e-3 + 0.0im
    E_kvec = 1.0 * k_0
end

σ_tot = SharedArray{Float64,5}((NMAX,NMAX,NMAX,NMAX,2));


@sync @showprogress @distributed for kkiijjmmnn in CartesianIndices((2, NMAX, NMAX, NMAX, NMAX))
    (kk, ii, jj, mm, nn) = Tuple(kkiijjmmnn)[1], Tuple(kkiijjmmnn)[2], Tuple(kkiijjmmnn)[3], Tuple(kkiijjmmnn)[4], Tuple(kkiijjmmnn)[5]

    delt = delt_iter[ii]
    E_ampl = E_iter[jj] + 0.0im
    d = d_iter[mm]
    L = L_iter[nn]

    d_1 = d
    d_2 = d + delt

    pos_1 = AtomicArrays.geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                                   position_0=[-(Nx-1)*d_1/2, 
                                                               -(Ny-1)*d_1/2,
                                                               -L/2])
    pos_2 = AtomicArrays.geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                                   position_0=[-(Nx-1)*d_2/2, 
                                                               -(Ny-1)*d_2/2,
                                                               L/2])
    pos = vcat(pos_1, pos_2)

    S = SpinCollection(pos,μ; gammas=γ_e, deltas=0.)

    E_width = 0.3*d*sqrt(Nx*Ny)
    if (kk == 1)
        E_pos0 = [0.0,0.0,0.0]
        E_polar = [1.0,0.0im,0.0]
        E_angle = [0.0,0.0]
    elseif (kk == 2)
        E_pos0 = [0.0,0.0,0.0]
        E_polar = [-1.0,0.0im,0.0]
        E_angle = [π,0.0]
    end
    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                             position_0 = E_pos0, waist_radius = E_width)

    """Dynamics: meanfield"""

    # E_field vector for Rabi constant computation
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:Nx*Ny*Nz]
    Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

    tmax = 1/minimum(abs.(GammaMatrix(S)))
    T = [0:tmax/2:tmax;]
    # Initial state (Bloch state)
    phi = 0.
    theta = pi/1.
    # Meanfield
    state0 = CollectiveSpins.meanfield.blochstate(phi, theta, Nx*Ny*Nz)
    tout, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S,
                                                                         Om_R,
                                                                         state0)

    # MPC
    #state0 = CollectiveSpins.mpc.blochstate(phi, theta, Nx*Ny*Nz)
    #tout, state_mpc_t = AtomicArrays.mpc_module.timeevolution_field(T, S, Om_R, state0);


    t_ind = length(T)
    sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices_mf(state_mf_t, t_ind)


    """Forward scattering"""

    r_lim = 1000.0
    σ_tot[ii,jj,mm,nn,kk] = AtomicArrays.field_module.forward_scattering(r_lim, E_inc,
                                                                S, sm_mat);

    #zlim = 40. #0.7*(d+delt)*(Nx)
    #n_samp = 400
    #σ_tot[ii,jj,mm,nn,kk], _ = AtomicArrays.field_module.transmission_reg(
    #                                        E_inc, em_inc_function,
    #                                        S, sm_mat; samples=n_samp, 
    #                                        zlim=zlim, angle=[π, π])
end

#efficiency = AtomicArrays.field_module.objective(σ_tot[:,:,:,:,1], σ_tot[:,:,:,:,2])
efficiency = abs.(σ_tot[:,:,:,:,1] - σ_tot[:,:,:,:,2]) ./ abs.(σ_tot[:,:,:,:,1] + σ_tot[:,:,:,:,2])
opt_idx = indexin(maximum(efficiency), efficiency)[1]

print("delt = ", delt_iter[opt_idx[1]], "\n",
      "E = ", E_iter[opt_idx[2]], "\n",
      "d = ", d_iter[opt_idx[3]], "\n",
      "L = ", L_iter[opt_idx[4]])

"""Plots"""

gcf()
fig_1, axs = PyPlot.subplots(ncols=3, nrows=2, figsize=(12, 9),
                        constrained_layout=true)
c11 = axs[1,1].contourf(delt_iter, E_iter, efficiency[:,:,opt_idx[3],opt_idx[4]]',  cmap="bwr")
axs[1,1].set_xlabel(L"\Delta / \lambda_0")
axs[1,1].set_ylabel(L"E_0")
axs[1,1].set_title("Objective (larger better)")

c12 = axs[1,2].contourf(delt_iter, d_iter, efficiency[:,opt_idx[2],:,opt_idx[4]]',  cmap="bwr")
axs[1,2].set_xlabel(L"\Delta / \lambda_0")
axs[1,2].set_ylabel(L"d/\lambda_0")
axs[1,2].set_title("Objective (larger better)")

c13 = axs[1,3].contourf(E_iter, d_iter, efficiency[opt_idx[1],:,:,opt_idx[4]]',  cmap="bwr")
axs[1,3].set_xlabel(L"E_0")
axs[1,3].set_ylabel(L"d/\lambda_0")
axs[1,3].set_title("Objective (larger better)")

c21 = axs[2,1].contourf(delt_iter, L_iter, efficiency[:,opt_idx[2],opt_idx[3],:]',  cmap="bwr")
axs[2,1].set_xlabel(L"\Delta / \lambda_0")
axs[2,1].set_ylabel(L"L /\lambda_0")
axs[2,1].set_title("Objective (larger better)")

c22 = axs[2,2].contourf(d_iter, L_iter, efficiency[opt_idx[1],opt_idx[2],:,:]',  cmap="bwr")
axs[2,2].set_xlabel(L"d/\lambda_0")
axs[2,2].set_ylabel(L"L/\lambda_0")
axs[2,2].set_title("Objective (larger better)")

c23 = axs[2,3].contourf(E_iter, L_iter, efficiency[opt_idx[1],:,opt_idx[3],:]',  cmap="bwr")
axs[2,3].set_xlabel(L"E_0")
axs[2,3].set_ylabel(L"L/\lambda_0")
axs[2,3].set_title("Objective (larger better)")
PyPlot.svg(true)
display(fig_1)

fig_1.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/obj4D_lattice_10x10_nit10.png", dpi=300)
write("/home/nikita/Documents/Work/Projects/two_arrays/Data/fs4D_lat_gauss.bin", σ_tot)
write("/home/nikita/Documents/Work/Projects/two_arrays/Data/obj4D_lat_gauss.bin", efficiency)

fs_0 = σ_tot[opt_idx[1],opt_idx[2],opt_idx[3], opt_idx[4], 1]
fs_π = σ_tot[opt_idx[1],opt_idx[2],opt_idx[3], opt_idx[4], 2]
obj_max = maximum(efficiency)

delt_iter[opt_idx[1]]
E_iter[opt_idx[2]]
d_iter[opt_idx[3]]
L_iter[opt_idx[4]]

#write("../Data/test.bin", I)
