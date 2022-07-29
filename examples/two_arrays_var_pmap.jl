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
    using LinearAlgebra, DifferentialEquations, Sundials, ODEInterfaceDiffEq

    using Revise
    using AtomicArrays
    const EMField = AtomicArrays.field_module.EMField
    const sigma_matrices_mf = AtomicArrays.meanfield_module.sigma_matrices
    const sigma_matrices_mpc = AtomicArrays.mpc_module.sigma_matrices


    #em_inc_function = AtomicArrays.field_module.gauss
    const em_inc_function = AtomicArrays.field_module.plane
    const NMAX = 10
    const NMAX_T = 41
    dir_list = ["right", "left"]
    delt_list = range(0.0, 0.7, NMAX)
    Delt_list = range(0.0, 0.7, NMAX)
    E_list = range(1e-3, 1.5e-2, NMAX)
    L_list = range(1.5e-1, 10e-1, NMAX)
    d_list = range(1.5e-1, 10e-1, NMAX)

    """Parameters"""
    const c_light = 1.0
    const lam_0 = 1.0
    const k_0 = 2 * π / lam_0
    const om_0 = 2.0 * pi * c_light / lam_0

    const Nx = 6
    const Ny = 6
    const Nz = 2  # number of arrays
    const N = Nx * Ny * Nz
    const M = 1 # Number of excitations
    #delt = 0.0035897435897435897 
    #d = 0.4666666
    #L = 0.81
    const μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny*Nz]
    const γ_e = [1e-2 for i = 1:Nx*Ny*Nz]

    # Incident field parameters
    #E_ampl = 4.5e-3 + 0.0im
    const E_kvec = 1.0 * k_0

    # Function for computing
    function total_scattering(DIRECTION, delt, Delt, d, L, E_ampl)
        d_1 = d
        d_2 = d + delt
        pos_1 = AtomicArrays.geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
            position_0=[-(Nx - 1) * d_1 / 2,
                -(Ny - 1) * d_1 / 2,
                -L / 2])
        pos_2 = AtomicArrays.geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
            position_0=[-(Nx - 1) * d_2 / 2,
                -(Ny - 1) * d_2 / 2,
                L / 2])
        pos = vcat(pos_1, pos_2)

        δ_S = [(ind < Nx * Ny) ? 0.0 : Delt for ind = 1:N]

        S = SpinCollection(pos, μ; gammas=γ_e, deltas=δ_S)

        E_width = 0.3 * d * sqrt(Nx * Ny)
        if (DIRECTION == "right")
            E_pos0 = [0.0, 0.0, 0.0]
            E_polar = [1.0, 0.0im, 0.0]
            E_angle = [0.0, 0.0]
        elseif (DIRECTION == "left")
            E_pos0 = [0.0, 0.0, 0.0]
            E_polar = [-1.0, 0.0im, 0.0]
            E_angle = [π, 0.0]
        end
        E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
            position_0=E_pos0, waist_radius=E_width)

        """Dynamics: meanfield"""

        # E_field vector for Rabi constant computation
        E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:Nx*Ny*Nz]
        Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

        tmax = 10000. #1. / minimum(abs.(GammaMatrix(S)))
        T = [0:tmax/2:tmax;]
        # Initial state (Bloch state)
        phi = 0.0
        theta = pi / 1.0
        # Meanfield
        state0_mf = CollectiveSpins.meanfield.blochstate(phi, theta, N)
        # state = AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, 
                # state0, alg=DynamicSS(AutoVern7(RadauIIA5(), nonstifftol=9//10)))
        _, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S,
           Om_R,
           state0_mf, alg=VCABM(), reltol=1e-10, abstol=1e-12)
        #state0 = state_t_0[end]
        #state = AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, 
        #        state0, alg=SSRootfind())
        # state_t = [CollectiveSpins.meanfield.ProductState(state.u)]

        # MPC
        # state0 = CollectiveSpins.mpc.blochstate(phi, theta, N)
        state0 = AtomicArrays.mpc_module.state_from_mf(state_mf_t[end], phi, theta, N)
 
        state = AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, 
               state0, alg=DynamicSS(VCABM()), reltol=1e-10, abstol=1e-12)
        state_t = [CollectiveSpins.mpc.MPCState(state.u)]
        # _, state_t = AtomicArrays.mpc_module.timeevolution_field(T, S, Om_R, state0, alg=Vern7());

        t_ind = 1
        # t_ind = length(T)
        # _, _, _, sm_mat, _ = sigma_matrices_mf(state_mf_t, t_ind)
        _, _, _, sm_mat, _ = sigma_matrices_mpc(state_t, t_ind)

        """Forward scattering"""
        r_lim = 1000.0
        return AtomicArrays.field_module.forward_scattering(r_lim, E_inc,
            S, sm_mat)
    end

    # Create collection of parameters
    arg_list = [
        [
            dir_list[(i-1) ÷ NMAX^4 + 1],
            delt_list[(i-1) ÷ NMAX^3 % NMAX + 1],
            Delt_list[1],
            d_list[(i-1) ÷ NMAX^2 % NMAX + 1],
            L_list[(i-1) ÷ NMAX % NMAX + 1],
            E_list[(i-1) % NMAX + 1]
        ]
     for i in 1:2*NMAX^4]
end

#σ_tot = pmap((args)->total_scattering(args...), arg_list)
σ_tot_vec = @showprogress pmap(arg_list) do x 
    total_scattering(x...)
end
σ_tot = reshape(σ_tot_vec, (NMAX,NMAX,NMAX,NMAX, 2));

#efficiency = AtomicArrays.field_module.objective(σ_tot[:,:,:,:,1], σ_tot[:,:,:,:,2])
efficiency = abs.(σ_tot[:, :, :, :, 1] - σ_tot[:, :, :, :, 2]) ./ abs.(σ_tot[:, :, :, :, 1] + σ_tot[:, :, :, :, 2]);
opt_idx = indexin(maximum(efficiency), efficiency)[1]

print("E = ", E_list[opt_idx[1]], "\n",
    "L = ", L_list[opt_idx[2]], "\n",
    "d = ", d_list[opt_idx[3]], "\n",
    "delt = ", delt_list[opt_idx[4]])

"""Plots"""

gcf()
fig_1, axs = PyPlot.subplots(ncols=3, nrows=2, figsize=(12, 9),
    constrained_layout=true)
c11 = axs[1, 1].contourf(delt_list, E_list, efficiency[:, opt_idx[2], opt_idx[3], :], cmap="bwr")
axs[1, 1].set_xlabel(L"\Delta / \lambda_0")
axs[1, 1].set_ylabel(L"E_0")
axs[1, 1].set_title("Objective (larger better)")

c12 = axs[1, 2].contourf(delt_list, d_list, efficiency[opt_idx[1], opt_idx[2], :, :], cmap="bwr")
axs[1, 2].set_xlabel(L"\Delta / \lambda_0")
axs[1, 2].set_ylabel(L"d/\lambda_0")
axs[1, 2].set_title("Objective (larger better)")

c13 = axs[1, 3].contourf(E_list, d_list, efficiency[:, opt_idx[2], :, opt_idx[4]]', cmap="bwr")
axs[1, 3].set_xlabel(L"E_0")
axs[1, 3].set_ylabel(L"d/\lambda_0")
axs[1, 3].set_title("Objective (larger better)")

c21 = axs[2, 1].contourf(delt_list, L_list, efficiency[opt_idx[1], :, opt_idx[3], :], cmap="bwr")
axs[2, 1].set_xlabel(L"\Delta / \lambda_0")
axs[2, 1].set_ylabel(L"L /\lambda_0")
axs[2, 1].set_title("Objective (larger better)")

c22 = axs[2, 2].contourf(d_list, L_list, efficiency[opt_idx[1], :, :, opt_idx[4]], cmap="bwr")
axs[2, 2].set_xlabel(L"d/\lambda_0")
axs[2, 2].set_ylabel(L"L/\lambda_0")
axs[2, 2].set_title("Objective (larger better)")

c23 = axs[2, 3].contourf(E_list, L_list, efficiency[:, :, opt_idx[3], opt_idx[4]]', cmap="bwr")
axs[2, 3].set_xlabel(L"E_0")
axs[2, 3].set_ylabel(L"L/\lambda_0")
axs[2, 3].set_title("Objective (larger better)")
PyPlot.svg(true)
display(fig_1)

fs_0 = σ_tot[opt_idx[1], opt_idx[2], opt_idx[3], opt_idx[4], 1]
fs_π = σ_tot[opt_idx[1], opt_idx[2], opt_idx[3], opt_idx[4], 2]
obj_max = maximum(efficiency)

E_list[opt_idx[1]]
L_list[opt_idx[2]]
d_list[opt_idx[3]]
delt_list[opt_idx[4]]

"""Writing DATA"""

using HDF5, FileIO

#fig_1.savefig(PATH_FIGS*"obj4D_lattice_4x4_mpc_nit10.png", dpi=300)

data_dict_obj = Dict("E" => collect(E_list), "L" => collect(L_list), 
                     "d" => collect(d_list), "delt" => collect(delt_list), 
                     "obj" => efficiency,
                     "order" => ["E", "L", "d", "delt"])
data_dict_sig = Dict("E" => collect(E_list), "L" => collect(L_list), 
                     "d" => collect(d_list), "delt" => collect(delt_list), 
                     "dir" => [1, 2],
                     "sigma_tot" => σ_tot,
                     "order" => ["E", "L", "d", "delt", "dir"])
save(PATH_DATA*"obj4D_lat_4x4_mpc.h5", data_dict_obj)
save(PATH_DATA*"fs4D_lat_4x4_mpc.h5", data_dict_sig)

data_dict_loaded = load(PATH_DATA*"obj4D_lat_4x4_mpc.h5")
# data_dict_loaded["obj"] == data_dict_obj["obj"]



#write("../Data/test.bin", I)

# fig_1.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/obj4D_lattice_6x6_mpc_nit10.png", dpi=300)
# write("/home/nikita/Documents/Work/Projects/two_arrays/Data/fs4D_lat_6x6_mpc_nit10.bin", σ_tot)
# write("/home/nikita/Documents/Work/Projects/two_arrays/Data/obj4D_lat_6x6_mpc_nit10.bin", efficiency)