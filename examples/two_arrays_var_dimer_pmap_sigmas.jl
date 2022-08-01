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
    using QuantumOptics
    using CairoMakie
    using LinearAlgebra, EllipsisNotation, DifferentialEquations, Sundials

    using Revise
    using AtomicArrays

    const EMField = AtomicArrays.field_module.EMField
    const sigma_matrices_mf = AtomicArrays.meanfield_module.sigma_matrices
    const sigma_matrices_mpc = AtomicArrays.mpc_module.sigma_matrices

    import EllipsisNotation: Ellipsis
    const .. = Ellipsis()

    PATH_FIGS, PATH_DATA = AtomicArrays.misc_module.path()

    # const em_inc_function = AtomicArrays.field_module.gauss
    const em_inc_function = AtomicArrays.field_module.plane
    const NMAX = 10
    const NMAX_T = 41
    dir_list = ["right", "left"]
    delt_list = range(0.0, 0.7, NMAX)
    a2_list = range(3.0e-1, 10e-1, NMAX)
    E_list = range(1e-3, 1.5e-2, NMAX)
    L_list = range(1.5e-1, 10e-1, NMAX)
    a1_list = range(3.0e-1, 10e-1, NMAX)

    """Parameters"""
    const c_light = 1.0
    const lam_0 = 1.0
    const k_0 = 2 * π / lam_0
    const om_0 = 2.0 * pi * c_light / lam_0

    const Nx = 10
    const Ny = 10
    const Nz = 2  # number of arrays
    const N = Nx * Ny * Nz
    const M = 1 # Number of excitations
    const μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny*Nz]
    const γ_e = [1e-2 for i = 1:Nx*Ny*Nz]

    # Incident field parameters
    const E_kvec = 1.0 * k_0

    # Function for computing
    function total_scattering(DIRECTION, delt, a_2, a_1, L, E_ampl)
        b_2 = a_2
        b_1 = a_1 + delt
        pos_1 = AtomicArrays.geometry_module.dimer_square_1(a_1, a_2; 
                                        Nx=Nx, Ny=Ny,
                                        position_0=[
                                          -0.5*((Nx÷2)*a_1 + (Nx-1)÷2*a_2),
                                          -0.5*((Ny÷2)*a_1 + (Ny-1)÷2*a_2),
                                          -0.5*L
                                        ])
        pos_2 = AtomicArrays.geometry_module.dimer_square_1(b_1, b_2; 
                                        Nx=Nx, Ny=Ny,
                                        position_0=[
                                          -0.5*((Nx÷2)*b_1 + (Nx-1)÷2*b_2),
                                          -0.5*((Ny÷2)*b_1 + (Ny-1)÷2*b_2),
                                          0.5*L
                                        ])
        pos = vcat(pos_1, pos_2)

        δ_S = 0.0

        S = SpinCollection(pos, μ; gammas=γ_e, deltas=δ_S)

        E_width = 0.3 * a_1 * sqrt(Nx * Ny)
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

        tmax = 50000. #1. / minimum(abs.(GammaMatrix(S)))
        T = [0:tmax/2:tmax;]
        # Initial state (Bloch state)
        phi = 0.0
        theta = pi / 1.0
        # Meanfield
        state0_mf = AtomicArrays.meanfield_module.blochstate(phi, theta, N)
        # state = AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, 
                # state0, alg=DynamicSS(AutoVern7(RadauIIA5(), nonstifftol=9//10)))
        _, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S,
           Om_R,
           state0_mf, alg=VCABM(), reltol=1e-10, abstol=1e-12)
        #state0 = state_t_0[end]
        #state = AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, 
        #        state0, alg=SSRootfind())
        # state_t = [AtomicArrays.meanfield_module.ProductState(state.u)]

        # MPC
        # state0 = AtomicArrays.mpc_module.blochstate(phi, theta, N)
        # state0 = AtomicArrays.mpc_module.state_from_mf(state_mf_t[end], phi, theta, N)
 
        # state = AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, 
        #        state0, alg=DynamicSS(VCABM()), reltol=1e-10, abstol=1e-12)
        # state_t = [CollectiveSpins.mpc.MPCState(state.u)]
        # _, state_t = AtomicArrays.mpc_module.timeevolution_field(T, S, Om_R, state0, alg=Vern7());

        # t_ind = 1
        t_ind = length(T)
        _, _, _, sm_mat, _ = sigma_matrices_mf(state_mf_t, t_ind)
        # _, _, _, sm_mat, _ = sigma_matrices_mpc(state_t, t_ind)
        return sm_mat
    end

    # Create collection of parameters
    arg_list = [
        [
            dir_list[(i-1) ÷ NMAX^4 + 1],
            delt_list[(i-1) ÷ NMAX^3 % NMAX + 1],
            a2_list[1],
            # a2_list[(i-1) ÷ NMAX^3 % NMAX + 1],
            a1_list[(i-1) ÷ NMAX^2 % NMAX + 1],
            L_list[(i-1) ÷ NMAX % NMAX + 1],
            E_list[(i-1) % NMAX + 1]
        ]
     for i in 1:2*NMAX^4]
end

sigmas_vec = @showprogress pmap(arg_list) do x 
    total_scattering(x...)
end
DIM = Int8(log(NMAX, length(arg_list)/2)) + 1
sigmas = reshape(sigmas_vec, 
                (N, NMAX, NMAX, NMAX, NMAX, 2));
    

"""Writing DATA"""

using HDF5, FileIO


data_dict_sig = Dict("E" => collect(E_list), "L" => collect(L_list), 
                     "a_1" => collect(a1_list), 
                    #  "a_2" => collect(a2_list), 
                     "delt" => collect(delt_list), 
                     "dir" => [1, 2],
                     "sigma_re" => real(sigmas),
                     "sigma_im" => imag(sigmas),
                     "order" => ["E", "L", "a_1", "delt", "dir"])
save(PATH_DATA*"sig4D_dimer_"*string(Nx)*"x"*string(Ny)*"_mf.h5", data_dict_t)

data_dict_loaded = load(PATH_DATA*"sig4D_dimer_"*string(Nx)*"x"*string(Ny)*"_mf.h5")
# data_dict_loaded["obj"] == data_dict_obj["obj"]



#write("../Data/test.bin", I)

# fig_1.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/obj4D_lattice_6x6_mpc_nit10.png", dpi=300)
# write("/home/nikita/Documents/Work/Projects/two_arrays/Data/fs4D_lat_6x6_mpc_nit10.bin", σ_tot)
# write("/home/nikita/Documents/Work/Projects/two_arrays/Data/obj4D_lat_6x6_mpc_nit10.bin", efficiency)
