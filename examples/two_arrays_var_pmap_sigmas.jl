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

    const PATH_FIGS, PATH_DATA = AtomicArrays.misc_module.path()

    const EQ_TYPE = "mf"

    # const em_inc_function = AtomicArrays.field_module.gauss
    const em_inc_function = AtomicArrays.field_module.plane
    const NMAX = 10
    const NMAX_T = 41
    dir_list = ["right", "left"]
    delt_list = range(0.0, 0.7, NMAX)
    Delt_list = range(0.0, 0.2, NMAX)
    E_list = range(5e-3, 2.5e-2, NMAX)
    L_list = range(1.5e-1, 10e-1, NMAX)
    d_list = range(1.5e-1, 10e-1, NMAX)

    """Parameters"""
    const c_light = 1.0
    const lam_0 = 1.0
    const k_0 = 2 * π / lam_0
    const om_0 = 2.0 * pi * c_light / lam_0

    const Nx =10 
    const Ny = 10
    const Nz = 2  # number of arrays
    const N = Nx * Ny * Nz
    const M = 1 # Number of excitations
    const μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny*Nz]
    const γ_e = [1e-2 for i = 1:Nx*Ny*Nz]

    # Incident field parameters
    const E_kvec = 1.0 * k_0

    # Function for computing
    function total_scattering(DIRECTION, delt, Delt, d, L, E_ampl)
        d_1 = d
        d_2 = d + delt
        pos_1 = geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
            position_0=[-(Nx - 1) * d_1 / 2,
                -(Ny - 1) * d_1 / 2,
                -L / 2])
        pos_2 = geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
            position_0=[-(Nx - 1) * d_2 / 2,
                -(Ny - 1) * d_2 / 2,
                L / 2])
        pos = vcat(pos_1, pos_2)

        δ_S = [(ind < Nx * Ny) ? -0.5*Delt : 0.5*Delt for ind = 1:N]

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

        tmax = 50000. #1. / minimum(abs.(GammaMatrix(S)))
        T = [0:tmax/2:tmax;]
        # Initial state (Bloch state)
        phi = 0.0
        theta = pi / 1.0
        # Meanfield
        state0_mf = AtomicArrays.meanfield_module.blochstate(phi, theta, N)
        _, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S,
           Om_R,
           state0_mf, alg=VCABM(), reltol=1e-10, abstol=1e-12)

        if EQ_TYPE == "mpc"
            state0 = AtomicArrays.mpc_module.state_from_mf(state_mf_t[end], phi, theta, N)
    
            # state = AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, 
            #     state0, alg=DynamicSS(VCABM()), reltol=1e-10, abstol=1e-12)
            # state_t = [AtomicArrays.mpc_module.MPCState(state.u)]

            _, state_t = AtomicArrays.mpc_module.timeevolution_field(T, S, Om_R, state0, alg=VCABM(), reltol=1e-10, abstol=1e-12);

            # t_ind = 1
            t_ind = length(T)
            _, _, _, sm_mat, _ = sigma_matrices_mpc(state_t, t_ind)
        elseif EQ_TYPE == "mf"
            t_ind = length(T)
            _, _, _, sm_mat, _ = sigma_matrices_mf(state_mf_t, t_ind)
        end

        return sm_mat
    end

    # Create collection of parameters
    arg_list = [
        [
            dir_list[(i-1) ÷ NMAX^4 + 1],
            delt_list[1],
            Delt_list[(i-1) ÷ NMAX^3 % NMAX + 1],
            d_list[(i-1) ÷ NMAX^2 % NMAX + 1],
            L_list[(i-1) ÷ NMAX % NMAX + 1],
            E_list[(i-1) % NMAX + 1]
        ]
     for i in 1:2*NMAX^4]
end

sigmas_vec = @showprogress pmap(arg_list) do x 
    total_scattering(x...)
end
sigmas_mat = mapreduce(permutedims, vcat, sigmas_vec); # convert to matrix
DIM = Int8(log(NMAX, length(arg_list)/2)) + 1
sigmas = reshape(sigmas_mat, 
                Tuple(push!([(i < DIM) ? NMAX : 2 
                        for i=1:DIM], N)));
    

"""Writing DATA"""

using HDF5, FileIO


data_dict_sig = Dict("E" => collect(E_list), "L" => collect(L_list), 
                     "d" => collect(d_list), "delt" => collect(Delt_list), 
                     "dir" => [1, 2],
                     "sigma_re" => real(sigmas),
                     "sigma_im" => imag(sigmas),
                     "order" => ["E", "L", "d", "delt", "dir"])

NAME_PART = string(Nx)*"x"*string(Ny)*"_"*EQ_TYPE*".h5"
save(PATH_DATA*"sig4D_freq_"*NAME_PART, data_dict_sig)

data_dict_loaded = load(PATH_DATA*"sig4D_freq_"*NAME_PART)
data_dict_loaded["sigma_re"] == data_dict_sig["sigma_re"]



#write("../Data/test.bin", I)

# fig_1.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/obj4D_lattice_6x6_mpc_nit10.png", dpi=300)
# write("/home/nikita/Documents/Work/Projects/two_arrays/Data/fs4D_lat_6x6_mpc_nit10.bin", σ_tot)
# write("/home/nikita/Documents/Work/Projects/two_arrays/Data/obj4D_lat_6x6_mpc_nit10.bin", efficiency)
