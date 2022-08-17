# Two arrays: nonreciprocity

using QuantumOptics
using PyPlot
using LinearAlgebra, OrdinaryDiffEq, DifferentialEquations
PyPlot.svg(true)

using Revise
using AtomicArrays
const EMField = AtomicArrays.field_module.EMField
const sigma_matrices_mf = AtomicArrays.meanfield_module.sigma_matrices
const sigma_matrices_mpc = AtomicArrays.mpc_module.sigma_matrices


dag(x) = conj(transpose(x))


const PATH_FIGS, PATH_DATA = AtomicArrays.misc_module.path()

const EQ_TYPE = "mf"
const LAT_TYPE = "freq"

# const em_inc_function = AtomicArrays.field_module.gauss
const em_inc_function = AtomicArrays.field_module.plane
const NMAX = 100
const NMAX_T = 41
σ_tot = zeros(NMAX, 2)
t_tot = zeros(NMAX, 2)
delt_iter = range(-0.1, 0.1, NMAX)
E_iter = (10.0) .^ (range(-3.0, -1.0, NMAX))

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
d = 0.24444
delt = 0. #0.156#0.0871859
Delt = 0.022222
d_1 = d
d_2 = d + delt
L = 0.62222 #0.338#0.335678

pos_1 = geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
    position_0=[-(Nx - 1) * d_1 / 2,
        -(Ny - 1) * d_1 / 2,
        -L / 2])
pos_2 = geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
    position_0=[-(Nx - 1) * d_2 / 2,
        -(Ny - 1) * d_2 / 2,
        L / 2])
pos = vcat(pos_1, pos_2)


fig_1 = PyPlot.figure(figsize=(5, 5))
x_at = [vec[1] for vec = pos]
y_at = [vec[2] for vec = pos]
z_at = [vec[3] for vec = pos]
PyPlot.scatter3D(x_at, y_at, z_at)
display(fig_1)

μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:N]
γ_e = [1e-2 for i = 1:Nx*Ny*Nz]
δ_S = [(i < Nx*Ny + 1) ? -0.5*Delt : 0.5*Delt for i = 1:N]
S = SpinCollection(pos, μ; gammas=γ_e, deltas=δ_S)
# Collective effects
Ωmat = OmegaMatrix(S)
Γmat = GammaMatrix(S)

Threads.@threads for kkii in CartesianIndices((2, NMAX))
    (kk, ii) = Tuple(kkii)[1], Tuple(kkii)[2]

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
    μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny*Nz]
    γ_e = [1e-2 for i = 1:Nx*Ny*Nz]
    δ_S = [(i < Nx*Ny + 1) ? 0.0 : Delt for i = 1:Nx*Ny*Nz]
    S = SpinCollection(pos, μ; gammas=γ_e, deltas=δ_S)
    if (kk == 1)
        E_pos0 = [0.0, 0.0, 0.0]
        E_polar = [1.0, 0.0im, 0.0]
        E_angle = [0.0, 0.0]
    elseif (kk == 2)
        E_pos0 = [0.0, 0.0, 0.0]
        E_polar = [-1.0, 0.0im, 0.0]
        E_angle = [π, 0.0]
    end
    # Incident field parameters
    E_ampl = E_iter[ii] + 0.0im
    E_kvec = 1.0 * k_0
    E_width = 0.3 * d * sqrt(Nx * Ny)

    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
        position_0=E_pos0, waist_radius=E_width)

    """Dynamics: meanfield"""

    # E_field vector for Rabi constant computation
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:Nx*Ny*Nz]
    Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

    T = [0:25000.0:50000;]
    # Initial state (Bloch state)
    phi = 0.0
    theta = pi

    "Meanfield"
    state0_mf = AtomicArrays.meanfield_module.blochstate(phi, theta, Nx * Ny * Nz)
    # tout, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S,
    #     Om_R, state0_mf, alg=Vern7(), reltol=1e-10, abstol=1e-12)
    state = AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, 
        state0_mf, alg=SSRootfind(), reltol=1e-10, abstol=1e-12)
    state_mf_t = [AtomicArrays.meanfield_module.ProductState(state.u)]

    if EQ_TYPE == "mpc"
        state0 = AtomicArrays.mpc_module.state_from_mf(state_mf_t[end], phi, theta, N)
    
        # state = AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, 
        #     state0, alg=SSRootfind(), reltol=1e-10, abstol=1e-12)
        # state_t = [AtomicArrays.mpc_module.MPCState(state.u)]

        _, state_t = AtomicArrays.mpc_module.timeevolution_field(T, S,
         Om_R, state0, alg=VCABM(), 
         reltol=1e-10, abstol=1e-12, maxiters=1e10);

        # t_ind = 1
        t_ind = length(T)
        _, _, _, sm_mat, _ = sigma_matrices_mpc(state_t, t_ind)
    elseif EQ_TYPE == "mf"
        t_ind = 1
        # t_ind = length(T)
        _, _, _, sm_mat, _ = sigma_matrices_mf(state_mf_t, t_ind)
    end

    """Forward scattering"""

    r_lim = 1000.0
    σ_tot[ii, kk] = AtomicArrays.field_module.forward_scattering(r_lim, E_inc,
        S, sm_mat)
    zlim = 500#0.7*(d+delt)*(Nx)
    n_samp = 5
    # t_tot[ii, kk], pnts = AtomicArrays.field_module.transmission_reg(
    #     E_inc, em_inc_function,
    #     S, sm_mat; samples=n_samp,
    #     zlim=zlim, angle=[π, π])
    t_tot[ii, kk], pnts = AtomicArrays.field_module.transmission_plane(
        E_inc, em_inc_function,
        S, sm_mat; samples=n_samp,
        zlim=zlim, size=[5, 5])

    println("$kk - $ii")
end

obj = AtomicArrays.field_module.objective(σ_tot[:, 1], σ_tot[:, 2])
efficiency = AtomicArrays.field_module.objective(t_tot[:, 1], t_tot[:, 2])

"""Plots"""

params_text = (
    L"d_1 = " * string(d_1) * "\n" *
    L"d_2 = " * string(round(d_2; digits=3)) * "\n" *
    L"\Delta = " * string(round(Delt; digits=3)) * "\n" *
    L"L = " * string(L) * "\n"
)

function allparam_fig(var, results)
    σ_tot, t_tot, obj, efficiency = results
    fig, ax = PyPlot.subplots(ncols=1, nrows=4, figsize=(6, 9),
        constrained_layout=true)
    ax[1].plot(var, σ_tot[:, 1], label=L"0")
    ax[1].plot(var, σ_tot[:, 2], label=L"\pi")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(L"E_0")
    ax[1].set_title("Scattering: 0, π")
    ax[1].legend()
    ax[1].text(var[NMAX÷6], maximum(σ_tot) / 2, params_text, fontsize=12, va="center")
    ax[2].plot(var, obj, label=L"M")
    ax[2].set_xlabel(L"E_0")
    ax[2].set_xscale("log")
    ax[2].set_title("Objective")
    ax[2].legend()
    ax[3].plot(var, t_tot[:, 1], label=L"0")
    ax[3].plot(var, t_tot[:, 2], label=L"\pi")
    ax[3].set_xscale("log")
    ax[3].set_xlabel(L"E_0")
    ax[3].set_title("Transmission: 0, π")
    ax[3].legend()
    ax[4].plot(var, efficiency, label=L"M")
    ax[4].set_xscale("log")
    ax[4].set_xlabel(L"E_0")
    ax[4].set_title("Diode efficiency")
    ax[4].legend()
    return fig
end


function scatt_fig(var, result)
    obj = AtomicArrays.field_module.objective(result[:, 1], result[:, 2])
    fig, ax = PyPlot.subplots(ncols=1, nrows=1, figsize=(6, 3),
        constrained_layout=true)
    ax.plot(var, result[:, 1], color="r", label=L"\sigma_0")
    ax.plot(var, result[:, 2], color="b", label=L"\sigma_\pi")
    ax.plot(var, obj, "--", color="black", label=L"\mathcal{M}")
    ax.set_xscale("log")
    ax.set_xlabel(L"E_0")
    ax.set_ylabel(L"\sigma_{0,\pi}")
    # ax.set_title("Scattering: 0, π")
    ax.legend()
    # ax.text(var[NMAX÷6], maximum(result) / 2, params_text, fontsize=12, va="center")
    fig.savefig(PATH_FIGS * "Evar_"*string(Nx)*"x"*string(Ny)*"_RL_"*LAT_TYPE*"_"*EQ_TYPE*".pdf", dpi=300)
    return fig
end


function fulleff_comparison_fig(var, results)
    full, efficient = results
    fig, ax = PyPlot.subplots(ncols=1, nrows=1, figsize=(6, 4),
        constrained_layout=true)
    ax.plot(E_iter, full[:, 1], label=L"0", color="r")
    ax.plot(E_iter, full[:, 2], label=L"\pi", color="b")
    ax.plot(10.0 .^ range(-3, -1, 100),
        maximum(full) / maximum(efficient) * efficient[:, 1],
        "--", label=L"0, eff", color="r")
    ax.plot(10.0 .^ range(-3, -1, 100),
        maximum(full) / maximum(efficient) * efficient[:, 2],
        "--", label=L"\pi, eff", color="b")
    ax.set_xscale("log")
    ax.set_xlabel(L"E_0")
    ax.set_title("Scattering: 0, π")
    ax.legend()
    ax.text(E_iter[NMAX÷4], maximum(full) / 3, params_text, fontsize=12, va="center")

    # fig.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/comp_Evar_lat.pdf", dpi=300)
    return fig
end


fig_1 = allparam_fig(E_iter, [σ_tot, t_tot, obj, efficiency])
display(fig_1)

fig_2 = scatt_fig(E_iter, σ_tot)
display(fig_2)