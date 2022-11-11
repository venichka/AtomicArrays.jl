# Calculate effective constants and interaction matrices for all parameters

if pwd()[end-14:end] == "AtomicArrays.jl"
    PATH_ENV = "."
else
    PATH_ENV = "../"
end

using Pkg
Pkg.activate(PATH_ENV)

using CollectiveSpins
using QuantumOptics
using FFTW
using PyPlot
#pygui(true)
PyPlot.svg(true)
using BenchmarkTools, ProgressMeter, Interpolations

using AtomicArrays
const EMField = AtomicArrays.field.EMField
const effective_constants = AtomicArrays.effective_interaction.effective_constants
const sigma_matrices = AtomicArrays.meanfield.sigma_matrices
const mapexpect = AtomicArrays.meanfield.mapexpect

dag(x) = conj(transpose(x))


const PATH_FIG_LS = "/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_eff_constants/"
const PATH_DATA_LS = "/home/nikita/Documents/Work/Projects/two_arrays/Data/effective_constants/"

"""Parameters"""
c_light = 1.0
lam_0 = 1.0
k_0 = 2*π / lam_0
om_0 = 2.0*pi*c_light / lam_0

Nx = 200
Ny = 200

NMAX = 200

"""Variables"""

d1_iter = range(0.1, 2.0, NMAX)
d2_iter = range(0.1, 2.0, NMAX)
L_iter = range(0.1, 2.0, NMAX)
gam_iter = range(0.001, 0.1, NMAX)
Delt_iter = range(-0.5, 0.5, NMAX)


"""Computing effective Γ and Ω"""

args_eff = [d1_iter, Delt_iter]
N_dim_eff = length(args_eff)

Omega_eff = Array{Float64, N_dim_eff}(undef, NMAX, NMAX);
Gamma_eff = Array{Float64, N_dim_eff}(undef, NMAX, NMAX);

N_prog = NMAX^N_dim_eff
p = Progress(N_prog);
update!(p,0)
jj = Threads.Atomic{Int}(0)

lock = Threads.SpinLock()
Threads.@threads for ij in CartesianIndices((NMAX, NMAX))
    (i, j) = Tuple(ij)[1], Tuple(ij)[2]

    d = d1_iter[i]
    Delt = Delt_iter[j]

    # Note that γ_e = 1, so the constants must be normalized by a proper gamma
    Omega_eff[i,j], Gamma_eff[i,j] = effective_constants(d, Delt, 1.0, Nx)

    # Update progress bar
    Threads.atomic_add!(jj, 1)
    Threads.lock(lock)
    update!(p, jj[])
    Threads.unlock(lock)
end

# Plots

fig_0, axs = PyPlot.plt.subplots(ncols=1, nrows=2, figsize=(6, 8),
                        constrained_layout=true)
for i = 1:NMAX
    axs[1].plot(d1_iter, Omega_eff[:, i])
    axs[2].plot(d1_iter, Gamma_eff[:, i])
end
axs[1].plot(d1_iter, zeros(NMAX), "--", color="black")
axs[1].plot(0.2*ones(NMAX), range(-3.0, 1.0, NMAX), "--", color="black")
axs[1].plot(0.8*ones(NMAX), range(-3.0, 1.0, NMAX), "--", color="black")
axs[1].set_ylim([-3,1])
display(fig_0)

# Write to files
fig_0.savefig(PATH_FIG_LS * "consts_eff_"*string(Nx)*".pdf", dpi=300)
write(PATH_DATA_LS * "gamma_eff_" * string(Nx) * ".bin", Gamma_eff)
write(PATH_DATA_LS * "omega_eff_" * string(Nx) * ".bin", Omega_eff)

# Read files
Omega_eff = Array{Float64, N_dim_eff}(undef, NMAX, NMAX);
Gamma_eff = Array{Float64, N_dim_eff}(undef, NMAX, NMAX);

read!(PATH_DATA_LS * "gamma_eff_" * string(Nx) * ".bin", Gamma_eff);
read!(PATH_DATA_LS * "omega_eff_" * string(Nx) * ".bin", Omega_eff);


# Interpolate Γ and Ω effective

Ω_int = LinearInterpolation((d1_iter, Delt_iter), Omega_eff)
Γ_int = LinearInterpolation((d1_iter, Delt_iter), Gamma_eff)

fig_1, axs = PyPlot.plt.subplots(ncols=1, nrows=1, figsize=(6, 4),
                        constrained_layout=true)
d_iter = range(0.1, 2.0, 1000);
axs.plot(d_iter, Ω_int(d_iter, 0.3), color="black")
axs.set_ylim([-3,1])
display(fig_1)


"""Computing Γ21 and Ω21"""

NMAX = 50

d1_iter = range(0.1, 2.0, NMAX)
d2_iter = range(0.1, 2.0, NMAX)
L_iter = range(0.1, 2.0, NMAX)
gam_iter = range(0.001, 0.1, NMAX)
Delt_iter = range(-0.5, 0.5, NMAX)

args = [d1_iter, d2_iter, L_iter, gam_iter, Delt_iter]
N_dim = length(args)

Ω21 = Array{Float32, N_dim}(undef, NMAX, NMAX, NMAX, NMAX, NMAX);
Γ21 = Array{Float32, N_dim}(undef, NMAX, NMAX, NMAX, NMAX, NMAX);

N_prog = NMAX^N_dim
p = Progress(N_prog);
update!(p,0)
jj = Threads.Atomic{Int}(0)

lock = Threads.SpinLock()
Threads.@threads for ijklm in CartesianIndices((NMAX, NMAX, NMAX, NMAX, NMAX))
    (i,j,k,l,m) = Tuple(ijklm)[1], Tuple(ijklm)[2], Tuple(ijklm)[3], Tuple(ijklm)[4], Tuple(ijklm)[5]

    d1 = d1_iter[i]
    d2 = d2_iter[j]
    L = L_iter[k]
    γ_e = gam_iter[l]
    Delt = Delt_iter[m]

    Omega_1, Gamma_1 = γ_e * [Ω_int(d1, -Delt), Γ_int(d1, -Delt)]#effective_constants(d1, -Delt, γ_e, Nx)
    Omega_2, Gamma_2 = γ_e * [Ω_int(d2, Delt), Γ_int(d2, Delt)]

    pos = [[0,0,0], [0,0,L]]
    S_1 = Spin(pos[1], delta=Omega_1 - Delt)
    S_2 = Spin(pos[2], delta=Omega_2 + Delt)
    S = SpinCollection([S_1, S_2], [[1,0,0],[1,0,0]], 
                        gammas=[abs(Gamma_1), abs(Gamma_2)])
    
    Ω21[i,j,k,l,m] = AtomicArrays.interaction.OmegaMatrix(S)[2,1]
    Γ21[i,j,k,l,m] = AtomicArrays.interaction.GammaMatrix(S)[2,1]

    # Update progress bar
    Threads.atomic_add!(jj, 1)
    Threads.lock(lock)
    update!(p, jj[])
    Threads.unlock(lock)
end


# Write to files
write(PATH_DATA_LS * "gamma_21_" * string(NMAX) * ".bin", Γ21)
write(PATH_DATA_LS * "omega_21_" * string(NMAX) * ".bin", Ω21)

minimum(Γ21)

@benchmark effective_constants(x, 0.1, 0.01, 1000) setup=(x=rand())

pos = [[0,0,0], [0,0,0.3]]
S_1 = Spin(pos[1], delta=0.001 - 0.2)
S_2 = Spin(pos[2], delta=-0.02 + 0.2)
S = SpinCollection([S_1, S_2], [[1,0,0],[1,0,0]], 
                    gammas=[0.02, 0.007])
@benchmark  AtomicArrays.interaction.OmegaMatrix(S)
@benchmark SpinCollection([S_1, S_2], [[1,0,0],[1,0,0]], 
                    gammas=[0.02, 0.007])