# One array: Fourier transform of the σ₋

using CollectiveSpins
using QuantumOptics
using PyPlot
using LinearAlgebra, BenchmarkTools

using AtomicArrays
const EMField = AtomicArrays.field.EMField
const sigma_matrices = AtomicArrays.meanfield.sigma_matrices
const mapexpect = AtomicArrays.meanfield.mapexpect

dag(x) = conj(transpose(x))

const NMAX = 100
const NMAX_T = 5


# Parameters
c_light = 1.0

delt_at = 0.0
om_0 = 2*π + delt_at
lam_0 = 2*π*c_light / om_0
Nx = 26
Ny = 26
Nz = 1  # number of arrays
d = 0.8 * lam_0
k_0 = 2*π / lam_0
pos = geometry.rectangle(d, d; Nx=Nx, Ny=Ny)
# shift the origin of the array
p_x0 = pos[1][1]
p_xN = pos[end][1]
p_y0 = pos[1][2]
p_yN = pos[end][2]
for i = 1:Nx*Ny
    pos[i][1] = pos[i][1] - 0.5*(p_x0 + p_xN)
    pos[i][2] = pos[i][2] - 0.5*(p_y0 + p_yN)
end

μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0 + 0.0im, 0.0]./sqrt(1) for i = 1:Nx*Ny]
γ_e = [1e-0 for i = 1:Nx*Ny]
S = SpinCollection(pos,μ; gammas=γ_e, deltas=delt_at)
# Plot arrays (atom distribution)
fig_1 = PyPlot.figure(figsize=(9,4))
x_at = [pos[i][1] for i = 1:Nx*Ny]
y_at = [pos[i][2] for i = 1:Nx*Ny]
z_at = [pos[i][3] for i = 1:Nx*Ny]
PyPlot.scatter3D(x_at, y_at, z_at)

# Collective effects
Ωmat = OmegaMatrix(S)
Γmat = GammaMatrix(S)

# Incident field parameters

om_f = om_0

E_ampl = 0.0001 + 0im
E_kvec = om_f/c_light
E_width = 0.3*d*sqrt(Nx*Ny)
E_pos0 = [0.0,0.0,0.0]
E_polar = [1.0, 0im, 0.0]
E_angle = [0.0*π/6, 0.0]  # {θ, φ}

E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                     position_0 = E_pos0, waist_radius = E_width)
em_inc_function = AtomicArrays.field.gauss
#em_inc_function = AtomicArrays.field.plane


"""Impinging field"""

x = range(-15., 15., NMAX)
y = 0.5*(pos[1][2] + pos[Nx*Ny][2])
z = range(-15., 15., NMAX)
e_field = Matrix{ComplexF64}(undef, length(x), length(z))
for i = 1:length(x)
    for j = 1:length(z)
        e_field[i,j] = em_inc_function([x[i],y,z[j]], E_inc)[1]
    end
end


fig_0 = PyPlot.figure(figsize=(6,5))
PyPlot.contourf(z, x, real(e_field), 30)
for p in pos
    PyPlot.plot(p[3],p[1],"o",color="w",ms=2)
end
PyPlot.xlabel("z")
PyPlot.ylabel("x")
PyPlot.colorbar(label="Amplitude",ticks=[])
PyPlot.tight_layout()


"""Dynamics: meanfield"""

# E_field vector for Rabi constant computation
E_vec = [em_inc_function(S.spins[k].position, E_inc)
         for k = 1:Nx*Ny*Nz]
Om_R = AtomicArrays.field.rabi(E_vec, μ)

const T = [0:1.0:1000;]
# Initial state (Bloch state)
const phi = 0.
const theta = pi/1.
# Meanfield
state0 = CollectiveSpins.meanfield.blochstate(phi, theta, Nx*Ny*Nz)
tout, state_mf_t = AtomicArrays.meanfield.timeevolution_field(T, S, Om_R, state0)


# Expectation values
num_at = 25

sx_mf = mapexpect(CollectiveSpins.meanfield.sx, state_mf_t, num_at)
sy_mf = mapexpect(CollectiveSpins.meanfield.sy, state_mf_t, num_at)
sz_mf = mapexpect(CollectiveSpins.meanfield.sz, state_mf_t, num_at)
sm_mf = 0.5*(sx_mf - 1im.*sy_mf)
sp_mf = 0.5*(sx_mf + 1im.*sy_mf)

fig = PyPlot.figure("Average values",figsize=(5,5))
PyPlot.subplot(311)
PyPlot.plot(T, sx_mf, label="mean field")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_x \rangle")
PyPlot.legend()

PyPlot.subplot(312)
PyPlot.plot(T, sy_mf, label="mean field")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_y \rangle")
PyPlot.legend()

PyPlot.subplot(313)
PyPlot.plot(T, sz_mf, label="mean field")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_z \rangle")
PyPlot.legend()


t_ind = length(T)
sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices(state_mf_t, t_ind)

PyPlot.figure()
PyPlot.plot(abs.(sm_mat),"o")
PyPlot.title(L"\sigma_-(\infty)")


"""Fourier transform of the σ₋ distribution"""

using FFTW

sm_2D = reshape(sm_mat, (Nx, Ny));
sm_2D_fft = sqrt(2)*fftshift(fft(ifftshift( sm_2D )))/sqrt(Nx*Ny)
x_fft = fftfreq(length(x_at[1:Nx]), 2*pi/d) |> fftshift
y_fft = x_fft

fig_1_1, axs = plt.subplots(ncols=2, nrows=1, figsize=(4, 3),
                        constrained_layout=true)
#axs[1].scatter(x_at[1:Nx*Ny]./d, y_at[1:Nx*Ny]./d,
#               c=(abs.(sm_mat[1:Nx*Ny])/
#                   (findmax(abs.(sm_mat[1:Nx*Ny]))[1])),
#        s=40, cmap="Reds")
levels = exp10.(range(log10(1e-8), log10(maximum(abs.(sm_2D_fft))),
                      length=100))
f1 = axs[1].contourf(x_at[1:Nx], x_at[1:Nx], abs.(sm_2D'), cmap="bwr")
axs[1].set_xlabel("x/d")
axs[1].set_ylabel("y/d")
axs[1].set_title(L"|\sigma_-^j|")
f2 = axs[2].contourf(x_fft, y_fft, abs.(sm_2D_fft)', levels=levels,
                norm=matplotlib.colors.LogNorm(), cmap="bwr")
axs[2].set_xlabel("q_x")
axs[2].set_ylabel("q_y")
axs[2].set_title(L"|\mathcal{F}(\sigma_-^j)|")
# colorbar
fig_1_1.colorbar(f1,ax=axs[1], location="bottom")
fig_1_1.colorbar(f2,ax=axs[2], location="bottom")

#write("/Users/jimi/Google Drive/Work/In process/Projects/\
#Collective_effects_QMS/Data/REsm_vec_1arr_E10_n26.bin", real(sm_mat))
#write("/Users/jimi/Google Drive/Work/In process/Projects/\
#Collective_effects_QMS/Data/IMsm_vec_1arr_E10_n26.bin", imag(sm_mat))
