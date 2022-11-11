# Comparison with Shahmoon results. 2D lattice of spins: incident field and evolution

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
d = 0.707 * lam_0
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

μ = [(i < 0) ? [0, 0, 1.0] : [0.0, 1.0 + 0.0im, 0.0]./sqrt(1) for i = 1:Nx*Ny]
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

E_ampl = 0.001 + 0im
E_kvec = om_f/c_light
E_width = 3.#0.3*d*sqrt(Nx*Ny)
E_pos0 = [0.0,0.0,0.0]
E_polar = [0.0, 1.0 + 0im, 0.0]
E_angle = [pi-1.0*π/6, 0.0]  # {θ, φ}

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
        e_field[i,j] = em_inc_function([x[i],y,z[j]], E_inc)[2]
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


"""Compute the radiation pattern"""

VIEW = "x-z"

if VIEW == "x-y"
# X-Y view
    x = range(pos[1][1] - 3d, pos[Nx*Ny][1] + 3d, NMAX)
    y = range(pos[1][2] - 3d, pos[Nx*Ny][2] + 3d, NMAX)
    z = 2d
    t_ind = length(T)
    I = zeros(length(x), length(y))
    Threads.@threads for i=1:length(x)
        for j=1:length(x)
            I[i,j] = (norm(AtomicArrays.field.total_field(em_inc_function,
                                                                 [x[i],y[j],z],
                                                                 E_inc, S,
                                                                 sm_mat))^2 /
                        abs(E_ampl)^2)
            print(i, "  ", j,"\n")
        end
    end

    # Plot
    I_arr = [I[(i - 1)%length(x) + 1, (i - 1)÷length(x)+1] for i = 1:length(x)^2]
    I_arr_sorted = sort(I_arr, rev=true)[Nx*Ny*Nz:end]  # sorted and truncated Intensity
    I_arr_av = (sum(I_arr_sorted) / (length(I_arr_sorted)))
    I_arr_max = findmax(I_arr)[1]


    fig_2 = PyPlot.figure(figsize=(9,4))
    # Lin scale
    levels = 0:I_arr_max*1e-2:I_arr_max*1e-0
    PyPlot.contourf(x,y,I',30, levels=levels)

    # Log scale
    #levels = exp10.(range(log10((I_arr_sorted[end])), log10(I_arr_sorted[1]),
    #                      length=100))
    #PyPlot.contourf(x,y,I',30, levels=levels, norm=matplotlib.colors.LogNorm())
    for p in pos
        PyPlot.plot(p[1],p[2],"o",color="w",ms=2)
    end
    PyPlot.xlabel(L"x/\lambda_0")
    PyPlot.ylabel(L"y/\lambda_0")
    PyPlot.colorbar(label=L"|E_{tot}|^2 / |E_0|^2")
    PyPlot.tight_layout()
elseif VIEW == "x-z"
# X-Z view
    x = range(-15., 15., NMAX)
    y = 0.5*(pos[1][2]+pos[Nx*Ny][2])
    z = range(-15., 15., NMAX)
    t_ind = length(T)
    I = zeros(length(x), length(z))
    Threads.@threads for i=1:length(x)
        for j=1:length(z)
            I[i,j] = (norm(AtomicArrays.field.total_field(em_inc_function,
                                                                 [x[i],y,z[j]],
                                                                 E_inc, S,
                                                                 sm_mat))^2 /
                        abs(E_ampl)^2)
            print(i, "  ", j,"\n")
        end
    end

    # Plot
    I_arr = [I[(i - 1)÷length(z) + 1, (i - 1)%length(z)+1]
             for i = 1:length(x)*length(z)]
    I_arr_sorted = sort(I_arr, rev=true)[Nx*Ny*Nz:end]  # sorted and truncated Intensity
    I_arr_av = (sum(I_arr_sorted) / (length(I_arr_sorted)))
    I_arr_max = findmax(I_arr)[1]


    fig_2 = PyPlot.figure(figsize=(9,4))
    # Lin scale
    levels = 0:I_arr_max*1e-2:I_arr_max*1e-0
    PyPlot.contourf(z,x,I,30, levels=levels)

    # Log scale
    #levels = exp10.(range(log10((I_arr_sorted[end])), log10(I_arr_sorted[1]),
    #                      length=100))
    #PyPlot.contourf(z,x,I,30, levels=levels, norm=matplotlib.colors.LogNorm())
    for p in pos
        PyPlot.plot(p[3],p[1],"o",color="w",ms=2)
    end
    PyPlot.xlabel(L"z/\lambda_0")
    PyPlot.ylabel(L"x/\lambda_0")
    PyPlot.colorbar(label=L"|E_{tot}|^2 / |E_0|^2")
    PyPlot.tight_layout()
end


"""Field study"""
fig_3 = PyPlot.figure()
PyPlot.plot(z, I[length(x)÷2,:])
PyPlot.title(L"|E_{tot}|^2 / |E_0|^2")


xf =0.2*d #0.5*(pos[1][1] + pos[Nx*Ny][1])
yf =0.2*d #0.5*(pos[1][2] + pos[Nx*Ny][2])
zf = range(-8., 8., 2*NMAX)
e_field_x = zeros(ComplexF64, length(zf))
e_tot_x = zeros(ComplexF64, length(zf))
for i = 1:length(zf)
    e_field_x[i] = (em_inc_function([xf,yf,zf[i]], E_inc)[1])
    e_tot_x[i] = (AtomicArrays.field.total_field(em_inc_function,
                                                                 [xf,yf,zf[i]],
                                                                 E_inc, S,
                                                                 sm_mat)[1])
end

fig_5, axs = plt.subplots(ncols=1, nrows=2, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].plot(zf, real(e_field_x))
axs[1].plot(zf, real(e_tot_x - e_field_x))
axs[1].set_xlabel(L"z/\lambda_0")
axs[1].set_ylabel(L"Re(E_{in}), Re(E_{sc})")
axs[2].plot(zf, imag(e_field_x))
axs[2].plot(zf, imag(e_tot_x - e_field_x))
axs[2].set_xlabel(L"z/\lambda_0")
axs[2].set_ylabel(L"Im(E_{in}), Im(E_{sc})")


"""Transmission"""
xlim = 0.0001
ylim = 0.0001
zlim = (E_angle[1] >= π/2) ? -1000.0 : 1000.0
x_t = range(-xlim, xlim, NMAX_T)
y_t = range(-ylim, ylim, NMAX_T)
E_out = sum(E_inc.polarisation'*AtomicArrays.field.total_field(em_inc_function,
                                                                 [x_t[i],y_t[j],zlim],
                                                                 E_inc, S,
                                                                 sm_mat)
            for i = 1:NMAX_T, j = 1:NMAX_T)
E_in = sum(E_inc.polarisation'*em_inc_function([x_t[i],y_t[j],zlim], E_inc)
            for i = 1:NMAX_T, j = 1:NMAX_T)
transmission = abs.(E_out)^2 ./ abs.(E_in)^2


const NMAX_T = 100
xlim = 5.0001
ylim = 5.0001
zlim = (E_angle[1] >= π/2) ? -d*Nx/2. : d*Nx/2
x_t = range(-xlim, xlim, NMAX_T)
y_t = range(-ylim, ylim, NMAX_T)
E_out = [norm(AtomicArrays.field.total_field(em_inc_function,
                                                           [x_t[i],y_t[j],zlim],
                                                           E_inc,
                                                           S, sm_mat))
            for i = 1:NMAX_T, j = 1:NMAX_T]
E_in = [norm(em_inc_function([x_t[i],y_t[j],zlim], E_inc))
            for i = 1:NMAX_T, j = 1:NMAX_T]
transmission = (E_out ./ E_in).^2
sum(transmission)/NMAX_T^2

PyPlot.figure()
PyPlot.contourf(x_t,y_t, transmission', 100)
PyPlot.colorbar()

zlim = 1*d*(Nx)
@time tran, points = AtomicArrays.field.transmission_reg(E_inc, em_inc_function,
                                       S, sm_mat; samples=40000, zlim=zlim, angle=[π, π]);
tran


"""Plot radiation pattern"""

using GLMakie

x_p = [points[i][1] for i = 1:length(points)]
y_p = [points[i][2] for i = 1:length(points)]
z_p = [points[i][3] for i = 1:length(points)]

#PyPlot.figure()
#PyPlot.scatter3D(x_p, y_p,z_p)

tf = AtomicArrays.field.total_field
positions = vec([(x_p[i], y_p[i], z_p[i]) for i in 1:length(x_p)])
vals = [norm(tf(em_inc_function,points[ip],E_inc,S, sm_mat))^2/norm(E_ampl)^2
        for ip in 1:length(points)]
fig, ax, pltobj = GLMakie.meshscatter(positions, color = vec(vals),
    marker = FRect3D(Vec3f0(3), Vec3f0(3)), # here, if you use less than 10, you will see smaller squares.
    colormap = :bwr, colorrange = (minimum(vals), maximum(vals)),
    transparency = false, # set to false, if you don't want the transparency.
    shading= true,
    figure = (; resolution = (800,800)),
    axis=(; type=Axis3, perspectiveness = 0.5,  azimuth = 7.19, elevation = 0.57,
        xlabel = "x", ylabel = "y", zlabel = "z",
        aspect = (1,1,1/2)))
cbar = GLMakie.Colorbar(fig, pltobj, label = L"|E_{tot}|^2 / |E_0|^2", height = Relative(0.5))
GLMakie.xlims!(ax,-1.2*zlim,1.2*zlim)
GLMakie.ylims!(ax,-1.2*zlim,1.2*zlim)
GLMakie.zlims!(ax, (E_angle[1] >= π/2) ? -1.2*zlim : 0,
               (E_angle[1] >= π/2) ? 0 : 0 + 1.2*zlim)
fig[1,2] = cbar
fig

#write("../Data/test.bin", I)
