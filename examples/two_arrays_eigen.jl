#  2D lattice of spins: incident field and evolution

# using CollectiveSpins
using Revise
using AtomicArrays
using QuantumOptics
using PyPlot
PyPlot.svg(true)

# Parameters

c_light = 1.0
lam_0 = 1.0
k_0 = 2*π / lam_0
om_0 = 2.0*pi*c_light / lam_0

Nx = 10
Ny = 10
Nz = 2  # number of arrays
M = 1 # Number of excitations
d = 0.1888
delt = 0.055
d_1 = d
d_2 = d + 0.0
L = 0.7
pos_1 = AtomicArrays.geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_1/2, 
                                                           -(Ny-1)*d_1/2,
                                                           -L/2])
pos_2 = AtomicArrays.geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_2/2, 
                                                           -(Ny-1)*d_2/2,
                                                           L/2])
pos = vcat(pos_1, pos_2)
μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0im, 0.0] for i = 1:Nx*Ny*Nz]
γ_e = [1e-2 for i = 1:Nx*Ny*Nz]
δ_S = [(i < Nx*Ny+1) ? 0.0 : delt for i = 1:Nx*Ny*Nz]
S = SpinCollection(pos,μ; gammas=γ_e, deltas=δ_S)

# Plot arrays (atom distribution)
fig_1 = figure(figsize=(7,3.5))
x_at = [pos[i][1] for i = 1:Nx*Ny*Nz]
y_at = [pos[i][2] for i = 1:Nx*Ny*Nz]
z_at = [pos[i][3] for i = 1:Nx*Ny*Nz]
scatter3D(x_at, y_at, z_at)

# Collective effects
Ωmat = OmegaMatrix(S)
Γmat = GammaMatrix(S)

# Hilbert space
b = ReducedSpinBasis(Nx*Ny*Nz,M,M) # Basis from M excitations up to M excitations

# b = ReducedSpinBasis(3,2,0) # Basis from M excitations up to M excitations
# sp = [reducedsigmap(b, i) for i=1:3]
# sm = [reducedsigmam(b, i) for i=1:3]
# spsm = [reducedsigmapsigmam(b, i, j) for i=1:3, j=1:3]

# kron([1,0],[0,1],[1,0])

# Effective Hamiltonian
spsm = [reducedsigmapsigmam(b, i, j) for i=1:Nx*Ny*Nz, j=1:Nx*Ny*Nz]
H_eff = dense(sum((Ωmat[i,j] - 0.5im*Γmat[i,j])*spsm[i, j]
                  for i=1:Nx*Ny*Nz, j=1:Nx*Ny*Nz))

# Find the most subradiant eigenstate
λ, states = eigenstates(H_eff; warning=false)
γ = -2 .* imag.(λ)
#s_ind = findmax(γ)[2]
s_ind = sortperm(γ)[1]
ψ = states[s_ind]


"""Plot eigenstates"""

# Parity (even: true, odd: false)
parity_n = [(sign(real(states[n].data[1]))==
    sign(real(states[n].data[Nx*Ny+1]))) ? true : false for n = 1:Nx*Ny*Nz]


fig_1_1, axs = plt.subplots(ncols=2, nrows=1, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].scatter(x_at[1:Nx*Ny]./d, y_at[1:Nx*Ny]./d,
               c=(abs.((states[s_ind].data)[1:Nx*Ny])/
                   (findmax(abs.((states[s_ind].data)[1:Nx*Ny]))[1])),
        s=200, cmap="Reds")
axs[1].set_xlabel("x/d")
axs[1].set_ylabel("y/d")
axs[1].set_title("array 1")
axs[2].scatter(x_at[Nx*Ny+1:end]./d, y_at[Nx*Ny+1:end]./d,
               c=(abs.((states[s_ind].data)[Nx*Ny+1:end])/
                   (findmax(abs.((states[s_ind].data)[Nx*Ny+1:end]))[1])),
        s=200, cmap="Reds")
axs[2].set_xlabel("x/d")
axs[2].set_ylabel("y/d")
axs[2].set_title("array 2")
display(fig_1_1)

figure()
PyPlot.plot((abs.((states[s_ind].data)[1:Nx*Ny])/
                   (findmax(abs.((states[s_ind].data)[1:Nx*Ny]))[1])))
PyPlot.plot((abs.((states[s_ind].data)[1:Nx*Ny])/
                   (findmax(abs.((states[s_ind].data)[1:Nx*Ny]))[1])), "o")
display(gcf())


"""Eigenvalues depending on quasimomentum"""

using FFTW

r_perp = [d.*[(j-1)%(Nx), (j-1)÷Nx] for j = 1:Nx*Ny]
q_perp = [-π/d .+ 2*π.*(r_perp[j]./d) ./ ((Nx)*d) for j = 1:Nx*Ny]
states_f = [sum([sqrt(2)*(states[i].data)[j]*exp(1im*r_perp[j]'*q_perp[k])/sqrt(Nx*Ny)
            for j = 1:Nx*Ny]) for i = 1:Nx*Ny*Nz, k = 1:Nx*Ny]
states_f_2D = [([states_f[i, (Nx)*(j-1) + (k-1)%(Ny) + 1] for j=1:Nx, k=1:Ny])
           for i=1:Nx*Ny*Nz]

q_quasi = [sum([abs(states_f[i, k])^2*norm(q_perp[k]) for k = 1:Nx*Ny])
               for i = 1:Nx*Ny*Nz]

# fft approach
q_perp_x = [-π/d + 2*π*(i-1) / d/(Nx) for i=1:Nx]
states_2D = [[states[i].data[(Nx)*(j-1) + (k-1)%(Ny) + 1]
              for j=1:Nx, k=1:Ny] for i = 1:Nx*Ny*Nz]
states_2D_fft = [sqrt(2)*fftshift(fft(states_2D[i]))/sqrt(Nx*Ny) for i=1:Nx*Ny*Nz]

q_quasi_fft = [sum([sum([abs(states_2D_fft[i][j,k])^2*
                         norm([q_perp_x[j], q_perp_x[k]])
                         for k = 1:Ny]) for j=1:Nx])
               for i = 1:Nx*Ny*Nz]

s_ind = sortperm(q_quasi)[1]

s_ind = sortperm(γ, rev=false)[1]
q_quasi[s_ind]
γ[s_ind]

ψ = states[s_ind]

PyPlot.figure()
PyPlot.contourf(abs.(states_2D[s_ind]))
PyPlot.xlabel("x/d")
PyPlot.ylabel("y/d")
display(gcf())

PyPlot.figure()
x_fft = fftfreq(length(x_at[1:Nx]), 2*pi/d) |> fftshift
y_fft = x_fft
PyPlot.contourf(x_fft, y_fft, abs.(states_2D_fft[s_ind]))
PyPlot.xlabel("q_x")
PyPlot.ylabel("q_y")
display(gcf())

PyPlot.figure()
PyPlot.contourf(q_perp_x, q_perp_x, abs.(states_f_2D[s_ind]))
PyPlot.xlabel("q_x")
PyPlot.ylabel("q_y")
display(gcf())

PyPlot.figure()
PyPlot.contourf(abs.(states_f'))
display(gcf())

PyPlot.figure()
PyPlot.plot(q_quasi)
PyPlot.plot(q_quasi_fft)
display(gcf())

PyPlot.figure()
PyPlot.plot(q_quasi[parity_n]./k_0, γ[parity_n]./γ_e[1], "o", color="red")
PyPlot.plot(q_quasi[.!parity_n]./k_0, γ[.!parity_n]./γ_e[1], "o", color="black")
PyPlot.plot(q_quasi_fft./k_0, γ./γ_e, "o", alpha=0.1)
PyPlot.xscale("log")
PyPlot.yscale("log")
PyPlot.xlabel(L"\bar{q}/k_0")
PyPlot.ylabel(L"\gamma_n/\gamma_e")
display(gcf())

PyPlot.figure()
PyPlot.plot(γ[sortperm(γ)[parity_n]]./γ_e[1], "o", color="red")
PyPlot.plot(γ[sortperm(γ)[.!parity_n]]./γ_e[1], "o", color="black")
PyPlot.xscale("log")
PyPlot.yscale("log")
PyPlot.xlabel(L"n")
PyPlot.ylabel(L"\gamma_n/\gamma_e")
display(gcf())

PyPlot.figure()
PyPlot.plot(γ[sortperm(γ)]./γ_e[1], "o", color="blue")
PyPlot.xscale("log")
PyPlot.yscale("log")
PyPlot.xlabel(L"n")
PyPlot.ylabel(L"\gamma_n/\gamma_e")
display(gcf())

"""Compute the radiation pattern"""

function G(r,i,j) # Green's Tensor overlap
    G_i = GreenTensor(r-pos[i])
    G_j = GreenTensor(r-pos[j])
    return μ[i]' * (G_i'*G_j) * μ[j]
end
function intensity(r) # The intensity ⟨E⁻(r)⋅E⁺(r)⟩
    real(sum(expect(spsm[i,j], ψ)*G(r,i,j) for i=1:Nx*Ny*Nz, j=1:Nx*Ny*Nz))
end

VIEW = "x-z"
#VIEW = "x-z"

if VIEW == "x-y"
# X-Y view
    x = range(-1d+pos[Nx*Ny+1][1], pos[end][1]+1d, 50)
    y = range(-1d+pos[Nx*Ny+1][1], pos[end][1]+1d, 50)
    z = L + 0.4d
    I = zeros(length(x), length(y))
    Threads.@threads for i=1:length(x)
        for j=1:length(y)
            I[i,j] = intensity([x[i],y[j],z])
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
    PyPlot.contourf(x./d,y./d,I',30, levels=levels)

    # Log scale
    #levels = exp10.(range(log10((I_arr_sorted[end])), log10(I_arr_sorted[1]),
    #                      length=100))
    #contourf(x./d,z./d,I',30, levels=levels, norm=matplotlib.colors.LogNorm())
    for p in pos
        PyPlot.plot(p[1]./d,p[2]./d,"o",color="w",ms=2)
    end
    PyPlot.xlabel("x/d")
    PyPlot.ylabel("y/d")
    PyPlot.colorbar(label="Intensity",ticks=[])
    PyPlot.tight_layout()
elseif VIEW == "x-z"
# X-Z view
    x = range(-1d+pos[Nx*Ny+1][1], pos[end][1]+1d, 50)
    y = 0.0#0.5*(pos[Nx*Ny+7*Nx+1][2] + pos[end][2])
    z = range(-3d+pos[1][3], pos[end][3]+3d, 50)
    I = zeros(length(x), length(z))
    Threads.@threads for i=1:length(x)
        for j=1:length(z)
            I[i,j] = intensity([x[i],y,z[j]])
            print(i, "  ", j,"\n")
        end
    end

    # Plot
    I_arr = [I[(i - 1)%length(x) + 1, (i - 1)÷length(z)+1]
             for i = 1:length(x)*length(z)]
    I_arr_sorted = sort(I_arr, rev=true)[Nx*Ny*Nz:end]  # sorted and truncated Intensity
    I_arr_av = (sum(I_arr_sorted) / (length(I_arr_sorted)))
    I_arr_max = findmax(I_arr)[1]


    fig_2 = PyPlot.figure(figsize=(9,4))
    # Lin scale
    levels = 0:I_arr_max*1e-2:I_arr_max*1e-0
    #levels = 0:1e-4:2e-2
    PyPlot.contourf(x./d,z./d,I',30, levels=levels)

    # Log scale
    #levels = exp10.(range(log10((I_arr_sorted[end])), log10(I_arr_sorted[1]),
    #                      length=100))
    #contourf(x./d,z./d,I',30, levels=levels, norm=matplotlib.colors.LogNorm())
    for p in pos
        PyPlot.plot(p[1]./d,p[3]./d,"o",color="w",ms=2)
    end
    PyPlot.xlabel("x/d")
    PyPlot.ylabel("z/d")
    PyPlot.colorbar(label="Intensity",ticks=[])
    PyPlot.tight_layout()
end
display(fig_2)

fig_2.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_eigen/ds_N10_freq_opt_0_xz.pdf", dpi=300)

#write("../Data/test.bin", I)

"""Calculate the collective shift depending on the lattice constant"""

function collective_shift(d, L, N)
    Nx = N
    Ny = N
    Nz = 2  # number of arrays
    delt = 0.0
    d_1 = d
    d_2 = d + delt
    pos_1 = geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny, position_0=[0.,0.,0.])
    pos_2 = geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny, position_0=[0.,0.,L])
    pos = vcat(pos_1, pos_2)
    # shift the origin of the array
    #p_x0 = pos[1][1]
    #p_xN = pos[Nx*Ny][1]
    #p_y0 = pos[1][2]
    #p_yN = pos[Nx*Ny][2]
    #for i = 1:Nx*Ny
    #    pos[i][1] = pos[i][1] - 0.5*(p_x0 + p_xN)
    #    pos[i][2] = pos[i][2] - 0.5*(p_y0 + p_yN)
    #    #pos[i+Nx*Ny][1] = (pos[i+Nx*Ny][1] - 0.5*(p_x0 + p_xN)) * d_2/d_1
    #    #pos[i+Nx*Ny][2] = (pos[i+Nx*Ny][2] - 0.5*(p_y0 + p_yN)) * d_2/d_1
    #end
    μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0im, 0.0] for i = 1:Nx*Ny*Nz]
    γ_e = [1e-0 for i = 1:Nx*Ny*Nz]
    S = SpinCollection(pos,μ; gammas=γ_e)
    Omega, Gamma = AtomicArrays.effective_interaction_module.effective_interactions(S)
end

NMAX = 100
d_iter = range(1e-1, 1e-0, NMAX)
Omega_arr = zeros(NMAX)
Gamma_arr = zeros(NMAX)

Threads.@threads for i=1:NMAX
    Omega_arr[i], Gamma_arr[i] = collective_shift(d_iter[i],0.49,100)
end


fig_3, axs = plt.subplots(ncols=2, nrows=1, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].plot(d_iter, Omega_arr)
axs[1].set_xlabel(L"a/\lambda_0")
axs[1].set_ylabel(L"\Delta_d")
axs[1].set_title("Collective shift")
axs[1].set_ylim([-3,1])
axs[2].plot(d_iter, Gamma_arr)
axs[2].set_xlabel(L"a/\lambda_0")
axs[2].set_ylabel(L"\Gamma_d")
axs[2].set_title("Collective decay")
display(fig_3)


"""Fidelity"""

# TODO: find the way to project the steady-state of mf model on reduced spin 
#       states

ρ_mf = AtomicArrays.meanfield_module.densityoperator(state_mf_t[end])