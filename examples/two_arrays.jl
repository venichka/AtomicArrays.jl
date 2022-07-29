#  2D lattice of spins: eigenmodes, eigenvalues, intensity

using CollectiveSpins
using QuantumOptics
using PyPlot

# Parameters
Nx = 10
Ny = 10
Nz = 2  # number of arrays
M = 1 # Number of excitations
d = 0.75
L = 20.0
k_0 = 2*π
pos = geometry.box(d, d, L; Nx=Nx, Ny=Ny, Nz=Nz)
μ = [(i < 0) ? [0, 0, 1.0] : [1.0, -1im, 0.0] for i = 1:Nx*Ny*Nz]
γ_e = [1e-2 for i = 1:Nx*Ny*Nz]
S = SpinCollection(pos,μ; gammas=γ_e)

# Plot arrays (atom distribution)
fig_1 = figure(figsize=(5,5))
x_at = [pos[i][1] for i = 1:Nx*Ny*Nz]
y_at = [pos[i][2] for i = 1:Nx*Ny*Nz]
z_at = [pos[i][3] for i = 1:Nx*Ny*Nz]
scatter3D(x_at, y_at, z_at)
show()

# Collective effects
Ωmat = OmegaMatrix(S)
Γmat = GammaMatrix(S)

# Hilbert space
b = ReducedSpinBasis(Nx*Ny*Nz,M,M) # Basis from M excitations up to M excitations

# Effective Hamiltonian
spsm = [reducedsigmapsigmam(b, i, j) for i=1:Nx*Ny*Nz, j=1:Nx*Ny*Nz]
H_eff = dense(sum((Ωmat[i,j] - 0.5im*Γmat[i,j])*spsm[i, j]
                  for i=1:Nx*Ny*Nz, j=1:Nx*Ny*Nz))

# Find the most subradiant eigenstate
λ, states = eigenstates(H_eff; warning=false)
γ = -2 .* imag.(λ)
s_ind = findmin(γ)[2]
s_ind = sortperm(γ)[1]
ψ = states[s_ind]


"""Plot eigenstates"""
s_ind = sortperm(q_quasi)[2]

eig_states = [(states[i].data)[j] for i = 1:Nx*Ny*Nz, j = 1:Nx*Ny]

# Parity (even: true, odd: false)
parity_n = [(sign(real(states[n].data[1]))==
    sign(real(states[n].data[Nx*Ny+1]))) ? true : false for n = 1:Nx*Ny*Nz]


fig_1_1 = figure(figsize=(9,4))
scatter(x_at[1:Nx*Ny]./d, y_at[1:Nx*Ny]./d, c="r",
        alpha=(abs.(eig_states[s_ind,:])/(findmax(abs.(eig_states[s_ind,:]))[1])),
        s=200)
xlabel("x/d")
ylabel("y/d")
tight_layout()


"""Eigenvalues depending on quasimomentum"""

using FFTW

r_perp = [pos[j][1:2] for j = 1:Nx*Ny]
q_perp = [-π/d .+ 2*π.*(r_perp[j]./d) ./ (Nx*d) for j = 1:Nx*Ny]
states_f = [sum([sqrt(2)*(states[i].data)[j]*exp(1im*r_perp[j]'*q_perp[k])/sqrt(Nx*Ny)
            for j = 1:Nx*Ny]) for i = 1:Nx*Ny*Nz, k = 1:Nx*Ny]
states_f_2D = [[states_f[i, (Nx)*(j-1) + (k-1)%(Ny) + 1] for j=1:Nx, k=1:Ny]
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

figure(figsize=(5,5))
contourf(abs.(states_2D[s_ind]))
xlabel("x/d")
ylabel("y/d")
tight_layout()

figure(figsize=(5,5))
x_fft = fftfreq(length(x_at[1:10]), 1.0/d) |> fftshift
y_fft = x_fft
#contourf(q_perp_x, q_perp_x, abs.(states_2D_fft[s_ind]))
contourf(x_fft, y_fft, abs.(states_2D_fft[s_ind]))
xlabel("q_x")
ylabel("q_y")
tight_layout()

figure(figsize=(5,5))
contourf(q_perp_x, q_perp_x, abs.(states_f_2D[s_ind]'))
xlabel("q_x")
ylabel("q_y")
tight_layout()

figure()
contourf(abs.(states_f'))

figure()
plot(q_quasi)
plot(q_quasi_fft)

figure()
plot(q_quasi[parity_n]./k_0, γ[parity_n]./γ_e[1], "o", color="red")
plot(q_quasi[.!parity_n]./k_0, γ[.!parity_n]./γ_e[1], "o", color="black")
plot(q_quasi_fft./k_0, γ./γ_e, "o", alpha=0.1)
xscale("log")
yscale("log")
xlabel(L"\bar{q}/k_0")
ylabel(L"\gamma_n/\gamma_e")



"""Compute the radiation pattern"""

function G(r,i,j) # Green's Tensor overlap
    G_i = GreenTensor(r-pos[i])
    G_j = GreenTensor(r-pos[j])
    return μ[i]' * (G_i'*G_j) * μ[j]
end
function intensity(r) # The intensity ⟨E⁻(r)⋅E⁺(r)⟩
    real(sum(expect(spsm[i,j], ψ)*G(r,i,j) for i=1:Nx*Ny*Nz, j=1:Nx*Ny*Nz))
end

# X-Z view
#x = -5d:0.1d:Nx*d+5d
#y = pos[Nx*Ny÷2][2]
#z = -5d:0.1d:L+5d

# X-Y view
x = -5d:0.2d:Nx*d+5d
y = -5d:0.2d:Ny*d+5d
z = L + 5d
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


fig_2 = figure(figsize=(9,4))
# Lin scale
levels = 0:I_arr_max*1e-2:I_arr_max*1e-0
contourf(x./d,y./d,I',30, levels=levels)

# Log scale
#levels = exp10.(range(log10((I_arr_sorted[end])), log10(I_arr_sorted[1]),
#                      length=100))
#contourf(x./d,z./d,I',30, levels=levels, norm=matplotlib.colors.LogNorm())
for p in pos
    plot(p[1]./d,p[2]./d,"o",color="w",ms=2)
end
xlabel("x/d")
ylabel("y/d")
colorbar(label="Intensity",ticks=[])
tight_layout()


#write("../Data/test.bin", I)
