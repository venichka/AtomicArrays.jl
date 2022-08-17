if pwd()[end-14:end] == "AtomicArrays.jl"
    PATH_ENV = "."
else
    PATH_ENV = "../"
end
using Pkg
Pkg.activate(PATH_ENV)

using QuantumOptics
using FFTW
using BenchmarkTools, Interpolations
using LaTeXStrings
using CairoMakie, GLMakie
using LinearAlgebra, EllipsisNotation
using HDF5, FileIO
using Markdown

using Revise
using AtomicArrays
const EMField = AtomicArrays.field_module.EMField
const effective_constants = AtomicArrays.effective_interaction_module.effective_constants

import EllipsisNotation: Ellipsis
const .. = Ellipsis()


"Load data"

"""
* `args`: `N_x`, `N_y`, `lat_type`, `eq_type`, `dim_vars`, `path_data`
* `output`: `filename`
"""
function filename_create(args)
    N_x, N_y, lat_type, eq_type, dim_vars, path_data = args
    part_name = (string(dim_vars) * "D" * "_" * lat_type * "_" * string(N_x)
                    * "x" * string(N_y) * "_" * eq_type * ".h5")
    filename_sig = ("sig" * part_name )
    filename_fs = ("fs" * part_name )
    @assert isfile(path_data * filename_sig) "There is no file $filename_sig in the directory $path_data"
    @assert isfile(path_data * filename_fs) "There is no file $filename_fs in the directory $path_data"
    return filename_sig, filename_fs
end


"""
* `args`: `path_data::String, file::String
* `output`: `dict`
"""
function load_dict(path_data::String, file_sig::String, file_fs::String)
    dict_sig = load(path_data * file_sig)
    if length(dict_sig["order"]) != length(size(dict_sig["sigma_re"])) - 1
        keys_dict_sig = [isassigned(dict_sig.keys, i) ? dict_sig.keys[i] : "0"
                        for i in 1:length(dict_sig.keys)]
        deleteat!(dict_sig["order"], findall(x -> x == true,
            [isempty(findall(x -> x == item, keys_dict_sig))
             for item in dict_sig["order"]]))
    end
    dict_fs = load(path_data * file_fs)
    if length(dict_fs["order"]) != length(size(dict_fs["sigma_tot"])) - 1
        keys_dict_fs = [isassigned(dict_fs.keys, i) ? dict_fs.keys[i] : "0"
                        for i in 1:length(dict_fs.keys)]
        deleteat!(dict_fs["order"], findall(x -> x == true,
            [isempty(findall(x -> x == item, keys_dict_fs))
             for item in dict_fs["order"]]))
    end
    return dict_sig, dict_fs
end

"Parameters"

begin

    """Parameters"""

    const c_light = 1.0
    const lam_0 = 1.0
    const k_0 = 2 * π / lam_0
    const om_0 = 2.0 * pi * c_light / lam_0

    const Nx = 10
    const Ny = 10
    const Nz = 2
    const N = Nx*Ny*Nz

    const LAT_TYPE = "freq"
    const EQ_TYPE = "mf"
    const DIM_VARS = 4

    const NMAX = 10

    const PATH_FIGS, PATH_DATA = AtomicArrays.misc_module.path()

end

"Load dicts and interpolate"

begin
    # File names
    args_files = [Nx, Ny, LAT_TYPE, EQ_TYPE, DIM_VARS, PATH_DATA]
    const FILE_SIG, FILE_FS = filename_create(args_files)

    # Loading data
    dict_sig, dict_fs = load_dict(PATH_DATA, FILE_SIG, FILE_FS)
    dict_sig["order"]
end

begin
    # Interpolation
    if LAT_TYPE == "dimer"

        σ_re = LinearInterpolation(
            (dict_sig["E"], dict_sig["L"], dict_sig["a_1"], dict_sig["delt"], [0, 1], collect(1:N)),
            dict_sig["sigma_re"])
        σ_im = LinearInterpolation(
            (dict_sig["E"], dict_sig["L"], dict_sig["a_1"], dict_sig["delt"], [0, 1], collect(1:N)),
            dict_sig["sigma_im"])
        fs_tot = LinearInterpolation(
            (dict_fs["E"], dict_fs["L"], dict_fs["a_1"], dict_fs["delt"], [0, 1]),
            dict_fs["sigma_tot"])

        # Variables
        E = range(dict_sig["E"][1], dict_sig["E"][end], NMAX)
        L = range(dict_sig["L"][1], dict_sig["L"][end], NMAX)
        a_1 = range(dict_sig["a_1"][1], dict_sig["a_1"][end], NMAX)
        delt = range(dict_sig["delt"][1], dict_sig["delt"][end], NMAX)
        "Interpolation's done"
    elseif LAT_TYPE == "lat" || LAT_TYPE == "freq"
        σ_re = LinearInterpolation(
            (dict_sig["E"], dict_sig["L"], dict_sig["d"], dict_sig["delt"], [0, 1], collect(1:N)),
            dict_sig["sigma_re"])
        σ_im = LinearInterpolation(
            (dict_sig["E"], dict_sig["L"], dict_sig["d"], dict_sig["delt"], [0, 1], collect(1:N)),
            dict_sig["sigma_im"])
        fs_tot = LinearInterpolation(
            (dict_fs["E"], dict_fs["L"], dict_fs["d"], dict_fs["delt"], [0, 1]),
            dict_fs["sigma_tot"])

        # Variables
        E = range(dict_sig["E"][1], dict_sig["E"][end], NMAX)
        L = range(dict_sig["L"][1], dict_sig["L"][end], NMAX)
        a_1 = range(dict_sig["d"][1], dict_sig["d"][end], NMAX)
        delt = range(dict_sig["delt"][1], dict_sig["delt"][end], NMAX)
        "Interpolation's done"
    end
end

"Optimal parameters"

begin
    # Find optimal parameters
    obj_data = AtomicArrays.field_module.objective(
        dict_fs["sigma_tot"][.., 1], dict_fs["sigma_tot"][.., 2])
    obj_max = maximum(obj_data)
    opt_idx = indexin(obj_max, obj_data)[1] # CartesianIndex(7, 6, 2, 2) # for 10 freq
    fs_0 = dict_fs["sigma_tot"][opt_idx, 1]
    fs_π = dict_fs["sigma_tot"][opt_idx, 2]
    if LAT_TYPE == "dimer"
        E_opt = dict_fs["E"][opt_idx[1][1]]
        L_opt = dict_fs["L"][opt_idx[2][1]]
        a_1_opt = dict_fs["a_1"][opt_idx[3][1]]
        delt_opt = dict_fs["delt"][opt_idx[4][1]]
        "optimal parameters"
    elseif LAT_TYPE == "lat" || LAT_TYPE == "freq"
        E_opt = dict_fs["E"][opt_idx[1][1]]
        L_opt = dict_fs["L"][opt_idx[2][1]]
        a_1_opt = dict_fs["d"][opt_idx[3][1]]
        delt_opt = dict_fs["delt"][opt_idx[4][1]]
        "optimal parameters"
    end
end

"Spin positions"

"""
Output: (x, y, z)
"""
function spin_positions(L, a_1, delt)
	if  LAT_TYPE == "lattice"
        d_1 = a_1
        d_2 = d_1 + delt
        pos_1 = geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                          position_0=[-(Nx-1)*d_1/2,
                                                      -(Ny-1)*d_1/2,
                                                      -L/2])
        pos_2 = geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                          position_0=[-(Nx-1)*d_2/2,
                                                      -(Ny-1)*d_2/2,
                                                      L/2])
        pos = vcat(pos_1, pos_2)
    elseif LAT_TYPE == "freq"
        d_1 = a_1
        d_2 = d_1
        pos_1 = geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                          position_0=[-(Nx-1)*d_1/2,
                                                      -(Ny-1)*d_1/2,
                                                      -L/2])
        pos_2 = geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                          position_0=[-(Nx-1)*d_2/2,
                                                      -(Ny-1)*d_2/2,
                                                      L/2])
        pos = vcat(pos_1, pos_2)
    elseif LAT_TYPE == "dimer"
        a_2 = 0.3
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
    end
    x = [p[1] for p in pos]
    y = [p[2] for p in pos]
    z = [p[3] for p in pos]
    return x, y, z
end


dict_sig["order"]

"""Plotting"""

function phases_unitcircle()
    GLMakie.activate!()
    f = Figure(resolution=(1850,800))

    sg = GLMakie.SliderGrid(
    f[1, 3],
    (label = "E", range = E, format = x -> string(round(x, digits = 4)), startvalue = E[opt_idx[1]]),
    (label = "L", range = L, format = x -> string(round(x, digits = 4)), startvalue = L[opt_idx[2]]),
    (label = "a_1", range = a_1, format = x -> string(round(x, digits = 4)), startvalue = a_1[opt_idx[3]]),
    (label = "delt", range = delt, format = x -> string(round(x, digits = 4)), startvalue = delt[opt_idx[4]]),
    (label = "dir", range = [0,1], format = x -> string(round(x, digits = 0)), startvalue = 0),
    width = 350,
    height = 175,
    tellheight = false)

    sl_E = sg.sliders[1]
    sl_L = sg.sliders[2]
    sl_a = sg.sliders[3]
    sl_delt = sg.sliders[4]
    sl_dir = sg.sliders[5]

    arr_re_1 = lift(sl_E.value, sl_L.value, sl_a.value,
                    sl_delt.value, sl_dir.value) do x, y, z, α, DIR
        sigms = σ_re(x, y, z, α, DIR, 1:N÷2) .+ im*σ_im(x, y, z, α, DIR, 1:N÷2)
        sigms = sigms ./ abs.(sigms)
        return real(sigms)
    end
    arr_im_1 = lift(sl_E.value, sl_L.value, sl_a.value,
                    sl_delt.value, sl_dir.value) do x, y, z, α, DIR
        sigms = σ_re(x, y, z, α, DIR, 1:N÷2) .+ im*σ_im(x, y, z, α, DIR, 1:N÷2)
        sigms = sigms ./ abs.(sigms)
        return imag(sigms)
    end
    arr_re_2 = lift(sl_E.value, sl_L.value, sl_a.value,
                    sl_delt.value, sl_dir.value) do x, y, z, α, DIR
        sigms = σ_re(x, y, z, α, DIR, N÷2+1:N) .+ im*σ_im(x, y, z, α, DIR, N÷2+1:N)
        sigms = sigms ./ abs.(sigms)
        return real(sigms)
    end
    arr_im_2 = lift(sl_E.value, sl_L.value, sl_a.value,
                    sl_delt.value, sl_dir.value) do x, y, z, α, DIR
        sigms = σ_re(x, y, z, α, DIR, N÷2+1:N) .+ im*σ_im(x, y, z, α, DIR, N÷2+1:N)
        sigms = sigms ./ abs.(sigms)
        return imag(sigms)
    end
    arr_arg_1 = lift(sl_E.value, sl_L.value, sl_a.value,
                    sl_delt.value, sl_dir.value) do x, y, z, α, DIR
        sigms = σ_re(x, y, z, α, DIR, 1:N÷2) .+ im*σ_im(x, y, z, α, DIR, 1:N÷2)
        return angle.(sigms)
    end
    arr_arg_2 = lift(sl_E.value, sl_L.value, sl_a.value,
                    sl_delt.value, sl_dir.value) do x, y, z, α, DIR
        sigms = σ_re(x, y, z, α, DIR, N÷2+1:N) .+ im*σ_im(x, y, z, α, DIR, N÷2+1:N)
        return angle.(sigms)
    end

     # Plot
    lines(f[1, 1], Circle(Point2f(0), 1),
        axis=(title=L"$\sigma$",
            xlabel=L"\Re \sigma", ylabel=L"\Im \sigma",
            xlabelsize=24, ylabelsize=24,
            titlesize=30), 
            color = :lightgray)
    scatter!(f[1, 1], arr_re_1, arr_im_1,
        color=:red, markersize=10)
    scatter!(f[1, 1], arr_re_2, arr_im_2,
        color=:blue, markersize=10)

    scatter(f[1, 2], 1:N÷2, arr_arg_1,
        axis=(title=L"arg$(\sigma)$",
            xlabel=L"Number of atom $$", ylabel=L"arg$(\sigma)$",
            xlabelsize=24, ylabelsize=24,
            titlesize=30),
            color = :red, markersize=16)
    scatter!(f[1, 2], 1:N÷2, arr_arg_2,
            color = :blue, markersize=16)

    # colsize!(f.layout, 1, Auto(0.5))
    return f
end


function phases_amplitudes_lattice()
    GLMakie.activate!()
    f = Figure(resolution=(1100,1200))

    sg = GLMakie.SliderGrid(
    f[1, 2],
    (label = "E", range = E, format = x -> string(round(x, digits = 4)), startvalue = E[opt_idx[1]]),
    (label = "L", range = L, format = x -> string(round(x, digits = 4)), startvalue = L[opt_idx[2]]),
    (label = "a_1", range = a_1, format = x -> string(round(x, digits = 4)), startvalue = a_1[opt_idx[3]]),
    (label = "delt", range = delt, format = x -> string(round(x, digits = 4)), startvalue = delt[opt_idx[4]]),
    (label = "dir", range = [0,1], format = x -> string(round(x, digits = 0)), startvalue = 0),
    width = 350,
    height = 175,
    tellheight = false)

    sl_E = sg.sliders[1]
    sl_L = sg.sliders[2]
    sl_a = sg.sliders[3]
    sl_delt = sg.sliders[4]
    sl_dir = sg.sliders[5]

    arr_abs = lift(sl_E.value, sl_L.value, sl_a.value,
                    sl_delt.value, sl_dir.value) do x, y, z, α, DIR
        sigms = σ_re(x, y, z, α, DIR, 1:N) .+ im*σ_im(x, y, z, α, DIR, 1:N)
        return abs.(sigms)
    end
    arr_arg = lift(sl_E.value, sl_L.value, sl_a.value,
                    sl_delt.value, sl_dir.value) do x, y, z, α, DIR
        sigms = σ_re(x, y, z, α, DIR, 1:N) .+ im*σ_im(x, y, z, α, DIR, 1:N)
        return angle.(sigms)
    end

    # Coordinates of spins
    x = lift(sl_L.value, sl_a.value,
                    sl_delt.value) do val_L, val_a, val_delt

        return spin_positions(val_L, val_a, val_delt)[1]
    end
    y = lift(sl_L.value, sl_a.value,
                    sl_delt.value) do val_L, val_a, val_delt

        return spin_positions(val_L, val_a, val_delt)[2]
    end
    z = lift(sl_L.value, sl_a.value,
                    sl_delt.value) do val_L, val_a, val_delt

        return spin_positions(val_L, val_a, val_delt)[3]
    end

    # Limits
    limits_0 = lift(sl_L.value, sl_a.value, sl_delt.value) do val_L, val_a, val_delt
        (
         minimum(spin_positions(val_L, val_a, val_delt)[1]),
         maximum(spin_positions(val_L, val_a, val_delt)[1]),
         minimum(spin_positions(val_L, val_a, val_delt)[2]),
         maximum(spin_positions(val_L, val_a, val_delt)[2]),
         minimum(spin_positions(val_L, val_a, val_delt)[3]),
         maximum(spin_positions(val_L, val_a, val_delt)[3]),
        )
    end


    # Plot
    Axis3(f[1, 1], title=L"|\sigma_j|",
        # limits=limits_0,
        viewmode=:fit,
        aspect=:data,
        xlabel=L"x",
        ylabel=L"y",
        zlabel=L"z",
        xlabelsize=28, ylabelsize=28, zlabelsize=28,
        titlesize=30,
        xticklabelsize=22,
        zticklabelsize=22,
        yticklabelsize=22)

    meshscatter!(x, y, z, markersize = 0.1, color = arr_abs, colormap=:Hiroshige)

    Axis3(f[2, 1], title=L"arg$(\sigma_j)$",
        # limits=limits_0,
        viewmode=:fit,
        aspect=:data,
        xlabel=L"x",
        ylabel=L"y",
        zlabel=L"z",
        xlabelsize=28, ylabelsize=28, zlabelsize=28,
        titlesize=30,
        xticklabelsize=22,
        zticklabelsize=22,
        yticklabelsize=22)

    sc2 = meshscatter!(x, y, z, markersize = 0.1, colorrange=(-pi, pi),
                       color = arr_arg, colormap=:phase)
    Colorbar(f[2, 2], sc2, label = L"arg$(\sigma_j)$", height = Relative(1))

    colsize!(f.layout, 1, Auto(0.2))
    return f
end


function phases_unitcircle_pub()

    CairoMakie.activate!()
    # GLMakie.activate!()

    fig_uc = Figure(resolution=(1150,600))

    # 0-direction
    sigms_0 = (σ_re(E[opt_idx[1]], L[opt_idx[2]], a_1[opt_idx[3]],
                 delt[opt_idx[4]], 0, 1:N) .+
              im*σ_im(E[opt_idx[1]], L[opt_idx[2]], a_1[opt_idx[3]],
                 delt[opt_idx[4]], 0, 1:N))
    sigms_0 = sigms_0 ./ abs.(sigms_0)
    arr_re_1_0 = real(sigms_0[1:N÷2])
    arr_im_1_0 = imag(sigms_0[1:N÷2])
    arr_re_2_0 = real(sigms_0[1+N÷2:N])
    arr_im_2_0 = imag(sigms_0[1+N÷2:N])

    # π-direction
    sigms_π = (σ_re(E[opt_idx[1]], L[opt_idx[2]], a_1[opt_idx[3]],
                 delt[opt_idx[4]], 1, 1:N) .+
              im*σ_im(E[opt_idx[1]], L[opt_idx[2]], a_1[opt_idx[3]],
                 delt[opt_idx[4]], 1, 1:N))
    sigms_π = sigms_π ./ abs.(sigms_π)
    arr_re_1_π = real(sigms_π[1:N÷2])
    arr_im_1_π = imag(sigms_π[1:N÷2])
    arr_re_2_π = real(sigms_π[1+N÷2:N])
    arr_im_2_π = imag(sigms_π[1+N÷2:N])

     # Plot
    ax1 = Axis(fig_uc[1, 1], title=L"Direction: $0$",
            xlabel=L"\Re \sigma / |\sigma|",
            ylabel=L"\Im \sigma / |\sigma|",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22)
    lines!(fig_uc[1, 1], Circle(Point2f(0), 1),
            color = :lightgray)
    scatter!(fig_uc[1, 1], arr_re_1_0, arr_im_1_0,
        color=:red, markersize=10, label=L"array $1$")
    scatter!(fig_uc[1, 1], arr_re_2_0, arr_im_2_0,
        color=:blue, markersize=10, label=L"array $2$")

    axislegend()

    ax2 = Axis(fig_uc[1, 2], title=L"Direction: $\pi$",
            xlabel=L"\Re \sigma / |\sigma|",
            ylabel=L"\Im \sigma / |\sigma|",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22)
    lines!(fig_uc[1, 2], Circle(Point2f(0), 1),
            color = :lightgray)
    scatter!(fig_uc[1, 2], arr_re_1_π, arr_im_1_π,
        color=:red, markersize=10, label=L"array $1$")
    scatter!(fig_uc[1, 2], arr_re_2_π, arr_im_2_π,
        color=:blue, markersize=10, label=L"array $2$")

    axislegend()

    # Adding letters
    ga = fig_uc[1, 1] = GridLayout()
    gb = fig_uc[1, 2] = GridLayout()
    for (label, layout) in zip([L"\mathrm{a})", L"\mathrm{b})"], [ga, gb])
        Label(layout[1, 1, TopLeft()], label,
              textsize = 30,
              # font = "TeX Gyre Heros Bold",
              padding = (0, 40, 20, 0),
              halign = :right)
    end

    colsize!(fig_uc.layout, 1, Auto(1.))

    save((PATH_FIGS * "sig_phases_" * LAT_TYPE * "_" * string(Nx)
                    * "x" * string(Ny) * "_" * EQ_TYPE * ".pdf"), fig_uc) # here, you save your figure.

    return fig_uc
end


function phases_amplitudes_lattice_pub(DIR)
    if DIR == "R"
        dir = 0
    elseif DIR == "L"
        dir = 1
    end

    CairoMakie.activate!()
    # GLMakie.activate!()
    f = Figure(resolution=(1200,1000))

    arr_abs_1 = abs.(σ_re(E_opt, L_opt, a_1_opt, delt_opt, dir, 1:N÷2) .+
        im*σ_im(E_opt, L_opt, a_1_opt, delt_opt, dir, 1:N÷2))
    arr_arg_1 = angle.(σ_re(E_opt, L_opt, a_1_opt, delt_opt, dir, 1:N÷2) .+
        im*σ_im(E_opt, L_opt, a_1_opt, delt_opt, dir, 1:N÷2))
    arr_abs_2 = abs.(σ_re(E_opt, L_opt, a_1_opt, delt_opt, dir, 1+N÷2:N) .+
        im*σ_im(E_opt, L_opt, a_1_opt, delt_opt, dir, 1+N÷2:N))
    arr_arg_2 = angle.(σ_re(E_opt, L_opt, a_1_opt, delt_opt, dir, 1+N÷2:N) .+
        im*σ_im(E_opt, L_opt, a_1_opt, delt_opt, dir, 1+N÷2:N))

    # Coordinates of spins
    x, y, _ = spin_positions(L_opt, a_1_opt, delt_opt)

    # Plot
    Axis(f[1, 1], title=L"Array 1$$",
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    sc11 = scatter!(x[1:N÷2], y[1:N÷2], markersize = 30,
                        color = arr_abs_1, colormap=:BuPu_6)
    Colorbar(f[1, 2], sc11, label = L"$|\sigma_j|$", height = Relative(1),
             labelsize=28, ticklabelsize=18)
    Axis(f[2, 1],
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    sc21 = scatter!(x[1:N÷2], y[1:N÷2], markersize = 30, colorrange=(-pi, pi),
                        color = arr_arg_1, colormap=:rainbow1)
    Colorbar(f[2, 2], sc21, label = L"arg$(\sigma_j)$", height = Relative(1),
             labelsize=28, ticklabelsize=18)

    Axis(f[1, 3], title=L"Array 2$$",
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    sc13 = scatter!(x[1:N÷2], y[1:N÷2], markersize = 30,
                        color = arr_abs_2, colormap=:BuPu_6)
    Colorbar(f[1, 4], sc13, label = L"$|\sigma_j|$", height = Relative(1),
             labelsize=28, ticklabelsize=18)

    Axis(f[2, 3],
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    sc23 = scatter!(x[1:N÷2], y[1:N÷2], markersize = 30, colorrange=(-pi, pi),
                        color = arr_arg_2, colormap=:rainbow1)
    Colorbar(f[2, 4], sc23, label = L"arg$(\sigma_j)$", height = Relative(1),
             labelsize=28, ticklabelsize=18)

    save((PATH_FIGS * "sig_abs_arg_" * LAT_TYPE * "_" * string(Nx)
                    * "x" * string(Ny) * "_" * EQ_TYPE * "_" * DIR * ".pdf"), f) # here, you save your figure.
    return f
end


md"""
### Maximum value of objective function and parameters

Objective max:
$(maximum(obj_data))

Scattering crosssection:
σ\_0 = $(fs_0); σ\_π = $(fs_π)

E = $(E_opt); L = $(L_opt); a\_1 = $(a_1_opt); δ = $(delt_opt)

σ\_0 = $(dict_fs["sigma_tot"][opt_idx, 1])

σ\_π = $(dict_fs["sigma_tot"][opt_idx, 2])

"""


phases_unitcircle()

phases_unitcircle_pub()

phases_amplitudes_lattice()

phases_amplitudes_lattice_pub("R")
