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
    opt_idx = indexin(obj_max, obj_data)[1]
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

dict_sig["order"]

"""Plotting"""

function phases_unitcircle()
    GLMakie.activate!()
    fig_uc = Figure(resolution=(1150,800))

    sg = GLMakie.SliderGrid(
    fig_uc[1, 2],
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

     # Plot
    lines(fig_uc[1, 1], Circle(Point2f(0), 1),
        axis=(title=L"$\sigma$",
            xlabel=L"\Re \sigma", ylabel=L"\Im \sigma",
            xlabelsize=24, ylabelsize=24,
            titlesize=30), 
            color = :lightgray)
    scatter!(fig_uc[1, 1], arr_re_1, arr_im_1,
        color=:red, markersize=10)
    scatter!(fig_uc[1, 1], arr_re_2, arr_im_2,
        color=:blue, markersize=10)

    colsize!(fig_uc.layout, 1, Auto(0.5))
    return fig_uc
end


function phases_unitcircle_pub()

    CairoMakie.activate!()

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
