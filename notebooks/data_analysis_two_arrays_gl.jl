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


GLMakie.activate!()

"Load data"

"""
* `args`: `N_x`, `N_y`, `lat_type`, `eq_type`, `dim_vars`, `path_data`
* `output`: `filename_fs`, `filename_obj`
"""
function filename_create(args)
    N_x, N_y, lat_type, eq_type, dim_vars, path_data = args
    filename_fs = ("fs" * string(dim_vars) * "D" * "_" * lat_type * "_" * string(N_x)
                   * "x" * string(N_y) * "_" * eq_type * ".h5")
    filename_obj = ("obj" * string(dim_vars) * "D" * "_" * lat_type * "_" * string(N_x)
                    * "x" * string(N_y) * "_" * eq_type * ".h5")
    @assert isfile(path_data * filename_fs) "There is no file $filename_fs in the directory $path_data"
    @assert isfile(path_data * filename_obj) "There is no file $filename_obj in the directory $path_data"
    return filename_fs, filename_obj
end


"""
* `args`: `path_data::String, file_fs::String, file_obj::String`
* `output`: `dict_fs, dict_obj`
"""
function load_dict(path_data::String, file_fs::String, file_obj::String)
    dict_fs = load(path_data * file_fs)
    dict_obj = load(path_data * file_obj)
    if length(dict_fs["order"]) != length(size(dict_fs["sigma_tot"]))
        keys_dict_fs = [isassigned(dict_fs.keys, i) ? dict_fs.keys[i] : "0"
                        for i in 1:length(dict_fs.keys)]
        deleteat!(dict_fs["order"], findall(x -> x == true,
            [isempty(findall(x -> x == item, keys_dict_fs))
             for item in dict_fs["order"]]))
    end
    if length(dict_obj["order"]) != length(size(dict_obj["obj"]))
        keys_dict_obj = [isassigned(dict_obj.keys, i) ? dict_obj.keys[i] : "0"
                         for i in 1:length(dict_obj.keys)]
        deleteat!(dict_obj["order"], findall(x -> x == true,
            [isempty(findall(x -> x == item, keys_dict_obj))
             for item in dict_obj["order"]]))
    end
    return dict_fs, dict_obj
end

"Parameters"

begin

    """Parameters"""

    const c_light = 1.0
    const lam_0 = 1.0
    const k_0 = 2 * π / lam_0
    const om_0 = 2.0 * pi * c_light / lam_0

    const Nx = 14
    const Ny = 14

    const LAT_TYPE = "dimer"
    const EQ_TYPE = "mf"
    const DIM_VARS = 4

    const NMAX = 30

    const PATH_FIGS, PATH_DATA = AtomicArrays.misc_module.path()

end

"Load dicts and interpolate"

begin
    # File names
    args_files = [Nx, Ny, LAT_TYPE, EQ_TYPE, DIM_VARS, PATH_DATA]
    const FILE_FS, FILE_OBJ = filename_create(args_files)

    # Loading data
    dict_fs, dict_obj = load_dict(PATH_DATA, FILE_FS, FILE_OBJ)
    dict_obj["order"]
end

begin
    # Interpolation
    if LAT_TYPE == "dimer"
        σ_tot = LinearInterpolation(
            (dict_fs["E"], dict_fs["L"], dict_fs["a_1"], dict_fs["delt"], [0, 1]),
            dict_fs["sigma_tot"])
        obj = LinearInterpolation(
            (dict_obj["E"], dict_obj["L"], dict_obj["a_1"], dict_obj["delt"]),
            dict_obj["obj"])

        # Variables
        E = range(dict_obj["E"][1], dict_obj["E"][end], NMAX)
        L = range(dict_obj["L"][1], dict_obj["L"][end], NMAX)
        a_1 = range(dict_obj["a_1"][1], dict_obj["a_1"][end], NMAX)
        delt = range(dict_obj["delt"][1], dict_obj["delt"][end], NMAX)

        "Interpolation's done"
    elseif LAT_TYPE == "lat"
        σ_tot = LinearInterpolation(
            (dict_fs["E"], dict_fs["L"], dict_fs["d"], dict_fs["delt"], [0, 1]),
            dict_fs["sigma_tot"])
        obj = LinearInterpolation(
            (dict_obj["E"], dict_obj["L"], dict_obj["d"], dict_obj["delt"]),
            dict_obj["obj"])

        # Variables
        E = range(dict_obj["E"][1], dict_obj["E"][end], NMAX)
        L = range(dict_obj["L"][1], dict_obj["L"][end], NMAX)
        a_1 = range(dict_obj["d"][1], dict_obj["d"][end], NMAX)
        delt = range(dict_obj["delt"][1], dict_obj["delt"][end], NMAX)
        "Interpolation's done"
    end
end


"Optimal parameters"

begin
    # Find optimal parameters
    if LAT_TYPE == "dimer"
        obj_max = maximum(dict_obj["obj"])
        opt_idx = indexin(obj_max, dict_obj["obj"])[1]
        E_opt = dict_obj["E"][opt_idx[1][1]]
        L_opt = dict_obj["L"][opt_idx[2][1]]
        a_1_opt = dict_obj["a_1"][opt_idx[3][1]]
        delt_opt = dict_obj["delt"][opt_idx[4][1]]
        "optimal parameters"
    elseif LAT_TYPE == "lat"
        obj_max = maximum(dict_obj["obj"])
        opt_idx = indexin(obj_max, dict_obj["obj"])[1]
        E_opt = dict_obj["E"][opt_idx[1][1]]
        L_opt = dict_obj["L"][opt_idx[2][1]]
        a_1_opt = dict_obj["d"][opt_idx[3][1]]
        delt_opt = dict_obj["delt"][opt_idx[4][1]]
        "optimal parameters"
    end
end

md"""
### Maximum value of objective function and parameters

Objective max:
$(maximum(dict_obj["obj"]))

E = $(E_opt)

L = $(L_opt)

a\_1 = $(a_1_opt)

δ = $(delt_opt)

"""

dict_obj["order"]

"""Plotting"""

"Plot objective"

function vol_obj()
    fig_Vobj = Figure(resolution=(1000,800))

    # sl_E = GLMakie.Slider(fig_Vobj[4, 1], range = E, 
    # startvalue = E[opt_idx[1]])
    # sl_delt = GLMakie.Slider(fig_Vobj[4, 2], range = delt, 
    # horizontal = true, startvalue = delt[Tuple(opt_idx)[end]])

    sg = GLMakie.SliderGrid(
    fig_Vobj[4, :],
    (label = "E", range = E, format = x -> string(round(x, digits = 4)), startvalue = E[opt_idx[1]]),
    (label = "delt", range = delt, format = x -> string(round(x, digits = 4)), startvalue = delt[Tuple(opt_idx)[end]]),
    width = 350,
    height = 75,
    tellheight = true)

    sl_E = sg.sliders[1]
    sl_delt = sg.sliders[2]

    cube = lift(sl_E.value) do x
        obj(x, L, a_1, delt)
    end
    colorrange = lift(sl_E.value) do x
        (minimum(obj(x, L, a_1, delt)), maximum(obj(x, L, a_1, delt)))
    end
    slice = lift(sl_E.value, sl_delt.value) do x, y
        obj(x, L, a_1, y)
    end
    line = lift(sl_E.value) do x
        [maximum(obj(x, L, a_1, delt_i)) for delt_i in delt]
    end

    colormap = to_colormap(:lightrainbow)  # colormap
    colormap[1] = Makie.RGBA{Float32}(1, 1, 1, 0)  # create opacity

     # Plot cube
     ax, cplot = volume(fig_Vobj[1, 1], L, a_1, delt, cube,
        algorithm=:absorption, absorption=4f0, colormap=colormap,
        axis=(type=Axis3, title = "Absorption", 
        xlabel=L"L", ylabel=L"a_1", zlabel=L"\delta"),
        colorrange=colorrange,
        transparency=true)

    # Plot rectangular
    rectplot = linesegments!(ax, Rect(minimum(L), minimum(a_1),
            maximum(L) - minimum(L),
            maximum(a_1) - minimum(a_1)),
        linewidth=2, color=:red)
    on(sl_delt.value) do x
        translate!(rectplot, 0, 0, x)
    end

    # Plot slice
    ax_1, cm = heatmap(fig_Vobj[1, 2], L, a_1, slice,
        colormap=:lightrainbow,
        axis=(; xlabel=L"$L$", ylabel=L"$a_1$", xlabelsize=25, ylabelsize=25))
    Colorbar(fig_Vobj[2, 2], cm, vertical=false)

    # Plot line
    ax_2, lin = lines(fig_Vobj[3, :], delt, line,
    linewidth=4, axis=(; xlabel=L"$\delta$", limits=(nothing, nothing, 0, maximum(obj(E, L, a_1, delt))), xlabelsize=25, height=150, tellheight=true))

    colsize!(fig_Vobj.layout, 1, Auto(1.5))
    fig_Vobj

end


"Plot scattering"

function vol_scatt()
    fig_Vsc = Figure(resolution=(1000,800))

    sg = GLMakie.SliderGrid(
    fig_Vsc[4, :],
    (label = "E", range = E, format = x -> string(round(x, digits = 4)), startvalue = E[opt_idx[1]]),
    (label = "delt", range = delt, format = x -> string(round(x, digits = 4)), startvalue = delt[Tuple(opt_idx)[end]]),
    width = 350,
    height = 75,
    tellheight = true)

    sl_E = sg.sliders[1]
    sl_delt = sg.sliders[2]

    cube_1 = lift(sl_E.value) do x
        σ_tot(x, L, a_1, delt, 0)
    end
    colorrange_1 = lift(sl_E.value) do x
        (minimum(σ_tot(x, L, a_1, delt, 0)), maximum(σ_tot(x, L, a_1, delt, 0)))
    end
    cube_2 = lift(sl_E.value) do x
        σ_tot(x, L, a_1, delt, 1)
    end
    colorrange_2 = lift(sl_E.value) do x
        (minimum(σ_tot(x, L, a_1, delt, 1)), maximum(σ_tot(x, L, a_1, delt, 1)))
    end
    slice_1 = lift(sl_E.value, sl_delt.value) do x, y
        σ_tot(x, L, a_1, y, 0)
    end
    slice_2 = lift(sl_E.value, sl_delt.value) do x, y
        σ_tot(x, L, a_1, y, 1)
    end

    colormap = to_colormap(:lightrainbow)  # colormap
    colormap[1] = Makie.RGBA{Float32}(1, 1, 1, 0)  # create opacity

     # Plot cube
     ax_1, cplot_1 = volume(fig_Vsc[1, 1], L, a_1, delt, cube_1,
        algorithm=:absorption, absorption=4f0, colormap=colormap,
        axis=(type=Axis3, title = L"\sigma_0", 
        xlabel=L"L", ylabel=L"a_1", zlabel=L"\delta"),
        colorrange=colorrange,
        transparency=true)
     ax_2, cplot_2 = volume(fig_Vsc[1, 2], L, a_1, delt, cube_2,
        algorithm=:absorption, absorption=4f0, colormap=colormap,
        axis=(type=Axis3, title = L"\sigma_\pi", 
        xlabel=L"L", ylabel=L"a_1", zlabel=L"\delta"),
        colorrange=colorrange,
        transparency=true)

    # Plot rectangular
    rectplot_1 = linesegments!(ax_1, Rect(minimum(L), minimum(a_1),
            maximum(L) - minimum(L),
            maximum(a_1) - minimum(a_1)),
        linewidth=2, color=:red)
    rectplot_2 = linesegments!(ax_2, Rect(minimum(L), minimum(a_1),
            maximum(L) - minimum(L),
            maximum(a_1) - minimum(a_1)),
        linewidth=2, color=:blue)
    on(sl_delt.value) do x
        translate!(rectplot_1, 0, 0, x)
        translate!(rectplot_2, 0, 0, x)
    end

    # Plot slice
    ax_11, cm_1 = heatmap(fig_Vsc[2, 1], L, a_1, slice_1,
        colormap=:lightrainbow,
        axis=(; xlabel=L"$L$", ylabel=L"$a_1$", xlabelsize=25, ylabelsize=25))
    Colorbar(fig_Vsc[3, 1], cm_1, vertical=false)
    ax_21, cm_2 = heatmap(fig_Vsc[2, 2], L, a_1, slice_2,
        colormap=:lightrainbow,
        axis=(; xlabel=L"$L$", ylabel=L"$a_1$", xlabelsize=25, ylabelsize=25))
    Colorbar(fig_Vsc[3, 2], cm_2, vertical=false)

    colsize!(fig_Vsc.layout, 1, Auto(0.5))
    fig_Vsc

end


"Plot scattering surfaces"

function surf_scatt()
    fig_Ssc = Figure(resolution=(1000,1000))

    sg = GLMakie.SliderGrid(
    fig_Ssc[2, :],
    (label = "E", range = E, format = x -> string(round(x, digits = 4)), startvalue = E[opt_idx[1]]),
    (label = "L", range = L, format = x -> string(round(x, digits = 4)), startvalue = L[Tuple(opt_idx)[2]]),
    width = 350,
    height = 75,
    tellheight = true)

    sl_E = sg.sliders[1]
    sl_L = sg.sliders[2]

    surf_1 = lift(sl_E.value, sl_L.value) do x, y
        σ_tot(x, y, a_1, delt, 0)
    end
    surf_2 = lift(sl_E.value, sl_L.value) do x, y
        σ_tot(x, y, a_1, delt, 1)
    end

     # Plot surfaces
    wireframe(fig_Ssc[1, 1], a_1, delt, surf_1,
        axis=(type=Axis3, title=L"$\sigma_0$, $\sigma_\pi$",
            xlabel=L"a_1", ylabel=L"\delta", zlabel=L"\sigma_0, \sigma_\pi",
            xlabelsize=24, ylabelsize=24, zlabelsize=24,
            titlesize=30, width=1000),
        color=:red)
    wireframe!(a_1, delt, surf_2,
        color=:blue)

    colsize!(fig_Ssc.layout, 1, Auto(0.5))
    fig_Ssc

end


"Plot scattering lines"

function lines_scatt()
    fig_Lsc = Figure(resolution=(1000,400))

    sg = GLMakie.SliderGrid(
    fig_Lsc[2, :],
    (label = "E", range = E, format = x -> string(round(x, digits = 4)), startvalue = E[opt_idx[1]]),
    (label = "L", range = L, format = x -> string(round(x, digits = 4)), startvalue = L[Tuple(opt_idx)[2]]),
    (label = "a_1", range = a_1, format = x -> string(round(x, digits = 4)), startvalue = a_1[Tuple(opt_idx)[3]]),
    width = 350,
    height = 75,
    tellheight = true)

    sl_E = sg.sliders[1]
    sl_L = sg.sliders[2]
    sl_a = sg.sliders[3]

    line_1 = lift(sl_E.value, sl_L.value, sl_a.value) do x, y, z
        σ_tot(x, y, z, delt, 0)
    end
    line_2 = lift(sl_E.value, sl_L.value, sl_a.value) do x, y, z
        σ_tot(x, y, z, delt, 1)
    end

    limits_0 = lift(sl_E.value, sl_L.value, sl_a.value) do x, y, z
        (nothing, nothing, 
        minimum(σ_tot(x, y, z, delt, [0,1])), 
        maximum(σ_tot(x, y, z, delt, [0,1])))
    end

    limits_1 = (nothing, nothing, minimum(σ_tot(E, L, a_1, delt, [0,1])), 
        maximum(σ_tot(E, L, a_1, delt, [0,1])))

     # Plot surfaces
    lines(fig_Lsc[1, 1], delt, line_1,
        axis=(title=L"$\sigma_0$, $\sigma_\pi$",
            xlabel=L"\delta", ylabel=L"\sigma_0, \sigma_\pi",
            xlabelsize=24, ylabelsize=24,
            titlesize=30, width=1000, limits=limits_1),
        color=:red, linewidth=4)
    lines!(delt, line_2,
        color=:blue, linewidth=4)

    colsize!(fig_Lsc.layout, 1, Auto(0.5))
    fig_Lsc

end


vol_obj()
vol_scatt()
surf_scatt()
lines_scatt()