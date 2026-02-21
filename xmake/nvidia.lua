target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("cuda")
    set_values("cuda.rdc", false)
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-device")
    add_deps("llaisys-device-nvidia")
target_end()

target("llaisys-ops-cuda")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("cuda")
    set_values("cuda.rdc", false)
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end
    add_links("cublas")
    add_files("../src/ops/*/cuda/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    add_deps("llaisys-ops-cuda")
target_end()
