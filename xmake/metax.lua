local maca_path = os.getenv("MACA_PATH")
local cucc_path = os.getenv("CUCC_PATH")
local cuda_path = os.getenv("CUDA_PATH")

local bridge_root = cuda_path
if not bridge_root and maca_path then
    local candidate = path.join(maca_path, "tools", "cu-bridge")
    if os.isdir(candidate) then
        bridge_root = candidate
        os.setenv("CUDA_PATH", bridge_root)
    end
end

if not bridge_root or not os.isdir(bridge_root) then
    raise("mx-gpu requires a valid cu-bridge root. Set CUDA_PATH to the cu-bridge directory, or set MACA_PATH so xmake can resolve MACA_PATH/tools/cu-bridge")
end

if cucc_path and os.isdir(cucc_path) then
    os.setenv("PATH", cucc_path .. path.envsep() .. (os.getenv("PATH") or ""))
end

local function add_metax_cuda_env()
    add_includedirs(path.join(bridge_root, "include"))
    if os.isdir(path.join(bridge_root, "lib64")) then
        add_linkdirs(path.join(bridge_root, "lib64"))
    end
    if os.isdir(path.join(bridge_root, "lib")) then
        add_linkdirs(path.join(bridge_root, "lib"))
    end
end

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("cuda")
    set_values("cuda.rdc", false)
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end
    add_metax_cuda_env()
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
    add_metax_cuda_env()
    add_links("cublas")
    add_links("cublasLt")
    add_files("../src/ops/*/cuda/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    add_deps("llaisys-ops-cuda")
target_end()
