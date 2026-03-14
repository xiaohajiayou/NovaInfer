add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("nv-cudnn")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to enable cuDNN backend integration for Nvidia GPU")
option_end()

option("maca-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile CUDA-compatible implementations with MXMACA cu-bridge")
option_end()

option("maca-cudnn")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to enable cuDNN-compatible integration through MXMACA cu-bridge")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

if has_config("maca-gpu") then
    add_defines("ENABLE_NVIDIA_API", "ENABLE_MACA_CUDA_BRIDGE")
    includes("xmake/maca.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    if has_config("nv-gpu") then
        add_links("nvToolsExt")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    if has_config("maca-gpu") then
        add_deps("llaisys-device-nvidia")
        add_deps("llaisys-ops-cuda")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_ldflags("-fopenmp")
        add_shflags(
            "-Wl,--export-dynamic-symbol=llaisysModelForward",
            "-Wl,--export-dynamic-symbol=llaisysSamplerSample"
        )
        add_syslinks("gomp")
    end
    if has_config("maca-gpu") then
        add_linkdirs("/opt/maca/lib", "/opt/maca/tools/cu-bridge/lib")
        add_rpathdirs("/opt/maca/lib", "/opt/maca/tools/cu-bridge/lib")
        add_shflags("-Wl,--no-as-needed")
        add_links("mcruntime", "mcblas", "mcblasLt", "mccompiler", "runtime_cu")
        add_shflags("-Wl,--as-needed")
    end
    add_files("src/llaisys/*.cc")
    add_files("src/llaisys/kv_cache/*.cpp")
    add_files("src/llaisys/workspace/*.cpp")
    add_files("src/llaisys/weights/*.cpp")
    add_files("src/llaisys/qwen2/*.cc")
    add_files("src/llaisys/qwen2/*.cpp")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
        if is_plat("macosx") then
            os.cp("lib/*.dylib", "python/llaisys/libllaisys/")
        end
    end)
    after_load(function (target)
        if has_config("maca-gpu") then
            local device = target:dep("llaisys-device-nvidia")
            local ops = target:dep("llaisys-ops-cuda")
            if device and ops then
                target:add("shflags", "-Wl,--whole-archive", device:targetfile(), ops:targetfile(), "-Wl,--no-whole-archive")
            end
        end
    end)
target_end()
