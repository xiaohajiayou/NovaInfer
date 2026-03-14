local maca_path = "/opt/maca"
local cu_bridge_path = maca_path .. "/tools/cu-bridge"
local mxcc = maca_path .. "/mxgpu_llvm/bin/mxcc"

rule("maca.cu")
    set_extensions(".maca")
    on_build_file(function (target, sourcefile, opt)
        import("core.project.depend")
        import("utils.progress")

        local objectfile = target:objectfile(sourcefile)
        table.insert(target:objectfiles(), objectfile)

        local argv = {
            "-x", "maca",
            "-offload-arch", "native",
            "--maca-path=" .. maca_path,
            "-c", sourcefile,
            "-o", objectfile
        }

        table.insert(argv, "-std=c++20")
        if not is_plat("windows") then
            table.insert(argv, "-fPIC")
            table.insert(argv, "-Wno-c++17-extensions")
        end

        for _, includedir in ipairs(target:get("includedirs") or {}) do
            table.insert(argv, "-I" .. includedir)
        end
        for _, includedir in ipairs(target:get("sysincludedirs") or {}) do
            table.insert(argv, "-I" .. includedir)
        end
        for _, define in ipairs(target:get("defines") or {}) do
            table.insert(argv, "-D" .. define)
        end
        for _, flag in ipairs(target:get("mxccflags") or {}) do
            table.insert(argv, flag)
        end

        os.mkdir(path.directory(objectfile))

        local dependfile = target:dependfile(objectfile)
        local dependinfo = target:is_rebuilt() and {} or (depend.load(dependfile) or {})
        if not depend.is_changed(dependinfo, {
            lastmtime = os.mtime(objectfile),
            files = {sourcefile},
            values = argv
        }) then
            return
        end

        progress.show(opt.progress, "${color.build.object}compiling.maca.$(mode) %s", sourcefile)
        os.vrunv(mxcc, argv)

        dependinfo.files = {sourcefile}
        dependinfo.values = argv
        depend.save(dependinfo, dependfile)
    end)

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx20")
    set_warnings("all", "error")
    add_includedirs(cu_bridge_path .. "/include", maca_path .. "/include", maca_path .. "/include/mcblas")
    add_linkdirs(maca_path .. "/lib")
    add_rpathdirs(maca_path .. "/lib")
    add_files(".maca_sources/device/*.maca", {rules = "maca.cu"})

    on_install(function (target) end)
target_end()

target("llaisys-device")
    add_deps("llaisys-device-nvidia", {public = true})
target_end()

target("llaisys-ops-cuda")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx20")
    set_warnings("all", "error")
    add_includedirs(cu_bridge_path .. "/include", maca_path .. "/include", maca_path .. "/include/mcblas")
    add_linkdirs(maca_path .. "/lib")
    add_rpathdirs(maca_path .. "/lib")
    add_links("mcruntime", "mcblas", "mcblasLt")
    if has_config("maca-cudnn") then
        add_defines("ENABLE_CUDNN_API")
        add_links("mcdnn")
        if os.isdir("../third_party/cudnn_frontend/include") then
            add_defines("ENABLE_CUDNN_FRONTEND")
            add_sysincludedirs("../third_party/cudnn_frontend/include")
            add_values("mxccflags", "-Wno-unused-function")
        end
    end
    add_files(".maca_sources/ops/*.maca", {rules = "maca.cu"})

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    add_deps("llaisys-ops-cuda", {public = true})
target_end()

for _, name in ipairs({"llaisys-device-nvidia", "llaisys-ops-cuda"}) do
    target(name)
        after_load(function (target)
            local syslinks = {}
            for _, link in ipairs(target:get("syslinks") or {}) do
                if link ~= "cudadevrt" and link ~= "cudart_static" then
                    table.insert(syslinks, link)
                end
            end
            target:set("syslinks", syslinks)
            target:add("links", "mcruntime", {public = true})
            target:add("linkdirs", maca_path .. "/lib", {public = true})
            target:add("rpathdirs", maca_path .. "/lib", {public = true})
        end)
    target_end()
end
