use std::path::Path;
use std::process::Command;

fn main() {
    // Only link ipopt when the ipopt-native feature is enabled
    if std::env::var("CARGO_FEATURE_IPOPT_NATIVE").is_ok() {
        let output = Command::new("pkg-config")
            .args(["--libs-only-L", "ipopt"])
            .output()
            .expect("pkg-config not found; install ipopt via homebrew");
        let lib_path = String::from_utf8(output.stdout).unwrap();
        let lib_path = lib_path.trim().trim_start_matches("-L").trim();
        // pkg-config returns no -L when Ipopt lives in a system path
        // (e.g. /usr/lib64 on Fedora). rustc 1.95+ rejects an empty -L,
        // so only emit a link-search line when we actually have a path.
        if !lib_path.is_empty() {
            println!("cargo:rustc-link-search=native={}", lib_path);
        }
        println!("cargo:rustc-link-lib=dylib=ipopt");

        let output = Command::new("pkg-config")
            .args(["--cflags-only-I", "ipopt"])
            .output()
            .unwrap();
        let inc_path = String::from_utf8(output.stdout).unwrap();
        let inc_path = inc_path.trim().trim_start_matches("-I");
        println!("cargo:rustc-env=IPOPT_INCLUDE={}", inc_path);
    }

    // Link CUTEst when the cutest feature is enabled
    if std::env::var("CARGO_FEATURE_CUTEST").is_ok() {
        let home = std::env::var("HOME").unwrap();
        let cutest_dir = std::env::var("CUTEST_ROOT")
            .unwrap_or_else(|_| format!("{}/.local/cutest", home));
        println!("cargo:rerun-if-env-changed=CUTEST_ROOT");
        let cutest_lib = format!("{}/install/lib", cutest_dir);
        let cutest_include = format!("{}/install/include", cutest_dir);
        let cutest_modules = format!("{}/install/modules", cutest_dir);

        // Link libcutest_double.a
        println!("cargo:rustc-link-search=native={}", cutest_lib);
        println!("cargo:rustc-link-lib=static=cutest_double");

        // Compile cutest_trampoline.f90 (provides cutest_load/unload_routines)
        let trampoline_src = format!("{}/cutest/src/tools/cutest_trampoline.f90", cutest_dir);
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let trampoline_obj = format!("{}/cutest_trampoline.o", out_dir);

        if Path::new(&trampoline_src).exists() {
            let status = Command::new("gfortran")
                .args([
                    "-cpp",
                    "-c",
                    "-fPIC",
                    &format!("-I{}", cutest_include),
                    &format!("-I{}", cutest_modules),
                    &format!("-J{}", cutest_modules),
                    &trampoline_src,
                    "-o",
                    &trampoline_obj,
                ])
                .status()
                .expect("gfortran not found; install via homebrew (brew install gcc)");

            if !status.success() {
                panic!("Failed to compile cutest_trampoline.f90");
            }

            // Compile fixed fortran_open wrapper
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            let fortran_open_src =
                format!("{}/benchmarks/cutest/fortran_open_fixed.f90", manifest_dir);
            let fortran_open_obj = format!("{}/fortran_open_fixed.o", out_dir);
            let status = Command::new("gfortran")
                .args(["-c", "-fPIC", &fortran_open_src, "-o", &fortran_open_obj])
                .status()
                .expect("gfortran not found");
            if !status.success() {
                panic!("Failed to compile fortran_open_fixed.f90");
            }

            // Create a static library from both objects
            let trampoline_lib = format!("{}/libcutest_trampoline.a", out_dir);
            let status = Command::new("ar")
                .args([
                    "rcs",
                    &trampoline_lib,
                    &trampoline_obj,
                    &fortran_open_obj,
                ])
                .status()
                .expect("ar not found");
            if !status.success() {
                panic!("Failed to create libcutest_trampoline.a");
            }

            println!("cargo:rustc-link-search=native={}", out_dir);
            println!("cargo:rustc-link-lib=static=cutest_trampoline");
        } else {
            panic!(
                "CUTEst trampoline not found at {}.\n\
                 Install the CUTEst toolchain with:\n  \
                 bash benchmarks/cutest/setup_cutest.sh\n\
                 (override location with CUTEST_ROOT=/path/to/cutest).",
                trampoline_src
            );
        }

        // Link gfortran runtime. Use the platform-correct shared-library
        // extension; gfortran's -print-file-name returns the input verbatim
        // when the file isn't found, so a wrong extension yields a relative
        // string whose parent is empty -- which rustc 1.95+ rejects as an
        // empty -L. Skip emitting a link-search if we can't resolve a path.
        let gfortran_libname = if cfg!(target_os = "macos") {
            "libgfortran.dylib"
        } else {
            "libgfortran.so"
        };
        let output = Command::new("gfortran")
            .arg(format!("-print-file-name={}", gfortran_libname))
            .output()
            .expect("gfortran not found");
        let gfortran_path = String::from_utf8(output.stdout).unwrap();
        let gfortran_path = gfortran_path.trim();
        let gfortran_dir = Path::new(gfortran_path)
            .parent()
            .and_then(|p| p.to_str())
            .map(str::to_string)
            .unwrap_or_default();
        if !gfortran_dir.is_empty() {
            println!("cargo:rustc-link-search=native={}", gfortran_dir);
        }
        println!("cargo:rustc-link-lib=dylib=gfortran");

        println!("cargo:rerun-if-changed={}", trampoline_src);
        let manifest_dir_for_rerun = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        println!("cargo:rerun-if-changed={}/benchmarks/cutest/fortran_open_fixed.f90", manifest_dir_for_rerun);
    }
}
