SHELL = /bin/bash
.PHONY: help test test-c test-iterative install uninstall clean cutest-install cutest-prepare cutest-run cutest-full cutest-report cutest cutest-maxiter cutest-smoke cutest-large large-scale electrolyte-run grid-run cho-run gas-run water-run benchmark benchmark-report

# Show available targets
help:
	@echo "Usage: make <target> [CUTEST_MAX_N=N]"
	@echo ""
	@echo "Build & Install:"
	@echo "  install          Build and install ripopt binary, shared library, and Pyomo plugin"
	@echo "  uninstall        Remove ripopt binary, shared library, and Pyomo plugin"
	@echo "  clean            Remove build artifacts (cargo clean)"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run unit/integration tests (cargo test)"
	@echo "  test-iterative   Run iterative/hybrid solver tests (release, large-scale)"
	@echo ""
	@echo "Benchmarks:"
	@echo "  benchmark        Full benchmark: CUTEst + domain + large-scale + report"
	@echo "  benchmark-report Generate unified report from existing results"
	@echo "  electrolyte-run  Run electrolyte thermodynamics benchmark"
	@echo "  grid-run         Run AC optimal power flow (electrical grid) benchmark"
	@echo "  cho-run          Run CHO parameter estimation benchmark"
	@echo "  large-scale      Run synthetic large-scale tests (up to 100K vars)"
	@echo "  gas-run          Run gas pipeline NLP benchmark (AMPL .nl files)"
	@echo "  water-run        Run water distribution network NLP benchmark (AMPL .nl files)"
	@echo "  test-c           Build shared library and run C API examples"
	@echo ""
	@echo "CUTEst Benchmarks:"
	@echo "  cutest-install   Print upstream CUTEst install instructions"
	@echo "  cutest           Full pipeline: prepare, run, report"
	@echo "  cutest-prepare   Compile SIF problems -> .dylib (reads problem_list.txt)"
	@echo "  cutest-run       Run benchmark on all prepared problems (results.json)"
	@echo "  cutest-report    Generate comparison report from results.json"
	@echo "  cutest-full      Full benchmark: all 1542 SIF problems (full_results.json)"
	@echo "  cutest-smoke     Quick smoke test (~20 small problems, n+m < 50)"
	@echo "  cutest-large     Large problems (n+m >= 100, sparse solver)"
	@echo "  cutest-maxiter   Re-run the 35 MaxIterations failures"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUTEST_MAX_N=N   Skip problems with n > N (default: 100)"

# Detect shared library extension
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  DYLIB_EXT := dylib
else
  DYLIB_EXT := so
endif

# Install ripopt binary and shared library
install:
	cargo build --release
	cargo install --path . --bin ripopt
	mkdir -p ~/.local/lib
	cp target/release/libripopt.$(DYLIB_EXT) ~/.local/lib/
	@echo ""
	@echo "Installed:"
	@echo "  ripopt binary      -> ~/.cargo/bin/ripopt"
	@echo "  libripopt.$(DYLIB_EXT)  -> ~/.local/lib/libripopt.$(DYLIB_EXT)"
	@echo ""
	@if echo "$$PATH" | tr ':' '\n' | grep -qx "$$HOME/.cargo/bin"; then \
		echo "Verify: ripopt --version"; \
	else \
		echo "WARNING: ~/.cargo/bin is not on your PATH."; \
		echo "Add it by appending this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"; \
		echo "  export PATH=\"\$$HOME/.cargo/bin:\$$PATH\""; \
		echo "Then restart your shell or run: source ~/.bashrc"; \
	fi
	@if command -v pip >/dev/null 2>&1; then \
		echo ""; \
		echo "Installing Pyomo solver plugin..."; \
		pip install ./pyomo-ripopt && \
		echo "  pyomo-ripopt       -> installed via pip"; \
	else \
		echo ""; \
		echo "NOTE: pip not found. To install the Pyomo solver plugin later, run:"; \
		echo "  pip install ./pyomo-ripopt"; \
	fi
	@if echo "$$LD_LIBRARY_PATH:$$DYLD_LIBRARY_PATH" | tr ':' '\n' | grep -qx "$$HOME/.local/lib"; then \
		true; \
	else \
		echo ""; \
		echo "NOTE: To use the shared library, ensure ~/.local/lib is in your library path:"; \
		echo "  export LD_LIBRARY_PATH=\"\$$HOME/.local/lib:\$$LD_LIBRARY_PATH\""; \
	fi

# Uninstall ripopt binary and shared library
uninstall:
	cargo uninstall ripopt 2>/dev/null || true
	pip uninstall -y pyomo-ripopt 2>/dev/null || true
	rm -f ~/.local/lib/libripopt.$(DYLIB_EXT)
	@echo "Uninstalled ripopt binary and shared library."

# Remove build artifacts
clean:
	cargo clean

# Run unit/integration tests (including C API tests via Rust FFI)
test:
	cargo test

# Run iterative/hybrid linear solver integration tests (includes ignored large-scale tests)
test-iterative:
	cargo test --release -- --ignored iterative hybrid

# Build shared library and compile/run all C API examples
test-c:
	cargo build --release
	@echo "=== Compiling and running C API examples ==="
	cc examples/c_api_test.c -I. -Ltarget/release -lripopt \
		-Wl,-rpath,$(CURDIR)/target/release -o target/release/c_api_test -lm
	target/release/c_api_test
	cc examples/c_rosenbrock.c -I. -Ltarget/release -lripopt \
		-Wl,-rpath,$(CURDIR)/target/release -o target/release/c_rosenbrock -lm
	target/release/c_rosenbrock
	cc examples/c_hs035.c -I. -Ltarget/release -lripopt \
		-Wl,-rpath,$(CURDIR)/target/release -o target/release/c_hs035 -lm
	target/release/c_hs035
	cc examples/c_example_with_options.c -I. -Ltarget/release -lripopt \
		-Wl,-rpath,$(CURDIR)/target/release -o target/release/c_example_with_options -lm
	target/release/c_example_with_options
	@echo "=== All C API examples passed ==="

# CUTEst toolchain install (manual — see benchmarks/cutest/README.md)
cutest-install:
	@echo "ripopt does not vendor a CUTEst installer."
	@echo ""
	@echo "Install the upstream CUTEst toolchain into ~/.local/cutest/ following the"
	@echo "instructions at https://github.com/ralna/CUTEst (ARCHDefs + SIFDecode +"
	@echo "CUTEst), then clone MASTSIF:"
	@echo "  git clone https://github.com/ralna/mastsif ~/.local/cutest/mastsif"
	@echo ""
	@echo "After install, source the env file and run 'make cutest-prepare':"
	@echo "  source ~/.local/cutest/env.sh"
	@echo "  make cutest-prepare"

# Backward-compatibility shims — real targets live in benchmarks/Makefile.
cutest-prepare cutest-run cutest-full cutest-report cutest-smoke cutest-large cutest-maxiter cutest:
	$(MAKE) -C benchmarks $@

large-scale electrolyte-run grid-run cho-run gas-run water-run benchmark benchmark-report:
	$(MAKE) -C benchmarks $@
