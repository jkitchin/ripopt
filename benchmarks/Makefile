# Benchmark suite Makefile.
#
# This Makefile drives the per-suite benchmarks under benchmarks/.
# The root Makefile delegates to this one for all benchmark targets so there's
# one source of truth. Cargo commands are invoked with --manifest-path pointing
# at the repo root so this Makefile works regardless of the user's CWD.
#
# See benchmarks/README.md for suite descriptions and per-suite READMEs for
# details on each benchmark.

SHELL = /bin/bash
REPO_ROOT := $(shell cd .. && pwd)
CARGO_MANIFEST := $(REPO_ROOT)/Cargo.toml
CARGO := cargo

.PHONY: help benchmark benchmark-report \
	cutest cutest-prepare cutest-prepare-full cutest-run cutest-full cutest-report cutest-smoke cutest-large cutest-maxiter \
	electrolyte-run grid-run cho-run large-scale gas-run water-run

# Show available targets
help:
	@echo "Benchmark suite targets (run from benchmarks/ or via root Makefile shims)."
	@echo ""
	@echo "  benchmark         Full sweep: CUTEst + domain + large-scale + report + gas"
	@echo "  benchmark-report  Regenerate benchmarks/BENCHMARK_REPORT.md"
	@echo ""
	@echo "  cutest            Full CUTEst pipeline: prepare, run, report"
	@echo "  cutest-prepare    Compile SIF problems to .dylib/.so (problem_list.txt, 727)"
	@echo "  cutest-prepare-full  Compile all SIF problems (problem_list_full.txt, 1542)"
	@echo "  cutest-run        Run CUTEst benchmark (writes benchmarks/cutest/results.json)"
	@echo "  cutest-full       Full CUTEst benchmark (all 1542 SIF problems)"
	@echo "  cutest-smoke      Quick smoke test (~20 problems)"
	@echo "  cutest-large      Large problems (n+m >= 100, sparse path)"
	@echo "  cutest-maxiter    Re-run MaxIterations failures"
	@echo "  cutest-report     Generate comparison report from results.json"
	@echo ""
	@echo "  electrolyte-run   Electrolyte thermodynamics benchmark"
	@echo "  grid-run          AC optimal power flow (electrical grid) benchmark"
	@echo "  cho-run           CHO parameter estimation benchmark"
	@echo "  large-scale       Synthetic large-scale NLPs (up to 100K vars)"
	@echo "  gas-run           Gas pipeline NLPs (AMPL .nl files)"
	@echo "  water-run         Water distribution network NLPs (AMPL .nl files)"
	@echo ""
	@echo "Most suites feed into benchmarks/BENCHMARK_REPORT.md. The gas and water"
	@echo "suites are standalone (see benchmarks/{gas,water}/README.md)."

# ---- CUTEst ----
cutest-prepare:
	source ~/.local/cutest/env.sh && bash $(REPO_ROOT)/benchmarks/cutest/prepare.sh

cutest-prepare-full:
	source ~/.local/cutest/env.sh && bash $(REPO_ROOT)/benchmarks/cutest/prepare.sh \
		--from-file $(REPO_ROOT)/benchmarks/cutest/problem_list_full.txt

cutest-run:
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --features cutest,ipopt-native --release \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/benchmark_stderr.txt >&2)

cutest-full:
	PROBLEM_LIST=$(REPO_ROOT)/benchmarks/cutest/problem_list_full.txt \
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/full_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --features cutest,ipopt-native --release \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/full_stderr.txt >&2)

cutest-report:
	python $(REPO_ROOT)/benchmarks/cutest/compare.py $(REPO_ROOT)/benchmarks/cutest/results.json

cutest-smoke:
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/smoke_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --features cutest,ipopt-native --release -- \
		ROSENBR BEALE CUBE DENSCHNB BROWNBS \
		HS6 HS10 HS35 HS71 HS106 \
		MARATOS BT1 HS13 HS57 S316-322 \
		BEALENE BROWNBSNE ENGVAL2NE \
		PALMER1D PALMER5D \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/smoke_stderr.txt >&2)
	@echo "Smoke test complete."

cutest-large:
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/large_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --features cutest,ipopt-native --release -- \
		HIMMELBI AIRPORT LAKES SWOPF QPCBLEND QPNBLEND SPANHYD LINSPANH \
		ACOPP14 ACOPR14 ACOPP30 ACOPR30 BATCH CORE1 MSS1 HAIFAM FEEDLOC \
		DISCS DECONVBNE DECONVNE NET1 HYDCAR20 \
		TFI1 TFI2 TFI3 EXPFITB ELATTAR CRESC50 EXPFITC \
		DUALC1 DUALC2 DUALC5 DUALC8 \
		PT HET-Z OET1 OET2 OET3 OET4 KSIP \
		SIPOW1 SIPOW1M SIPOW2 SIPOW2M SIPOW3 SIPOW4 \
		TAXR13322 GOFFIN \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/large_stderr.txt >&2)
	@echo "Large problem benchmark complete."

cutest-maxiter:
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/maxiter_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --features cutest,ipopt-native --release -- \
		AIRCRFTB ALLINIT BENNETT5LS BIGGS3 BIGGS5 BOX2 BQPGABIM CHWIRUT2LS \
		DJTL HAHN1LS MEYER3 MGH10LS MINSURF OSBORNEA RAT43LS SIM2BQP THURBERLS \
		AIRCRFTA CONCON DECONVC DECONVNE DECONVU DNIEPER HS109 HS67 HS83 HS84 \
		HS85 HS99EXP MCONCON OPTCNTRL QC QCNEW SSINE TRIGGER \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/maxiter_stderr.txt >&2)

cutest: cutest-prepare cutest-run cutest-report
	@echo "CUTEst benchmark complete. Report: benchmarks/cutest/CUTEST_REPORT.md"

# ---- Domain benchmarks ----
electrolyte-run:
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/electrolyte/electrolyte_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --release --features ipopt-native --example electrolyte_benchmark \
		2> >(tee $(REPO_ROOT)/benchmarks/electrolyte/electrolyte_stderr.txt >&2)
	@echo "Electrolyte benchmark complete."

grid-run:
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/grid/grid_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --release --features ipopt-native --example grid_benchmark \
		2> >(tee $(REPO_ROOT)/benchmarks/grid/grid_stderr.txt >&2)
	@echo "Grid benchmark complete."

cho-run:
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cho/cho_results.json \
	CHO_NL_PATH=$(REPO_ROOT)/benchmarks/cho/nl_export_results/cho_parmest.nl \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --release --features ipopt-native --example cho_benchmark \
		2> >(tee $(REPO_ROOT)/benchmarks/cho/cho_stderr.txt >&2)
	@echo "CHO benchmark complete."

# ---- Large scale ----
large-scale:
	$(CARGO) test --manifest-path $(CARGO_MANIFEST) --release --features ipopt-native -- --ignored large_scale_vs_ipopt --nocapture \
		2>&1 | tee $(REPO_ROOT)/benchmarks/large_scale/large_scale_results.txt
	@echo "Large-scale benchmark complete. Output: benchmarks/large_scale/large_scale_results.txt"

# ---- Gas pipeline ----
# Solves each .nl file with ripopt directly. Results land in per-problem .sol
# files alongside the .nl. No JSON aggregation and does NOT feed BENCHMARK_REPORT.
gas-run:
	@set -e; \
	for nl in $(REPO_ROOT)/benchmarks/gas/*.nl; do \
		echo "=== $$(basename $$nl) ==="; \
		$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin ripopt --release -- $$nl print_level=5 || true; \
	done
	@echo "Gas benchmark complete. Solver output is in benchmarks/gas/*.sol."

# ---- Water network ----
# Solves each .nl file with ripopt directly. Results land in per-problem .sol
# files alongside the .nl. No JSON aggregation and does NOT feed BENCHMARK_REPORT.
water-run:
	@set -e; \
	for nl in $(REPO_ROOT)/benchmarks/water/*.nl; do \
		echo "=== $$(basename $$nl) ==="; \
		$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin ripopt --release -- $$nl print_level=5 || true; \
	done
	@echo "Water benchmark complete. Solver output is in benchmarks/water/*.sol."

# ---- Aggregation ----
# Full sweep. Note: gas-run and water-run run after benchmark-report because
# they do not contribute to the composite report.
benchmark: cutest-run electrolyte-run grid-run large-scale benchmark-report gas-run water-run
	@echo "Full benchmark complete. Report: benchmarks/BENCHMARK_REPORT.md"

# Generate composite report from existing results JSONs.
benchmark-report:
	python $(REPO_ROOT)/benchmarks/benchmark_report.py
	@echo "Report: benchmarks/BENCHMARK_REPORT.md"
