# Mittelmann ampl-nlp benchmark harness
#
# Mirrors the structure of gams/Makefile. Pulls .mod sources from
# plato.asu.edu/ftp/ampl-nlp-source/, translates them to .nl with
# the AMPL CE binary in ../.venv-ampl, runs ripopt + ipopt, and
# emits a per-version markdown report.
#
# The .mod sources, the .nl translations, and per-instance run logs
# are all gitignored. Only the harness files (Makefile, problems.txt,
# scripts) and the final reports/*.md live in git.
#
# Usage:
#   make fetch         download all .mod files from plato (cached)
#   make translate     .mod -> .nl via ampl (cached)
#   make run-ripopt    run ripopt on every translated .nl
#   make run-ipopt     run ipopt on every translated .nl
#   make report        regenerate reports/BENCHMARK_REPORT_<version>.md
#   make all           fetch + translate + run-ripopt + run-ipopt + report
#   make clean         remove logs and per-version results
#   make clean-cache   also remove cached .mod and .nl

# --- Paths ---
RIPOPT_ROOT  ?= ../..
AMPL         ?= $(RIPOPT_ROOT)/.venv-ampl/lib/python3.12/site-packages/ampl_module_base/bin/ampl
RIPOPT_BIN   ?= $(RIPOPT_ROOT)/target/release/ripopt
IPOPT_BIN    ?= ipopt
SOURCE_URL   ?= https://plato.asu.edu/ftp/ampl-nlp-source

# --- Per-instance limits ---
TIMELIMIT    ?= 7200       # seconds, matches Mittelmann
MAX_ITER     ?= 3000

# --- Version tag for report filenames ---
VERSION      ?= $(shell awk '/^version/{gsub(/[" ]/,""); split($$0,a,"="); print "v"a[2]; exit}' $(RIPOPT_ROOT)/Cargo.toml)

# --- Problem set ---
PROBLEMS := $(shell cat problems.txt)
MOD_FILES := $(addprefix source/,$(addsuffix .mod,$(PROBLEMS)))
NL_FILES  := $(addprefix nl/,$(addsuffix .nl,$(PROBLEMS)))

.PHONY: help all fetch translate run-ripopt run-ipopt report clean clean-cache

help:
	@echo "Mittelmann ampl-nlp benchmark targets:"
	@echo "  fetch         download .mod files from plato.asu.edu (cached)"
	@echo "  translate     .mod -> .nl via local AMPL (cached)"
	@echo "  run-ripopt    run ripopt on every .nl, write results/ripopt_<version>.json"
	@echo "  run-ipopt     run ipopt on every .nl, write results/ipopt_<version>.json"
	@echo "  report        regenerate reports/BENCHMARK_REPORT_<version>.md"
	@echo "  all           fetch + translate + run-ripopt + run-ipopt + report"
	@echo "  clean         remove logs and per-version results"
	@echo "  clean-cache   also remove cached .mod and .nl"
	@echo ""
	@echo "Variables:"
	@echo "  TIMELIMIT=<s>   per-instance timeout (default 7200, matches Mittelmann)"
	@echo "  VERSION=vX.Y.Z  report tag (default: read from Cargo.toml)"

all: fetch translate run-ripopt run-ipopt report

# --- fetch ---

source/%.mod:
	@mkdir -p source
	@echo "  fetching $*.mod"
	@curl -sf -o $@ $(SOURCE_URL)/$*.mod 2>/dev/null \
	  || (curl -sf -o $@.gz $(SOURCE_URL)/$*.mod.gz && gunzip -f $@.gz) \
	  || (echo "    FAILED: $*"; rm -f $@ $@.gz; exit 1)

fetch: $(MOD_FILES)
	@echo "fetched $$(ls source/*.mod 2>/dev/null | wc -l) of $(words $(PROBLEMS)) .mod files"

# --- translate ---

nl/%.nl: source/%.mod | $(AMPL)
	@mkdir -p nl logs/translate
	@echo "  translating $*"
	@cd nl && (echo "model ../source/$*.mod; option auxfiles cr; write g$*;" \
	  | $(abspath $(AMPL)) > $(abspath logs/translate)/$*.log 2>&1) \
	  || (echo "    FAILED: see logs/translate/$*.log"; rm -f $@; exit 1)

translate: $(NL_FILES)
	@echo "translated $$(ls nl/*.nl 2>/dev/null | wc -l) of $(words $(PROBLEMS)) problems"

# --- run ripopt ---

run-ripopt: translate $(RIPOPT_BIN)
	@mkdir -p logs/ripopt results
	@./run_solver.sh ripopt "$(RIPOPT_BIN)" $(TIMELIMIT) $(VERSION)

# --- run ipopt ---

run-ipopt: translate
	@mkdir -p logs/ipopt results
	@./run_solver.sh ipopt "$(IPOPT_BIN)" $(TIMELIMIT) $(VERSION)

# --- report ---

report:
	@mkdir -p reports
	@./make_report.py $(VERSION) reports/BENCHMARK_REPORT_$(VERSION).md
	@echo "wrote reports/BENCHMARK_REPORT_$(VERSION).md"

$(RIPOPT_BIN):
	@echo "building $(RIPOPT_BIN)..."
	@cd $(RIPOPT_ROOT) && cargo build --release --bin ripopt

# --- clean ---

clean:
	rm -rf logs results

clean-cache: clean
	rm -rf source nl
