PY := .venv/bin/python

.PHONY: setup audit test main founder shap sim source stem macro h4 survival audit-sample all legacy-pilot legacy-aipw

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

audit:
	$(PY) scripts/run_audit.py

test:
	.venv/bin/pytest -q tests

main:
	$(PY) scripts/run_main_analysis.py

founder:
	$(PY) scripts/run_founder_hypotheses.py

shap:
	$(PY) scripts/run_shap.py

sim:
	$(PY) scripts/run_sim.py

source:
	$(PY) scripts/run_source_checks.py

stem:
	$(PY) scripts/run_stem_sensitivity.py

macro:
	$(PY) scripts/run_macro_ablation.py

h4:
	$(PY) scripts/run_h4_diagnostics.py

survival:
	$(PY) scripts/run_survival.py

audit-sample:
	$(PY) scripts/make_audit_sample.py

all: audit test main founder shap sim source stem macro h4 survival

legacy-pilot:
	$(PY) scripts/run_pilot.py

legacy-aipw:
	$(PY) scripts/run_aipw.py

.PHONY: check appendix ci
check:
	$(PY) scripts/check_manuscript_numbers.py

appendix:
	$(PY) scripts/make_appendix_tables.py

ci:
	$(PY) scripts/run_ci_methods.py
