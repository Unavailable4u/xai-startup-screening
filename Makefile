PY := .venv/bin/python

.PHONY: setup audit test main founder shap sim survival audit-sample all legacy-pilot legacy-aipw

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

survival:
	$(PY) scripts/run_survival.py

audit-sample:
	$(PY) scripts/make_audit_sample.py

all: audit test main founder shap sim survival

legacy-pilot:
	$(PY) scripts/run_pilot.py

legacy-aipw:
	$(PY) scripts/run_aipw.py
