PY := .venv/bin/python

.PHONY: setup audit pilot shap aipw sim km test all

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

audit:
	$(PY) scripts/run_audit.py

pilot:
	$(PY) scripts/run_pilot.py

shap:
	$(PY) scripts/run_shap.py

aipw:
	$(PY) scripts/run_aipw.py

sim:
	$(PY) scripts/run_sim.py

km:
	$(PY) scripts/run_km.py

test:
	.venv/bin/pytest -q tests

all: audit test pilot shap aipw sim km
