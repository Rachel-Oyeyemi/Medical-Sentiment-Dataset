PYTHON ?= python

.PHONY: pipeline test app presentation

pipeline:
	$(PYTHON) run_pipeline.py

test:
	pytest -q

app:
	streamlit run app/app.py

presentation:
	$(PYTHON) presentation/generate_presentation.py
