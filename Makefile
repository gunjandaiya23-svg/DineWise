PY := .venv/bin/python
PIP := uv pip install --python $(PY)

.PHONY: setup check download prep docs index app test lint clean

setup:
	UV_HTTP_TIMEOUT=600 uv venv --python 3.12 .venv
	UV_HTTP_TIMEOUT=600 $(PIP) torch --index-url https://download.pytorch.org/whl/cu128
	UV_HTTP_TIMEOUT=600 $(PIP) -e ".[dev]"

check:
	$(PY) scripts/00_setup_check.py

download:
	$(PY) scripts/01_download_data.py

prep:
	$(PY) scripts/02_prepare_data.py

docs:
	$(PY) scripts/03_build_documents.py

chunks:
	$(PY) scripts/04_build_chunks.py

index: chunks
	$(PY) scripts/05_build_index.py

query:
	$(PY) scripts/06_query.py "$(Q)"

app:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src scripts tests

clean:
	rm -rf data/processed data/indexes data/docs/*.pdf data/docs/*.docx data/docs/*.txt
