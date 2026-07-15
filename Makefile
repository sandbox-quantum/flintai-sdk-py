.PHONY: install install-dev test lint typecheck build clean publish release-audit license-check sbom vuln-scan secret-scan lock-deps check-metadata

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"

test:
	pytest

lint:
	ruff check src/ tests/ examples/
	ruff format --check src/ tests/ examples/

lint-fix:
	ruff check --fix src/ tests/ examples/
	ruff format src/ tests/ examples/

typecheck:
	mypy src/

build: clean
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info

publish: release-audit build
	twine check dist/*
	twine upload dist/*

# ---------------------------------------------------------------------------
# Metadata consistency (README ranges must match pyproject.toml pins)
# ---------------------------------------------------------------------------

check-metadata:
	@echo "=== Metadata consistency check ==="
	@errors=0; \
	for dep in openai anthropic google-genai; do \
		pyproject_range=$$(grep "\"$${dep}>=" pyproject.toml | sed "s/.*\"\($${dep}[^\"]*\)\".*/\1/" | head -1); \
		readme_range=$$(grep "\`$${dep}>=" README.md | sed "s/.*\`\($${dep}[^\`]*\)\`.*/\1/" | head -1); \
		if [ -z "$$pyproject_range" ] || [ -z "$$readme_range" ]; then \
			echo "SKIP: $${dep} (not found in both files)"; \
		elif [ "$$pyproject_range" != "$$readme_range" ]; then \
			echo "FAIL: $${dep} range mismatch"; \
			echo "  pyproject.toml: $${pyproject_range}"; \
			echo "  README.md:      $${readme_range}"; \
			errors=$$((errors + 1)); \
		else \
			echo "OK: $${dep} $${pyproject_range}"; \
		fi; \
	done; \
	if [ "$$errors" -gt 0 ]; then \
		echo ""; \
		echo "FAILED: $$errors metadata inconsistency(ies). Update README.md to match pyproject.toml."; \
		exit 1; \
	fi
	@echo "All metadata checks passed."
	@echo ""

# ---------------------------------------------------------------------------
# Release audit (mirrors flintai-cli)
# ---------------------------------------------------------------------------

release-audit: check-metadata lock-deps license-check sbom vuln-scan secret-scan
	@echo ""
	@echo "=== Release audit complete ==="
	@echo "Outputs in release/:"
	@ls -1 release/*.json release/*.csv release/*.txt 2>/dev/null
	@echo ""

# Packages whose license field carries full license text rather than a short
# SPDX identifier (so --allow-only can't match them). Manually verified permissive:
#   tiktoken       — MIT License
#   flintai-sdk-py — Apache-2.0 (this project)
IGNORE_PACKAGES := --ignore-packages flintai-sdk-py tiktoken

license-check:
	@mkdir -p release
	@echo "=== License review ==="
	pip-licenses \
		--format=csv \
		--with-urls \
		--with-authors \
		--output-file=release/licenses.csv
	pip-licenses \
		--format=json \
		--with-urls \
		--with-authors \
		--output-file=release/licenses.json
	@echo "License summary:"
	@pip-licenses --summary --order=count
	@echo ""
	@echo "Checking for copyleft/restricted licenses..."
	@pip-licenses $(IGNORE_PACKAGES) --allow-only="Apache Software License;\
Apache Software License; BSD License;\
Apache Software License; MIT License;\
Apache License 2.0;\
Apache 2.0;\
Apache-2.0;\
Apache-2.0 AND CNRI-Python;\
Apache-2.0 AND MIT;\
Apache-2.0 OR BSD-2-Clause;\
Apache-2.0 OR BSD-3-Clause;\
Apache-2.0 OR MIT;\
Apache 2.0 License;\
Apache License;\
Apache;\
BSD License;\
BSD-2-Clause;\
BSD-3-Clause;\
BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0;\
BSD 3-Clause OR Apache-2.0;\
3-Clause BSD License;\
MIT License;\
MIT;\
MIT-0;\
MIT OR Apache-2.0;\
MIT-CMU;\
ISC License (ISCL);\
ISC;\
Mozilla Public License 2.0 (MPL 2.0);\
MPL-2.0 AND (Apache-2.0 OR MIT);\
MPL-2.0 AND MIT;\
Python Software Foundation License;\
PSF;\
PSF-2.0;\
Public Domain;\
The Unlicense (Unlicense);\
Historical Permission Notice and Disclaimer (HPND);\
GNU Lesser General Public License v2 or later (LGPLv2+);\
GNU Lesser General Public License v3 or later (LGPLv3+);\
Zope Public License;\
ZPL-2.1;\
UNKNOWN" \
		2>&1 || (echo "WARNING: Some packages have licenses that need manual review — see above" && exit 1)
	@echo ""
	@echo "Packages with UNKNOWN license (need manual verification):"
	@pip-licenses --format=csv | grep UNKNOWN || echo "  (none)"
	@echo ""
	@echo "All known licenses are Apache-2.0 compatible."

sbom:
	@mkdir -p release
	@echo "=== SBOM generation (CycloneDX) ==="
	cyclonedx-py environment \
		-o release/sbom.json \
		--of json \
		--pyproject pyproject.toml
	@echo "SBOM written to release/sbom.json"

vuln-scan:
	@mkdir -p release
	@echo "=== Dependency vulnerability scan ==="
	pip-audit --format=json --output=release/vulnerabilities.json || true
	pip-audit --desc 2>&1 | tee release/vulnerabilities.txt
	@echo ""

secret-scan:
	@mkdir -p release
	@echo "=== Secret scan (gitleaks) ==="
	gitleaks detect --source . --no-git \
		--report-path=release/gitleaks-report.json \
		--report-format=json \
		--exit-code 1 \
		|| (echo "FAIL: secrets detected — see release/gitleaks-report.json" && exit 1)
	@echo "=== Secret scan (gitleaks) - git history ==="
	gitleaks detect --source . \
		--report-path=release/gitleaks-history-report.json \
		--report-format=json \
		--exit-code 1 \
		|| (echo "FAIL: secrets detected in git history - see release/gitleaks-history-report.json" && exit 1)
	@echo "No secrets detected."

lock-deps:
	@echo "=== Locking dependencies ==="
	pip-compile --all-extras --strip-extras --output-file=requirements.lock pyproject.toml
	@echo "Locked dependencies written to requirements.lock"
