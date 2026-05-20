# Minimal Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html dirhtml singlehtml htmlview linkcheck doctest coverage strict serve livehtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" "$(SOURCEDIR)/_autosummary"

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

dirhtml:
	@$(SPHINXBUILD) -M dirhtml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

singlehtml:
	@$(SPHINXBUILD) -M singlehtml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build HTML then open the result in $BROWSER (or xdg-open).
htmlview: html
	@xdg-open "$(BUILDDIR)/html/index.html" 2>/dev/null || open "$(BUILDDIR)/html/index.html" 2>/dev/null || true

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doctest:
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

coverage:
	@$(SPHINXBUILD) -M coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Treat warnings as errors. Use this in CI.
strict:
	@$(SPHINXBUILD) -b html -W --keep-going -n "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

serve: html
	@cd "$(BUILDDIR)/html" && uv run python -m http.server 8000

livehtml:
	uv run sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --open-browser $(O)

# `make latexpdf` chains latexmk → xelatex (multiple passes). The intermediate
# passes emit forward-reference and rerun-required warnings that ALWAYS resolve
# on the final pass; they're noise, not real failures. Run the target then
# audit the *final* xpcsjax.log for any genuine remaining warnings.
latexpdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) \
	  > "$(BUILDDIR)/latexpdf.stdout.log" 2>&1 \
	  && echo "make latexpdf: PDF written to $(BUILDDIR)/latex/xpcsjax.pdf" \
	  || (cat "$(BUILDDIR)/latexpdf.stdout.log"; exit 1)
	@if grep -qE "(Warning|Error|undefined|multiply|Underfull|Overfull|^!)" \
	   "$(BUILDDIR)/latex/xpcsjax.log" 2>/dev/null \
	   | grep -vE "^(Package: |\\\\TU/|infwarerr|FreeMono|FreeSerif)"; then \
	  echo "ERROR: real warnings in $(BUILDDIR)/latex/xpcsjax.log"; \
	  exit 1; \
	fi

%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
