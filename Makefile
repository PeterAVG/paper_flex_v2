VENV_DIR = .venv

clean:
	rm -rf $(VENV_DIR) poetry.lock venv_julia .tmp .mypy_cache .pytest_cache

clean-tex:
	rm -rf tex/ieee/*.glo tex/ieee/*.aux tex/ieee/*.fls tex/ieee/*.ist tex/ieee/*-glg tex/ieee/*-glo tex/ieee/*-gls tex/ieee/*.synctex.gz tex/ieee/*.dvi tex/ieee/*.log tex/ieee/*.pdf tex/ieee/*.bbl tex/ieee/*.blg tex/ieee/*.fdb_latexmk
	rm -rf tex/*.glo tex/*.aux tex/*.fls tex/*.ist tex/*-glg tex/*-glo tex/*-gls tex/*.synctex.gz tex/*.dvi tex/*.log tex/*.pdf tex/*.bbl tex/*.blg tex/*.fdb_latexmk
	rm -rf tex/figures/*.glo tex/figures/*.aux tex/figures/*.fls tex/figures/*.ist tex/figures/*-glg tex/figures/*-glo tex/figures/*-gls tex/figures/*.synctex.gz tex/figures/*.dvi tex/figures/*.log tex/figures/*.pdf tex/figures/*.bbl tex/figures/*.blg tex/figures/*.fdb_latexmk

dev-setup: $(VENV_DIR)/.made
	$(VENV_DIR)/bin/pre-commit install -ft pre-push
	$(VENV_DIR)/bin/pre-commit install -ft pre-commit

$(VENV_DIR)/.made:
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	poetry install
	touch $@

julia-setup:
	julia --project=. src/julia/setup.jl "venv_julia"
