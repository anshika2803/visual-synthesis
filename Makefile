# visual-synthesis -- the common tasks, one command each. `make help` lists them.
#
# Every target is a thin wrapper around the `visual-synthesis` command, which is
# on PATH once the environment is active (poetry shell / conda activate).

.DEFAULT_GOAL := help

VS        ?= visual-synthesis
MODEL     ?= sd-turbo
DETECTOR  ?= nudenet
STAGE     ?= generate
LIMIT     ?=
RUN_NAME  ?=
CONFIG    ?=
KEEP      ?= 20
IMAGE     ?= visual-synthesis:local
# CPU wheels by default; for CUDA: make docker-build TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cpu

# CONFIG="a.json b.json" becomes "-c a.json -c b.json"
FLAGS = $(foreach file,$(CONFIG),-c $(file)) \
        $(if $(RUN_NAME),--run-name $(RUN_NAME)) \
        $(if $(LIMIT),--limit $(LIMIT))

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- setup -------------------------------------------------------------------

.PHONY: install install-core conda-env test lint
install:  ## Install everything (models, detectors, API, UI, dev tools)
	pip install -e ".[all]"

install-core:  ## Install only what the data stages need (no torch)
	pip install -e ".[dev]"

conda-env:  ## Build the conda environment (Linux / HPC)
	conda env create -f conda/environment.yml
	@echo 'now run: conda activate visual-synthesis && pip install -e ".[all]"'

test:  ## Run the tests (no GPU, weights or network needed)
	pytest

lint:  ## Lint with ruff
	ruff check src tests

# --- pipeline ----------------------------------------------------------------

.PHONY: validate stages run-all download standardize unify dedup generate nsfw pilot dry-run
validate:  ## Check the config
	$(VS) config validate $(foreach file,$(CONFIG),-c $(file))

stages:  ## List the pipeline stages
	$(VS) stages

run-all:  ## Run every stage, chained
	$(VS) run all $(FLAGS)

download:  ## Download the datasets and models listed under `download`
	$(VS) download $(FLAGS)

standardize:  ## Convert the raw datasets to the standard schema
	$(VS) standardize $(FLAGS)

unify:  ## Merge the standardized datasets
	$(VS) unify $(FLAGS)

dedup:  ## Deduplicate the unified table
	$(VS) postprocess $(FLAGS)

generate:  ## Generate images: make generate MODEL=sd-xl LIMIT=100
	$(VS) generate --model $(MODEL) $(FLAGS)

nsfw:  ## Detect NSFW content: make nsfw DETECTOR=nudenet
	$(VS) nsfw --detector $(DETECTOR) $(FLAGS)

pilot:  ## Generate a small stratified pilot: make pilot LIMIT=60
	$(VS) generate --model $(MODEL) --sample stratified --stratify-by standard_category \
	  --limit $(or $(LIMIT),60) --run-name $(or $(RUN_NAME),pilot)

dry-run:  ## Show what the whole chain would do, without writing
	$(VS) run all --dry-run $(FLAGS)

# --- front doors -------------------------------------------------------------

.PHONY: interact ui serve
interact:  ## Interactive terminal session
	$(VS) interact --model $(MODEL) --detector $(DETECTOR)

ui:  ## Launch the web UI
	$(VS) ui

serve:  ## Launch the HTTP API
	$(VS) serve

# --- cluster, docker, cleanup ------------------------------------------------

.PHONY: slurm-render slurm-submit docker-build docker-run clean clean-runs
slurm-render:  ## Write a job script: make slurm-render STAGE=generate MODEL=sd-xl  (ARRAY=1: one task per model)
	$(VS) slurm render $(STAGE) --overwrite $(if $(ARRAY),--array,--model $(MODEL)) $(if $(LIMIT),--limit $(LIMIT)) $(foreach file,$(CONFIG),-c $(file))

slurm-submit:  ## Write a job script and submit it
	$(VS) slurm submit $(STAGE) --overwrite $(if $(ARRAY),--array,--model $(MODEL)) $(if $(LIMIT),--limit $(LIMIT)) $(foreach file,$(CONFIG),-c $(file))

docker-build:  ## Build the container image
	docker build --build-arg TORCH_INDEX_URL=$(TORCH_INDEX_URL) -t $(IMAGE) .

docker-run:  ## Run a command in the container: make docker-run CMD="run all --limit 10"
	docker run --rm -it -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models \
	  -v $(PWD)/logs:/app/logs -v $(PWD)/results:/app/results -v $(PWD)/config:/app/config:ro \
	  $(IMAGE) $(CMD)

clean:  ## Delete caches, old logs (keeping the newest KEEP) and rendered jobs
	$(VS) clean all --keep $(KEEP) --yes

clean-runs:  ## DESTRUCTIVE: delete all but the newest run of every stage (needs CONFIRM=1)
	@test "$(CONFIRM)" = "1" || { echo "this deletes pipeline output; run: make clean-runs CONFIRM=1"; exit 1; }
	$(VS) clean runs --yes
