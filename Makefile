.PHONY: init train test update eval split-data help

# ── Defaults (override on the command line) ────────────────────────────────
CONFIG   ?= config/UrbanElementsReID_train.yml
LOG      ?= train.log
PIDFILE  ?= train.pid
TRACK    ?= outputs/track

# ── PAT training ──────────────────────────────────────────────────────────
# Run in background; tail the log with `make tail-log`
train:
	@python scripts/train.py --config_file "$(CONFIG)" > "$(LOG)" 2>&1 & echo $$! > "$(PIDFILE)"
	@echo "Training started — PID: $$(cat '$(PIDFILE)')"
	@echo "Log: $(LOG)  |  tail with: make tail-log"

tail-log:
	@tail -n 200 -f "$(LOG)"

stop:
	@if [ -f "$(PIDFILE)" ]; then \
		PID="$$(cat '$(PIDFILE)')"; \
		echo "Stopping PID $$PID"; \
		kill $$PID || true; \
	else \
		echo "No PID file at $(PIDFILE)"; \
	fi

# ── DINOv3 pipeline (staged fine-tune) ────────────────────────────────────
train-dinov3:
	python scripts/train_dinov3.py --config_file "$(CONFIG)"

extract:
	python scripts/extract_dinov3.py --config_file "$(CONFIG)"

eval-variants:
	python scripts/eval_variants.py --config_file "$(CONFIG)"

# ── Submission generation (PAT) ────────────────────────────────────────────
update:
	python scripts/update.py --config_file "$(CONFIG)" --track "$(TRACK)"

# ── Build reproducible val split ──────────────────────────────────────────
split-data:
	python src/urban_elements_reid_challenge/data/build_split_v2.py

# ── Local mAP evaluation ───────────────────────────────────────────────────
eval:
	python scripts/evaluate_csv.py \
		--path "$(DATA_ROOT)/csv_folder/" \
		--track "$(TRACK)/track_submission.csv"

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo "Targets:"
	@echo "  init            Install Python dependencies"
	@echo "  train           Train PAT model (CONFIG=...)"
	@echo "  train-dinov3    Train DINOv3 model (CONFIG=...)"
	@echo "  extract         Extract & cache DINOv3 features"
	@echo "  eval-variants   Run post-processing variants"
	@echo "  update          Generate PAT submission CSV"
	@echo "  split-data      Build reproducible val split CSV"
	@echo "  eval            Compute local mAP (DATA_ROOT=...)"
	@echo ""
	@echo "Variables (set on command line):"
	@echo "  CONFIG   config YAML file (default: $(CONFIG))"
	@echo "  LOG      training log path (default: $(LOG))"
	@echo "  TRACK    submission output directory (default: $(TRACK))"
	@echo "  DATA_ROOT  path to dataset root for evaluation"
