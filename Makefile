.PHONY: technique-api technique-check-light technique-test-light technique-smoke-manifest technique-strategy-audit technique-preflight technique-status technique-app-collection-plan technique-app-materialize-collection technique-app-export-collection-packet technique-app-check-collection technique-app-prepare-review technique-app-review-progress technique-app-manifests technique-app-adapted-plan technique-app-adapted-train technique-app-baseline-eval technique-app-adapted-eval technique-app-adapted-compare technique-app-adapted-package

TECHNIQUE_APP_AUDIO_DIR ?= ./gt_singer_grader/data/app_recordings/raw
TECHNIQUE_APP_AUDIO_ROOT ?= .
TECHNIQUE_APP_LABELS ?= ./gt_singer_grader/data/app_recordings/review_labels.csv
TECHNIQUE_APP_COLLECTION_PLAN ?= ./gt_singer_grader/data/app_recordings/collection_plan.csv
TECHNIQUE_APP_COLLECTION_PLAN_JSON ?= ./gt_singer_grader/data/app_recordings/collection_plan.json
TECHNIQUE_APP_COLLECTION_ROOT ?= ./gt_singer_grader/data/app_recordings
TECHNIQUE_APP_COLLECTION_CHECKLIST ?= ./gt_singer_grader/data/app_recordings/collection_checklist.csv
TECHNIQUE_APP_COLLECTION_MISSING ?= ./gt_singer_grader/data/app_recordings/collection_missing.csv
TECHNIQUE_APP_COLLECTION_MATERIALIZE_REPORT ?= ./gt_singer_grader/data/app_recordings/collection_materialize_report.json
TECHNIQUE_APP_COLLECTION_PACKET_DIR ?= ./gt_singer_grader/data/app_recordings/collection_packet
TECHNIQUE_APP_COLLECTION_PACKET_SUMMARY ?= ./gt_singer_grader/data/app_recordings/collection_packet_summary.json
TECHNIQUE_APP_PREPARE_REPORT ?= ./gt_singer_grader/data/app_recordings/prepare_report.json
TECHNIQUE_APP_LABEL_COVERAGE_REPORT ?= ./gt_singer_grader/data/app_recordings/label_coverage_report.json
TECHNIQUE_APP_MANIFEST ?= ./gt_singer_grader/manifests/app_recordings.jsonl
TECHNIQUE_APP_TRAINABLE_MANIFEST ?= ./gt_singer_grader/manifests/app_recordings_trainable.jsonl
TECHNIQUE_APP_EVAL_ONLY_MANIFEST ?= ./gt_singer_grader/manifests/app_recordings_eval_only.jsonl
TECHNIQUE_APP_TRAIN_MANIFEST ?= ./gt_singer_grader/manifests/app_recordings_train.jsonl
TECHNIQUE_APP_VAL_MANIFEST ?= ./gt_singer_grader/manifests/app_recordings_val.jsonl
TECHNIQUE_APP_EVAL_MANIFEST ?= ./gt_singer_grader/manifests/app_recordings_eval.jsonl
TECHNIQUE_APP_VALIDATION_AUDIT ?= ./gt_singer_grader/manifests/app_recordings_eval_audit.json
TECHNIQUE_APP_ADAPTED_TRAIN_MANIFEST ?= ./gt_singer_grader/manifests/app_adapted_train.jsonl
TECHNIQUE_BASELINE_RUN_DIR ?= ./gt_singer_grader/runs/gtsinger_song_aug_v1
TECHNIQUE_APP_BASELINE_EVAL_DIR ?= ./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_app
TECHNIQUE_APP_ADAPTED_RUN_DIR ?= ./gt_singer_grader/runs/gtsinger_app_adapted_v1
TECHNIQUE_APP_ADAPTED_PLAN ?= ./gt_singer_grader/runs/gtsinger_app_adapted_v1/training_plan.json
TECHNIQUE_APP_ADAPTED_EVAL_DIR ?= ./gt_singer_grader/runs/gtsinger_app_adapted_v1/eval_app
TECHNIQUE_APP_ADAPTED_COMPARISON ?= ./gt_singer_grader/runs/run_comparison_app_adapted.json
TECHNIQUE_OUTPUT_CHECKPOINT ?= ./gt_singer_grader/models/technique_demo_best.pth
TECHNIQUE_OUTPUT_METADATA ?= ./gt_singer_grader/models/technique_demo_metadata.json
TECHNIQUE_EPOCHS ?= 50
TECHNIQUE_BATCH_SIZE ?= 8
TECHNIQUE_PYTHON ?= .venv/bin/python
TECHNIQUE_API_HOST ?= 127.0.0.1
TECHNIQUE_API_PORT ?= 8765

technique-api:
	cd server/technique && $(TECHNIQUE_PYTHON) api.py --host $(TECHNIQUE_API_HOST) --port $(TECHNIQUE_API_PORT)

technique-check-light:
	python3 -m py_compile \
		server/technique/api.py \
		server/technique/gt_singer_grader/app_label_coverage.py \
		server/technique/gt_singer_grader/audit_app_validation.py \
		server/technique/gt_singer_grader/build_manifest.py \
		server/technique/gt_singer_grader/compare_runs.py \
		server/technique/gt_singer_grader/constants.py \
		server/technique/gt_singer_grader/data.py \
		server/technique/gt_singer_grader/dataset_strategy.py \
		server/technique/gt_singer_grader/download_dataset.py \
		server/technique/gt_singer_grader/evaluation_artifacts.py \
		server/technique/gt_singer_grader/experiment_status.py \
		server/technique/gt_singer_grader/evaluate.py \
		server/technique/gt_singer_grader/features.py \
		server/technique/gt_singer_grader/export_app_collection_packet.py \
		server/technique/gt_singer_grader/filter_manifest.py \
		server/technique/gt_singer_grader/feedback.py \
		server/technique/gt_singer_grader/infer.py \
		server/technique/gt_singer_grader/manifest.py \
		server/technique/gt_singer_grader/materialize_app_collection.py \
		server/technique/gt_singer_grader/merge_manifest.py \
		server/technique/gt_singer_grader/model.py \
		server/technique/gt_singer_grader/package_candidate.py \
		server/technique/gt_singer_grader/plan_app_collection.py \
		server/technique/gt_singer_grader/plan_training.py \
		server/technique/gt_singer_grader/preflight.py \
		server/technique/gt_singer_grader/prepare_app_recordings.py \
		server/technique/gt_singer_grader/run_metadata.py \
		server/technique/gt_singer_grader/sample_manifest.py \
		server/technique/gt_singer_grader/split_health.py \
		server/technique/gt_singer_grader/split_manifest.py \
		server/technique/gt_singer_grader/train.py \
		server/technique/gt_singer_grader/verify_evaluation.py \
		server/technique/gt_singer_grader/verify_package.py \
		server/technique/gt_singer_grader/verify_run.py
	node --check coach/web/analyzer.js
	node --check coach/web/coach.js
	$(MAKE) technique-test-light
	$(MAKE) technique-strategy-audit
	$(MAKE) technique-smoke-manifest

technique-test-light:
	PYTHONPATH=server/technique python3 -m unittest discover -s server/technique/gt_singer_grader/tests

technique-strategy-audit:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.dataset_strategy --strict

technique-smoke-manifest:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.app_label_coverage \
		--csv gt_singer_grader/tests/fixtures/app_recordings_review_smoke.csv \
		--target-family breathy \
		--target-family vibrato \
		--min-per-family 1 \
		--min-negative 2 \
		--min-groups 4 \
		--strict
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.plan_app_collection \
		--csv gt_singer_grader/tests/fixtures/app_recordings_review_smoke.csv \
		--target-family breathy \
		--target-family vibrato \
		--min-per-family 1 \
		--min-negative 2 \
		--min-groups 4 \
		--output-csv /tmp/nanopitch-technique/app_collection_plan.csv
	mkdir -p /tmp/nanopitch-technique/raw/smoke_singer_001
	mkdir -p /tmp/nanopitch-technique/raw/smoke_singer_002
	python3 -c "import wave; paths = ['/tmp/nanopitch-technique/raw/smoke_singer_001/vibrato_0001.wav', '/tmp/nanopitch-technique/raw/smoke_singer_002/control_0002.wav']; frames = b'\x00\x00' * (16000 * 6); [((audio := wave.open(path, 'wb')).setnchannels(1), audio.setsampwidth(2), audio.setframerate(16000), audio.writeframes(frames), audio.close()) for path in paths]"
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.plan_app_collection \
		--csv /tmp/nanopitch-technique/missing_review_labels.csv \
		--target-family vibrato \
		--min-per-family 1 \
		--min-negative 1 \
		--min-groups 2 \
		--singer-prefix smoke_singer \
		--output-csv /tmp/nanopitch-technique/app_collection_plan_prepare.csv
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.materialize_app_collection \
		--plan /tmp/nanopitch-technique/app_collection_plan_prepare.csv \
		--root /tmp/nanopitch-technique \
		--checklist /tmp/nanopitch-technique/collection_checklist.csv \
		--missing-csv /tmp/nanopitch-technique/collection_missing.csv \
		--validate-wav-files \
		--strict
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.export_app_collection_packet \
		--checklist /tmp/nanopitch-technique/collection_checklist.csv \
		--output-dir /tmp/nanopitch-technique/collection_packet \
		--summary-json /tmp/nanopitch-technique/collection_packet_summary.json \
		--strict
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.prepare_app_recordings \
		--audio-dir /tmp/nanopitch-technique/raw \
		--output /tmp/nanopitch-technique/prepared_review_labels.csv \
		--report-json /tmp/nanopitch-technique/prepare_report.json \
		--relative-to /tmp/nanopitch-technique \
		--collection-plan /tmp/nanopitch-technique/app_collection_plan_prepare.csv \
		--strict-collection-plan \
		--singer-id-from-parent \
		--force
	cd server/technique && python3 -m json.tool /tmp/nanopitch-technique/prepare_report.json
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.app_label_coverage \
		--csv /tmp/nanopitch-technique/prepared_review_labels.csv \
		--target-family vibrato \
		--min-per-family 1 \
		--min-negative 1 \
		--min-groups 2
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.build_manifest app-recordings \
		--csv gt_singer_grader/tests/fixtures/app_recordings_review_smoke.csv \
		--output /tmp/nanopitch-technique/app_recordings_manifest.jsonl
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.manifest /tmp/nanopitch-technique/app_recordings_manifest.jsonl
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.filter_manifest \
		--input /tmp/nanopitch-technique/app_recordings_manifest.jsonl \
		--trainable-output /tmp/nanopitch-technique/app_recordings_trainable.jsonl \
		--eval-only-output /tmp/nanopitch-technique/app_recordings_eval_only.jsonl \
		--summary-output /tmp/nanopitch-technique/app_recordings_filter_summary.json
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.manifest \
		/tmp/nanopitch-technique/app_recordings_trainable.jsonl \
		--require-trainable
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.sample_manifest \
		--input /tmp/nanopitch-technique/app_recordings_trainable.jsonl \
		--output /tmp/nanopitch-technique/app_recordings_trainable_sampled.jsonl \
		--summary-output /tmp/nanopitch-technique/app_recordings_sample_summary.json \
		--max-per-family 1
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.manifest \
		/tmp/nanopitch-technique/app_recordings_trainable_sampled.jsonl \
		--require-trainable
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.split_manifest \
		--input /tmp/nanopitch-technique/app_recordings_trainable.jsonl \
		--train-output /tmp/nanopitch-technique/app_recordings_train.jsonl \
		--val-output /tmp/nanopitch-technique/app_recordings_val.jsonl \
		--summary-output /tmp/nanopitch-technique/app_recordings_split_summary.json \
		--val-ratio 0.5 \
		--strict-non-empty \
		--strict-family-coverage
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.merge_manifest \
		--input /tmp/nanopitch-technique/app_recordings_val.jsonl \
		--input /tmp/nanopitch-technique/app_recordings_eval_only.jsonl \
		--output /tmp/nanopitch-technique/app_recordings_eval.jsonl \
		--summary-output /tmp/nanopitch-technique/app_recordings_eval_summary.json
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.manifest /tmp/nanopitch-technique/app_recordings_eval.jsonl
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.audit_app_validation \
		--manifest /tmp/nanopitch-technique/app_recordings_eval.jsonl \
		--output-json /tmp/nanopitch-technique/app_recordings_eval_audit.json \
		--target-family breathy \
		--target-family vibrato \
		--min-per-family 1 \
		--min-negative 2 \
		--min-groups 4 \
		--strict

technique-preflight:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.preflight

technique-status:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.experiment_status \
		--output-json ./gt_singer_grader/runs/experiment_status.json

technique-app-collection-plan:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.plan_app_collection \
		--csv $(TECHNIQUE_APP_LABELS) \
		--output-json $(TECHNIQUE_APP_COLLECTION_PLAN_JSON) \
		--output-csv $(TECHNIQUE_APP_COLLECTION_PLAN) \
		--clips-per-singer 7

technique-app-materialize-collection:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.materialize_app_collection \
		--plan $(TECHNIQUE_APP_COLLECTION_PLAN) \
		--root $(TECHNIQUE_APP_COLLECTION_ROOT) \
		--checklist $(TECHNIQUE_APP_COLLECTION_CHECKLIST) \
		--missing-csv $(TECHNIQUE_APP_COLLECTION_MISSING) \
		--report-json $(TECHNIQUE_APP_COLLECTION_MATERIALIZE_REPORT) \
		--strict

technique-app-export-collection-packet:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.export_app_collection_packet \
		--checklist $(TECHNIQUE_APP_COLLECTION_CHECKLIST) \
		--output-dir $(TECHNIQUE_APP_COLLECTION_PACKET_DIR) \
		--summary-json $(TECHNIQUE_APP_COLLECTION_PACKET_SUMMARY) \
		--strict

technique-app-check-collection:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.materialize_app_collection \
		--plan $(TECHNIQUE_APP_COLLECTION_PLAN) \
		--root $(TECHNIQUE_APP_COLLECTION_ROOT) \
		--checklist $(TECHNIQUE_APP_COLLECTION_CHECKLIST) \
		--missing-csv $(TECHNIQUE_APP_COLLECTION_MISSING) \
		--report-json $(TECHNIQUE_APP_COLLECTION_MATERIALIZE_REPORT) \
		--strict \
		--require-audio-files \
		--validate-wav-files \
		--min-wav-seconds 5 \
		--max-wav-seconds 10

technique-app-prepare-review:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.prepare_app_recordings \
		--audio-dir $(TECHNIQUE_APP_AUDIO_DIR) \
		--output $(TECHNIQUE_APP_LABELS) \
		--report-json $(TECHNIQUE_APP_PREPARE_REPORT) \
		--relative-to . \
		--collection-plan $(TECHNIQUE_APP_COLLECTION_PLAN) \
		--strict-collection-plan \
		--singer-id-from-parent

technique-app-review-progress:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.app_label_coverage \
		--csv $(TECHNIQUE_APP_LABELS) \
		--audio-root $(TECHNIQUE_APP_AUDIO_ROOT) \
		--require-audio-files \
		--output-json $(TECHNIQUE_APP_LABEL_COVERAGE_REPORT)

technique-app-manifests:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.app_label_coverage \
		--csv $(TECHNIQUE_APP_LABELS) \
		--audio-root $(TECHNIQUE_APP_AUDIO_ROOT) \
		--require-audio-files \
		--output-json $(TECHNIQUE_APP_LABEL_COVERAGE_REPORT) \
		--strict
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.build_manifest app-recordings \
		--csv $(TECHNIQUE_APP_LABELS) \
		--output $(TECHNIQUE_APP_MANIFEST)
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.manifest $(TECHNIQUE_APP_MANIFEST)
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.filter_manifest \
		--input $(TECHNIQUE_APP_MANIFEST) \
		--trainable-output $(TECHNIQUE_APP_TRAINABLE_MANIFEST) \
		--eval-only-output $(TECHNIQUE_APP_EVAL_ONLY_MANIFEST) \
		--summary-output ./gt_singer_grader/manifests/app_recordings_filter_summary.json
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.manifest \
		$(TECHNIQUE_APP_TRAINABLE_MANIFEST) \
		--require-trainable
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.split_manifest \
		--input $(TECHNIQUE_APP_TRAINABLE_MANIFEST) \
		--train-output $(TECHNIQUE_APP_TRAIN_MANIFEST) \
		--val-output $(TECHNIQUE_APP_VAL_MANIFEST) \
		--summary-output ./gt_singer_grader/manifests/app_recordings_split_summary.json \
		--val-ratio 0.2 \
		--strict-non-empty \
		--strict-family-coverage
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.merge_manifest \
		--input $(TECHNIQUE_APP_VAL_MANIFEST) \
		--input $(TECHNIQUE_APP_EVAL_ONLY_MANIFEST) \
		--output $(TECHNIQUE_APP_EVAL_MANIFEST) \
		--summary-output ./gt_singer_grader/manifests/app_recordings_eval_summary.json
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.audit_app_validation \
		--manifest $(TECHNIQUE_APP_EVAL_MANIFEST) \
		--output-json $(TECHNIQUE_APP_VALIDATION_AUDIT) \
		--strict

technique-app-adapted-plan:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.merge_manifest \
		--input $(TECHNIQUE_BASELINE_RUN_DIR)/train_manifest.jsonl \
		--input $(TECHNIQUE_APP_TRAIN_MANIFEST) \
		--output $(TECHNIQUE_APP_ADAPTED_TRAIN_MANIFEST) \
		--summary-output ./gt_singer_grader/manifests/app_adapted_train_summary.json
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.plan_training \
		--train-manifest $(TECHNIQUE_APP_ADAPTED_TRAIN_MANIFEST) \
		--val-manifest $(TECHNIQUE_APP_VAL_MANIFEST) \
		--require-train-dataset gtsinger \
		--require-train-dataset app_recordings \
		--require-val-dataset app_recordings \
		--output-json $(TECHNIQUE_APP_ADAPTED_PLAN) \
		--strict

technique-app-adapted-train:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.train \
		--train-manifest $(TECHNIQUE_APP_ADAPTED_TRAIN_MANIFEST) \
		--val-manifest $(TECHNIQUE_APP_VAL_MANIFEST) \
		--output-dir $(TECHNIQUE_APP_ADAPTED_RUN_DIR) \
		--training-plan $(TECHNIQUE_APP_ADAPTED_PLAN) \
		--user-audio-augmentation \
		--epochs $(TECHNIQUE_EPOCHS) \
		--batch-size $(TECHNIQUE_BATCH_SIZE) \
		--quiet

technique-app-baseline-eval:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.evaluate \
		--checkpoint $(TECHNIQUE_BASELINE_RUN_DIR)/checkpoints/best.pth \
		--manifest $(TECHNIQUE_APP_EVAL_MANIFEST) \
		--run-config $(TECHNIQUE_BASELINE_RUN_DIR)/run_config.json \
		--output-dir $(TECHNIQUE_APP_BASELINE_EVAL_DIR) \
		--max-control-fpr 0.25 \
		--max-non-technique-fpr 0.25
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.verify_evaluation \
		--eval-dir $(TECHNIQUE_APP_BASELINE_EVAL_DIR) \
		--strict

technique-app-adapted-eval:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.evaluate \
		--checkpoint $(TECHNIQUE_APP_ADAPTED_RUN_DIR)/checkpoints/best.pth \
		--manifest $(TECHNIQUE_APP_EVAL_MANIFEST) \
		--run-config $(TECHNIQUE_APP_ADAPTED_RUN_DIR)/run_config.json \
		--output-dir $(TECHNIQUE_APP_ADAPTED_EVAL_DIR) \
		--max-control-fpr 0.25 \
		--max-non-technique-fpr 0.25
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.verify_evaluation \
		--eval-dir $(TECHNIQUE_APP_ADAPTED_EVAL_DIR) \
		--strict

technique-app-adapted-compare:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.compare_runs \
		--baseline $(TECHNIQUE_APP_BASELINE_EVAL_DIR) \
		--candidate $(TECHNIQUE_APP_ADAPTED_EVAL_DIR) \
		--output-json $(TECHNIQUE_APP_ADAPTED_COMPARISON)

technique-app-adapted-package:
	cd server/technique && $(TECHNIQUE_PYTHON) -m gt_singer_grader.package_candidate \
		--checkpoint $(TECHNIQUE_APP_ADAPTED_RUN_DIR)/checkpoints/best.pth \
		--comparison $(TECHNIQUE_APP_ADAPTED_COMPARISON) \
		--candidate-eval-dir $(TECHNIQUE_APP_ADAPTED_EVAL_DIR) \
		--app-validation-audit $(TECHNIQUE_APP_VALIDATION_AUDIT) \
		--output-checkpoint $(TECHNIQUE_OUTPUT_CHECKPOINT) \
		--metadata $(TECHNIQUE_OUTPUT_METADATA)
