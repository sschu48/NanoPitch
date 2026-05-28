.PHONY: technique-check-light technique-test-light technique-smoke-manifest

technique-check-light:
	python3 -m py_compile \
		server/technique/api.py \
		server/technique/gt_singer_grader/build_manifest.py \
		server/technique/gt_singer_grader/constants.py \
		server/technique/gt_singer_grader/data.py \
		server/technique/gt_singer_grader/features.py \
		server/technique/gt_singer_grader/feedback.py \
		server/technique/gt_singer_grader/infer.py \
		server/technique/gt_singer_grader/manifest.py \
		server/technique/gt_singer_grader/model.py \
		server/technique/gt_singer_grader/train.py
	node --check coach/web/analyzer.js
	node --check coach/web/coach.js
	$(MAKE) technique-test-light
	$(MAKE) technique-smoke-manifest

technique-test-light:
	PYTHONPATH=server/technique python3 -m unittest discover -s server/technique/gt_singer_grader/tests

technique-smoke-manifest:
	cd server/technique && python3 -m gt_singer_grader.build_manifest app-recordings \
		--csv gt_singer_grader/app_recordings_labels_template.csv \
		--output /tmp/nanopitch-technique/app_recordings_manifest.jsonl
	cd server/technique && python3 -m gt_singer_grader.manifest /tmp/nanopitch-technique/app_recordings_manifest.jsonl
