IMAGE ?= blast-radius-entry
PLATFORM ?= linux/amd64
SAMPLE ?= $(CURDIR)/sample
OUTPUT := $(CURDIR)/output.jsonl

RUN_FLAGS := --rm --platform $(PLATFORM) --network none --cpus 2 \
	--memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
	--tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
	--security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
	--volume $(SAMPLE)/commits:/data/commits:ro \
	--volume $(SAMPLE)/tests.json:/data/tests.json:ro

.PHONY: build sample check
build:
	docker build --platform $(PLATFORM) --tag $(IMAGE) .
sample: build
	docker run $(RUN_FLAGS) $(EXTRA) $(IMAGE) > $(OUTPUT)
	python3 score.py $(SAMPLE) $(SAMPLE)/answers.json $(OUTPUT)
