PYTHON := venv/bin/python

.PHONY: setup test info data train generate clean

setup:
	uv venv venv 2>/dev/null || true
	uv pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

info:
	$(PYTHON) scripts/info.py --config configs/kuon-13m.json

data:
	$(PYTHON) scripts/prepare_data.py --source tinystories --out data/tinystories

train:
	$(PYTHON) scripts/train.py --config configs/kuon-13m.json

generate:
	$(PYTHON) scripts/generate.py --checkpoint checkpoints/kuon-13m/kuon-13m-last.pt --prompt "Once upon a time"

clean:
	rm -rf checkpoints logs
