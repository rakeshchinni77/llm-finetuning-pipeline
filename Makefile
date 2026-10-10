# LLM Fine-Tuning Pipeline Makefile
.PHONY: help install prepare-data train evaluate test clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install        - Install python dependencies"
	@echo "  make prepare-data   - Download, clean, format and split dataset"
	@echo "  make train          - Run QLoRA fine-tuning with SFTTrainer"
	@echo "  make evaluate       - Run comparative evaluation on held-out test set"
	@echo "  make test           - Run pytest unit tests"
	@echo "  make docker-build   - Build reproducible Docker image"
	@echo "  make docker-run     - Launch container with GPU support"
	@echo "  make clean          - Remove temporary files and caches"

install:
	pip install -r requirements.txt

prepare-data:
	python src/data_pipeline.py --config configs/training_config.yaml

train:
	python src/train.py --config configs/training_config.yaml

evaluate:
	python src/evaluate.py --config configs/evaluation_config.yaml

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
