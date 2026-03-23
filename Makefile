.PHONY: install fetch train dashboard evaluate test lint clean

install:
pip install -r requirements.txt

fetch:
python data/dataset_builder.py --fetch-real

train:
python training/train_aquila.py

dashboard:
python dashboard/app.py

evaluate:
python evaluation/evaluate.py

test:
pytest tests/ -v --color=yes

lint:
black . --line-length 100 && flake8 . --max-line-length 100

clean:
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
