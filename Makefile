PYTHON = python3
ENV_NAME = venv
REQUIREMENTS = requirements.txt

# Exécuter toutes les étapes
all: lint test prepare_data train evaluate predict

# 2. Qualité du code, formattage automatique du code, sécurité du code, etc.
lint:
	@echo "Vérification de la qualité du code..."
	@bash -c "source $(ENV_NAME)/bin/activate && black ."
	@bash -c "source $(ENV_NAME)/bin/activate && flake8 --exclude=$(ENV_NAME),.ipynb_checkpoints ."
	@bash -c "source $(ENV_NAME)/bin/activate && pylint --fail-under=6 --ignore=$(ENV_NAME),.ipynb_checkpoints main.py model_pipeline.py"
# Prepare data
prepare_data:
	@echo "Preparing data..."
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) main.py --prepare"

# Train model
train:
	@echo "Entraînement du modèle..."
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) main.py --train"

evaluate:
	@echo "Evaluating model..."
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) main.py --evaluate"



#watch
watch:
	python watch_make.py
test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover tests
	@echo "Testing complete."
# Lancer l'API FastAPI
run-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

test-api:
		curl -X 'POST' \
		'http://127.0.0.1:8000/predict' \
		-H 'accept: application/json' \
		-H 'Content-Type: application/json' \
		-d '{"features": [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10, 10, 110, 12, 13, 41, 15, 16, 17, 18, 9]}'

run-web:
	@echo "Démarrage de l'interface Flask..."
	@bash -c "source $(ENV_NAME)/bin/activate && python web_app.py"


predict: $(ENV_NAME)/bin/activate
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) main.py --predict"

