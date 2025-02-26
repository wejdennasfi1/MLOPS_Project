PYTHON = python3
ENV_NAME = venv
REQUIREMENTS = requirements.txt

# Exécuter toutes les étapes
all: lint prepare_data train evaluate  

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

