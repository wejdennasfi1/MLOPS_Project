PYTHON = python3
ENV_NAME = venv
REQUIREMENTS = requirements.txt
PYTHON_EXEC = $(ENV_NAME)/bin/python
DOCKER_IMAGE = wejdennasfi/wejden_nasfi_mlops_1
TAG = v1
# Variables
DOCKER_COMPOSE = docker-compose
MLFLOW_PORT = 5000
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
# Démarrer MLflow
start-mlflow:
	@echo "Démarrage du serveur MLflow..."
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port $(MLFLOW_PORT) &





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
		-d '{"features": [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]}'

run-web:
	@echo "Démarrage de l'interface Flask..."
	@bash -c "source $(ENV_NAME)/bin/activate && python web_app.py"


predict: $(ENV_NAME)/bin/activate
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) main.py --predict"

#  Build the Docker image
build-docker:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(TAG) .

#  Push the Docker image to Docker Hub
push-docker:
	@echo "Pushing Docker image..."
	docker push $(DOCKER_IMAGE):$(TAG)

#  Run the Docker container
run-docker:
	@echo "Running Docker container..."
	docker run -d -p 8000:8000 --name mlops_container $(DOCKER_IMAGE):$(TAG)

#  Remove the container (if needed)
clean-docker:
	@echo "Removing Docker container..."
	docker rm -f mlops_container || true
	docker rmi -f $(DOCKER_IMAGE):$(TAG) || true
