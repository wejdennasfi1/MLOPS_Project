# Utiliser une image Python officielle comme base
FROM python:3.9

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel FastAPI tourne (8000 par défaut)
EXPOSE 8000

# Commande pour exécuter l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
