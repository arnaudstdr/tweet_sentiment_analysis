# Utilise l'image officielle Python basée sur Debian Bullseye
FROM python:3.10-bullseye

# Copie l'intégralité de ton projet directement dans le dossier /app
COPY . /app

# Définit /app comme répertoire par défaut
WORKDIR /app

# Installe tes dépendances Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Commande par défaut (change avec ton script principal)
CMD ["python3", "twitter_sentiment_analysis.py"]
