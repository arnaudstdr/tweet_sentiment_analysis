# 🐦 Analyse des Sentiments sur Twitter avec Machine Learning & Transformers

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-310/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Dockerfile](https://img.shields.io/badge/Docker-ready-blue?logo=docker)

Ce projet propose une **analyse des sentiments** sur des tweets en combinant deux approches complémentaires :
- 🔤 **Traditionnelle** : TF-IDF + régression logistique (scikit-learn)
- 🤖 **Deep Learning** : Modèles Transformers (type BERT)

L’objectif est de prédire si un tweet est positif, négatif ou neutre, en comparant la performance des méthodes classiques et avancées du NLP.

## ✨ Fonctionnalités
- Prédiction du sentiment d’un tweet (positif, négatif, neutre)
- Deux pipelines : traditionnel (TF-IDF + logreg) et deep learning (BERT)
- Visualisations : nuages de mots, matrices de confusion, rapports de classification
- Notebook interactif pour l’exploration et la comparaison
- Dockerisation complète
- Prêt pour Dev Container VS Code

## 📦 Installation locale

### Option 1 : Environnement Python local

#### 1. Cloner le dépôt
```bash
git clone https://github.com/ton_nom_utilisateur/Twitter_sentiments_analysis.git
cd Twitter_sentiments_analysis
```

#### 2. (Optionnel) Créer un environnement virtuel
```bash
python -m venv env
source env/bin/activate
```

#### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### Option 2 : 🐳 Dockerisation

#### 1. Build de l’image Docker
```bash
docker build -t twitter-sentiments .
```

#### 2. Lancement du conteneur
```bash
docker run -it --rm -p 8888:8888 twitter-sentiments
```
➡️ Le notebook Jupyter sera accessible sur : http://localhost:8888

## 🐳 Utilisation avec Dev Container

Ce projet est prêt à être utilisé avec [Dev Containers](https://containers.dev/) de VS Code.

- Installez l’extension **Dev Containers** sur VS Code.
- Ouvrez le dossier du projet dans VS Code.
- Cliquez sur `Reopen in Container` ou utilisez la palette de commandes (`F1`) :
  - `Dev Containers: Reopen in Container`

L’environnement de développement (Python, dépendances, outils) sera automatiquement configuré dans le conteneur.

## 🚀 Utilisation du projet

### 1. Lancer le notebook Jupyter
```bash
jupyter notebook twitter_sentiment_analysis.ipynb
```

### 2. Choisir l’approche
- **Traditionnelle** : TF-IDF + régression logistique
- **Deep Learning** : BERT
- **Comparaison** : Les deux pipelines

## 📊 Visualisations & Résultats
- Répartition des sentiments
- Nuages de mots
- Matrices de confusion
- Rapports de classification
- Scores : Accuracy, Precision, Recall, F1-score, ROC-AUC

## 🛣️ Roadmap
- ✅ Pipeline traditionnel (TF-IDF + logreg)
- ✅ Pipeline BERT (transformers)
- ✅ Visualisations
- ✅ Dockerisation
- ✅ Dev Container
- ⬜️ API REST (FastAPI/Flask)
- ⬜️ Ajout d’autres modèles Transformers (RoBERTa, DistilBERT)

## 🧠 Auteur
👤 Arnaud Stadler - Passionné de NLP, Machine Learning et Data Science

## 📄 Licence
Ce projet est open-source sous licence [MIT](LICENSE). Vous pouvez l’utiliser, le modifier et le redistribuer librement dans le respect de cette licence.