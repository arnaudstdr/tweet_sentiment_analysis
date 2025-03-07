# Analyse des sentiments sur Twitter avec Machine Learning & Transformers
Ce projet permet d’effectuer une **analyse des sentiments** sur des tweets en utilisant deux approches complémentaires :
- **Traditionnelle** : TF_IDF avec une régression logistique.
- **Deep Learning** : Modèles transformers (type BERT).
 
 L’objectif est de prédire si un tweet est positif, négatif ou neutre, avec des outils adaptés à chaque besoin, du classique au dernier cri en NLP !

## Objectif du projet
- Apprendre à traiter, visualiser et modéliser des données textuelles issues de réseaux sociaux.
- Comparer les performances des approches traditionnelles et avancées en NLP.
- Acquérir des compétences pratiques en prétraitement de texte, TF-IDF et Transformers.

## Structure du projet
```bash
Twitter_sentiments_analysis/
├── data/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── twitter_sentiment_analysis.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation et pré-requis

Clone le projet :
```bash
git clone https://github.com/ton_nom_utilisateur/Twitter_sentiments_analysis.git
cd Twitter_sentiments_analysis
```

Crée un environnement virtuel :
```bash
python -m venv env
source env/bin/activate
```

Installe les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation du projet

### Exécuter l'analyse
Depuis le dossier racine du projet, lancer le notebook Jupyter :
```bash
jupyter notebook twitter_sentiment_analysis.ipynb
```

### Options proposées
Lors de l'exécution, choisissez entre les deux approches :
1. **Traditionnelle** : TF_IDF avec une régression logistique.
2. **Deep Learning** : Modèles transformers (type BERT).
3. **Les deux** : Comparaison des deux approches.

## Détails techniques

### Prétraitement des données
- Nettoyage de texte (suppression des stopwords, ponctuations, mentions et liens)
- Tokenisation et lemmatisation avec NLTK
- Vectorisation TF-IDF pour l’approche traditionnelle

### Modèles utilisés
- **Régression logistique** (scikit-learn) pour l’approche traditionnelle
- **BERT** (transformers) pour l’approche deep learning

### Visualisations
- Répartition des sentiments.
- Nuage de mots pour les termes fréquents.
- Matrice de confusion et rapport de classification.

## Résultats attendus
On obtiendra des métriques telles que :
- Accuracy
- Precision, Recall et F1-score
- Courbe ROC-AUC

Ainsi que des visualisations permettant de mieux comprendre les résultats obtenus par les deux approches.

## Amélioration futures

- Testert d'autres modèles Transformers (RoBERTa, DistilBERT, etc.)
- Déployer l'application sur une plateforme cloud (Streamlit, FastAPI)

## Auteurs
- [Arnaud Stadler](https://github.com/arnaudstdr) - Développeur ML/DL

## Licence
Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
