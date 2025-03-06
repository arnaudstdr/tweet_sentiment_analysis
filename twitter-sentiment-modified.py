"""
ANALYSE DES SENTIMENTS SUR TWITTER
==================================
Ce projet permet d'analyser les sentiments exprimés dans des tweets en utilisant des techniques
de traitement du langage naturel (NLP) et de deep learning.
Adapté pour un dataset Kaggle avec fichiers d'entraînement et de validation séparés.
"""

# === IMPORTATION DES BIBLIOTHÈQUES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import gradio as gr

# === TÉLÉCHARGEMENT DES RESSOURCES NLTK ===
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === CHARGEMENT ET EXPLORATION DES DONNÉES ===
def load_data(train_path, val_path=None):
    """Charge les datasets Twitter d'entraînement et de validation"""
    train_data = pd.read_csv(train_path)
    print(f"Dimensions du dataset d'entraînement: {train_data.shape}")
    print("Aperçu du dataset d'entraînement:")
    print(train_data.head())
    
    # Vérifier les colonnes du dataset
    print("\nColonnes du dataset d'entraînement:")
    print(train_data.columns.tolist())
    
    # Si un fichier de validation est fourni, le charger également
    if val_path:
        val_data = pd.read_csv(val_path)
        print(f"\nDimensions du dataset de validation: {val_data.shape}")
        print("Aperçu du dataset de validation:")
        print(val_data.head())
        return train_data, val_data
    else:
        return train_data, None

# === PRÉTRAITEMENT DU TEXTE ===
def preprocess_text(text):
    """Prétraite le texte des tweets"""
    # Vérifier si le texte est une chaîne de caractères
    if not isinstance(text, str):
        return ""
    
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Supprimer les mentions utilisateurs (@user)
    text = re.sub(r'@\w+', '', text)
    
    # Supprimer les hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Supprimer les caractères non-alphanumériques
    text = re.sub(r'[^\w\s]', '', text)
    
    # Supprimer les chiffres
    text = re.sub(r'\d+', '', text)
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoindre les tokens
    return ' '.join(tokens)

def prepare_dataset(data, text_column, label_column):
    """Applique le prétraitement à l'ensemble du dataset"""
    # Vérifier que les colonnes existent
    if text_column not in data.columns:
        raise ValueError(f"La colonne de texte '{text_column}' n'existe pas dans le dataset")
    if label_column not in data.columns:
        raise ValueError(f"La colonne d'étiquette '{label_column}' n'existe pas dans le dataset")
    
    # Appliquer le prétraitement au texte
    data['clean_text'] = data[text_column].apply(preprocess_text)
    
    # Identifier les valeurs uniques dans la colonne des sentiments
    unique_sentiments = data[label_column].unique()
    print(f"Valeurs uniques de sentiment trouvées: {unique_sentiments}")
    
    # Créer un mapping des sentiments basé sur les valeurs trouvées
    # Cette partie peut nécessiter un ajustement selon le format de votre dataset
    sentiment_map = {}
    
    # Essayer de détecter automatiquement le format du sentiment
    if set(unique_sentiments).issubset({0, 1}) or set(unique_sentiments).issubset({'0', '1'}):
        # Dataset binaire (positif/négatif)
        sentiment_map = {0: 0, 1: 1, '0': 0, '1': 1}
        print("Format détecté: Binaire (négatif/positif)")
    elif set(unique_sentiments).issubset({-1, 0, 1}) or set(unique_sentiments).issubset({'-1', '0', '1'}):
        # Dataset ternaire avec -1, 0, 1
        sentiment_map = {-1: 0, 0: 2, 1: 1, '-1': 0, '0': 2, '1': 1}
        print("Format détecté: Ternaire (-1=négatif, 0=neutre, 1=positif)")
    elif any(isinstance(x, str) and x.lower() in ['positive', 'negative', 'neutral'] for x in unique_sentiments):
        # Dataset avec texte
        sentiment_map = {'negative': 0, 'neutral': 2, 'positive': 1}
        print("Format détecté: Textuel (negative/neutral/positive)")
    else:
        # Format non reconnu, créer un mapping générique
        sentiment_map = {val: idx for idx, val in enumerate(unique_sentiments)}
        print(f"Format non reconnu. Mapping créé: {sentiment_map}")
    
    # Appliquer le mapping
    data['sentiment_label'] = data[label_column].map(sentiment_map)
    
    # Vérifier qu'il n'y a pas de NaN dans les labels après le mapping
    if data['sentiment_label'].isna().any():
        print("ATTENTION: Certaines valeurs de sentiment n'ont pas pu être converties!")
        print("Valeurs problématiques:", data[data['sentiment_label'].isna()][label_column].unique())
        # Remplir les NaN avec une valeur par défaut (par exemple, 0 pour négatif)
        data['sentiment_label'] = data['sentiment_label'].fillna(0)
    
    return data

# === VISUALISATION DES DONNÉES ===
def visualize_data(data, sentiment_column, text_column):
    """Crée des visualisations pour explorer le dataset"""
    # Distribution des sentiments
    plt.figure(figsize=(10, 6))
    sentiment_counts = data[sentiment_column].value_counts()
    
    # Créer une palette de couleurs plus attrayante
    colors = ['#ff9999', '#66b3ff', '#99ff99'][:len(sentiment_counts)]
    
    # Afficher le graphique
    ax = sentiment_counts.plot(kind='bar', color=colors)
    for i, v in enumerate(sentiment_counts):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    plt.title('Distribution des Sentiments', fontsize=14)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Nombre de Tweets', fontsize=12)
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    print("Graphique de distribution des sentiments sauvegardé dans 'sentiment_distribution.png'")
    
    # Longueur des tweets par sentiment
    data['text_length'] = data[text_column].astype(str).apply(len)
    
    plt.figure(figsize=(12, 7))
    
    # Utilisation de boxplot avec swarmplot pour une meilleure visualisation
    ax = sns.boxplot(x=sentiment_column, y='text_length', data=data, palette='Set2')
    sns.swarmplot(x=sentiment_column, y='text_length', data=data, color='0.25', size=4, alpha=0.5)
    
    plt.title('Longueur des Tweets par Sentiment', fontsize=14)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Longueur du Tweet (caractères)', fontsize=12)
    plt.tight_layout()
    plt.savefig('tweet_length_by_sentiment.png')
    print("Graphique de longueur des tweets sauvegardé dans 'tweet_length_by_sentiment.png'")
    
    # Analyse des mots les plus fréquents par sentiment
    from collections import Counter
    import matplotlib.cm as cm
    
    # Créer un DataFrame pour les mots les plus fréquents par sentiment
    plt.figure(figsize=(15, 12))
    
    # Définir le nombre de sentiments dans le dataset
    num_sentiments = data['sentiment_label'].nunique()
    
    # Ajuster le nombre de sous-graphiques en fonction du nombre de sentiments
    fig, axes = plt.subplots(1, num_sentiments, figsize=(15, 6))
    if num_sentiments == 1:
        axes = [axes]  # Assurer que axes est toujours une liste
    
    sentiment_names = {0: 'Négatif', 1: 'Positif', 2: 'Neutre'}
    
    # Pour chaque sentiment, trouver les mots les plus fréquents
    for i, sentiment_value in enumerate(sorted(data['sentiment_label'].unique())):
        # Filtrer les tweets par sentiment
        sentiment_data = data[data['sentiment_label'] == sentiment_value]
        
        # Joindre tous les textes nettoyés
        all_words = ' '.join(sentiment_data['clean_text'].astype(str)).split()
        
        # Compter les mots
        word_counts = Counter(all_words)
        
        # Prendre les 15 mots les plus fréquents
        most_common = word_counts.most_common(15)
        
        # Créer des listes pour le graphique
        words = [word for word, count in most_common]
        counts = [count for word, count in most_common]
        
        # Tracer le graphique à barres horizontales
        sentiment_name = sentiment_names.get(sentiment_value, f"Sentiment {sentiment_value}")
        axes[i].barh(words, counts, color=cm.Set3(i / num_sentiments))
        axes[i].set_title(f'Mots fréquents - {sentiment_name}')
        axes[i].set_xlabel('Fréquence')
        
    plt.tight_layout()
    plt.savefig('frequent_words_by_sentiment.png')
    print("Graphique des mots fréquents sauvegardé dans 'frequent_words_by_sentiment.png'")
    
    return

# === APPROCHE TRADITIONNELLE AVEC TF-IDF ===
def train_tfidf_model(train_data, val_data=None):
    """Entraîne un modèle utilisant TF-IDF et régression logistique"""
    # Si pas de données de validation, utiliser une partie des données d'entraînement
    if val_data is None:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            train_data['clean_text'], train_data['sentiment_label'], 
            test_size=0.2, random_state=42
        )
    else:
        X_train = train_data['clean_text']
        y_train = train_data['sentiment_label']
        X_val = val_data['clean_text']
        y_val = val_data['sentiment_label']
    
    print(f"Taille des données d'entraînement: {len(X_train)}")
    print(f"Taille des données de validation: {len(X_val)}")
    
    # Vectorisation TF-IDF
    print("Vectorisation TF-IDF des données...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    
    # Entraînement du modèle (Régression Logistique)
    print("Entraînement du modèle de régression logistique...")
    lr_model = LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1)
    lr_model.fit(X_train_tfidf, y_train)
    
    # Évaluation du modèle
    y_pred = lr_model.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Précision du modèle TF-IDF + Régression Logistique: {accuracy:.4f}")
    
    # Rapport de classification détaillé
    print("\nRapport de classification:")
    print(classification_report(y_val, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Utiliser un heatmap plus informatif
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Négatif', 'Positif', 'Neutre'][:lr_model.classes_.size],
        yticklabels=['Négatif', 'Positif', 'Neutre'][:lr_model.classes_.size]
    )
    plt.title('Matrice de Confusion - Modèle TF-IDF', fontsize=14)
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Réalité', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_tfidf.png')
    print("Matrice de confusion sauvegardée dans 'confusion_matrix_tfidf.png'")
    
    # Analyse des caractéristiques les plus importantes
    if hasattr(lr_model, 'coef_'):
        for i, class_label in enumerate(lr_model.classes_):
            # Obtenir les coefficients pour cette classe
            coefficients = lr_model.coef_[i]
            
            # Obtenir les noms des caractéristiques
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Créer un DataFrame pour faciliter le tri
            coefficients_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients
            })
            
            # Trier par coefficient absolu décroissant
            sorted_coefficients = coefficients_df.reindex(
                coefficients_df['coefficient'].abs().sort_values(ascending=False).index
            )
            
            # Afficher les 10 caractéristiques les plus importantes pour cette classe
            sentiment_name = ['Négatif', 'Positif', 'Neutre'][i] if i < 3 else f"Classe {i}"
            print(f"\n10 mots les plus importants pour la classe {sentiment_name}:")
            print(sorted_coefficients.head(10))
    
    return lr_model, tfidf_vectorizer

# === APPROCHE DEEP LEARNING AVEC TRANSFORMERS ===
def train_transformer_model(train_data, val_data=None, model_name="distilbert-base-uncased"):
    """Entraîne un modèle BERT pour l'analyse des sentiments"""
    print(f"Chargement du modèle pré-entraîné: {model_name}")
    
    # Déterminer le nombre de classes
    num_labels = train_data['sentiment_label'].nunique()
    print(f"Nombre de classes détecté: {num_labels}")
    
    # Chargement du modèle pré-entraîné
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    
    # Si pas de données de validation, utiliser une partie des données d'entraînement
    if val_data is None:
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_data['clean_text'].tolist(), train_data['sentiment_label'].tolist(), 
            test_size=0.2, random_state=42
        )
    else:
        train_texts = train_data['clean_text'].tolist()
        train_labels = train_data['sentiment_label'].tolist()
        val_texts = val_data['clean_text'].tolist()
        val_labels = val_data['sentiment_label'].tolist()
    
    print(f"Taille des données d'entraînement: {len(train_texts)}")
    print(f"Taille des données de validation: {len(val_texts)}")
    
    # Fonction de tokenisation pour BERT
    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True, max_length=128)
    
    print("Tokenisation des données...")
    # Création des encodages
    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)
    
    # Création d'une classe de dataset compatible avec Hugging Face
    class TwitterDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
    
    train_dataset = TwitterDataset(train_encodings, train_labels)
    val_dataset = TwitterDataset(val_encodings, val_labels)
    
    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Désactiver les rapports pour simplifier
    )
    
    # Création du Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Entraînement du modèle
    print("Début de l'entraînement du modèle Transformer...")
    trainer.train()
    
    # Évaluation
    print("Évaluation du modèle...")
    eval_results = trainer.evaluate()
    print(f"Résultats de l'évaluation: {eval_results}")
    
    # Sauvegarde du modèle et du tokenizer
    print("Sauvegarde du modèle et du tokenizer...")
    model.save_pretrained("./twitter_sentiment_model")
    tokenizer.save_pretrained("./twitter_sentiment_tokenizer")
    
    return model, tokenizer

# === DÉPLOIEMENT D'UN CHATBOT D'ANALYSE DES SENTIMENTS ===
def create_sentiment_chatbot(model, tokenizer, num_labels=3):
    """Crée une interface Gradio pour le chatbot d'analyse des sentiments"""
    def predict_sentiment(tweet):
        # Prétraitement
        clean_tweet = preprocess_text(tweet)
        
        # Tokenisation pour le modèle
        inputs = tokenizer(clean_tweet, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
        
        # Conversion de la prédiction numérique en texte
        # Adapter selon le nombre de classes dans votre modèle
        if num_labels == 2:
            sentiment_map = {0: "Négatif", 1: "Positif"}
        else:  # num_labels == 3
            sentiment_map = {0: "Négatif", 1: "Positif", 2: "Neutre"}
        
        predicted_sentiment = sentiment_map.get(predictions.item(), f"Classe {predictions.item()}")
        
        # Calcul des probabilités
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Création du dictionnaire de probabilités en fonction du nombre de classes
        prob_dict = {}
        for i in range(num_labels):
            sentiment_name = sentiment_map.get(i, f"Classe {i}")
            prob_dict[sentiment_name] = f"{probs[0][i].item():.2%}"
        
        return predicted_sentiment, prob_dict
    
    # Création de l'interface Gradio
    with gr.Blocks() as demo:
        gr.Markdown("# Analyseur de Sentiments pour Tweets")
        gr.Markdown("Entrez un tweet et l'IA analysera son sentiment.")
        
        with gr.Row():
            with gr.Column():
                tweet_input = gr.Textbox(
                    label="Entrez un tweet", 
                    placeholder="J'adore ce nouvel outil d'analyse de sentiments!",
                    lines=3
                )
                analyze_btn = gr.Button("Analyser le sentiment", variant="primary")
            
            with gr.Column():
                sentiment_output = gr.Textbox(label="Sentiment")
                confidence_output = gr.JSON(label="Confiance (%)")
        
        # Exemples pour faciliter les tests
        gr.Examples(
            [
                ["Je suis vraiment déçu par la qualité de ce service. Terrible expérience!"],
                ["Ce produit est absolument incroyable! Je le recommande vivement."],
                ["J'ai reçu ma commande aujourd'hui. Elle est arrivée à l'heure."],
                ["Je ne sais pas quoi penser de ce nouveau film. Il avait des bons et des mauvais moments."]
            ],
            inputs=[tweet_input],
            outputs=[sentiment_output, confidence_output],
            fn=predict_sentiment,
            cache_examples=True
        )
        
        # Configuration du bouton d'analyse
        analyze_btn.click(
            predict_sentiment, 
            inputs=[tweet_input], 
            outputs=[sentiment_output, confidence_output]
        )
    
    # Lancement de l'interface
    print("Lancement de l'interface Gradio...")
    demo.launch(share=True)  # L'option share=True crée un lien public temporaire

# === FONCTION PRINCIPALE ===
def main():
    print("=== ANALYSE DES SENTIMENTS SUR TWITTER ===")
    print("Début du traitement...\n")
    
    # Chemins vers les datasets
    train_path = "path_to_training_data.csv"  # À remplacer par votre chemin
    val_path = "path_to_validation_data.csv"  # À remplacer par votre chemin
    
    # Noms des colonnes (à ajuster selon votre dataset)
    text_column = "text"  # Colonne contenant le texte du tweet
    label_column = "sentiment"  # Colonne contenant l'étiquette de sentiment
    
    # Chargement des données
    print("Chargement des données...")
    train_data, val_data = load_data(train_path, val_path)
    
    # Prétraitement
    print("\nPrétraitement des données...")
    processed_train = prepare_dataset(train_data, text_column, label_column)
    
    if val_data is not None:
        processed_val = prepare_dataset(val_data, text_column, label_column)
    else:
        processed_val = None
    
    # Visualisation
    print("\nCréation des visualisations...")
    visualize_data(processed_train, label_column, text_column)
    
    # Demander à l'utilisateur quelle approche utiliser
    print("\nChoisissez votre approche d'analyse:")
    print("1. Approche traditionnelle (TF-IDF + Régression Logistique)")
    print("2. Approche Deep Learning (Transformers)")
    print("3. Les deux approches")
    
    choice = input("Votre choix (1, 2 ou 3): ")
    
    if choice in ["1", "3"]:
        print("\n=== APPROCHE TRADITIONNELLE ===")
        lr_model, tfidf_vectorizer = train_tfidf_model(processed_train, processed_val)
    
    if choice in ["2", "3"]:
        print("\n=== APPROCHE DEEP LEARNING ===")
        transformer_model, tokenizer = train_transformer_model(processed_train, processed_val)
    
    # Déploiement du chatbot (avec le modèle disponible)
    print("\n=== DÉPLOIEMENT DU CHATBOT ===")
    if choice == "2" or choice == "3":
        create_sentiment_chatbot(transformer_model, tokenizer)
    elif choice == "1":
        print("Le chatbot avec le modèle TF-IDF n'est pas disponible dans cette version.")
        print("Utilisez l'approche Deep Learning pour déployer le chatbot.")
    
    print("\n=== TRAITEMENT TERMINÉ ===")

if __name__ == "__main__":
    main()
