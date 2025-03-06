def prepare_dataset(data, text_column, label_column):
    """
    Prépare le dataset en excluant les tweets "Irrelevant" et en appliquant le prétraitement.
    
    Args:
        data (pd.DataFrame): Le DataFrame contenant les données
        text_column (str): Le nom de la colonne contenant le texte des tweets
        label_column (str): Le nom de la colonne contenant les labels de sentiment
        
    Returns:
        pd.DataFrame: Le DataFrame prétraité
    """
    # Vérifier que les colonnes existent
    if text_column not in data.columns:
        raise ValueError(f"La colonne de texte '{text_column}' n'existe pas dans le dataset")
    if label_column not in data.columns:
        raise ValueError(f"La colonne d'étiquette '{label_column}' n'existe pas dans le dataset")

    # Filtrer les tweets "Irrelevant"
    data = data[data[label_column] != 'Irrelevant'].copy()
    print(f"Nombre de tweets après filtrage des 'Irrelevant': {len(data)}")

    # Appliquer le prétraitement au texte
    data['clean_text'] = data[text_column].apply(preprocess_text)

    # Identifier les valeurs uniques dans la colonne des sentiments
    unique_sentiments = data[label_column].unique()
    print(f"Valeurs uniques de sentiment trouvées : {unique_sentiments}")

    # Créer un mapping des sentiments basé sur les valeurs trouvées
    sentiment_map = {}

    # Essayer de détecter automatiquement le format du sentiment
    if set(unique_sentiments).issubset({0, 1}) or set(unique_sentiments).issubset({'0', '1'}):
        # Dataset binaire (positif/négatif)
        sentiment_map = {0: 0, 1: 1, '0': 0, '1': 1}
        print("Format détecté : Binaire (négatif/positif)")
    elif set(unique_sentiments).issubset({-1, 0, 1}) or set(unique_sentiments).issubset({'-1', '0', '1'}):
        # Dataset ternaire avec -1, 0, 1
        sentiment_map = {-1: 0, 0: 2, 1:1, '-1':0, '0':2, '1':1}
        print("Format détecté : Ternaire (-1=négatif, 0=neutre, 1=positif)")
    elif any(isinstance(x, str) and x.lower() in ['positive', 'negative', 'neutral'] for x in unique_sentiments):
        # Dataset avec texte
        sentiment_map = {'Negative': 0, 'Neutral': 2, 'Positive': 1}
        print("Format détecté : Textuel (negative/neutral/positive)")
    else:
        # Format non reconnu, créer un mapping générique
        sentiment_map = {val: idx for idx, val in enumerate(unique_sentiments)}
        print(f"Format non reconnu. Mapping créé : {sentiment_map}")

    # Appliquer le mapping
    data['sentiment_label'] = data[label_column].map(sentiment_map)

    # Vérifier qu'il n'y a pas de NaN dans les labels après le mapping
    if data['sentiment_label'].isna().any():
        print("ATTENTION : Certaines valeurs de sentiment n'ont pas pu être converties !")
        print("Valeurs problématiques : ", data[data['sentiment_label'].isna()][label_column].unique())
        # Remplir les NaN avec une valeur par défaut (par exemple, 0 pour négatif)
        data['sentiment_label'] = data['sentiment_label'].fillna(0)

    return data 