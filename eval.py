import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'API
HOST = os.getenv("API_HOST", "https://fb-1021317796643.europe-west1.run.app")
API_ENDPOINT_ANSWER = f"{HOST}/answer"

def load_test_data(csv_path="C:\Users\sarra\Desktop\medical-chatbot - Copie\data\medquad.csv", n_samples=10):
    """
    Charger et échantillonner les données de test à partir du dataset MedQuAD.
    
    Args:
        csv_path: Chemin vers le fichier CSV contenant les données
        n_samples: Nombre d'échantillons à sélectionner
        
    Returns:
        Liste de dictionnaires contenant les questions et réponses
    """
    try:
        # Vérifier si le fichier CSV existe
        if not os.path.exists(csv_path):
            print(f"ERREUR: Le fichier CSV n'existe pas à: {csv_path}")
            # Utiliser un dataset de test factice en cas d'erreur
            test_data = [
                {"question": "Quels sont les symptômes du diabète?", 
                 "answer": "Les symptômes courants du diabète incluent une soif excessive, une miction fréquente et une fatigue inhabituelle."},
                {"question": "Comment traite-t-on l'hypertension?", 
                 "answer": "L'hypertension est généralement traitée par des changements de mode de vie et des médicaments comme les diurétiques."}
            ]
            # Dupliquer pour atteindre n_samples
            while len(test_data) < n_samples:
                test_data.append(random.choice(test_data))
            return test_data[:n_samples]
        
        # Charger le CSV et sélectionner des exemples aléatoires
        df = pd.read_csv(csv_path)
        if len(df) < n_samples:
            print(f"ATTENTION: Le dataset contient moins de {n_samples} exemples, utilisation de tous les exemples disponibles.")
            n_samples = len(df)
        
        # Sélectionner des exemples aléatoires
        sample_indices = random.sample(range(len(df)), n_samples)
        test_data = []
        
        for idx in sample_indices:
            row = df.iloc[idx]
            test_data.append({
                "question": row["question"],
                "answer": row["answer"],
                "source": row.get("source", "Non spécifié")
            })
        
        return test_data
    
    except Exception as e:
        print(f"Erreur lors du chargement des données de test: {str(e)}")
        return []

def calculate_similarity(true_answer, predicted_answer):
    """
    Calculer la similarité cosinus entre la réponse réelle et la réponse prédite.
    
    Args:
        true_answer: Réponse de référence
        predicted_answer: Réponse générée par le chatbot
        
    Returns:
        Score de similarité entre 0 et 1
    """
    try:
        # Si l'une des réponses est vide, retourner 0
        if not true_answer or not predicted_answer:
            return 0.0
        
        # Utiliser TF-IDF et similarité cosinus
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([true_answer, predicted_answer])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    except Exception as e:
        print(f"Erreur lors du calcul de similarité: {str(e)}")
        return 0.0

def evaluate_response_time(start_time, end_time):
    """Évaluer le temps de réponse en secondes."""
    return end_time - start_time

def evaluate_chatbot(test_data, temperature=0.2, language="Français"):
    """
    Évaluer le chatbot sur des données de test.
    
    Args:
        test_data: Liste de dictionnaires contenant les questions et réponses
        temperature: Paramètre de température pour le modèle
        language: Langue pour les réponses
        
    Returns:
        Dictionnaire contenant les métriques d'évaluation
    """
    results = []
    
    print(f"\nÉvaluation du Chatbot Médical sur {len(test_data)} exemples")
    print("="*50)
    
    for i, test_case in enumerate(test_data):
        print(f"\nTest {i+1}/{len(test_data)}: {test_case['question'][:50]}...")
        
        try:
            # Mesurer le temps de début
            start_time = time.time()
            
            # Envoyer la requête à l'API
            response = requests.post(
                API_ENDPOINT_ANSWER,
                json={
                    "question": test_case["question"],
                    "temperature": temperature,
                    "language": language
                },
                timeout=30
            )
            
            # Mesurer le temps de fin
            end_time = time.time()
            
            if response.status_code != 200:
                print(f"Erreur API: {response.status_code} - {response.text}")
                predicted_answer = ""
                similarity_score = 0.0
            else:
                response_data = response.json()
                predicted_answer = response_data.get("answer", "")
                
                # Calculer la similarité
                similarity_score = calculate_similarity(test_case["answer"], predicted_answer)
            
            # Temps de réponse en secondes
            response_time = evaluate_response_time(start_time, end_time)
            
            # Évaluation qualitative (simplifiée pour l'automatisation)
            if similarity_score >= 0.8:
                quality_rating = "Excellente"
            elif similarity_score >= 0.6:
                quality_rating = "Bonne"
            elif similarity_score >= 0.4:
                quality_rating = "Acceptable"
            else:
                quality_rating = "Faible"
            
            # Stocker les résultats
            result = {
                "question": test_case["question"],
                "true_answer": test_case["answer"],
                "predicted_answer": predicted_answer,
                "similarity_score": similarity_score,
                "response_time": response_time,
                "quality_rating": quality_rating
            }
            
            results.append(result)
            
            print(f"Score de similarité: {similarity_score:.2f}")
            print(f"Temps de réponse: {response_time:.2f}s")
            print(f"Évaluation qualitative: {quality_rating}")
            
        except Exception as e:
            print(f"Erreur lors de l'évaluation: {str(e)}")
            results.append({
                "question": test_case["question"],
                "error": str(e),
                "similarity_score": 0.0,
                "response_time": 0.0,
                "quality_rating": "Erreur"
            })
    
    # Calculer les métriques globales
    df_results = pd.DataFrame(results)
    
    metrics = {
        "average_similarity": df_results["similarity_score"].mean(),
        "min_similarity": df_results["similarity_score"].min(),
        "max_similarity": df_results["similarity_score"].max(),
        "median_similarity": df_results["similarity_score"].median(),
        "std_similarity": df_results["similarity_score"].std(),
        
        "average_response_time": df_results["response_time"].mean(),
        "min_response_time": df_results["response_time"].min(),
        "max_response_time": df_results["response_time"].max(),
        
        "quality_ratings": df_results["quality_rating"].value_counts().to_dict()
    }
    
    return results, metrics

def generate_visual_report(metrics, results, output_dir="evaluation_results"):
    """
    Générer un rapport visuel avec des graphiques.
    
    Args:
        metrics: Dictionnaire des métriques calculées
        results: Liste des résultats détaillés
        output_dir: Répertoire de sortie pour les graphiques
    """
    try:
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convertir les résultats en DataFrame
        df = pd.DataFrame(results)
        
        # 1. Histogramme des scores de similarité
        plt.figure(figsize=(10, 6))
        plt.hist(df["similarity_score"], bins=10, alpha=0.7, color='blue')
        plt.title("Distribution des scores de similarité")
        plt.xlabel("Score de similarité")
        plt.ylabel("Nombre de réponses")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/similarity_distribution.png")
        
        # 2. Graphique du temps de réponse vs score de similarité
        plt.figure(figsize=(10, 6))
        plt.scatter(df["response_time"], df["similarity_score"], alpha=0.7)
        plt.title("Temps de réponse vs Score de similarité")
        plt.xlabel("Temps de réponse (s)")
        plt.ylabel("Score de similarité")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/response_time_vs_similarity.png")
        
        # 3. Camembert des évaluations qualitatives
        quality_counts = df["quality_rating"].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title("Distribution des évaluations qualitatives")
        plt.savefig(f"{output_dir}/quality_distribution.png")
        
        print(f"\nGraphiques enregistrés dans le répertoire '{output_dir}'")
        
    except Exception as e:
        print(f"Erreur lors de la génération du rapport visuel: {str(e)}")

def save_results(results, metrics, output_file="evaluation_results.json"):
    """
    Sauvegarder les résultats et métriques dans un fichier JSON.
    
    Args:
        results: Liste des résultats détaillés
        metrics: Dictionnaire des métriques calculées
        output_file: Chemin du fichier de sortie
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": metrics,
                "detailed_results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nRésultats enregistrés dans '{output_file}'")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des résultats: {str(e)}")

def print_summary(metrics):
    """Afficher un résumé des métriques d'évaluation."""
    print("\n" + "="*50)
    print("RÉSUMÉ DE L'ÉVALUATION")
    print("="*50)
    
    print(f"\nScores de similarité:")
    print(f"  Moyenne: {metrics['average_similarity']:.2f}")
    print(f"  Médiane: {metrics['median_similarity']:.2f}")
    print(f"  Minimum: {metrics['min_similarity']:.2f}")
    print(f"  Maximum: {metrics['max_similarity']:.2f}")
    print(f"  Écart-type: {metrics['std_similarity']:.2f}")
    
    print(f"\nTemps de réponse (secondes):")
    print(f"  Moyen: {metrics['average_response_time']:.2f}s")
    print(f"  Minimum: {metrics['min_response_time']:.2f}s")
    print(f"  Maximum: {metrics['max_response_time']:.2f}s")
    
    print(f"\nÉvaluations qualitatives:")
    for rating, count in metrics['quality_ratings'].items():
        print(f"  {rating}: {count}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    # Charger les données de test
    test_data = load_test_data(n_samples=10)
    
    if not test_data:
        print("Aucune donnée de test disponible. Fin de l'évaluation.")
        exit(1)
    
    # Évaluer le chatbot
    results, metrics = evaluate_chatbot(test_data)
    
    # Afficher le résumé
    print_summary(metrics)
    
    # Sauvegarder les résultats
    save_results(results, metrics)
    
    # Générer des visualisations
    generate_visual_report(metrics, results)