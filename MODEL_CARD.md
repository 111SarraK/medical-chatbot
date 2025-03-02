# Model Card for Medical Chatbot with LayoutLM

## 1. Model Details
- **Model Name**: Medical Chatbot with LayoutLM
- **Version**: 1.0.0
- **Date**: 28.02.2025
- **Developers**: Sarra Hammami 
- **Repository**: [Lien vers le repository Git]
- **Contact Information**: [Email ou autre moyen de contact]

## 2. Intended Use
- **Primary Use Case**: Répondre à des questions spécifiques dans le domaine médical en utilisant un dataset structuré avec des informations de layout (disposition spatiale des éléments dans les documents).
- **Target Audience**: Professionnels de santé, étudiants en médecine, ou toute personne cherchant des informations médicales techniques.
- **Out-of-Scope Use Cases**: Diagnostic médical direct, conseils médicaux personnalisés, ou utilisation dans des contextes critiques sans supervision médicale.

## 3. Data
- **Dataset**: LayoutLM Dataset (Kaggle)
- **Source**: (https://www.kaggle.com/datasets/jpmiller/layoutlm/data)
- **Data Preprocessing**:
  - Chargement des documents médicaux depuis un fichier CSV (`medquad.csv`).
  - Stockage des documents dans une base de données Cloud SQL.
  - Vectorisation des documents pour la recherche de similarité.
  - Extraction des informations de layout (bounding boxes) et des images associées.

## 4. Model Architecture
- **Model Type**: Modèle de langage basé sur Google Gemini (Gemini-1.5-pro).
- **Framework**: LangChain, FastAPI, Streamlit.
- **Pre-trained Model**: Gemini-1.5-pro (Google Generative AI).
- **Fine-tuning**: Aucun fine-tuning effectué, utilisation directe du modèle pré-entraîné.
- **Vector Store**: Utilisation de VertexAIEmbeddings pour la vectorisation des documents et stockage dans une base de données PostgreSQL via Cloud SQL.

## 5. Performance
- **Metrics**:
  - **Similarity Score**: Score de similarité entre la question posée et les documents pertinents.
  - **Response Time**: Temps de réponse moyen pour générer une réponse.
  - **Feedback Score**: Score de satisfaction des utilisateurs (entre 0 et 1).
- **Evaluation Results**:
  - Le modèle est évalué sur 10 exemples aléatoires du dataset via un script `eval.py`.
  - Les résultats incluent la précision des réponses, le temps de réponse, et les scores de similarité.

## 6. Ethical Considerations
- **Bias**: Les biais potentiels dans les données peuvent affecter les réponses du modèle.
- **Fairness**: Le modèle doit être utilisé de manière équitable pour tous les utilisateurs.
- **Privacy**: Les données des utilisateurs sont stockées dans une base de données sécurisée (Cloud SQL).

## 7. Deployment
- **Infrastructure**:
  - **Cloud SQL**: Stockage des documents et des requêtes.
  - **Cloud Run**: Déploiement de l'application FastAPI.
  - **Streamlit**: Interface utilisateur pour interagir avec le chatbot.
- **Monitoring**: Utilisation de LangFuse pour surveiller les appels LLM et collecter les feedbacks des utilisateurs.

## 8. Usage Instructions
- **How to Use**:
  - Pour exécuter l'application Streamlit : `streamlit run streamlit_app.py`.
  - Pour tester le modèle : `python eval.py`.
- **Requirements**:
  - Les dépendances sont listées dans `requirements.txt`.
  - Les variables d'environnement doivent être configurées (par exemple, `OPENAI_API_KEY`, `INSTANCE_CONNECTION_NAME`, etc.).

## 9. References
- **Papers**:
  - LayoutLM: Pre-training of Text and Layout for Document Image Understanding (https://arxiv.org/abs/1912.13318)
- **Documentation**:
  - LangChain: https://python.langchain.com/
  - Google Generative AI: https://cloud.google.com/vertex-ai/docs/generative-ai
  - Streamlit: https://docs.streamlit.io/