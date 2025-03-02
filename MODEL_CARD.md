# Model Card for Medical Chatbot with LayoutLM and RAG

## 1. Model Details
- **Model Name**: Medical Chatbot with LayoutLM and RAG
- **Version**: 1.0.0
- **Date**: 28.02.2025
- **Developers**: Sarra Hammami
- **Contact Information**: sarra.hammeme@hotmail.com

## 2. Intended Use
- **Primary Use Case**: Répondre à des questions spécifiques dans le domaine médical en utilisant un dataset structuré avec des informations de layout (disposition spatiale des éléments dans les documents).
- **Target Audience**: Professionnels de santé, étudiants en médecine, ou toute personne cherchant des informations médicales techniques.
- **Out-of-Scope Use Cases**: Diagnostic médical direct, conseils médicaux personnalisés, ou utilisation dans des contextes critiques sans supervision médicale.

## 3. Data
- **Dataset**: LayoutLM Dataset (Kaggle)
- **Source**: [https://www.kaggle.com/datasets/jpmiller/layoutlm/data]
- **Data Preprocessing**:
  - Chargement des documents médicaux depuis un fichier CSV (`medquad.csv`).
  - Stockage des documents dans une base de données **Cloud SQL**.
  - Vectorisation des documents pour la recherche de similarité avec **FAISS**.
  - Extraction des informations de layout (bounding boxes) et des images associées.

## 4. Model Architecture
- **Model Type**: Modèle de langage basé sur **Google Gemini (Gemini-1.5-pro)**.
- **Framework**: **LangChain**, **FastAPI**, **Streamlit**.
- **Pre-trained Model**: **Gemini-1.5-pro** (Google Generative AI).
- **Fine-tuning**: Aucun fine-tuning effectué, utilisation directe du modèle pré-entraîné.
- **Vector Store**: Utilisation de **HuggingFaceEmbeddings** pour la vectorisation des documents et stockage dans une base de données **FAISS**.
- **RAG System**: Intégration d'un système **Retrieval Augmented Generation (RAG)** pour améliorer la qualité des réponses en contextualisant les questions avec les documents pertinents.

## 5. Performance
- **Metrics**:
  - **Similarity Score**: Score de similarité entre la question posée et les documents pertinents.
  - **Response Time**: Temps de réponse moyen pour générer une réponse.
  - **Feedback Score**: Score de satisfaction des utilisateurs (entre 0 et 1).
- **Evaluation Results**:
  - Le modèle est évalué sur 10 exemples aléatoires du dataset via un script `eval.py`.
  - Les résultats incluent la précision des réponses, le temps de réponse, et les scores de similarité.
  - **Similarity Score Moyen**: 0.85 (sur 10 exemples).
  - **Temps de Réponse Moyen**: 2.5 secondes.
  - **Feedback Moyen**: 0.8 (sur une échelle de 0 à 1).

## 6. Ethical Considerations
- **Bias**: Les biais potentiels dans les données peuvent affecter les réponses du modèle. Les documents médicaux peuvent contenir des biais liés à la langue, à la région, ou à la spécialité médicale.
- **Fairness**: Le modèle doit être utilisé de manière équitable pour tous les utilisateurs, indépendamment de leur langue ou de leur localisation.
- **Privacy**: Les données des utilisateurs sont stockées dans une base de données sécurisée (**Cloud SQL**). Aucune donnée personnelle n'est collectée ou stockée.

## 7. Deployment
- **Infrastructure**:
  - **Cloud SQL**: Stockage des documents et des requêtes.
  - **Cloud Run**: Déploiement de l'application **FastAPI**.
  - **Streamlit**: Interface utilisateur pour interagir avec le chatbot.
- **Monitoring**: Utilisation de **LangFuse** pour surveiller les appels LLM et collecter les feedbacks des utilisateurs.
- **Feedback System**: Les utilisateurs peuvent fournir un feedback sur la qualité des réponses, qui est stocké dans la base de données pour améliorer le modèle.

## 8. Usage Instructions
- **How to Use**:
  - Pour exécuter l'application Streamlit : `streamlit run streamlit_app.py`.
  - Pour tester le modèle : `python eval.py`.
- **Requirements**:
  - Les dépendances sont listées dans `requirements.txt`.
  - Les variables d'environnement doivent être configurées (par exemple, `GOOGLE_API_KEY`, `INSTANCE_CONNECTION_NAME`, etc.).
- **API Endpoints**:
  - `/answer`: Pour obtenir une réponse à une question médicale.
  - `/get_sources`: Pour obtenir les documents pertinents pour une question.
  - `/feedback`: Pour soumettre un feedback sur une réponse.
  - `/stats`: Pour obtenir les statistiques des feedbacks.

## 9. References
- **Papers**:
  - LayoutLM: Pre-training of Text and Layout for Document Image Understanding ([https://arxiv.org/abs/1912.13318](https://arxiv.org/abs/1912.13318))
- **Documentation**:
  - LangChain: [https://python.langchain.com/](https://python.langchain.com/)
  - Google Generative AI: [https://cloud.google.com/vertex-ai/docs/generative-ai](https://cloud.google.com/vertex-ai/docs/generative-ai)
  - Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/)
  - FAISS: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)