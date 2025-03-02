import os
from typing import Dict, List
import streamlit as st
import requests
from dotenv import load_dotenv
import time
import base64
from PIL import Image, ImageDraw
import io
import numpy as np
import pandas as pd
import logging
import yaml


# Charger les variables d'environnement
load_dotenv()

# Charger le fichier config.yaml
with open("C:/Users/sarra/Desktop/medical-chatbot/deployment/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de l'API 
HOST = os.getenv("API_HOST", "https://fb-1021317796643.europe-west1.run.app")
API_ENDPOINT_ANSWER = f"{HOST}/answer"
API_ENDPOINT_SOURCES = f"{HOST}/get_sources"
API_ENDPOINT_FEEDBACK = f"{HOST}/feedback"
API_ENDPOINT_DOCUMENT = f"{HOST}/get_document_image"

# Configuration de la page Streamlit
st.set_page_config(
    page_title=config["streamlit"]["page_title"],
    page_icon=config["streamlit"]["page_icon"],
    layout=config["streamlit"]["layout"]
)

# Fonction pour visualiser les bounding boxes sur une image
def visualize_layout(image_bytes, boxes):
    try:
        # Ouvrir l'image depuis les bytes
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        
        # Dessiner les bounding boxes
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            color = colors[i % len(colors)]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        
        # Convertir l'image en bytes pour l'affichage
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation du layout: {str(e)}")
        st.error(f"Erreur lors de la visualisation du layout: {str(e)}")
        return None

# Titre de l'application
st.title(config["streamlit"]["page_title"])
st.write("Posez vos questions médicales et obtenez des réponses basées sur des documents médicaux avec layout.")

# Barre latérale pour les paramètres
with st.sidebar:
    st.header("Paramètres")
    temperature = st.slider(
        "Température",
        min_value=0.0,
        max_value=1.0,
        value=config["model"]["temperature"],
        step=0.05,
        help="Contrôle la créativité des réponses. Une valeur plus basse donne des réponses plus précises."
    )
    language = st.selectbox(
        "Langue",
        options=["Français", "English", "Arabic"],
        index=0,
        help="Sélectionnez la langue de la réponse."
    )
    
    use_layout = st.checkbox(
        "Utiliser les informations de layout",
        value=True,
        help="Prendre en compte la disposition spatiale des éléments dans les documents"
    )
    
    show_layout_details = st.checkbox(
        "Afficher les détails du layout",
        value=True,
        help="Afficher les bounding boxes et les informations spatiales"
    )
    
    # option pour le système RAG
    rag_mode = st.selectbox(
        "Mode de recherche",
        options=["Sémantique (RAG avancé)", "Mots-clés (Basique)", "Hybride"],
        index=0,
        help="Sélectionnez la méthode de recherche des documents pertinents"
    )
    
    rag_documents_limit = st.slider(
        "Nombre de documents à analyser",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Nombre maximum de documents à utiliser comme contexte"
    )
    
    st.header("À propos des données")
    st.info("""
    Cette application utilise le dataset LayoutLM de Kaggle qui contient des documents médicaux avec 
    leurs informations de layout. Ces informations permettent de comprendre la structure spatiale 
    des documents, ce qui est particulièrement utile pour les formulaires médicaux, les 
    rapports de laboratoire, etc.
    
    Le système RAG (Retrieval Augmented Generation) permet d'améliorer la qualité des réponses
    en contextualisant le modèle d'IA avec les documents médicaux les plus pertinents.
    """)

# Fonction pour envoyer le feedback à l'API
def submit_feedback(question: str, feedback_score: float):
    try:
        response = requests.post(
            API_ENDPOINT_FEEDBACK,
            json={"question": question, "feedback_score": feedback_score},
            timeout=10
        )
        if response.status_code == 200:
            st.success("Merci pour votre feedback !")
        else:
            st.error(f"Erreur lors de l'envoi du feedback : {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de connexion à l'API : {str(e)}")
        st.error(f"Erreur de connexion à l'API : {str(e)}")

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Quelle est votre question médicale ? Vous pouvez poser des questions spécifiques sur les documents médicaux, leurs structures ou contenus."}
    ]

# Affichage de l'historique des messages
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        
        # Afficher des sources utilisées si disponibles
        if message.get("role") == "assistant" and message.get("sources"):
            with st.expander("Sources utilisées pour cette réponse"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"**Source {i+1}:** {source.get('title', 'Document')}")
                    st.write(f"Pertinence: {source.get('score', 0):.2f}")
                    st.write(f"Extrait: {source.get('excerpt', '')}")

# Saisie de la question
if question := st.chat_input("Quelle est votre question médicale ?"):
    # Ajouter la question à l'historique
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user", avatar="🧑‍💻").write(question)

    # Afficher un indicateur de chargement pendant le traitement
    with st.spinner("Recherche de la réponse avec RAG..."):
        start_time = time.time()
        try:
            # Envoyer la question à l'API pour obtenir les sources d'abord
            documents_response = requests.post(
                API_ENDPOINT_SOURCES,
                json={
                    "question": question,
                    "temperature": temperature,
                    "language": language,
                    "consider_layout": use_layout,
                    "rag_mode": rag_mode.lower().split(" ")[0],  # Extraire le premier mot (sémantique, mots-clés, hybride)
                    "limit": rag_documents_limit
                },
                timeout=20
            )
            
            sources_data = {}
            if documents_response.status_code == 200:
                sources_data = documents_response.json()
            
            # Envoyer la question à l'API pour obtenir une réponse avec RAG
            response = requests.post(
                API_ENDPOINT_ANSWER,
                json={
                    "question": question,
                    "temperature": temperature,
                    "language": language,
                    "consider_layout": use_layout,
                    "rag_mode": rag_mode.lower().split(" ")[0],
                    "limit": rag_documents_limit,
                    "sources": sources_data.get("sources", [])  # Passer les sources trouvées au LLM
                },
                timeout=30  #  le timeout pour RAG
            )

            # Traiter la réponse de l'API
            if response.status_code == 200:
                answer_data = response.json()
                answer = answer_data["answer"]
                
                # Récupérer les sources utilisées
                used_sources = answer_data.get("used_sources", [])
                
                # Ajouter la réponse à l'historique avec les sources
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": used_sources
                })
                
                # Afficher la réponse
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(answer)
                    
                    # Afficher les sources utilisées
                    if used_sources:
                        with st.expander("Sources utilisées pour cette réponse"):
                            for i, source in enumerate(used_sources):
                                st.write(f"**Source {i+1}:** {source.get('title', 'Document')}")
                                st.write(f"Pertinence: {source.get('score', 0):.2f}")
                                st.write(f"Extrait: {source.get('excerpt', '')}")

                # Ajouter un système de feedback
                st.write("### Feedback")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Bonne réponse"):
                        submit_feedback(question, 1.0)
                with col2:
                    if st.button("👎 Mauvaise réponse"):
                        submit_feedback(question, 0.0)
                with col3:
                    if st.button("🤔 Neutre"):
                        submit_feedback(question, 0.5)
            else:
                st.error(f"Erreur API (réponse) : {response.status_code} - {response.text}")

            # Afficher les sources avec visualisation du layout
            if documents_response.status_code == 200:
                sources: List[Dict] = sources_data.get("sources", [])
                layout_data = sources_data.get("layout_data", {})
                
                if not sources:
                    st.warning("Aucune source trouvée pour cette question.")
                else:
                    st.write("### Sources détaillées :")
                    
                    # Afficher un tableau récapitulatif des sources et leur pertinence
                    source_info = []
                    for i, source in enumerate(sources):
                        metadata = source.get("metadata", {})
                        source_info.append({
                            "Source": f"Source {i + 1}",
                            "Document": metadata.get("source", "Inconnu"),
                            "Domaine": metadata.get("focus_area", "Général"),
                            "Score": source.get("similarity_score", 0),
                            "Layout": "Oui" if metadata.get("has_layout", False) else "Non"
                        })
                    
                    if source_info:
                        st.dataframe(pd.DataFrame(source_info))
                        
                        # Afficher métrique de qualité RAG
                        rag_metrics = sources_data.get("rag_metrics", {})
                        if rag_metrics:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Précision RAG", f"{rag_metrics.get('precision', 0):.2f}")
                            with col2:
                                st.metric("Rappel RAG", f"{rag_metrics.get('recall', 0):.2f}")
                            with col3:
                                st.metric("Score-F1 RAG", f"{rag_metrics.get('f1_score', 0):.2f}")
                    
                    # Afficher chaque source avec ses détails
                    for i, source in enumerate(sources):
                        with st.expander(f"Source {i + 1}: {source.get('metadata', {}).get('source', 'Document')}"):
                            metadata = source.get("metadata", {})
                            
                            # Afficher les métadonnées
                            st.write("**Métadonnées :**")
                            st.json(metadata)
                            
                            # Afficher le contenu textuel
                            st.write("**Contenu :**")
                            st.write(source.get("page_content", "Contenu non disponible"))
                            
                            # Afficher les chunks RAG si disponibles
                            if source.get("chunks"):
                                st.write("**Segments RAG :**")
                                for j, chunk in enumerate(source.get("chunks", [])):
                                    st.write(f"*Segment {j+1}:* {chunk}")
                            
                            # Afficher le layout si disponible
                            if show_layout_details and metadata.get("has_layout", False):
                                st.write("**Informations de layout :**")
                                
                                # Récupérer l'image du document si disponible
                                doc_id = metadata.get("source")
                                if doc_id:
                                    try:
                                        doc_response = requests.get(
                                            API_ENDPOINT_DOCUMENT,
                                            params={"document_id": doc_id},
                                            timeout=10
                                        )
                                        
                                        if doc_response.status_code == 200:
                                            image_data = doc_response.content
                                            boxes = layout_data.get(doc_id, {}).get("boxes", [])
                                            
                                            # Afficher les informations de layout
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.write("**Bounding Boxes :**")
                                                if boxes:
                                                    for j, box in enumerate(boxes):
                                                        st.write(f"Box {j+1}: {box}")
                                                else:
                                                    st.write("Aucune information de bounding box disponible")
                                            
                                            with col2:
                                                st.write("**Document avec Layout :**")
                                                if boxes and image_data:
                                                    # Visualiser le layout sur l'image
                                                    img_with_boxes = visualize_layout(image_data, boxes)
                                                    if img_with_boxes:
                                                        st.markdown(f'<img src="{img_with_boxes}" width="100%"/>', unsafe_allow_html=True)
                                                    else:
                                                        st.image(image_data, caption=f"Document: {doc_id}")
                                                else:
                                                    st.image(image_data, caption=f"Document: {doc_id}")
                                        else:
                                            st.warning(f"Impossible de récupérer l'image du document {doc_id}")
                                    except Exception as e:
                                        logger.error(f"Erreur lors de la récupération de l'image: {str(e)}")
                                        st.error(f"Erreur lors de la récupération de l'image: {str(e)}")
            else:
                st.error(f"Erreur API (sources) : {documents_response.status_code} - {documents_response.text}")

            # Gérer un délai minimal d'affichage du spinner
            elapsed_time = time.time() - start_time
            if elapsed_time < 2:  # Délai minimal de 2 secondes
                time.sleep(2 - elapsed_time)

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de connexion à l'API : {str(e)}")
            st.error(f"Erreur de connexion à l'API : {str(e)}")

# Section pour explorer le dataset LayoutLM
st.sidebar.header("Explorer le Dataset")
if st.sidebar.checkbox("Afficher l'explorateur de dataset"):
    st.subheader("Explorateur du Dataset LayoutLM")
    
    # Ajout d'une option pour visualiser les embeddings RAG
    tabs = st.tabs(["Documents", "Visualisation des Embeddings"])
    
    with tabs[0]:
        # Simulation de l'explorateur de dataset (à remplacer par des appels API réels)
        documents_types = config["layoutlm"]["document_types"]
        selected_doc_type = st.selectbox("Type de document", documents_types)
        
        # Ici, vous pourriez ajouter un appel API pour récupérer les documents de ce type
        st.write(f"Documents de type: {selected_doc_type}")
        
        # Simulation d'exemples de documents (à remplacer par des données réelles)
        doc_examples = []
        for i in range(1, 6):
            doc_examples.append(f"Document {i} - {selected_doc_type}")
        
        selected_doc = st.selectbox("Sélectionner un document", doc_examples)
        
        if st.button("Afficher le document"):
            with st.spinner("Chargement du document..."):
                # Ici, vous feriez un appel API pour récupérer les détails du document
                st.success(f"Document chargé: {selected_doc}")
                st.write("**Cet exemple montre comment explorer le dataset LayoutLM. Dans une implémentation complète, vous pourriez visualiser les documents réels avec leurs informations de layout.**")
    
    with tabs[1]:
        st.write("### Visualisation des embeddings de RAG")
        st.write("Cette visualisation montre comment les documents sont positionnés dans l'espace vectoriel.")
        
        # Placeholder pour la visualisation des embeddings
        if st.button("Générer visualisation des embeddings"):
            with st.spinner("Génération de la visualisation..."):
                # Ici, vous feriez un appel API pour récupérer les données d'embedding
                st.write("Visualisation des embeddings (simulée):")
                
                # Créer une simulation de carte d'embeddings 2D
                np.random.seed(42)
                embeddings = np.random.rand(20, 2)
                
                # Créer un DataFrame pour le scatter plot
                embedding_df = pd.DataFrame({
                    'x': embeddings[:, 0],
                    'y': embeddings[:, 1],
                    'document': [f"Doc {i}" for i in range(1, 21)],
                    'type': np.random.choice(['Rapport', 'Formulaire', 'Article'], 20)
                })
                
                # Afficher le scatter plot
                st.write("Projection t-SNE des embeddings des documents:")
                st.scatter_chart(embedding_df, x='x', y='y', color='type', size=20, use_container_width=True)
                
                st.info("Cette visualisation permet de comprendre la similarité sémantique entre les documents. Les documents proches dans cet espace sont sémantiquement similaires.")