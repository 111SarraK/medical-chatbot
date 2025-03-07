# Medical Chatbot

## Description

Le **Medical Chatbot** est un chatbot technique conçu pour répondre à des questions spécifiques dans le domaine médical. Le projet utilise des modèles d'IA avancés, y compris des architectures comme **LayoutLM** et **RAG (Retrieval Augmented Generation)**, pour extraire des informations à partir de documents médicaux en tenant compte de leur mise en page. Le chatbot est capable de fournir des réponses précises en s'appuyant sur des documents médicaux structurés et en utilisant des techniques de recherche sémantique.

## Fonctionnalités clés

- **Répondre à des questions médicales** : Le chatbot extrait des informations à partir de documents médicaux pour fournir des réponses précises.
- **Utilisation de LayoutLM** : Le modèle comprend et analyse les documents en tenant compte de leur mise en page (layout).
- **Interface interactive** : Une interface utilisateur est disponible via **Streamlit** pour interagir avec le chatbot.
- **Stockage des données** : Les données sources et les réponses sont stockées dans une base de données **Cloud SQL**, avec des détails comme le score de similarité et la zone de focus.
- **Système RAG** : Le chatbot utilise un système de **Retrieval Augmented Generation** pour améliorer la qualité des réponses en contextualisant les questions avec les documents pertinents.

## Objectifs du Projet

- **Création de l'interface Streamlit** : Une interface permettant de poser des questions et d'afficher les réponses extraites des documents.
- **Formulation de la réponse** : Le chatbot doit être capable de fournir une réponse exacte à une question donnée, avec les sources utilisées pour formuler cette réponse.
- **Stockage dans Cloud SQL** : Toutes les sources utilisées pour répondre aux questions sont stockées dans une base de données **Cloud SQL** avec des informations sur la source et la zone de focus.
- **Déploiement sur Cloud Run** : Le chatbot est déployé sur **Cloud Run** avec un nom de projet spécifique.

## Prérequis

- **Python 3.10 ou version supérieure**
- **Clé API Google** : Pour accéder aux modèles de génération de texte de Google (Gemini).
- **Compte Google Cloud** : Pour utiliser **Cloud SQL** et **Cloud Run**.
- **Fichier de configuration** : Un fichier `.env` doit être configuré avec les variables d'environnement nécessaires (clé API, paramètres de connexion à la base de données, etc.).

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/111SarraK/medical-document-chatbot.git
   cd medical-document-chatbot#   m e d i c a l - c h a t b o t  
 