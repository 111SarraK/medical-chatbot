import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import streamlit as st

# Ajuster le chemin pour importer les modules du projet

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Patcher Streamlit avant l'importation du module
for attr in dir(st):
    if attr.startswith("_") or attr.startswith("cache") or attr == "spinner":
        continue
    setattr(st, attr, MagicMock())

# Maintenant on peut importer l'application
from src.chatbot.streamlit_app import initialize_chatbot, main

class TestStreamlitUI(unittest.TestCase):
    """Tests pour l'interface utilisateur Streamlit."""

    def setUp(self):
        """Initialiser les mocks pour les tests."""
        # Patcher les dépendances
        self.ui_patches = [
            patch('src.chatbot.streamlit_app.MedicalChatbot'),
            patch('src.chatbot.streamlit_app.DatabaseManager'),
            patch('src.chatbot.streamlit_app.st')
        ]
        
        # Démarrer les patches
        self.mock_chatbot_cls, self.mock_db_cls, self.mock_st = [p.start() for p in self.ui_patches]
        
        # Configurer les instances mock
        self.mock_chatbot = MagicMock()
        self.mock_db = MagicMock()
        self.mock_chatbot_cls.return_value = self.mock_chatbot
        self.mock_db_cls.return_value = self.mock_db
        
        # Configurer le comportement du mock Streamlit
        self.mock_st.text_input.return_value = ""
        self.mock_st.slider.return_value = 3
        self.mock_st.button.return_value = False
        
    def tearDown(self):
        """Arrêter tous les patches."""
        for patch in self.ui_patches:
            patch.stop()
    
    @patch('src.chatbot.streamlit_app.os')
    @patch('src.chatbot.streamlit_app.load_dotenv')
    def test_initialize_chatbot(self, mock_load_dotenv, mock_os):
        """Tester l'initialisation du chatbot."""
        # Configurer les variables d'environnement
        mock_os.getenv.side_effect = lambda key: {
            "OPENAI_API_KEY": "mock-api-key",
            "INSTANCE_CONNECTION_NAME": "mock-instance",
            "DB_USER": "mock-user",
            "DB_PASS": "mock-pass",
            "DB_NAME": "mock-db"
        }.get(key)
        
        # Initialiser le chatbot
        chatbot, db = initialize_chatbot()
        
        # Vérifier que les méthodes sont appelées
        mock_load_dotenv.assert_called_once()
        self.mock_chatbot_cls.assert_called_once_with(api_key="mock-api-key")
        self.mock_db_cls.assert_called_once_with(
            instance_connection_name="mock-instance",
            db_user="mock-user",
            db_pass="mock-pass",
            db_name="mock-db"
        )
        
        self.mock_db.initialize_connection.assert_called_once()
        self.mock_db.get_all_documents.assert_called_once()
        self.mock_chatbot.initialize_vectorstore.assert_called_once()
        self.mock_chatbot.setup_qa_chain.assert_called_once()
    
    @patch('src.chatbot.streamlit_app.initialize_chatbot')
    def test_main_no_question(self, mock_initialize_chatbot):
        """Tester l'interface principale sans question."""
        # Configurer le mock du chatbot
        mock_chatbot = MagicMock()
        mock_db = MagicMock()
        mock_initialize_chatbot.return_value = (mock_chatbot, mock_db)
        
        # Aucune question n'est entrée
        self.mock_st.text_input.return_value = ""
        
        # Exécuter la fonction principale
        main()
        
        # Vérifier que le titre est affiché mais pas de réponse
        self.mock_st.title.assert_called_once()
        mock_chatbot.get_answer.assert_not_called()
    
    @patch('src.chatbot.streamlit_app.initialize_chatbot')
    def test_main_with_question(self, mock_initialize_chatbot):
        """Tester l'interface principale avec une question."""
        # Configurer le mock du chatbot
        mock_chatbot = MagicMock()
        mock_db = MagicMock()
        mock_initialize_chatbot.return_value = (mock_chatbot, mock_db)
        
        # Configurer la réponse du chatbot
        mock_chatbot.get_answer.return_value = (
            "Réponse de test",
            [{"source": "source1", "focus_area": "area1", "similarity_score": 0.9}],
            0.05
        )
        
        # Simuler une question
        self.mock_st.text_input.return_value = "Question test?"
        
        # Exécuter la fonction principale
        main()
        
        # Vérifier que le chatbot est interrogé
        mock_chatbot.get_answer.assert_called_once_with("Question test?")
        
        # Vérifier que la réponse est affichée
        self.mock_st.write.assert_any_call("### Réponse:")
        self.mock_st.write.assert_any_call("Réponse de test")
        
        # Vérifier que les sources sont affichées
        self.mock_st.write.assert_any_call("### Sources:")
        
        # Vérifier que la question est stockée
        mock_db.store_query.assert_called_once()
    
    @patch('src.chatbot.streamlit_app.initialize_chatbot')
    def test_feedback_submission(self, mock_initialize_chatbot):
        """Tester la soumission de feedback."""
        # Configurer les mocks
        mock_chatbot = MagicMock()
        mock_db = MagicMock()
        mock_initialize_chatbot.return_value = (mock_chatbot, mock_db)
        
        # Configurer une question et une réponse
        self.mock_st.text_input.return_value = "Question test?"
        mock_chatbot.get_answer.return_value = (
            "Réponse de test",
            [{"source": "source1", "focus_area": "area1", "similarity_score": 0.9}],
            0.05
        )
        
        # Simuler un clic sur le bouton de feedback
        self.mock_st.button.return_value = True
        
        # Exécuter la fonction principale
        main()
        
        # Vérifier que le feedback est stocké
        mock_db.store_feedback.assert_called_once_with("Question test?", 3)

if __name__ == '__main__':
    unittest.main()