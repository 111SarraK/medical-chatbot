import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ajuster le chemin pour importer les modules du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.chatbot.model import MedicalChatbot

class TestMedicalChatbot(unittest.TestCase):
    """Tests pour la classe MedicalChatbot."""

    def setUp(self):
        """Initialiser les mocks et le chatbot pour les tests."""
        self.mock_api_key = "mock-api-key"
        
        # Patcher les dépendances externes
        self.patcher_chatopenai = patch('src.chatbot.model.ChatOpenAI')
        self.patcher_openaiembeddings = patch('src.chatbot.model.OpenAIEmbeddings')
        self.patcher_faiss = patch('src.chatbot.model.FAISS')
        self.patcher_retrievalqa = patch('src.chatbot.model.RetrievalQA')
        
        # Démarrer les patches
        self.mock_llm = self.patcher_chatopenai.start()
        self.mock_embeddings = self.patcher_openaiembeddings.start()
        self.mock_faiss = self.patcher_faiss.start()
        self.mock_qa = self.patcher_retrievalqa.start()
        
        # Créer l'instance du chatbot avec les mocks
        self.chatbot = MedicalChatbot(api_key=self.mock_api_key)
        
    def tearDown(self):
        """Arrêter tous les patches."""
        self.patcher_chatopenai.stop()
        self.patcher_openaiembeddings.stop()
        self.patcher_faiss.stop()
        self.patcher_retrievalqa.stop()
    
    def test_initialization(self):
        """Tester l'initialisation correcte du chatbot."""
        self.mock_llm.assert_called_once()
        self.mock_embeddings.assert_called_once_with(api_key=self.mock_api_key)
        
        self.assertEqual(self.chatbot.api_key, self.mock_api_key)
        self.assertIsNotNone(self.chatbot.llm)
        self.assertIsNotNone(self.chatbot.embeddings)
        self.assertIsNone(self.chatbot.vectorstore)
        self.assertIsNone(self.chatbot.qa_chain)
    
    def test_initialize_vectorstore(self):
        """Tester l'initialisation du vectorstore avec des documents LayoutLM."""
        test_docs = [
            {"text": "test text 1", "source": "source1", "focus_area": "area1", "layout": {"bounding_boxes": []}},
            {"text": "test text 2", "source": "source2", "focus_area": "area2", "layout": {"bounding_boxes": []}}
        ]
        
        self.chatbot.initialize_vectorstore(test_docs)
        
        self.mock_faiss.from_texts.assert_called_once()
        texts_arg, embeddings_arg = self.mock_faiss.from_texts.call_args[0][:2]
        metadatas_arg = self.mock_faiss.from_texts.call_args[1]['metadatas']
        
        self.assertEqual(len(texts_arg), 2)
        self.assertEqual(texts_arg[0], "test text 1")
        self.assertEqual(embeddings_arg, self.chatbot.embeddings)
        self.assertEqual(len(metadatas_arg), 2)
        self.assertEqual(metadatas_arg[0]['source'], "source1")
        self.assertEqual(metadatas_arg[0]['focus_area'], "area1")
        self.assertEqual(metadatas_arg[0]['layout'], {"bounding_boxes": []})
    
    def test_get_answer_without_setup(self):
        """Tester que get_answer lève une erreur si qa_chain n'est pas initialisé."""
        with self.assertRaises(ValueError):
            self.chatbot.get_answer("test question")
    
    def test_get_answer(self):
        """Tester la méthode get_answer avec des documents LayoutLM."""
        self.chatbot.qa_chain = MagicMock()
        self.chatbot.qa_chain.return_value = {"result": "test answer"}
        
        self.chatbot.vectorstore = MagicMock()
        mock_docs = [
            (MagicMock(metadata={"source": "src1", "focus_area": "area1", "layout": {"bounding_boxes": []}}), 0.9),
            (MagicMock(metadata={"source": "src2", "focus_area": "area2", "layout": {"bounding_boxes": []}}), 0.8)
        ]
        self.chatbot.vectorstore.similarity_search_with_score.return_value = mock_docs
        
        mock_cb = MagicMock()
        mock_cb.total_cost = 0.05
        
        with patch('src.chatbot.model.get_openai_callback', return_value=mock_cb):
            answer, sources, cost = self.chatbot.get_answer("test question")
        
        self.assertEqual(answer, "test answer")
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["source"], "src1")
        self.assertEqual(sources[0]["focus_area"], "area1")
        self.assertEqual(sources[0]["similarity_score"], 0.9)
        self.assertEqual(sources[0]["layout"], {"bounding_boxes": []})
        self.assertEqual(cost, 0.05)

if __name__ == '__main__':
    unittest.main()