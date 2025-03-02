from setuptools import setup, find_packages
import os
from typing import List

# Lire le contenu du fichier README.md
def read_file(filename: str) -> str:
    """Lit le contenu d'un fichier."""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

# Lire les dépendances depuis requirements.txt
def load_requirements(filename: str) -> List[str]:
    """Charge les dépendances depuis un fichier."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Le fichier {filename} est introuvable.")
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().splitlines()

# Informations du projet
NAME = "medical-document-chatbot"
VERSION = "1.0.0"
AUTHOR = Sarra Hammami
AUTHOR_EMAIL = sarra.hammemi@hotmail.com
DESCRIPTION = "Un chatbot médical utilisant RAG et LayoutLM pour traiter des documents avec leur mise en page"
LONG_DESCRIPTION = read_file("README.md")
URL = "https://github.com/111SarraK/medical-document-chatbot"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.9"

# Charger les dépendances
try:
    REQUIREMENTS = load_requirements("requirements.txt")
except FileNotFoundError as e:
    print(f"Erreur : {str(e)}")
    REQUIREMENTS = [
        "transformers>=4.20.0",
        "pytorch-lightning>=1.6.0",
        "torch>=1.10.0",
        "pytesseract>=0.3.9",
        "pdf2image>=1.16.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.5",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "langchain>=0.0.200",
        "google-generativeai>=0.1.0",
        "pandas>=1.3.0",
        "streamlit>=1.12.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "yaml>=5.4.1",
    ]

# Configuration du package
setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    packages=find_packages(where="src"),  # Inclut tous les packages dans le répertoire src
    package_dir={"": "src"},  # Indique que les packages sont dans le répertoire src
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Document Analysis",
    ],
    python_requires=PYTHON_REQUIRES,
    install_requires=REQUIREMENTS,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "medical-chatbot=chatbot.streamlit_app:main",  
            "evaluate-chatbot=eval:evaluate_chatbot",  
            "process-documents=document_processing.process:main",  
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0",
            "flake8>=5.0",
            "black>=22.0",
            "mypy>=0.9",
        ],
        "docs": [
            "mkdocs>=1.4",
            "mkdocs-material>=8.0",
        ],
        "layoutlm": [
            "datasets>=2.0.0",
            "nltk>=3.7",
            "scikit-learn>=1.0.0",
            "scipy>=1.8.0",
            "wandb>=0.13.0",  # Pour le tracking des expériences
        ],
    },
)