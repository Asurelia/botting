"""
Gestionnaire de la base de données avec SQLAlchemy.
Ce module centralise la connexion et la gestion des sessions
pour l'ensemble de l'application.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Chemin vers la base de données SQLite
DATABASE_URL = "sqlite:///G:/Botting/data/memory_palace.db"

# Création du moteur SQLAlchemy
# `check_same_thread=False` est nécessaire pour SQLite avec des applications multi-threads.
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    echo=False  # Mettre à True pour voir les requêtes SQL générées
)

# Création d'une fabrique de sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base déclarative pour les modèles de données
Base = declarative_base()

def get_db():
    """
    Fonction générateur pour fournir une session de base de données
    à chaque requête qui en a besoin.
    Assure que la session est toujours fermée après usage.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialise la base de données en créant toutes les tables
    définies dans les modèles.
    """
    # Importer ici tous les modèles pour qu'ils soient enregistrés
    # dans les métadonnées de la Base.
    from . import database_models
    
    # S'assurer que le répertoire data existe
    data_dir = os.path.dirname(DATABASE_URL.replace("sqlite:///", ""))
    os.makedirs(data_dir, exist_ok=True)
    
    print("Initialisation de la base de données...")
    Base.metadata.create_all(bind=engine)
    print("Base de données initialisée avec succès.")

if __name__ == "__main__":
    # Permet d'initialiser la base de données en exécutant ce fichier directement
    # python -m core.database
    init_db()
