"""
Définition des modèles de données (tables) pour la base de données SQLAlchemy.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float
from sqlalchemy.sql import func
from .database import Base

class MonsterAI(Base):
    """
    Table pour stocker la connaissance sur l'intelligence artificielle des monstres.
    """
    __tablename__ = "monster_ai"

    id = Column(Integer, primary_key=True, index=True)
    monster_name = Column(String, unique=True, nullable=False, index=True)
    ai_type = Column(String, nullable=False)  # Ex: "Fuyard", "Agressif", "Soigneur"
    behavior_summary = Column(Text, nullable=True)
    special_abilities = Column(JSON, nullable=True) # Ex: ["Mot d'Immobilisation", "Piqûre Motivante"]
    elemental_weaknesses = Column(JSON, nullable=True) # Ex: {"feu": 20, "air": -10}
    source = Column(String, nullable=True) # Ex: "dofus.fandom.com" ou "datamining"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ProgressionLog(Base):
    """
    Table pour enregistrer des "clichés" de la progression du joueur/bot.
    Format de type JSON Lines pour une grande flexibilité.
    """
    __tablename__ = "progression_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    snapshot_id = Column(String, nullable=False, index=True) # Ex: "character_stats", "inventory_summary"
    data = Column(JSON, nullable=False) # Les données du snapshot, ex: {"kamas": 12345, "level": 42}

class HumanFeedback(Base):
    """
    Table pour enregistrer les leçons apprises du joueur en mode semi-assisté.
    """
    __tablename__ = "human_feedback"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    game_state = Column(JSON, nullable=False) # Un snapshot de l'état du jeu au moment de la décision
    bot_suggestion = Column(JSON, nullable=False) # L'action suggérée par le bot
    user_action = Column(JSON, nullable=False) # L'action réellement effectuée par le joueur
    outcome = Column(JSON, nullable=True) # Le résultat (victoire, défaite, gain, etc.)
