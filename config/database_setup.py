"""
Module de configuration et initialisation de la base de données SQLite
=====================================================================

Ce module gère la création, configuration et migration de la base de données
SQLite utilisée par le bot DOFUS. Il inclut toutes les tables nécessaires
pour le stockage des données de jeu, configuration et logs.

Créé le: 2025-08-31
Version: 1.0.0
"""

import sqlite3
import os
import logging
import hashlib
import json
import shutil
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


class DatabaseManager:
    """Gestionnaire principal de la base de données"""
    
    def __init__(self, db_path: str = "data/databases/dofus_bot.db"):
        """
        Initialise le gestionnaire de base de données
        
        Args:
            db_path: Chemin vers le fichier de base de données
        """
        self.db_path = Path(db_path)
        self.backup_path = Path("data/backups/")
        self.logger = logging.getLogger(__name__)
        self.version = "1.0.0"
        
        # Créer les répertoires si nécessaire
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
    def connect(self) -> sqlite3.Connection:
        """
        Crée une connexion à la base de données avec optimisations
        
        Returns:
            Connexion SQLite configurée
        """
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False
        )
        
        # Configuration des pragma pour optimiser les performances
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        conn.execute("PRAGMA foreign_keys=ON")
        
        # Activer les fonctions personnalisées
        conn.row_factory = sqlite3.Row
        
        return conn
    
    def initialize_database(self) -> bool:
        """
        Initialise la base de données avec toutes les tables
        
        Returns:
            True si l'initialisation réussit, False sinon
        """
        try:
            with self.connect() as conn:
                self.logger.info("Initialisation de la base de données...")
                
                # Créer les tables dans l'ordre des dépendances
                self._create_system_tables(conn)
                self._create_character_tables(conn)
                self._create_game_data_tables(conn)
                self._create_combat_tables(conn)
                self._create_profession_tables(conn)
                self._create_economy_tables(conn)
                self._create_navigation_tables(conn)
                self._create_automation_tables(conn)
                self._create_log_tables(conn)
                
                # Créer les index pour optimiser les performances
                self._create_indexes(conn)
                
                # Insérer la version de la base de données
                self._update_database_version(conn)
                
                self.logger.info("Base de données initialisée avec succès")
                return True
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            return False
    
    def _create_system_tables(self, conn: sqlite3.Connection):
        """Crée les tables système"""
        
        # Table de version et métadonnées
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_info (
                id INTEGER PRIMARY KEY,
                version TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                schema_hash TEXT,
                backup_count INTEGER DEFAULT 0
            )
        """)
        
        # Table de configuration
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table des migrations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                migration_name TEXT NOT NULL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT TRUE
            )
        """)
    
    def _create_character_tables(self, conn: sqlite3.Connection):
        """Crée les tables liées aux personnages"""
        
        # Table des personnages
        conn.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                class TEXT NOT NULL,
                breed_id INTEGER,
                level INTEGER DEFAULT 1,
                experience BIGINT DEFAULT 0,
                server TEXT,
                kamas BIGINT DEFAULT 0,
                alignment TEXT,
                honor INTEGER DEFAULT 0,
                dishonor INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Table des statistiques
        conn.execute("""
            CREATE TABLE IF NOT EXISTS character_stats (
                character_id INTEGER PRIMARY KEY,
                vitality INTEGER DEFAULT 0,
                wisdom INTEGER DEFAULT 0,
                strength INTEGER DEFAULT 0,
                intelligence INTEGER DEFAULT 0,
                chance INTEGER DEFAULT 0,
                agility INTEGER DEFAULT 0,
                action_points INTEGER DEFAULT 0,
                movement_points INTEGER DEFAULT 0,
                summon_count INTEGER DEFAULT 0,
                range INTEGER DEFAULT 0,
                critical_hit INTEGER DEFAULT 0,
                critical_damage INTEGER DEFAULT 0,
                initiative INTEGER DEFAULT 0,
                heal_bonus INTEGER DEFAULT 0,
                damage_bonus INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
        
        # Table de l'équipement
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equipment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                slot INTEGER NOT NULL, -- Position d'équipement
                item_id INTEGER NOT NULL,
                item_name TEXT,
                item_type TEXT,
                level INTEGER,
                effects TEXT, -- JSON des effets
                equipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
        
        # Table des sorts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS character_spells (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                spell_id INTEGER NOT NULL,
                level INTEGER DEFAULT 1,
                position INTEGER, -- Position dans la barre de sorts
                cooldown_remaining INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
    
    def _create_game_data_tables(self, conn: sqlite3.Connection):
        """Crée les tables de données de jeu"""
        
        # Table des sorts (référence)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spells (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                class TEXT,
                element TEXT,
                level_required INTEGER,
                ap_cost INTEGER,
                range_min INTEGER DEFAULT 0,
                range_max INTEGER,
                cast_per_turn INTEGER DEFAULT 1,
                cast_per_target INTEGER DEFAULT 1,
                line_of_sight BOOLEAN DEFAULT FALSE,
                linear BOOLEAN DEFAULT FALSE,
                cooldown INTEGER DEFAULT 0,
                critical_hit_rate INTEGER DEFAULT 0,
                effects TEXT, -- JSON des effets par niveau
                icon_path TEXT
            )
        """)
        
        # Table des objets (référence)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                level INTEGER,
                description TEXT,
                effects TEXT, -- JSON des effets
                conditions TEXT, -- JSON des conditions
                recipe TEXT, -- JSON de la recette si craftable
                market_category TEXT,
                weight INTEGER DEFAULT 1,
                stackable BOOLEAN DEFAULT FALSE,
                max_stack INTEGER DEFAULT 1,
                icon_path TEXT,
                rarity TEXT DEFAULT 'common'
            )
        """)
        
        # Table des cartes et zones
        conn.execute("""
            CREATE TABLE IF NOT EXISTS maps (
                id INTEGER PRIMARY KEY,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                world_id INTEGER DEFAULT 1,
                area_id INTEGER,
                sub_area_id INTEGER,
                name TEXT,
                indoor BOOLEAN DEFAULT FALSE,
                has_zaap BOOLEAN DEFAULT FALSE,
                has_zaapi BOOLEAN DEFAULT FALSE,
                has_prism BOOLEAN DEFAULT FALSE,
                monsters TEXT, -- JSON liste des monstres
                resources TEXT, -- JSON liste des ressources
                npcs TEXT, -- JSON liste des PNJ
                shops TEXT, -- JSON liste des magasins
                capabilities INTEGER DEFAULT 0 -- Bitmask des capacités
            )
        """)
        
        # Table des monstres
        conn.execute("""
            CREATE TABLE IF NOT EXISTS monsters (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                level INTEGER,
                race TEXT,
                element TEXT,
                health INTEGER,
                ap INTEGER,
                mp INTEGER,
                resistances TEXT, -- JSON des résistances
                spells TEXT, -- JSON des sorts
                drops TEXT, -- JSON des drops possibles
                experience INTEGER,
                kamas_min INTEGER DEFAULT 0,
                kamas_max INTEGER DEFAULT 0,
                alignment TEXT,
                aggro_range INTEGER DEFAULT 0
            )
        """)
    
    def _create_combat_tables(self, conn: sqlite3.Connection):
        """Crée les tables liées au combat"""
        
        # Table des combats
        conn.execute("""
            CREATE TABLE IF NOT EXISTS combats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                map_id INTEGER,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration INTEGER, -- en secondes
                result TEXT, -- victory, defeat, flee, disconnect
                experience_gained INTEGER DEFAULT 0,
                kamas_gained INTEGER DEFAULT 0,
                drops TEXT, -- JSON des objets obtenus
                enemies TEXT, -- JSON des ennemis
                allies TEXT, -- JSON des alliés
                turn_count INTEGER DEFAULT 0,
                damage_dealt INTEGER DEFAULT 0,
                damage_received INTEGER DEFAULT 0,
                healing_done INTEGER DEFAULT 0,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
        
        # Table des actions de combat
        conn.execute("""
            CREATE TABLE IF NOT EXISTS combat_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combat_id INTEGER NOT NULL,
                turn_number INTEGER NOT NULL,
                actor_type TEXT, -- player, monster, ally
                actor_id INTEGER,
                action_type TEXT, -- spell, move, pass_turn, item
                spell_id INTEGER,
                target_cell INTEGER,
                target_id INTEGER,
                damage_dealt INTEGER DEFAULT 0,
                healing_done INTEGER DEFAULT 0,
                ap_used INTEGER DEFAULT 0,
                mp_used INTEGER DEFAULT 0,
                critical_hit BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (combat_id) REFERENCES combats (id) ON DELETE CASCADE
            )
        """)
        
        # Table des stratégies de combat
        conn.execute("""
            CREATE TABLE IF NOT EXISTS combat_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                character_class TEXT,
                priority INTEGER DEFAULT 0,
                conditions TEXT, -- JSON des conditions d'activation
                actions TEXT, -- JSON de la séquence d'actions
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0
            )
        """)
    
    def _create_profession_tables(self, conn: sqlite3.Connection):
        """Crée les tables liées aux professions"""
        
        # Table des niveaux de profession
        conn.execute("""
            CREATE TABLE IF NOT EXISTS character_professions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                profession_name TEXT NOT NULL,
                level INTEGER DEFAULT 1,
                experience BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE,
                UNIQUE(character_id, profession_name)
            )
        """)
        
        # Table des sessions de récolte
        conn.execute("""
            CREATE TABLE IF NOT EXISTS harvest_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                profession TEXT NOT NULL,
                map_id INTEGER,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration INTEGER, -- en secondes
                resources_harvested TEXT, -- JSON des ressources récoltées
                experience_gained INTEGER DEFAULT 0,
                total_value INTEGER DEFAULT 0, -- valeur estimée en kamas
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
        
        # Table des ressources
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT, -- wood, ore, cereals, etc.
                profession TEXT NOT NULL,
                level_required INTEGER DEFAULT 1,
                respawn_time INTEGER DEFAULT 300, -- en secondes
                locations TEXT, -- JSON des emplacements
                market_value INTEGER DEFAULT 0,
                last_price_update TIMESTAMP
            )
        """)
        
        # Table des recettes de craft
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                profession TEXT NOT NULL,
                level_required INTEGER,
                ingredients TEXT NOT NULL, -- JSON des ingrédients requis
                crafting_time INTEGER DEFAULT 0, -- en secondes
                experience_gained INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                cost_estimate INTEGER DEFAULT 0,
                profit_estimate INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_economy_tables(self, conn: sqlite3.Connection):
        """Crée les tables liées à l'économie"""
        
        # Table des prix du marché
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                item_name TEXT,
                category TEXT,
                quantity INTEGER DEFAULT 1,
                unit_price INTEGER NOT NULL,
                total_lots INTEGER DEFAULT 0,
                seller_name TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        # Table de l'historique des prix
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                server TEXT NOT NULL,
                date DATE NOT NULL,
                min_price INTEGER,
                max_price INTEGER,
                avg_price INTEGER,
                median_price INTEGER,
                total_volume INTEGER DEFAULT 0,
                price_change_percent REAL DEFAULT 0.0
            )
        """)
        
        # Table des transactions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                transaction_type TEXT NOT NULL, -- buy, sell, craft, drop
                item_id INTEGER,
                item_name TEXT,
                quantity INTEGER DEFAULT 1,
                unit_price INTEGER,
                total_price INTEGER,
                profit INTEGER DEFAULT 0, -- Pour les ventes
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
        
        # Table de l'inventaire
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                item_id INTEGER NOT NULL,
                item_name TEXT,
                quantity INTEGER DEFAULT 1,
                position INTEGER, -- Position dans l'inventaire
                acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acquisition_price INTEGER DEFAULT 0,
                estimated_value INTEGER DEFAULT 0,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
    
    def _create_navigation_tables(self, conn: sqlite3.Connection):
        """Crée les tables liées à la navigation"""
        
        # Table des chemins sauvegardés
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                start_map_id INTEGER NOT NULL,
                end_map_id INTEGER NOT NULL,
                path_data TEXT NOT NULL, -- JSON du chemin
                distance INTEGER,
                estimated_time INTEGER, -- en secondes
                path_type TEXT DEFAULT 'walking', -- walking, zaap, recall
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0
            )
        """)
        
        # Table des obstacles dynamiques
        conn.execute("""
            CREATE TABLE IF NOT EXISTS map_obstacles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                map_id INTEGER NOT NULL,
                cell_id INTEGER NOT NULL,
                obstacle_type TEXT, -- monster, player, resource, temporary
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                severity INTEGER DEFAULT 1 -- 1=minor, 5=critical
            )
        """)
        
        # Table des temps de trajet
        conn.execute("""
            CREATE TABLE IF NOT EXISTS travel_times (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_map_id INTEGER NOT NULL,
                end_map_id INTEGER NOT NULL,
                transport_type TEXT, -- walking, zaap, recall, dragoturkey
                duration INTEGER NOT NULL, -- en secondes
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_automation_tables(self, conn: sqlite3.Connection):
        """Crée les tables liées à l'automatisation"""
        
        # Table des tâches automatisées
        conn.execute("""
            CREATE TABLE IF NOT EXISTS automation_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                task_type TEXT NOT NULL, -- farming, leveling, crafting, trading
                character_id INTEGER NOT NULL,
                priority INTEGER DEFAULT 0,
                config TEXT, -- JSON de la configuration
                status TEXT DEFAULT 'pending', -- pending, running, completed, failed, paused
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                last_error TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
        
        # Table des routines quotidiennes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_routines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                routine_name TEXT NOT NULL,
                schedule_time TEXT, -- Format HH:MM
                days_of_week TEXT, -- JSON array [1,2,3,4,5,6,7]
                tasks TEXT, -- JSON array des tâches
                estimated_duration INTEGER, -- en minutes
                last_executed TIMESTAMP,
                next_execution TIMESTAMP,
                execution_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                enabled BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
        
        # Table des objectifs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS objectives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                objective_type TEXT NOT NULL, -- level, kamas, profession, achievement
                target_value INTEGER NOT NULL,
                current_value INTEGER DEFAULT 0,
                description TEXT,
                priority INTEGER DEFAULT 0,
                deadline TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                progress_percent REAL DEFAULT 0.0,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
    
    def _create_log_tables(self, conn: sqlite3.Connection):
        """Crée les tables de logging"""
        
        # Table des logs système
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL,
                module TEXT,
                message TEXT NOT NULL,
                character_id INTEGER,
                session_id TEXT,
                exception_details TEXT
            )
        """)
        
        # Table des événements de jeu
        conn.execute("""
            CREATE TABLE IF NOT EXISTS game_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                event_type TEXT NOT NULL, -- level_up, death, disconnection, achievement
                event_data TEXT, -- JSON des données de l'événement
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                map_id INTEGER,
                importance INTEGER DEFAULT 1, -- 1=low, 5=critical
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
        
        # Table des sessions de jeu
        conn.execute("""
            CREATE TABLE IF NOT EXISTS game_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_end TIMESTAMP,
                duration INTEGER, -- en secondes
                maps_visited INTEGER DEFAULT 0,
                combats_fought INTEGER DEFAULT 0,
                experience_gained INTEGER DEFAULT 0,
                kamas_gained INTEGER DEFAULT 0,
                items_gathered INTEGER DEFAULT 0,
                disconnection_count INTEGER DEFAULT 0,
                bot_detection_alerts INTEGER DEFAULT 0,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE
            )
        """)
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Crée les index pour optimiser les performances"""
        
        indexes = [
            # Index sur les tables principales
            "CREATE INDEX IF NOT EXISTS idx_characters_name ON characters(name)",
            "CREATE INDEX IF NOT EXISTS idx_characters_server ON characters(server)",
            "CREATE INDEX IF NOT EXISTS idx_characters_level ON characters(level)",
            
            # Index sur les combats
            "CREATE INDEX IF NOT EXISTS idx_combats_character_id ON combats(character_id)",
            "CREATE INDEX IF NOT EXISTS idx_combats_started_at ON combats(started_at)",
            "CREATE INDEX IF NOT EXISTS idx_combat_actions_combat_id ON combat_actions(combat_id)",
            
            # Index sur les professions
            "CREATE INDEX IF NOT EXISTS idx_char_professions_char_id ON character_professions(character_id)",
            "CREATE INDEX IF NOT EXISTS idx_harvest_sessions_char_id ON harvest_sessions(character_id)",
            "CREATE INDEX IF NOT EXISTS idx_harvest_sessions_profession ON harvest_sessions(profession)",
            
            # Index sur l'économie
            "CREATE INDEX IF NOT EXISTS idx_market_prices_item_id ON market_prices(item_id)",
            "CREATE INDEX IF NOT EXISTS idx_market_prices_server ON market_prices(server)",
            "CREATE INDEX IF NOT EXISTS idx_market_prices_recorded_at ON market_prices(recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_price_history_item_server ON price_history(item_id, server)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_character_id ON transactions(character_id)",
            "CREATE INDEX IF NOT EXISTS idx_inventory_character_id ON inventory(character_id)",
            
            # Index sur la navigation
            "CREATE INDEX IF NOT EXISTS idx_maps_coordinates ON maps(x, y)",
            "CREATE INDEX IF NOT EXISTS idx_maps_area_id ON maps(area_id)",
            "CREATE INDEX IF NOT EXISTS idx_saved_paths_start_end ON saved_paths(start_map_id, end_map_id)",
            
            # Index sur l'automatisation
            "CREATE INDEX IF NOT EXISTS idx_automation_tasks_character_id ON automation_tasks(character_id)",
            "CREATE INDEX IF NOT EXISTS idx_automation_tasks_status ON automation_tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_daily_routines_character_id ON daily_routines(character_id)",
            
            # Index sur les logs
            "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level)",
            "CREATE INDEX IF NOT EXISTS idx_game_events_character_id ON game_events(character_id)",
            "CREATE INDEX IF NOT EXISTS idx_game_events_timestamp ON game_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_game_sessions_character_id ON game_sessions(character_id)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.Error as e:
                self.logger.warning(f"Erreur lors de la création d'un index: {e}")
    
    def _update_database_version(self, conn: sqlite3.Connection):
        """Met à jour la version de la base de données"""
        
        # Calculer le hash du schéma
        schema_hash = self._calculate_schema_hash(conn)
        
        # Insérer ou mettre à jour la version
        conn.execute("""
            INSERT OR REPLACE INTO system_info 
            (id, version, updated_at, schema_hash) 
            VALUES (1, ?, CURRENT_TIMESTAMP, ?)
        """, (self.version, schema_hash))
        
        conn.commit()
    
    def _calculate_schema_hash(self, conn: sqlite3.Connection) -> str:
        """Calcule un hash du schéma de la base de données"""
        
        cursor = conn.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        
        schema_sql = ""
        for row in cursor:
            if row[0]:
                schema_sql += row[0] + "\n"
        
        return hashlib.md5(schema_sql.encode()).hexdigest()
    
    def backup_database(self, backup_name: Optional[str] = None) -> str:
        """
        Crée une sauvegarde de la base de données
        
        Args:
            backup_name: Nom optionnel de la sauvegarde
            
        Returns:
            Chemin du fichier de sauvegarde créé
        """
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        backup_file = self.backup_path / backup_name
        
        try:
            shutil.copy2(self.db_path, backup_file)
            
            # Mettre à jour le compteur de sauvegardes
            with self.connect() as conn:
                conn.execute("""
                    UPDATE system_info 
                    SET backup_count = backup_count + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """)
            
            self.logger.info(f"Sauvegarde créée: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    def restore_database(self, backup_path: str) -> bool:
        """
        Restaure la base de données depuis une sauvegarde
        
        Args:
            backup_path: Chemin vers le fichier de sauvegarde
            
        Returns:
            True si la restauration réussit, False sinon
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            self.logger.error(f"Fichier de sauvegarde introuvable: {backup_path}")
            return False
        
        try:
            # Créer une sauvegarde de sécurité avant restauration
            current_backup = self.backup_database("before_restore")
            
            # Remplacer la base de données courante
            shutil.copy2(backup_file, self.db_path)
            
            self.logger.info(f"Base de données restaurée depuis: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la restauration: {e}")
            return False
    
    def vacuum_database(self) -> bool:
        """
        Optimise et compacte la base de données
        
        Returns:
            True si l'opération réussit, False sinon
        """
        try:
            with self.connect() as conn:
                self.logger.info("Optimisation de la base de données...")
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                self.logger.info("Optimisation terminée")
                return True
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'optimisation: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de la base de données
        
        Returns:
            Dictionnaire contenant les statistiques
        """
        stats = {}
        
        try:
            with self.connect() as conn:
                # Taille du fichier
                stats['file_size'] = self.db_path.stat().st_size
                
                # Nombre de tables
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM sqlite_master WHERE type='table'
                """)
                stats['table_count'] = cursor.fetchone()[0]
                
                # Informations système
                cursor = conn.execute("SELECT * FROM system_info WHERE id = 1")
                system_info = cursor.fetchone()
                if system_info:
                    stats['version'] = system_info['version']
                    stats['created_at'] = system_info['created_at']
                    stats['updated_at'] = system_info['updated_at']
                    stats['backup_count'] = system_info['backup_count']
                
                # Compte des enregistrements par table principale
                tables_to_count = [
                    'characters', 'combats', 'market_prices', 'transactions',
                    'automation_tasks', 'system_logs', 'game_events'
                ]
                
                for table in tables_to_count:
                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        stats[f'{table}_count'] = cursor.fetchone()[0]
                    except sqlite3.Error:
                        stats[f'{table}_count'] = 0
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des stats: {e}")
            stats['error'] = str(e)
        
        return stats


# Fonctions utilitaires pour l'initialisation
def init_database(config_path: str = "config/bot_config.yaml") -> bool:
    """
    Initialise la base de données selon la configuration
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        True si l'initialisation réussit, False sinon
    """
    try:
        # Charger la configuration si elle existe
        db_path = "data/databases/dofus_bot.db"  # Valeur par défaut
        
        if Path(config_path).exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                db_path = config.get('database', {}).get('path', db_path)
        
        # Initialiser la base de données
        db_manager = DatabaseManager(db_path)
        success = db_manager.initialize_database()
        
        if success:
            print(f"[OK] Base de données initialisée avec succès: {db_path}")
            
            # Afficher les statistiques
            stats = db_manager.get_database_stats()
            print(f"[INFO] Tables créées: {stats.get('table_count', 0)}")
            print(f"[INFO] Taille: {stats.get('file_size', 0)} bytes")
        else:
            print("[ERROR] Erreur lors de l'initialisation de la base de données")
        
        return success
        
    except Exception as e:
        print(f"[ERROR] Erreur critique: {e}")
        return False


if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
    )
    
    # Initialiser la base de données
    success = init_database()
    exit(0 if success else 1)