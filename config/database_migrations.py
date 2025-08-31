"""
Système de migrations de base de données
=======================================

Ce module gère les migrations de la base de données pour maintenir
la compatibilité lors des mises à jour du bot. Il permet d'ajouter
de nouvelles tables, colonnes, index et de modifier la structure
existante de manière sûre.

Créé le: 2025-08-31
Version: 1.0.0
"""

import sqlite3
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class Migration:
    """Représente une migration de base de données"""
    version: str
    name: str
    description: str
    up_func: Callable[[sqlite3.Connection], None]
    down_func: Optional[Callable[[sqlite3.Connection], None]] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class DatabaseMigrationManager:
    """Gestionnaire des migrations de base de données"""
    
    def __init__(self, db_path: str = "data/databases/dofus_bot.db"):
        """
        Initialise le gestionnaire de migrations
        
        Args:
            db_path: Chemin vers la base de données
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self.migrations: List[Migration] = []
        self.current_version = "1.0.0"
        
        # Enregistrer toutes les migrations
        self._register_migrations()
    
    def connect(self) -> sqlite3.Connection:
        """Crée une connexion à la base de données"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Base de données non trouvée: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
    
    def _register_migrations(self):
        """Enregistre toutes les migrations disponibles"""
        
        # Migration 1.0.0 -> 1.0.1: Ajout de colonnes de performance
        self.migrations.append(Migration(
            version="1.0.1",
            name="add_performance_tracking",
            description="Ajouter colonnes de tracking de performance aux combats",
            up_func=self._migration_1_0_1_up,
            down_func=self._migration_1_0_1_down
        ))
        
        # Migration 1.0.1 -> 1.0.2: Optimisations index
        self.migrations.append(Migration(
            version="1.0.2",
            name="optimize_indexes",
            description="Optimiser les index pour de meilleures performances",
            up_func=self._migration_1_0_2_up,
            down_func=self._migration_1_0_2_down,
            dependencies=["1.0.1"]
        ))
        
        # Migration 1.0.2 -> 1.0.3: Système de cache
        self.migrations.append(Migration(
            version="1.0.3",
            name="add_cache_system",
            description="Ajouter tables de cache pour les données fréquemment utilisées",
            up_func=self._migration_1_0_3_up,
            down_func=self._migration_1_0_3_down,
            dependencies=["1.0.2"]
        ))
        
        # Migration 1.0.3 -> 1.1.0: Nouvelles fonctionnalités économie
        self.migrations.append(Migration(
            version="1.1.0",
            name="economy_enhancements",
            description="Améliorations système économique et market tracker",
            up_func=self._migration_1_1_0_up,
            down_func=self._migration_1_1_0_down,
            dependencies=["1.0.3"]
        ))
        
        # Migration 1.1.0 -> 1.2.0: Système de guildes
        self.migrations.append(Migration(
            version="1.2.0",
            name="guild_system",
            description="Ajouter support pour la gestion de guildes",
            up_func=self._migration_1_2_0_up,
            dependencies=["1.1.0"]
        ))
    
    def get_current_database_version(self) -> str:
        """
        Récupère la version actuelle de la base de données
        
        Returns:
            Version actuelle ou "1.0.0" si non définie
        """
        try:
            with self.connect() as conn:
                cursor = conn.execute("SELECT version FROM system_info WHERE id = 1")
                row = cursor.fetchone()
                return row['version'] if row else "1.0.0"
        except sqlite3.Error:
            return "1.0.0"
    
    def get_pending_migrations(self, target_version: Optional[str] = None) -> List[Migration]:
        """
        Récupère la liste des migrations en attente
        
        Args:
            target_version: Version cible (si None, utilise la dernière version)
            
        Returns:
            Liste des migrations à appliquer
        """
        current_version = self.get_current_database_version()
        target = target_version or self.current_version
        
        pending = []
        for migration in sorted(self.migrations, key=lambda m: m.version):
            if self._version_compare(current_version, migration.version) < 0:
                if self._version_compare(migration.version, target) <= 0:
                    pending.append(migration)
        
        return pending
    
    def _version_compare(self, v1: str, v2: str) -> int:
        """
        Compare deux versions
        
        Returns:
            -1 si v1 < v2, 0 si v1 == v2, 1 si v1 > v2
        """
        def version_tuple(v):
            return tuple(map(int, v.split('.')))
        
        t1, t2 = version_tuple(v1), version_tuple(v2)
        return -1 if t1 < t2 else (1 if t1 > t2 else 0)
    
    def migrate(self, target_version: Optional[str] = None, dry_run: bool = False) -> bool:
        """
        Exécute les migrations vers la version cible
        
        Args:
            target_version: Version cible
            dry_run: Si True, simule les migrations sans les exécuter
            
        Returns:
            True si toutes les migrations réussissent
        """
        pending_migrations = self.get_pending_migrations(target_version)
        
        if not pending_migrations:
            self.logger.info("Aucune migration en attente")
            return True
        
        if dry_run:
            self.logger.info("=== SIMULATION DES MIGRATIONS ===")
            for migration in pending_migrations:
                self.logger.info(f"Migration {migration.version}: {migration.description}")
            return True
        
        self.logger.info(f"Exécution de {len(pending_migrations)} migrations...")
        
        # Créer une sauvegarde avant les migrations
        backup_path = self._create_backup()
        self.logger.info(f"Sauvegarde créée: {backup_path}")
        
        success_count = 0
        try:
            with self.connect() as conn:
                for migration in pending_migrations:
                    if self._execute_migration(conn, migration):
                        success_count += 1
                        self._record_migration(conn, migration)
                        self.logger.info(f"✅ Migration {migration.version} appliquée")
                    else:
                        self.logger.error(f"❌ Échec migration {migration.version}")
                        break
                
                if success_count == len(pending_migrations):
                    conn.commit()
                    self.logger.info("🎉 Toutes les migrations ont réussi!")
                    return True
                else:
                    conn.rollback()
                    self.logger.error("❌ Migrations interrompues - rollback effectué")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Erreur critique lors des migrations: {e}")
            return False
    
    def _execute_migration(self, conn: sqlite3.Connection, migration: Migration) -> bool:
        """
        Exécute une migration spécifique
        
        Args:
            conn: Connexion à la base de données
            migration: Migration à exécuter
            
        Returns:
            True si la migration réussit
        """
        try:
            # Vérifier les dépendances
            if not self._check_dependencies(migration):
                self.logger.error(f"Dépendances non satisfaites pour {migration.version}")
                return False
            
            # Exécuter la migration
            migration.up_func(conn)
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur dans la migration {migration.version}: {e}")
            return False
    
    def _check_dependencies(self, migration: Migration) -> bool:
        """Vérifie que les dépendances d'une migration sont satisfaites"""
        current_version = self.get_current_database_version()
        
        for dep_version in migration.dependencies:
            if self._version_compare(current_version, dep_version) < 0:
                return False
        
        return True
    
    def _record_migration(self, conn: sqlite3.Connection, migration: Migration):
        """Enregistre l'exécution d'une migration"""
        
        # Enregistrer dans la table des migrations
        conn.execute("""
            INSERT INTO migrations (version, migration_name, executed_at, success)
            VALUES (?, ?, CURRENT_TIMESTAMP, TRUE)
        """, (migration.version, migration.name))
        
        # Mettre à jour la version système
        conn.execute("""
            UPDATE system_info 
            SET version = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (migration.version,))
    
    def _create_backup(self) -> str:
        """Crée une sauvegarde avant migration"""
        backup_name = f"migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_dir = self.db_path.parent / "../backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / backup_name
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        return str(backup_path)
    
    def rollback_migration(self, target_version: str) -> bool:
        """
        Effectue un rollback vers une version antérieure
        
        Args:
            target_version: Version vers laquelle revenir
            
        Returns:
            True si le rollback réussit
        """
        current_version = self.get_current_database_version()
        
        if self._version_compare(target_version, current_version) >= 0:
            self.logger.warning("Version cible supérieure ou égale à la version actuelle")
            return True
        
        # Récupérer les migrations à annuler (dans l'ordre inverse)
        rollback_migrations = []
        for migration in reversed(self.migrations):
            if (self._version_compare(target_version, migration.version) < 0 and
                self._version_compare(migration.version, current_version) <= 0):
                rollback_migrations.append(migration)
        
        if not rollback_migrations:
            self.logger.info("Aucune migration à annuler")
            return True
        
        self.logger.warning(f"🔄 Rollback de {len(rollback_migrations)} migrations...")
        
        # Créer une sauvegarde
        backup_path = self._create_backup()
        self.logger.info(f"Sauvegarde créée: {backup_path}")
        
        try:
            with self.connect() as conn:
                for migration in rollback_migrations:
                    if migration.down_func:
                        try:
                            migration.down_func(conn)
                            self.logger.info(f"✅ Rollback {migration.version}")
                        except Exception as e:
                            self.logger.error(f"❌ Erreur rollback {migration.version}: {e}")
                            return False
                    else:
                        self.logger.warning(f"⚠️ Pas de rollback pour {migration.version}")
                
                # Mettre à jour la version
                conn.execute("""
                    UPDATE system_info 
                    SET version = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (target_version,))
                
                conn.commit()
                self.logger.info(f"🎉 Rollback vers {target_version} terminé")
                return True
                
        except Exception as e:
            self.logger.error(f"Erreur critique lors du rollback: {e}")
            return False
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des migrations
        
        Returns:
            Liste des migrations exécutées
        """
        try:
            with self.connect() as conn:
                cursor = conn.execute("""
                    SELECT version, migration_name, executed_at, success
                    FROM migrations 
                    ORDER BY executed_at DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except sqlite3.Error as e:
            self.logger.error(f"Erreur récupération historique: {e}")
            return []
    
    # =====================================
    # MIGRATIONS SPÉCIFIQUES
    # =====================================
    
    def _migration_1_0_1_up(self, conn: sqlite3.Connection):
        """Migration 1.0.1: Colonnes de performance"""
        
        # Ajouter colonnes de performance aux combats
        conn.execute("""
            ALTER TABLE combats 
            ADD COLUMN average_turn_time REAL DEFAULT 0.0
        """)
        
        conn.execute("""
            ALTER TABLE combats 
            ADD COLUMN ai_decision_time REAL DEFAULT 0.0
        """)
        
        conn.execute("""
            ALTER TABLE combats 
            ADD COLUMN spell_success_rate REAL DEFAULT 1.0
        """)
        
        # Ajouter table de performance des sorts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spell_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                spell_id INTEGER NOT NULL,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                total_damage INTEGER DEFAULT 0,
                average_damage REAL DEFAULT 0.0,
                critical_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                FOREIGN KEY (character_id) REFERENCES characters (id),
                UNIQUE(character_id, spell_id)
            )
        """)
    
    def _migration_1_0_1_down(self, conn: sqlite3.Connection):
        """Rollback 1.0.1"""
        # Note: SQLite ne supporte pas DROP COLUMN, donc on recrée la table
        conn.execute("DROP TABLE IF EXISTS spell_performance")
    
    def _migration_1_0_2_up(self, conn: sqlite3.Connection):
        """Migration 1.0.2: Optimisation des index"""
        
        # Index composites pour de meilleures performances
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_combats_char_date ON combats(character_id, started_at)",
            "CREATE INDEX IF NOT EXISTS idx_market_prices_item_date ON market_prices(item_id, recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_harvest_char_prof ON harvest_sessions(character_id, profession)",
            "CREATE INDEX IF NOT EXISTS idx_automation_status_priority ON automation_tasks(status, priority)",
            "CREATE INDEX IF NOT EXISTS idx_game_events_char_type ON game_events(character_id, event_type)",
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    def _migration_1_0_2_down(self, conn: sqlite3.Connection):
        """Rollback 1.0.2"""
        indexes_to_drop = [
            "idx_combats_char_date", "idx_market_prices_item_date",
            "idx_harvest_char_prof", "idx_automation_status_priority",
            "idx_game_events_char_type"
        ]
        
        for index_name in indexes_to_drop:
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")
    
    def _migration_1_0_3_up(self, conn: sqlite3.Connection):
        """Migration 1.0.3: Système de cache"""
        
        # Table de cache générique
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_data (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index sur les caches
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_category ON cache_data(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_data(expires_at)")
        
        # Table de cache des chemins
        conn.execute("""
            CREATE TABLE IF NOT EXISTS path_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_map INTEGER NOT NULL,
                end_map INTEGER NOT NULL,
                path_hash TEXT NOT NULL,
                path_data TEXT NOT NULL,
                cost INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                UNIQUE(start_map, end_map, path_hash)
            )
        """)
    
    def _migration_1_0_3_down(self, conn: sqlite3.Connection):
        """Rollback 1.0.3"""
        conn.execute("DROP TABLE IF EXISTS cache_data")
        conn.execute("DROP TABLE IF EXISTS path_cache")
    
    def _migration_1_1_0_up(self, conn: sqlite3.Connection):
        """Migration 1.1.0: Améliorations économie"""
        
        # Ajouter colonnes à market_prices
        conn.execute("ALTER TABLE market_prices ADD COLUMN trend TEXT DEFAULT 'stable'")
        conn.execute("ALTER TABLE market_prices ADD COLUMN volatility REAL DEFAULT 0.0")
        conn.execute("ALTER TABLE market_prices ADD COLUMN prediction_confidence REAL DEFAULT 0.0")
        
        # Table d'analyse de marché
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                server TEXT NOT NULL,
                analysis_date DATE NOT NULL,
                trend_7d TEXT, -- up, down, stable
                trend_30d TEXT,
                volatility_score REAL DEFAULT 0.0,
                buy_recommendation TEXT, -- strong_buy, buy, hold, sell, strong_sell
                sell_recommendation TEXT,
                profit_potential REAL DEFAULT 0.0,
                risk_level TEXT DEFAULT 'medium',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(item_id, server, analysis_date)
            )
        """)
        
        # Table des alertes de marché
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                item_id INTEGER NOT NULL,
                alert_type TEXT NOT NULL, -- price_drop, price_spike, volume_change
                threshold_value INTEGER,
                current_value INTEGER,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (character_id) REFERENCES characters (id)
            )
        """)
    
    def _migration_1_1_0_down(self, conn: sqlite3.Connection):
        """Rollback 1.1.0"""
        conn.execute("DROP TABLE IF EXISTS market_analysis")
        conn.execute("DROP TABLE IF EXISTS market_alerts")
        # Note: Les colonnes ajoutées à market_prices restent (limitation SQLite)
    
    def _migration_1_2_0_up(self, conn: sqlite3.Connection):
        """Migration 1.2.0: Système de guildes"""
        
        # Table des guildes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS guilds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                level INTEGER DEFAULT 1,
                experience BIGINT DEFAULT 0,
                member_count INTEGER DEFAULT 0,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table des membres de guilde
        conn.execute("""
            CREATE TABLE IF NOT EXISTS guild_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                character_id INTEGER NOT NULL,
                rank TEXT DEFAULT 'member',
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                contribution_points INTEGER DEFAULT 0,
                last_active TIMESTAMP,
                FOREIGN KEY (guild_id) REFERENCES guilds (id) ON DELETE CASCADE,
                FOREIGN KEY (character_id) REFERENCES characters (id) ON DELETE CASCADE,
                UNIQUE(guild_id, character_id)
            )
        """)
        
        # Table des activités de guilde
        conn.execute("""
            CREATE TABLE IF NOT EXISTS guild_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                activity_type TEXT NOT NULL, -- member_join, member_leave, level_up
                character_id INTEGER,
                description TEXT,
                experience_gained INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (guild_id) REFERENCES guilds (id) ON DELETE CASCADE,
                FOREIGN KEY (character_id) REFERENCES characters (id)
            )
        """)
        
        # Ajouter colonne guild_id aux personnages
        conn.execute("ALTER TABLE characters ADD COLUMN guild_id INTEGER")
        conn.execute("ALTER TABLE characters ADD COLUMN guild_rank TEXT")
    
    # Pas de rollback pour 1.2.0 (trop complexe avec les FK)
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Récupère les informations complètes sur la base de données
        
        Returns:
            Informations sur la version, migrations, etc.
        """
        info = {
            'current_version': self.get_current_database_version(),
            'target_version': self.current_version,
            'pending_migrations': len(self.get_pending_migrations()),
            'total_migrations': len(self.migrations),
            'migration_history': self.get_migration_history()[:5],  # 5 dernières
            'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0
        }
        
        return info


# Fonctions utilitaires pour l'utilisation en CLI
def main():
    """Interface en ligne de commande pour les migrations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gestionnaire de migrations DOFUS Bot")
    parser.add_argument('--migrate', action='store_true', help='Exécuter les migrations')
    parser.add_argument('--rollback', metavar='VERSION', help='Rollback vers une version')
    parser.add_argument('--info', action='store_true', help='Informations sur la base')
    parser.add_argument('--history', action='store_true', help='Historique des migrations')
    parser.add_argument('--dry-run', action='store_true', help='Simulation des migrations')
    parser.add_argument('--target', metavar='VERSION', help='Version cible')
    parser.add_argument('--db-path', default='data/databases/dofus_bot.db', help='Chemin DB')
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    manager = DatabaseMigrationManager(args.db_path)
    
    if args.info:
        info = manager.get_database_info()
        print("📊 INFORMATIONS BASE DE DONNÉES")
        print(f"Version actuelle: {info['current_version']}")
        print(f"Version cible: {info['target_version']}")
        print(f"Migrations en attente: {info['pending_migrations']}")
        print(f"Taille DB: {info['database_size'] / 1024 / 1024:.2f} MB")
        
    elif args.history:
        history = manager.get_migration_history()
        print("📜 HISTORIQUE DES MIGRATIONS")
        for migration in history:
            status = "✅" if migration['success'] else "❌"
            print(f"{status} {migration['version']} - {migration['migration_name']} "
                  f"({migration['executed_at']})")
    
    elif args.migrate:
        success = manager.migrate(args.target, args.dry_run)
        exit(0 if success else 1)
    
    elif args.rollback:
        success = manager.rollback_migration(args.rollback)
        exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()