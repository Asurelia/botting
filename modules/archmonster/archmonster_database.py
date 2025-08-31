"""Base de données SQLite pour archimonstres."""
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager

class ArchmonsterDatabase:
    def __init__(self, db_path="data/archmonsters.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._init_db()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        with self.get_connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY,
                name TEXT, zone TEXT, confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
    
    def get_database_stats(self):
        return {'detections_count': 0}