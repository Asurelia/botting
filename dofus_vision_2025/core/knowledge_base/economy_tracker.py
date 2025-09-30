"""
DOFUS Economy Tracker - Surveillance du Marche et Prix
Suivi temps reel des prix, tendances et opportunites economiques
Approche 100% vision - Analyse automatique hotel de vente via OCR
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import statistics
import requests
from collections import defaultdict

logger = logging.getLogger(__name__)

class PriceCategory(Enum):
    """Categories de prix pour analyse"""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"

class TrendDirection(Enum):
    """Direction des tendances de prix"""
    FALLING = "falling"
    STABLE = "stable"
    RISING = "rising"
    VOLATILE = "volatile"

class MarketOpportunity(Enum):
    """Types d'opportunites de marche"""
    BUY_LOW = "buy_low"
    SELL_HIGH = "sell_high"
    ARBITRAGE = "arbitrage"
    CRAFT_PROFIT = "craft_profit"
    RESOURCE_SPIKE = "resource_spike"

@dataclass
class PriceEntry:
    """Entree de prix pour un objet"""
    item_id: int
    item_name: str
    server: str
    price: int
    quantity: int
    timestamp: datetime
    seller_name: Optional[str] = None
    category: PriceCategory = PriceCategory.NORMAL

@dataclass
class PriceTrend:
    """Tendance de prix pour un objet"""
    item_id: int
    item_name: str
    server: str

    # Prix actuels
    current_min_price: int
    current_avg_price: int
    current_max_price: int

    # Historique (7 jours)
    avg_price_7d: int
    min_price_7d: int
    max_price_7d: int

    # Tendances
    trend_direction: TrendDirection
    price_change_percent: float
    volatility_score: float

    # Volume
    daily_sales_avg: int
    total_quantity_available: int

    # Derniere mise a jour
    last_updated: datetime

@dataclass
class MarketOpportunityAlert:
    """Alerte d'opportunite de marche"""
    opportunity_type: MarketOpportunity
    item_id: int
    item_name: str
    server: str

    # Details opportunite
    current_price: int
    target_price: int
    profit_potential: int
    profit_percent: float

    # Recommandations
    action: str  # "buy", "sell", "craft", "wait"
    urgency: str  # "low", "medium", "high", "critical"

    # Validite
    expires_at: datetime
    confidence_score: float

@dataclass
class CraftProfitAnalysis:
    """Analyse rentabilite craft"""
    recipe_id: int
    recipe_name: str
    craft_cost: int
    sell_price: int
    profit: int
    profit_margin: float

    # Details ingredients
    ingredients_cost: Dict[str, int]
    ingredients_availability: Dict[str, int]

    # Facteurs
    craft_time_minutes: int
    success_rate: float
    market_demand: str  # "low", "medium", "high"

class DofusEconomyTracker:
    """
    Tracker economique DOFUS complet
    Surveillance marche + detection opportunites + alertes
    """

    def __init__(self, db_path: str = "data/dofus_economy.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Cache des donnees
        self.price_cache: Dict[Tuple[int, str], List[PriceEntry]] = defaultdict(list)
        self.trend_cache: Dict[Tuple[int, str], PriceTrend] = {}
        self.opportunities_cache: List[MarketOpportunityAlert] = []

        # Configuration
        self.servers = ["Julith", "Draconiros", "Hécate", "Rubilax"]
        self.tracked_items: Set[int] = set()

        # OCR et vision pour lecture prix
        self.setup_price_ocr()

        self._init_database()
        self._load_tracked_items()

        logger.info("DofusEconomyTracker initialise")

    def setup_price_ocr(self):
        """Configure OCR pour lecture prix hotel de vente"""
        try:
            import pytesseract
            # Configuration optimisee pour chiffres et noms items
            self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\'éèêëàâäùûüôöîïç '
            logger.info("OCR prix configure")
        except ImportError:
            logger.warning("OCR non disponible pour lecture prix automatique")

    def _init_database(self):
        """Initialise la base de donnees economique"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table des prix
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER,
                item_name TEXT,
                server TEXT,
                price INTEGER,
                quantity INTEGER,
                seller_name TEXT,
                timestamp DATETIME,
                category TEXT
            )
        ''')

        # Table des tendances
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_trends (
                item_id INTEGER,
                server TEXT,
                current_min_price INTEGER,
                current_avg_price INTEGER,
                trend_direction TEXT,
                price_change_percent REAL,
                last_updated DATETIME,
                PRIMARY KEY (item_id, server)
            )
        ''')

        # Table des opportunites
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                opportunity_type TEXT,
                item_id INTEGER,
                server TEXT,
                current_price INTEGER,
                profit_potential INTEGER,
                action TEXT,
                urgency TEXT,
                created_at DATETIME,
                expires_at DATETIME
            )
        ''')

        # Index pour performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_item_server ON price_history(item_id, server)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_history(timestamp)')

        conn.commit()
        conn.close()

    def _load_tracked_items(self):
        """Charge la liste des objets suivis"""
        # Items populaires par defaut
        default_tracked = [
            # Ressources de base
            289, 371, 401, 423, 445,  # Bles, orges, avoines, houblons, lins
            312, 313, 314, 315, 316,  # Minerais fer, cuivre, bronze, kobalt, manganèse

            # Ressources rares
            1301, 1302, 1303, 1304,  # Minerais précieux
            2326, 2327, 2328, 2329,  # Bois rares

            # Parchemins caracteristiques
            10207, 10208, 10209, 10210,  # PA, PM, PO, Invocations

            # Consommables
            544, 515, 545, 546,  # Potions recall, pain

            # Items échangeables populaires
            7303, 7304, 7305,  # Panoplies populaires
        ]

        self.tracked_items.update(default_tracked)
        logger.info(f"Items suivis: {len(self.tracked_items)}")

    def add_price_entry(self, price_entry: PriceEntry):
        """Ajoute une entree de prix"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO price_history
            (item_id, item_name, server, price, quantity, seller_name, timestamp, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            price_entry.item_id, price_entry.item_name, price_entry.server,
            price_entry.price, price_entry.quantity, price_entry.seller_name,
            price_entry.timestamp, price_entry.category.value
        ))

        conn.commit()
        conn.close()

        # Mise a jour cache
        key = (price_entry.item_id, price_entry.server)
        self.price_cache[key].append(price_entry)

        # Nettoyage cache (garde 100 dernieres entrees par item/serveur)
        if len(self.price_cache[key]) > 100:
            self.price_cache[key] = self.price_cache[key][-100:]

    def get_price_history(self, item_id: int, server: str, days: int = 7) -> List[PriceEntry]:
        """Recupere l'historique des prix"""
        since_date = datetime.now() - timedelta(days=days)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT item_id, item_name, server, price, quantity, seller_name, timestamp, category
            FROM price_history
            WHERE item_id = ? AND server = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (item_id, server, since_date))

        entries = []
        for row in cursor.fetchall():
            entry = PriceEntry(
                item_id=row[0], item_name=row[1], server=row[2],
                price=row[3], quantity=row[4], seller_name=row[5],
                timestamp=datetime.fromisoformat(row[6]),
                category=PriceCategory(row[7])
            )
            entries.append(entry)

        conn.close()
        return entries

    def calculate_price_trend(self, item_id: int, server: str) -> Optional[PriceTrend]:
        """Calcule la tendance de prix pour un objet"""
        prices_7d = self.get_price_history(item_id, server, 7)
        prices_1d = self.get_price_history(item_id, server, 1)

        if not prices_7d:
            return None

        # Calculs statistiques
        all_prices = [p.price for p in prices_7d]
        recent_prices = [p.price for p in prices_1d] if prices_1d else all_prices[-10:]

        current_min = min(recent_prices) if recent_prices else 0
        current_avg = int(statistics.mean(recent_prices)) if recent_prices else 0
        current_max = max(recent_prices) if recent_prices else 0

        avg_7d = int(statistics.mean(all_prices))
        min_7d = min(all_prices)
        max_7d = max(all_prices)

        # Calcul tendance
        if len(all_prices) >= 2:
            old_avg = statistics.mean(all_prices[:len(all_prices)//2])
            new_avg = statistics.mean(all_prices[len(all_prices)//2:])
            price_change = ((new_avg - old_avg) / old_avg) * 100
        else:
            price_change = 0.0

        # Direction tendance
        if abs(price_change) < 5:
            trend_direction = TrendDirection.STABLE
        elif price_change > 0:
            trend_direction = TrendDirection.RISING
        else:
            trend_direction = TrendDirection.FALLING

        # Volatilite
        volatility = (statistics.stdev(all_prices) / avg_7d) * 100 if avg_7d > 0 else 0

        if volatility > 30:
            trend_direction = TrendDirection.VOLATILE

        # Volume
        daily_sales = len(prices_1d)
        total_quantity = sum(p.quantity for p in prices_1d)

        trend = PriceTrend(
            item_id=item_id,
            item_name=prices_7d[0].item_name,
            server=server,
            current_min_price=current_min,
            current_avg_price=current_avg,
            current_max_price=current_max,
            avg_price_7d=avg_7d,
            min_price_7d=min_7d,
            max_price_7d=max_7d,
            trend_direction=trend_direction,
            price_change_percent=price_change,
            volatility_score=volatility,
            daily_sales_avg=daily_sales,
            total_quantity_available=total_quantity,
            last_updated=datetime.now()
        )

        # Sauvegarde en cache et DB
        self.trend_cache[(item_id, server)] = trend
        self._save_trend_to_db(trend)

        return trend

    def _save_trend_to_db(self, trend: PriceTrend):
        """Sauvegarde tendance en base"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO price_trends
            (item_id, server, current_min_price, current_avg_price,
             trend_direction, price_change_percent, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            trend.item_id, trend.server, trend.current_min_price,
            trend.current_avg_price, trend.trend_direction.value,
            trend.price_change_percent, trend.last_updated
        ))

        conn.commit()
        conn.close()

    def detect_market_opportunities(self, server: str) -> List[MarketOpportunityAlert]:
        """Detecte les opportunites de marche"""
        opportunities = []

        for item_id in self.tracked_items:
            trend = self.calculate_price_trend(item_id, server)
            if not trend:
                continue

            # Opportunite achat bas prix
            if (trend.current_min_price < trend.avg_price_7d * 0.7 and
                trend.trend_direction != TrendDirection.FALLING):

                opportunity = MarketOpportunityAlert(
                    opportunity_type=MarketOpportunity.BUY_LOW,
                    item_id=item_id,
                    item_name=trend.item_name,
                    server=server,
                    current_price=trend.current_min_price,
                    target_price=trend.avg_price_7d,
                    profit_potential=trend.avg_price_7d - trend.current_min_price,
                    profit_percent=((trend.avg_price_7d - trend.current_min_price) / trend.current_min_price) * 100,
                    action="buy",
                    urgency="medium",
                    expires_at=datetime.now() + timedelta(hours=6),
                    confidence_score=0.7
                )
                opportunities.append(opportunity)

            # Opportunite vente prix eleve
            if (trend.current_max_price > trend.avg_price_7d * 1.3 and
                trend.trend_direction == TrendDirection.RISING):

                opportunity = MarketOpportunityAlert(
                    opportunity_type=MarketOpportunity.SELL_HIGH,
                    item_id=item_id,
                    item_name=trend.item_name,
                    server=server,
                    current_price=trend.current_max_price,
                    target_price=trend.avg_price_7d,
                    profit_potential=trend.current_max_price - trend.avg_price_7d,
                    profit_percent=((trend.current_max_price - trend.avg_price_7d) / trend.avg_price_7d) * 100,
                    action="sell",
                    urgency="high",
                    expires_at=datetime.now() + timedelta(hours=3),
                    confidence_score=0.8
                )
                opportunities.append(opportunity)

        # Tri par potentiel profit
        opportunities.sort(key=lambda x: x.profit_potential, reverse=True)
        self.opportunities_cache = opportunities[:20]  # Top 20

        return self.opportunities_cache

    def analyze_craft_profitability(self, recipe_id: int) -> Optional[CraftProfitAnalysis]:
        """Analyse la rentabilite d'une recette de craft"""
        # Importation base donnees recettes
        try:
            from .spells_database import get_spells_database
            # Cette fonction serait etendue avec une vraie base de recettes
            # Pour l'instant, exemple avec donnees fictives

            example_analysis = CraftProfitAnalysis(
                recipe_id=recipe_id,
                recipe_name="Epee en Fer",
                craft_cost=5000,
                sell_price=7500,
                profit=2500,
                profit_margin=50.0,
                ingredients_cost={"Minerai Fer": 3000, "Bois Frene": 2000},
                ingredients_availability={"Minerai Fer": 150, "Bois Frene": 200},
                craft_time_minutes=5,
                success_rate=0.95,
                market_demand="high"
            )

            return example_analysis

        except Exception as e:
            logger.error(f"Erreur analyse craft: {e}")
            return None

    def get_server_economy_overview(self, server: str) -> Dict[str, Any]:
        """Vue d'ensemble de l'economie d'un serveur"""
        trends = []
        total_opportunities = 0
        avg_inflation = 0.0

        for item_id in list(self.tracked_items)[:20]:  # Echantillon
            trend = self.calculate_price_trend(item_id, server)
            if trend:
                trends.append(trend)

        if trends:
            avg_inflation = statistics.mean([t.price_change_percent for t in trends])
            total_opportunities = len(self.detect_market_opportunities(server))

        overview = {
            "server": server,
            "total_tracked_items": len(self.tracked_items),
            "active_trends": len(trends),
            "average_inflation": round(avg_inflation, 2),
            "market_opportunities": total_opportunities,
            "economy_health": "good" if -5 <= avg_inflation <= 10 else "unstable",
            "last_updated": datetime.now().isoformat(),
            "top_opportunities": self.opportunities_cache[:5]
        }

        return overview

    def export_market_report(self, server: str, output_path: str):
        """Exporte un rapport de marche complet"""
        overview = self.get_server_economy_overview(server)
        opportunities = self.detect_market_opportunities(server)

        report = {
            "server": server,
            "generated_at": datetime.now().isoformat(),
            "overview": overview,
            "opportunities": [asdict(opp) for opp in opportunities],
            "recommendations": {
                "investment_focus": "resources" if overview["average_inflation"] > 5 else "equipment",
                "market_timing": "buy" if overview["average_inflation"] < 0 else "sell",
                "risk_level": "high" if overview["economy_health"] == "unstable" else "medium"
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Rapport marche exporte: {output_path}")

# Instance globale
_economy_tracker_instance = None

def get_economy_tracker() -> DofusEconomyTracker:
    """Retourne l'instance singleton du tracker economique"""
    global _economy_tracker_instance
    if _economy_tracker_instance is None:
        _economy_tracker_instance = DofusEconomyTracker()
    return _economy_tracker_instance

# Test du module
if __name__ == "__main__":
    tracker = DofusEconomyTracker()

    # Ajout donnees test
    test_entry = PriceEntry(
        item_id=289,
        item_name="Ble",
        server="Julith",
        price=150,
        quantity=10,
        timestamp=datetime.now(),
        category=PriceCategory.NORMAL
    )

    tracker.add_price_entry(test_entry)

    # Test tendance
    trend = tracker.calculate_price_trend(289, "Julith")
    if trend:
        print(f"Tendance Ble: {trend.trend_direction.value}, Prix moyen: {trend.current_avg_price}")

    # Test opportunites
    opportunities = tracker.detect_market_opportunities("Julith")
    print(f"Opportunites detectees: {len(opportunities)}")

    # Vue d'ensemble
    overview = tracker.get_server_economy_overview("Julith")
    print(f"Economie serveur: {overview['economy_health']}")

    # Export rapport
    tracker.export_market_report("Julith", "test_market_report.json")