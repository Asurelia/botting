"""
Analyseur de marché intelligent pour Dofus - Scan HDV, prédictions prix, arbitrage
Module d'analyse économique avec Machine Learning et détection d'opportunités
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import threading
import time

@dataclass
class MarketItem:
    """Représente un item sur le marché avec ses données"""
    item_id: int
    name: str
    current_price: float
    quantity_available: int
    server_id: str
    timestamp: datetime
    category: str = ""
    level: int = 0
    rarity: str = ""
    historical_avg: float = 0.0
    volatility: float = 0.0
    trend: str = "stable"  # up, down, stable

@dataclass
class ArbitrageOpportunity:
    """Opportunité d'arbitrage détectée"""
    item_id: int
    buy_server: str
    sell_server: str
    buy_price: float
    sell_price: float
    profit_margin: float
    roi_percentage: float
    risk_level: str  # low, medium, high
    confidence: float
    estimated_volume: int

@dataclass
class PricePrediction:
    """Prédiction de prix avec métriques de confiance"""
    item_id: int
    predicted_price: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: timedelta
    accuracy_score: float
    trend_direction: str
    volatility_forecast: float

class MarketDatabase:
    """Gestionnaire de base de données pour les données de marché"""
    
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialise les tables de la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table des prix historiques
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                server_id TEXT NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                category TEXT,
                level INTEGER,
                rarity TEXT
            )
        ''')
        
        # Table des prédictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                prediction_date DATETIME NOT NULL,
                target_date DATETIME NOT NULL,
                accuracy REAL,
                model_used TEXT
            )
        ''')
        
        # Table des opportunités d'arbitrage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                buy_server TEXT NOT NULL,
                sell_server TEXT NOT NULL,
                buy_price REAL NOT NULL,
                sell_price REAL NOT NULL,
                profit_margin REAL NOT NULL,
                roi_percentage REAL NOT NULL,
                detected_at DATETIME NOT NULL,
                executed BOOLEAN DEFAULT FALSE,
                actual_profit REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_market_data(self, items: List[MarketItem]):
        """Stocke les données de marché"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in items:
            cursor.execute('''
                INSERT INTO price_history 
                (item_id, server_id, price, quantity, timestamp, category, level, rarity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.item_id, item.server_id, item.current_price, 
                item.quantity_available, item.timestamp, item.category, 
                item.level, item.rarity
            ))
        
        conn.commit()
        conn.close()
    
    def get_price_history(self, item_id: int, server_id: str = None, 
                         days: int = 30) -> pd.DataFrame:
        """Récupère l'historique des prix"""
        conn = sqlite3.connect(self.db_path)
        
        since_date = datetime.now() - timedelta(days=days)
        
        if server_id:
            query = '''
                SELECT * FROM price_history 
                WHERE item_id = ? AND server_id = ? AND timestamp >= ?
                ORDER BY timestamp
            '''
            params = (item_id, server_id, since_date)
        else:
            query = '''
                SELECT * FROM price_history 
                WHERE item_id = ? AND timestamp >= ?
                ORDER BY timestamp
            '''
            params = (item_id, since_date)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df

class MLPricePredictor:
    """Prédicteur de prix utilisant le Machine Learning"""
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.trained_items = set()
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prépare les features pour le ML"""
        if df.empty:
            return np.array([])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        features = []
        
        # Features temporelles
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Features de prix
        df['price_lag_1'] = df['price'].shift(1)
        df['price_lag_2'] = df['price'].shift(2)
        df['price_ma_3'] = df['price'].rolling(window=3).mean()
        df['price_ma_7'] = df['price'].rolling(window=7).mean()
        
        # Volatilité
        df['price_volatility'] = df['price'].rolling(window=5).std()
        
        # Volume features
        df['quantity_ma_3'] = df['quantity'].rolling(window=3).mean()
        df['quantity_volatility'] = df['quantity'].rolling(window=5).std()
        
        # Ratio prix/quantité
        df['price_quantity_ratio'] = df['price'] / (df['quantity'] + 1)
        
        feature_columns = [
            'hour', 'day_of_week', 'day_of_month',
            'price_lag_1', 'price_lag_2', 'price_ma_3', 'price_ma_7',
            'price_volatility', 'quantity_ma_3', 'quantity_volatility',
            'price_quantity_ratio'
        ]
        
        return df[feature_columns].fillna(0).values
    
    def train_model(self, item_id: int, market_db: MarketDatabase):
        """Entraîne le modèle pour un item spécifique"""
        df = market_db.get_price_history(item_id, days=90)
        
        if len(df) < 20:  # Pas assez de données
            return False
        
        features = self.prepare_features(df)
        if features.size == 0:
            return False
        
        prices = df['price'].values
        
        # On prédit le prix suivant
        X = features[:-1]
        y = prices[1:]
        
        if len(X) < 10:
            return False
        
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement des modèles
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
            except Exception as e:
                print(f"Erreur lors de l'entraînement du modèle {name}: {e}")
                return False
        
        self.trained_items.add(item_id)
        return True
    
    def predict_price(self, item_id: int, market_db: MarketDatabase, 
                     horizon_days: int = 1) -> Optional[PricePrediction]:
        """Prédit le prix futur d'un item"""
        if item_id not in self.trained_items:
            if not self.train_model(item_id, market_db):
                return None
        
        df = market_db.get_price_history(item_id, days=30)
        if len(df) < 5:
            return None
        
        features = self.prepare_features(df)
        if features.size == 0:
            return None
        
        # Utilisation des dernières features
        last_features = features[-1:].reshape(1, -1)
        last_features_scaled = self.scaler.transform(last_features)
        
        # Prédictions des différents modèles
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(last_features_scaled)[0]
                predictions[name] = pred
            except:
                continue
        
        if not predictions:
            return None
        
        # Moyenne pondérée des prédictions
        final_prediction = np.mean(list(predictions.values()))
        
        # Calcul de la confiance basée sur la variance des prédictions
        confidence = 1.0 / (1.0 + np.var(list(predictions.values())))
        
        # Intervalle de confiance basé sur la volatilité historique
        recent_prices = df['price'].tail(10).values
        volatility = np.std(recent_prices)
        confidence_interval = (
            final_prediction - 1.96 * volatility,
            final_prediction + 1.96 * volatility
        )
        
        # Détection de la tendance
        recent_trend = df['price'].tail(5).values
        if len(recent_trend) >= 2:
            if recent_trend[-1] > recent_trend[0] * 1.05:
                trend = "up"
            elif recent_trend[-1] < recent_trend[0] * 0.95:
                trend = "down"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return PricePrediction(
            item_id=item_id,
            predicted_price=final_prediction,
            confidence_interval=confidence_interval,
            prediction_horizon=timedelta(days=horizon_days),
            accuracy_score=confidence,
            trend_direction=trend,
            volatility_forecast=volatility
        )

class ArbitrageDetector:
    """Détecteur d'opportunités d'arbitrage"""
    
    def __init__(self, min_profit_margin: float = 0.1, min_roi: float = 0.05):
        self.min_profit_margin = min_profit_margin
        self.min_roi = min_roi
        self.server_fees = {
            "default": 0.02  # 2% de frais par défaut
        }
    
    def calculate_fees(self, price: float, server: str = "default") -> float:
        """Calcule les frais de transaction"""
        fee_rate = self.server_fees.get(server, self.server_fees["default"])
        return price * fee_rate
    
    def detect_opportunities(self, market_data: Dict[str, List[MarketItem]]) -> List[ArbitrageOpportunity]:
        """Détecte les opportunités d'arbitrage entre serveurs"""
        opportunities = []
        
        # Grouper les items par ID
        items_by_id = defaultdict(list)
        for server, items in market_data.items():
            for item in items:
                items_by_id[item.item_id].append((server, item))
        
        # Analyser chaque item
        for item_id, server_items in items_by_id.items():
            if len(server_items) < 2:
                continue
            
            # Trier par prix
            server_items.sort(key=lambda x: x[1].current_price)
            
            # Comparer tous les pairs
            for i in range(len(server_items)):
                for j in range(i + 1, len(server_items)):
                    buy_server, buy_item = server_items[i]
                    sell_server, sell_item = server_items[j]
                    
                    if buy_server == sell_server:
                        continue
                    
                    buy_price = buy_item.current_price
                    sell_price = sell_item.current_price
                    
                    # Calcul des frais
                    buy_fees = self.calculate_fees(buy_price, buy_server)
                    sell_fees = self.calculate_fees(sell_price, sell_server)
                    
                    net_buy_price = buy_price + buy_fees
                    net_sell_price = sell_price - sell_fees
                    
                    if net_sell_price <= net_buy_price:
                        continue
                    
                    profit = net_sell_price - net_buy_price
                    roi = profit / net_buy_price
                    
                    if profit < self.min_profit_margin or roi < self.min_roi:
                        continue
                    
                    # Évaluation du risque
                    risk_level = self._evaluate_risk(buy_item, sell_item)
                    confidence = self._calculate_confidence(buy_item, sell_item)
                    
                    # Volume estimé
                    estimated_volume = min(buy_item.quantity_available, sell_item.quantity_available)
                    
                    opportunity = ArbitrageOpportunity(
                        item_id=item_id,
                        buy_server=buy_server,
                        sell_server=sell_server,
                        buy_price=net_buy_price,
                        sell_price=net_sell_price,
                        profit_margin=profit,
                        roi_percentage=roi * 100,
                        risk_level=risk_level,
                        confidence=confidence,
                        estimated_volume=estimated_volume
                    )
                    
                    opportunities.append(opportunity)
        
        # Trier par ROI décroissant
        opportunities.sort(key=lambda x: x.roi_percentage, reverse=True)
        return opportunities
    
    def _evaluate_risk(self, buy_item: MarketItem, sell_item: MarketItem) -> str:
        """Évalue le niveau de risque d'une opportunité"""
        # Facteurs de risque
        risk_score = 0
        
        # Volatilité
        if buy_item.volatility > 0.3 or sell_item.volatility > 0.3:
            risk_score += 2
        
        # Quantité disponible
        if buy_item.quantity_available < 5 or sell_item.quantity_available < 5:
            risk_score += 1
        
        # Écart de prix trop important (peut indiquer une erreur)
        price_ratio = sell_item.current_price / buy_item.current_price
        if price_ratio > 3:
            risk_score += 2
        
        # Âge des données
        now = datetime.now()
        if (now - buy_item.timestamp).seconds > 3600:  # Plus d'1 heure
            risk_score += 1
        
        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence(self, buy_item: MarketItem, sell_item: MarketItem) -> float:
        """Calcule le niveau de confiance"""
        confidence = 1.0
        
        # Réduction basée sur la volatilité
        avg_volatility = (buy_item.volatility + sell_item.volatility) / 2
        confidence *= max(0.1, 1.0 - avg_volatility)
        
        # Réduction basée sur l'âge des données
        now = datetime.now()
        max_age = max(
            (now - buy_item.timestamp).seconds,
            (now - sell_item.timestamp).seconds
        )
        age_factor = max(0.1, 1.0 - max_age / 7200)  # Dégradation sur 2h
        confidence *= age_factor
        
        # Bonus pour les quantités importantes
        min_quantity = min(buy_item.quantity_available, sell_item.quantity_available)
        quantity_factor = min(1.0, min_quantity / 10)
        confidence *= (0.7 + 0.3 * quantity_factor)
        
        return confidence

class MarketAnalyzer:
    """Analyseur de marché principal"""
    
    def __init__(self, db_path: str = "market_data.db"):
        self.db = MarketDatabase(db_path)
        self.predictor = MLPricePredictor()
        self.arbitrage_detector = ArbitrageDetector()
        self.running = False
        self.scan_interval = 300  # 5 minutes
        self._scan_thread = None
    
    def start_continuous_scan(self):
        """Démarre le scan continu du marché"""
        if self.running:
            return
        
        self.running = True
        self._scan_thread = threading.Thread(target=self._continuous_scan)
        self._scan_thread.daemon = True
        self._scan_thread.start()
    
    def stop_continuous_scan(self):
        """Arrête le scan continu"""
        self.running = False
        if self._scan_thread:
            self._scan_thread.join()
    
    def _continuous_scan(self):
        """Scan continu en arrière-plan"""
        while self.running:
            try:
                # Ici vous intégreriez le scan réel du HDV
                market_data = self._scan_hdv()
                if market_data:
                    self._process_market_data(market_data)
                
                time.sleep(self.scan_interval)
            except Exception as e:
                print(f"Erreur lors du scan continu: {e}")
                time.sleep(60)  # Attendre 1 minute en cas d'erreur
    
    def _scan_hdv(self) -> Dict[str, List[MarketItem]]:
        """
        Interface pour scanner le HDV
        À implémenter avec votre système de vision
        """
        # TODO: Implémenter le scan réel du HDV
        # Cette méthode devrait utiliser votre système de vision pour scanner
        # les prix et quantités dans l'interface du HDV
        
        # Exemple de données fictives pour les tests
        sample_data = {
            "server1": [
                MarketItem(
                    item_id=1234,
                    name="Blé",
                    current_price=50.0,
                    quantity_available=100,
                    server_id="server1",
                    timestamp=datetime.now(),
                    category="Ressource",
                    level=1
                )
            ]
        }
        
        return sample_data
    
    def _process_market_data(self, market_data: Dict[str, List[MarketItem]]):
        """Traite les données de marché scannées"""
        all_items = []
        for server_items in market_data.values():
            all_items.extend(server_items)
        
        # Stockage en base
        self.db.store_market_data(all_items)
        
        # Détection d'arbitrage
        opportunities = self.arbitrage_detector.detect_opportunities(market_data)
        
        # Log des meilleures opportunités
        if opportunities:
            print(f"\n=== {len(opportunities)} opportunités d'arbitrage détectées ===")
            for opp in opportunities[:5]:  # Top 5
                print(f"Item {opp.item_id}: {opp.buy_server} → {opp.sell_server}")
                print(f"  Profit: {opp.profit_margin:.2f} ({opp.roi_percentage:.1f}%)")
                print(f"  Risque: {opp.risk_level} | Confiance: {opp.confidence:.2f}")
    
    def get_item_analysis(self, item_id: int) -> Dict[str, Any]:
        """Analyse complète d'un item"""
        analysis = {
            "item_id": item_id,
            "timestamp": datetime.now(),
            "price_prediction": None,
            "arbitrage_opportunities": [],
            "market_statistics": {}
        }
        
        # Prédiction de prix
        prediction = self.predictor.predict_price(item_id, self.db)
        if prediction:
            analysis["price_prediction"] = {
                "predicted_price": prediction.predicted_price,
                "confidence_interval": prediction.confidence_interval,
                "trend": prediction.trend_direction,
                "accuracy": prediction.accuracy_score
            }
        
        # Statistiques de marché
        df = self.db.get_price_history(item_id, days=30)
        if not df.empty:
            analysis["market_statistics"] = {
                "avg_price_30d": df['price'].mean(),
                "min_price_30d": df['price'].min(),
                "max_price_30d": df['price'].max(),
                "volatility": df['price'].std(),
                "avg_volume": df['quantity'].mean(),
                "total_transactions": len(df)
            }
        
        return analysis
    
    def get_top_opportunities(self, limit: int = 10) -> List[ArbitrageOpportunity]:
        """Retourne les meilleures opportunités d'arbitrage actuelles"""
        # TODO: Implémenter la récupération depuis la base de données
        # Pour l'instant retourne une liste vide
        return []
    
    def export_market_report(self, filepath: str):
        """Exporte un rapport de marché complet"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "market_summary": {},
            "top_opportunities": [
                opp.__dict__ for opp in self.get_top_opportunities()
            ],
            "trending_items": [],
            "server_comparison": {}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

# Exemple d'utilisation
if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    
    # Démarrage du scan continu
    print("Démarrage de l'analyseur de marché...")
    analyzer.start_continuous_scan()
    
    try:
        # Simulation de fonctionnement
        time.sleep(10)
        
        # Analyse d'un item spécifique
        analysis = analyzer.get_item_analysis(1234)
        print(f"Analyse de l'item 1234: {analysis}")
        
        # Export d'un rapport
        analyzer.export_market_report("market_report.json")
        print("Rapport exporté: market_report.json")
        
    except KeyboardInterrupt:
        print("\nArrêt de l'analyseur...")
    finally:
        analyzer.stop_continuous_scan()