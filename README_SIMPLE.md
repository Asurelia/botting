# 🤖 Bot DOFUS Simplifié - Version Streamlined

Version simplifiée du bot DOFUS pour **usage personnel unique**.  
Toute la complexité inutile a été supprimée !

## 🎯 Pourquoi cette version ?

- ✅ **Plus simple** : Pas de blockchain, multi-comptes, chiffrement complexe
- ✅ **Plus rapide** : Démarrage instantané, moins de ressources
- ✅ **Plus pratique** : Une seule commande pour tout faire
- ✅ **Tout aussi efficace** : Protection anti-détection conservée
- ✅ **Moins de bugs** : Code simplifié = moins de problèmes

## 🚀 Démarrage Ultra-Rapide

```bash
# Lancement simple
python simple_bot_main.py

# Mode farming uniquement  
python simple_bot_main.py --mode farming

# Session de 5 heures max
python simple_bot_main.py --max-hours 5

# Test sans DOFUS
python simple_bot_main.py --test
```

## 📁 Structure Simplifiée

```
G:/Botting/
├── simple_bot_main.py           # 🎯 Point d'entrée principal SIMPLE
├── core/simple_security.py      # 🛡️ Sécurité basique mais efficace
├── modules/                     # 🎮 Modules de gameplay (inchangés)
├── engine/                      # ⚙️ Moteur (si besoin des fonctions avancées)
└── README_SIMPLE.md            # 📖 Ce fichier !
```

## 🛡️ Sécurité Simplifiée (Mais Efficace !)

### Ce qui est GARDÉ ✅
- **Délais variables** (100-250ms au lieu de toujours pareil)
- **Variation position souris** (±3 pixels)
- **Pauses naturelles** (toutes les 500-800 actions)
- **Limite de session** (3h par défaut)
- **Logs simples** pour débugger

### Ce qui est SUPPRIMÉ ❌
- ~~Chiffrement AES-256~~
- ~~Authentification multi-facteurs~~
- ~~Blockchain/audit trail complexe~~
- ~~Multi-comptes~~
- ~~Analyse comportementale avancée~~

## 💡 Utilisation Basique

### Dans ton code Python :
```python
from core.simple_security import SimpleSecurity

# Système de sécurité simple
security = SimpleSecurity(max_hours=3.0)

# Clic sécurisé (avec délai et variation)
final_x, final_y = security.safe_click(100, 200, "attack_monster")

# Attente sécurisée (avec variation)
security.safe_wait(2.0, 0.5)  # 2±0.5 secondes

# Vérifier si on devrait s'arrêter
if security.should_stop():
    print("Session trop longue, arrêt recommandé")
```

### Logs automatiques :
```
2024-08-31 15:30:15 - INFO - Action: attack_monster - {'target': (100, 200), 'actual': (102, 198)}
2024-08-31 15:30:17 - INFO - Action: wait - {'requested': 2.0, 'actual': 2.3}
2024-08-31 15:35:22 - INFO - Pause humaine de 8.2s après 534 actions
```

## 📊 Monitoring Simple

```python
# État du système
status = security.get_status()
print(f"Actions: {status['actions_count']}")
print(f"Session: {status['session']['duration_minutes']:.1f} min")

# Statistiques détaillées
security.print_status()
```

Sortie :
```
==================================================
📊 STATISTIQUES SESSION
==================================================
⏱️  Durée: 45.2 minutes
🎮 Actions: 1234
⚡ Actions/min: 27.3
❌ Erreurs: 2
==================================================
⏰ Temps restant recommandé: 135 min
```

## ⚙️ Configuration

Tout se configure facilement dans le code :

```python
# Personnalisation du comportement
security = SimpleSecurity(max_hours=5.0)

# Modifier les délais de base
security.behavior.click_delay_base = 200  # Plus lent
security.behavior.click_delay_variation = 100  # Plus de variation
security.behavior.mouse_variation = 5  # Plus de variation souris
```

## 🔧 Extension Facile

Besoin d'une fonction ? Ajoute-la simplement :

```python
class MyCustomBot(SimpleDofusBot):
    def _handle_farming(self):
        # Ta logique de farming custom
        print("Mon farming personnalisé !")
        self.security.safe_click(x, y, "harvest_wheat")
        
# Utilisation
bot = MyCustomBot()
bot.start()
```

## 🆚 Comparaison Versions

| Fonctionnalité | Version Complexe | Version Simple |
|---|---|---|
| **Fichiers** | 50+ fichiers | 3 fichiers principaux |
| **Démarrage** | ~10 secondes | ~1 seconde |
| **RAM utilisée** | ~500 MB | ~50 MB |
| **Configuration** | YAML/JSON complexe | Code Python simple |
| **Multi-comptes** | ✅ | ❌ (pas besoin) |
| **Sécurité anti-détection** | ✅ Ultra | ✅ Efficace |
| **Facilité d'usage** | 🔴 Complexe | 🟢 Simple |
| **Pour distribuer** | ✅ | ❌ |
| **Usage personnel** | ✅ | ✅ **Parfait** |

## 🎮 Modes Disponibles

```bash
# Mode automatique (recommandé)
python simple_bot_main.py --mode auto

# Mode farming seulement
python simple_bot_main.py --mode farming

# Mode combat seulement  
python simple_bot_main.py --mode combat

# Session courte (1 heure)
python simple_bot_main.py --max-hours 1
```

## ❓ FAQ

**Q: C'est moins sécurisé que la version complexe ?**  
R: Non ! La protection anti-détection essentielle est conservée. On a juste retiré le superflu.

**Q: Je peux quand même utiliser les modules avancés ?**  
R: Oui ! Tous les modules (combat IA, vision, métiers) sont toujours disponibles.

**Q: Ça marche avec ma config existante ?**  
R: Oui, c'est rétro-compatible. Tu peux mixer simple et complexe.

**Q: Je peux revenir à la version complexe ?**  
R: Oui, les deux versions coexistent. Utilise `main.py` pour la version complète.

## 🏁 Conclusion

Cette version **simple mais efficace** est parfaite pour :
- ✅ Usage personnel unique
- ✅ Apprentissage du système  
- ✅ Tests rapides
- ✅ Moins de maintenance

**Démarre avec `python simple_bot_main.py` et c'est parti !** 🚀