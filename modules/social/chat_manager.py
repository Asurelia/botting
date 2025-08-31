"""
Module de gestion du chat avec OCR avancé et intelligence artificielle
Fonctionnalités:
- OCR pour lecture des messages de chat
- Filtrage intelligent du spam et des publicités
- Réponses automatiques contextuelles
- Détection de mots-clés importants
- Patterns NLP pour analyse sémantique
"""

import cv2
import numpy as np
import re
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import easyocr
from datetime import datetime, timedelta
import threading
from collections import deque, defaultdict

class MessageType(Enum):
    """Types de messages détectés"""
    NORMAL = "normal"
    SPAM = "spam"
    PUBLICITE = "publicite"
    KAMA = "kama"
    ECHANGE = "echange"
    GROUPE = "groupe"
    GUILDE = "guilde"
    COMBAT = "combat"
    IMPORTANT = "important"
    AIDE = "aide"
    FLOOD = "flood"

class ResponseType(Enum):
    """Types de réponses automatiques"""
    SALUTATION = "salutation"
    REMERCIEMENT = "remerciement"
    AIDE = "aide"
    ECHANGE = "echange"
    EXCUSE = "excuse"
    DISPONIBILITE = "disponibilite"
    INFORMATION = "information"

@dataclass
class ChatMessage:
    """Structure d'un message de chat"""
    author: str
    content: str
    timestamp: datetime
    channel: str = "general"
    message_type: MessageType = MessageType.NORMAL
    confidence: float = 0.0
    auto_responded: bool = False
    important_keywords: List[str] = field(default_factory=list)

@dataclass
class ResponseTemplate:
    """Template de réponse automatique"""
    response_type: ResponseType
    triggers: List[str]
    responses: List[str]
    cooldown: int = 60  # secondes
    probability: float = 0.8
    context_required: List[str] = field(default_factory=list)

class ChatOCR:
    """Système OCR avancé pour la lecture des messages"""
    
    def __init__(self):
        self.reader = easyocr.Reader(['fr', 'en'], gpu=True)
        self.chat_regions = {
            'general': (50, 400, 600, 200),
            'guilde': (50, 300, 600, 100),
            'groupe': (50, 200, 600, 100),
            'prive': (50, 100, 600, 100)
        }
        self.last_messages = {}
        
    def extract_chat_region(self, screenshot: np.ndarray, channel: str) -> np.ndarray:
        """Extrait la région du chat spécifiée"""
        if channel not in self.chat_regions:
            channel = 'general'
            
        x, y, w, h = self.chat_regions[channel]
        return screenshot[y:y+h, x:x+w]
    
    def preprocess_chat_image(self, image: np.ndarray) -> np.ndarray:
        """Prétraite l'image pour améliorer l'OCR"""
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Débruitage
        denoised = cv2.medianBlur(enhanced, 3)
        
        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def parse_chat_line(self, text: str) -> Optional[Tuple[str, str]]:
        """Parse une ligne de chat pour extraire auteur et message"""
        # Patterns pour différents formats de chat
        patterns = [
            r'^\[(\d{2}:\d{2})\]\s*([^:]+):\s*(.+)$',  # [HH:MM] Joueur: message
            r'^([^:]+):\s*(.+)$',                       # Joueur: message
            r'^\*([^*]+)\*\s*(.+)$',                    # *Joueur* message (émote)
            r'^<([^>]+)>\s*(.+)$',                      # <Joueur> message
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text.strip())
            if match:
                if len(match.groups()) == 3:  # Avec timestamp
                    _, author, content = match.groups()
                else:  # Sans timestamp
                    author, content = match.groups()
                return author.strip(), content.strip()
        
        return None
    
    def read_chat_messages(self, screenshot: np.ndarray, channel: str = 'general') -> List[ChatMessage]:
        """Lit les messages du chat depuis une capture d'écran"""
        chat_region = self.extract_chat_region(screenshot, channel)
        processed_image = self.preprocess_chat_image(chat_region)
        
        # OCR sur la région
        results = self.reader.readtext(processed_image)
        
        messages = []
        current_time = datetime.now()
        
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Seuil de confiance minimum
                parsed = self.parse_chat_line(text)
                if parsed:
                    author, content = parsed
                    # Éviter les doublons
                    message_key = f"{author}:{content[:20]}"
                    if message_key not in self.last_messages or \
                       (current_time - self.last_messages[message_key]).seconds > 5:
                        
                        message = ChatMessage(
                            author=author,
                            content=content,
                            timestamp=current_time,
                            channel=channel,
                            confidence=confidence
                        )
                        messages.append(message)
                        self.last_messages[message_key] = current_time
        
        return messages

class SpamFilter:
    """Système de filtrage intelligent du spam et des publicités"""
    
    def __init__(self):
        self.spam_patterns = self._load_spam_patterns()
        self.whitelist_users = set()
        self.blacklist_users = set()
        self.user_message_history = defaultdict(deque)
        self.flood_threshold = 3  # messages par minute
        
    def _load_spam_patterns(self) -> Dict[MessageType, List[str]]:
        """Charge les patterns de détection de spam"""
        return {
            MessageType.SPAM: [
                r'(?i)visite.{0,10}site',
                r'(?i)gratuit.{0,10}kamas?',
                r'(?i)livraison.{0,10}rapide',
                r'(?i)prix.{0,10}imbattable',
                r'(?i)www\.\w+\.\w+',
                r'(?i)http[s]?://',
                r'(?i)achete.{0,10}compte',
                r'(?i)vend.{0,10}compte',
            ],
            MessageType.PUBLICITE: [
                r'(?i)boutique.{0,10}en.{0,10}ligne',
                r'(?i)promo.{0,10}exceptionnel',
                r'(?i)réduction.{0,10}\d+%',
                r'(?i)offre.{0,10}spécial',
                r'(?i)code.{0,10}promo',
                r'(?i)discount',
            ],
            MessageType.KAMA: [
                r'(?i)vend.{0,10}kamas?',
                r'(?i)achète.{0,10}kamas?',
                r'(?i)échange.{0,10}kamas?',
                r'(?i)\d+k\s*kamas?',
                r'(?i)\d+m\s*kamas?',
                r'(?i)farm.{0,10}kamas?',
            ],
            MessageType.FLOOD: [
                r'^(.)\1{10,}$',  # Répétition de caractères
                r'^[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]{5,}$',  # Symboles répétés
                r'(?i)^(haha|lol|mdr){3,}',
            ]
        }
    
    def analyze_message(self, message: ChatMessage) -> MessageType:
        """Analyse un message pour détecter son type"""
        content = message.content.lower()
        
        # Vérifier les listes noires/blanches
        if message.author in self.blacklist_users:
            return MessageType.SPAM
        if message.author in self.whitelist_users:
            return MessageType.NORMAL
        
        # Vérifier le flood
        if self._is_flooding(message):
            return MessageType.FLOOD
        
        # Vérifier les patterns de spam
        for msg_type, patterns in self.spam_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message.content):
                    message.message_type = msg_type
                    return msg_type
        
        return MessageType.NORMAL
    
    def _is_flooding(self, message: ChatMessage) -> bool:
        """Détecte si un utilisateur fait du flood"""
        user_history = self.user_message_history[message.author]
        
        # Ajouter le message actuel
        user_history.append(message.timestamp)
        
        # Nettoyer l'historique (garder seulement la dernière minute)
        cutoff_time = message.timestamp - timedelta(minutes=1)
        while user_history and user_history[0] < cutoff_time:
            user_history.popleft()
        
        return len(user_history) > self.flood_threshold
    
    def add_to_blacklist(self, username: str):
        """Ajoute un utilisateur à la liste noire"""
        self.blacklist_users.add(username)
        if username in self.whitelist_users:
            self.whitelist_users.remove(username)
    
    def add_to_whitelist(self, username: str):
        """Ajoute un utilisateur à la liste blanche"""
        self.whitelist_users.add(username)
        if username in self.blacklist_users:
            self.blacklist_users.remove(username)

class NLPProcessor:
    """Processeur NLP pour analyse sémantique des messages"""
    
    def __init__(self):
        self.important_keywords = self._load_important_keywords()
        self.context_patterns = self._load_context_patterns()
        self.sentiment_words = self._load_sentiment_words()
    
    def _load_important_keywords(self) -> Dict[str, List[str]]:
        """Charge les mots-clés importants par catégorie"""
        return {
            'combat': [
                'combat', 'fight', 'pvp', 'koliseum', 'agression', 'attaque',
                'défense', 'koli', 'duel', 'guerre'
            ],
            'echange': [
                'echange', 'trade', 'vend', 'achète', 'prix', 'kama', 'hdv',
                'hôtel', 'vente', 'achat', 'négocie', 'troc'
            ],
            'aide': [
                'aide', 'help', 'question', 'comment', 'pourquoi', 'problème',
                'bug', 'erreur', 'astuce', 'conseil', 'tip'
            ],
            'groupe': [
                'groupe', 'team', 'équipe', 'recrutement', 'cherche', 'rejoindre',
                'invitation', 'invite', 'place', 'slot'
            ],
            'guilde': [
                'guilde', 'guild', 'alliance', 'recrutement', 'candidature',
                'postule', 'membre', 'prisme', 'percepteur'
            ],
            'donjon': [
                'donjon', 'dungeon', 'boss', 'clé', 'drop', 'réussite',
                'échec', 'difficulté', 'niveau'
            ]
        }
    
    def _load_context_patterns(self) -> List[str]:
        """Charge les patterns de contexte"""
        return [
            r'(?i)quelqu\'un.{0,20}(aide|help)',
            r'(?i)cherche.{0,20}(groupe|team)',
            r'(?i)vend.{0,20}\w+.{0,20}(\d+k|\d+m)',
            r'(?i)où.{0,10}trouve.{0,10}\w+',
            r'(?i)comment.{0,10}faire',
            r'(?i)problème.{0,10}avec',
        ]
    
    def _load_sentiment_words(self) -> Dict[str, List[str]]:
        """Charge les mots de sentiment"""
        return {
            'positif': [
                'merci', 'super', 'génial', 'parfait', 'excellent', 'top',
                'cool', 'sympa', 'bien', 'bravo', 'félicitation'
            ],
            'negatif': [
                'nul', 'merde', 'chiant', 'relou', 'énervant', 'bug',
                'cassé', 'pourri', 'ras le bol', 'marre'
            ],
            'neutre': [
                'ok', 'oui', 'non', 'peut-être', 'normal', 'standard'
            ]
        }
    
    def extract_keywords(self, message: ChatMessage) -> List[str]:
        """Extrait les mots-clés importants d'un message"""
        content = message.content.lower()
        found_keywords = []
        
        for category, keywords in self.important_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    found_keywords.append(f"{category}:{keyword}")
        
        message.important_keywords = found_keywords
        return found_keywords
    
    def analyze_context(self, message: ChatMessage) -> str:
        """Analyse le contexte d'un message"""
        content = message.content
        
        for pattern in self.context_patterns:
            if re.search(pattern, content):
                return "question_aide"
        
        keywords = message.important_keywords
        if any('aide:' in kw for kw in keywords):
            return "demande_aide"
        elif any('echange:' in kw for kw in keywords):
            return "proposition_echange"
        elif any('groupe:' in kw for kw in keywords):
            return "recherche_groupe"
        elif any('combat:' in kw for kw in keywords):
            return "discussion_combat"
        
        return "conversation_generale"
    
    def detect_sentiment(self, message: ChatMessage) -> str:
        """Détecte le sentiment d'un message"""
        content = message.content.lower()
        
        positive_score = sum(1 for word in self.sentiment_words['positif'] if word in content)
        negative_score = sum(1 for word in self.sentiment_words['negatif'] if word in content)
        
        if positive_score > negative_score:
            return "positif"
        elif negative_score > positive_score:
            return "negatif"
        else:
            return "neutre"

class AutoResponder:
    """Système de réponses automatiques contextuelles"""
    
    def __init__(self):
        self.response_templates = self._load_response_templates()
        self.last_responses = {}
        self.conversation_context = defaultdict(list)
        self.user_preferences = {}
        
    def _load_response_templates(self) -> List[ResponseTemplate]:
        """Charge les templates de réponses automatiques"""
        return [
            ResponseTemplate(
                response_type=ResponseType.SALUTATION,
                triggers=['salut', 'hello', 'bonjour', 'bonsoir', 'coucou', 'yo'],
                responses=[
                    'Salut {author} !',
                    'Hello {author} !',
                    'Bonjour {author}, comment ça va ?',
                    'Coucou {author} !',
                    'Yo {author} ! Quoi de neuf ?'
                ],
                cooldown=300,
                probability=0.7
            ),
            ResponseTemplate(
                response_type=ResponseType.REMERCIEMENT,
                triggers=['merci', 'thanks', 'thx', 'ty'],
                responses=[
                    'De rien {author} !',
                    'Avec plaisir !',
                    'Pas de souci {author} !',
                    'Content d\'avoir pu aider !',
                    'No problem !'
                ],
                cooldown=60,
                probability=0.8
            ),
            ResponseTemplate(
                response_type=ResponseType.AIDE,
                triggers=['aide', 'help', 'comment', 'où', 'problème'],
                responses=[
                    'Je ne suis pas sûr, mais tu peux essayer de demander sur le canal guilde ?',
                    'Hmm, bonne question ! Quelqu\'un d\'autre a une idée ?',
                    'Je ne connais pas la réponse exacte, désolé.',
                    'Tu as essayé de regarder sur le wiki Dofus ?'
                ],
                cooldown=180,
                probability=0.5,
                context_required=['question_aide', 'demande_aide']
            ),
            ResponseTemplate(
                response_type=ResponseType.EXCUSE,
                triggers=['désolé', 'excuse', 'pardon', 'sorry'],
                responses=[
                    'Pas de problème {author} !',
                    'Aucun souci !',
                    'T\'inquiète pas pour ça !',
                    'C\'est rien du tout !'
                ],
                cooldown=120,
                probability=0.9
            ),
            ResponseTemplate(
                response_type=ResponseType.DISPONIBILITE,
                triggers=['libre', 'disponible', 'dispo', 'tu fais quoi'],
                responses=[
                    'Je suis en train de farmer, mais je peux te parler !',
                    'Je fais un peu de craft, pourquoi ?',
                    'Pas grand chose, tu as besoin de quelque chose ?',
                    'Je navigue un peu, qu\'est-ce qu\'il y a ?'
                ],
                cooldown=600,
                probability=0.6
            )
        ]
    
    def should_respond(self, message: ChatMessage, template: ResponseTemplate) -> bool:
        """Détermine si on devrait répondre à un message"""
        import random
        
        # Vérifier le cooldown
        key = f"{message.author}:{template.response_type.value}"
        last_response_time = self.last_responses.get(key)
        if last_response_time and \
           (datetime.now() - last_response_time).seconds < template.cooldown:
            return False
        
        # Vérifier la probabilité
        if random.random() > template.probability:
            return False
        
        # Vérifier le contexte si requis
        if template.context_required:
            context = self.conversation_context.get(message.author, [])
            if not any(req_context in context for req_context in template.context_required):
                return False
        
        # Vérifier les triggers
        content_lower = message.content.lower()
        return any(trigger in content_lower for trigger in template.triggers)
    
    def generate_response(self, message: ChatMessage, template: ResponseTemplate) -> str:
        """Génère une réponse basée sur le template"""
        import random
        
        response = random.choice(template.responses)
        
        # Remplacer les variables
        response = response.replace('{author}', message.author)
        response = response.replace('{timestamp}', message.timestamp.strftime('%H:%M'))
        
        return response
    
    def process_message(self, message: ChatMessage, context: str) -> Optional[str]:
        """Traite un message et génère une réponse si appropriée"""
        # Mettre à jour le contexte de conversation
        user_context = self.conversation_context[message.author]
        user_context.append(context)
        if len(user_context) > 5:  # Garder seulement les 5 derniers contextes
            user_context.pop(0)
        
        # Chercher un template approprié
        for template in self.response_templates:
            if self.should_respond(message, template):
                response = self.generate_response(message, template)
                
                # Enregistrer la réponse
                key = f"{message.author}:{template.response_type.value}"
                self.last_responses[key] = datetime.now()
                
                return response
        
        return None

class ChatManager:
    """Gestionnaire principal du système de chat"""
    
    def __init__(self):
        self.ocr = ChatOCR()
        self.spam_filter = SpamFilter()
        self.nlp_processor = NLPProcessor()
        self.auto_responder = AutoResponder()
        
        self.active_channels = ['general', 'guilde', 'groupe']
        self.message_history = defaultdict(deque)
        self.important_messages = deque(maxlen=50)
        
        self.running = False
        self.update_thread = None
        
        # Callbacks pour événements
        self.on_important_message = None
        self.on_spam_detected = None
        self.on_response_generated = None
        
    def start_monitoring(self):
        """Démarre la surveillance du chat"""
        self.running = True
        self.update_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.update_thread.start()
        
    def stop_monitoring(self):
        """Arrête la surveillance du chat"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Boucle principale de surveillance"""
        while self.running:
            try:
                # Simuler une capture d'écran (à remplacer par votre système)
                screenshot = self._get_screenshot()
                if screenshot is not None:
                    self.process_screenshot(screenshot)
                
                time.sleep(1)  # Vérifier toutes les secondes
                
            except Exception as e:
                print(f"Erreur dans la boucle de surveillance: {e}")
                time.sleep(5)
    
    def _get_screenshot(self) -> Optional[np.ndarray]:
        """Récupère une capture d'écran (à implémenter selon votre système)"""
        # À remplacer par votre logique de capture d'écran
        return None
    
    def process_screenshot(self, screenshot: np.ndarray):
        """Traite une capture d'écran pour extraire les messages"""
        for channel in self.active_channels:
            messages = self.ocr.read_chat_messages(screenshot, channel)
            
            for message in messages:
                self.process_message(message)
    
    def process_message(self, message: ChatMessage):
        """Traite un message de chat complet"""
        try:
            # Analyser le type de message (spam, etc.)
            message.message_type = self.spam_filter.analyze_message(message)
            
            # Si c'est du spam, l'ignorer
            if message.message_type in [MessageType.SPAM, MessageType.PUBLICITE, MessageType.FLOOD]:
                if self.on_spam_detected:
                    self.on_spam_detected(message)
                return
            
            # Extraire les mots-clés
            keywords = self.nlp_processor.extract_keywords(message)
            
            # Analyser le contexte
            context = self.nlp_processor.analyze_context(message)
            
            # Détecter le sentiment
            sentiment = self.nlp_processor.detect_sentiment(message)
            
            # Ajouter à l'historique
            channel_history = self.message_history[message.channel]
            channel_history.append(message)
            if len(channel_history) > 100:  # Garder seulement les 100 derniers
                channel_history.popleft()
            
            # Vérifier si c'est un message important
            if self._is_important_message(message):
                self.important_messages.append(message)
                if self.on_important_message:
                    self.on_important_message(message)
            
            # Générer une réponse automatique si appropriée
            response = self.auto_responder.process_message(message, context)
            if response:
                message.auto_responded = True
                if self.on_response_generated:
                    self.on_response_generated(message, response)
                
                # Ici, vous intégreriez l'envoi du message dans le jeu
                self._send_chat_message(response, message.channel)
            
        except Exception as e:
            print(f"Erreur lors du traitement du message: {e}")
    
    def _is_important_message(self, message: ChatMessage) -> bool:
        """Détermine si un message est important"""
        important_indicators = [
            'aide:', 'combat:', 'groupe:', 'guilde:', 'echange:'
        ]
        
        return any(indicator in kw for kw in message.important_keywords 
                  for indicator in important_indicators) or \
               message.author in self.spam_filter.whitelist_users
    
    def _send_chat_message(self, message: str, channel: str = 'general'):
        """Envoie un message dans le chat (à implémenter selon votre système)"""
        # À remplacer par votre logique d'envoi de message
        print(f"[{channel.upper()}] Bot: {message}")
    
    def add_custom_response(self, triggers: List[str], responses: List[str], 
                          response_type: ResponseType = ResponseType.INFORMATION):
        """Ajoute une réponse personnalisée"""
        template = ResponseTemplate(
            response_type=response_type,
            triggers=triggers,
            responses=responses,
            cooldown=120,
            probability=0.8
        )
        self.auto_responder.response_templates.append(template)
    
    def blacklist_user(self, username: str):
        """Ajoute un utilisateur à la liste noire"""
        self.spam_filter.add_to_blacklist(username)
    
    def whitelist_user(self, username: str):
        """Ajoute un utilisateur à la liste blanche"""
        self.spam_filter.add_to_whitelist(username)
    
    def get_recent_messages(self, channel: str = 'general', count: int = 10) -> List[ChatMessage]:
        """Récupère les messages récents d'un canal"""
        history = self.message_history.get(channel, deque())
        return list(history)[-count:]
    
    def get_important_messages(self, count: int = 10) -> List[ChatMessage]:
        """Récupère les messages importants récents"""
        return list(self.important_messages)[-count:]
    
    def get_statistics(self) -> Dict[str, int]:
        """Récupère les statistiques du chat"""
        stats = {
            'total_messages': sum(len(history) for history in self.message_history.values()),
            'spam_blocked': len(self.spam_filter.blacklist_users),
            'important_messages': len(self.important_messages),
            'auto_responses_sent': sum(1 for history in self.message_history.values() 
                                     for msg in history if msg.auto_responded)
        }
        return stats

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer le gestionnaire de chat
    chat_manager = ChatManager()
    
    # Définir les callbacks
    def on_important_message(message):
        print(f"MESSAGE IMPORTANT de {message.author}: {message.content}")
    
    def on_spam_detected(message):
        print(f"SPAM détecté de {message.author}: {message.content[:50]}...")
    
    def on_response_generated(message, response):
        print(f"Réponse automatique à {message.author}: {response}")
    
    # Configurer les callbacks
    chat_manager.on_important_message = on_important_message
    chat_manager.on_spam_detected = on_spam_detected
    chat_manager.on_response_generated = on_response_generated
    
    # Ajouter des réponses personnalisées
    chat_manager.add_custom_response(
        triggers=['bot', 'automatique'],
        responses=[
            'Oui, je suis un bot ! 😊',
            'En effet, je suis automatisé !',
            'Bot à votre service !'
        ]
    )
    
    # Démarrer la surveillance
    chat_manager.start_monitoring()
    
    try:
        # Garder le programme actif
        while True:
            time.sleep(10)
            stats = chat_manager.get_statistics()
            print(f"Stats: {stats}")
    except KeyboardInterrupt:
        print("Arrêt du gestionnaire de chat...")
        chat_manager.stop_monitoring()