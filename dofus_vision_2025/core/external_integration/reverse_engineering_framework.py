"""
Framework de Reverse Engineering - DOFUS Unity World Model AI
Analyse et intégration d'applications externes (Ganymede, DOFUS Guide)
"""

import os
import json
import time
import psutil
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import socket
import struct
from datetime import datetime

# Pour l'analyse réseau
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP

# Pour l'interception Windows
try:
    import win32gui
    import win32process
    import win32api
    import win32con
except ImportError:
    print("⚠️ Modules Windows non disponibles - fonctionnalités limitées")
    win32gui = None

@dataclass
class NetworkCapture:
    """Capture réseau d'une application"""
    timestamp: float
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str
    data_size: int
    payload: bytes
    decoded_payload: Optional[str] = None

@dataclass
class APIEndpoint:
    """Endpoint API détecté"""
    url: str
    method: str
    parameters: Dict[str, Any]
    response_format: str
    success_rate: float
    last_seen: float
    example_request: Optional[str] = None
    example_response: Optional[str] = None

@dataclass
class ApplicationProcess:
    """Processus d'application analysé"""
    pid: int
    name: str
    executable_path: str
    command_line: str
    network_connections: List[Dict[str, Any]]
    open_files: List[str]
    memory_usage: int
    cpu_usage: float

class WindowsProcessAnalyzer:
    """Analyseur de processus Windows"""

    def __init__(self):
        self.target_processes = [
            "ganymede", "dofus_guide", "dofusguide",
            "dofus", "java", "electron", "chrome"
        ]

    def find_dofus_related_processes(self) -> List[ApplicationProcess]:
        """Trouve les processus liés à DOFUS"""
        dofus_processes = []

        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
                try:
                    process_info = proc.info
                    process_name = process_info['name'].lower() if process_info['name'] else ""

                    # Vérifier si c'est un processus DOFUS/Ganymede/Guide
                    is_target = any(target in process_name for target in self.target_processes)

                    if is_target:
                        # Obtenir informations détaillées
                        try:
                            process = psutil.Process(proc.pid)

                            # Connexions réseau
                            connections = []
                            try:
                                for conn in process.connections():
                                    connections.append({
                                        "local_address": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "",
                                        "remote_address": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                                        "status": conn.status,
                                        "family": conn.family.name,
                                        "type": conn.type.name
                                    })
                            except (psutil.AccessDenied, psutil.NoSuchProcess):
                                pass

                            # Fichiers ouverts
                            open_files = []
                            try:
                                for file in process.open_files():
                                    open_files.append(file.path)
                            except (psutil.AccessDenied, psutil.NoSuchProcess):
                                pass

                            app_process = ApplicationProcess(
                                pid=proc.pid,
                                name=process_info['name'] or "Unknown",
                                executable_path=process_info['exe'] or "",
                                command_line=" ".join(process_info['cmdline']) if process_info['cmdline'] else "",
                                network_connections=connections,
                                open_files=open_files,
                                memory_usage=process.memory_info().rss,
                                cpu_usage=process.cpu_percent()
                            )

                            dofus_processes.append(app_process)

                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            continue

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        except Exception as e:
            print(f"❌ Erreur analyse processus: {e}")

        return dofus_processes

    def analyze_process_behavior(self, pid: int, duration: int = 60) -> Dict[str, Any]:
        """Analyse le comportement d'un processus sur une durée"""
        print(f"🔍 Analyse du processus PID {pid} pendant {duration}s...")

        behavior_data = {
            "pid": pid,
            "start_time": time.time(),
            "duration": duration,
            "network_activity": [],
            "file_activity": [],
            "memory_usage": [],
            "cpu_usage": [],
            "api_calls_detected": []
        }

        try:
            process = psutil.Process(pid)
            start_time = time.time()

            while (time.time() - start_time) < duration:
                try:
                    # Métriques système
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent()

                    behavior_data["memory_usage"].append({
                        "timestamp": time.time(),
                        "rss": memory_info.rss,
                        "vms": memory_info.vms
                    })

                    behavior_data["cpu_usage"].append({
                        "timestamp": time.time(),
                        "cpu_percent": cpu_percent
                    })

                    # Connexions réseau
                    current_connections = []
                    for conn in process.connections():
                        if conn.raddr:  # Connexion établie
                            conn_info = {
                                "timestamp": time.time(),
                                "remote_ip": conn.raddr.ip,
                                "remote_port": conn.raddr.port,
                                "status": conn.status,
                                "protocol": conn.type.name
                            }
                            current_connections.append(conn_info)

                    behavior_data["network_activity"].extend(current_connections)

                    time.sleep(2)  # Échantillonnage toutes les 2 secondes

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        except Exception as e:
            print(f"❌ Erreur analyse comportement: {e}")

        return behavior_data

class NetworkTrafficAnalyzer:
    """Analyseur de trafic réseau"""

    def __init__(self):
        self.captured_packets: List[NetworkCapture] = []
        self.api_endpoints: Dict[str, APIEndpoint] = {}
        self.is_capturing = False
        self.capture_thread: Optional[threading.Thread] = None

    def start_network_capture(self, interface: str = None, filter_ports: List[int] = None):
        """Démarre la capture réseau"""
        if self.is_capturing:
            print("⚠️ Capture déjà en cours")
            return

        self.is_capturing = True

        # Ports typiques des applications web/API
        if not filter_ports:
            filter_ports = [80, 443, 8080, 8443, 3000, 5000, 9000]

        # Créer filtre Scapy
        port_filter = " or ".join([f"port {port}" for port in filter_ports])
        scapy_filter = f"tcp and ({port_filter})"

        print(f"🔍 Démarrage capture réseau avec filtre: {scapy_filter}")

        def capture_packets():
            try:
                scapy.sniff(
                    iface=interface,
                    filter=scapy_filter,
                    prn=self._process_packet,
                    stop_filter=lambda p: not self.is_capturing
                )
            except Exception as e:
                print(f"❌ Erreur capture réseau: {e}")
                self.is_capturing = False

        self.capture_thread = threading.Thread(target=capture_packets, daemon=True)
        self.capture_thread.start()

    def _process_packet(self, packet):
        """Traite un paquet capturé"""
        try:
            if packet.haslayer(TCP) and packet.haslayer(IP):
                # Extraire informations
                ip_layer = packet[IP]
                tcp_layer = packet[TCP]

                # Payload
                payload = bytes(tcp_layer.payload) if tcp_layer.payload else b""

                capture = NetworkCapture(
                    timestamp=time.time(),
                    source_ip=ip_layer.src,
                    dest_ip=ip_layer.dst,
                    source_port=tcp_layer.sport,
                    dest_port=tcp_layer.dport,
                    protocol="TCP",
                    data_size=len(payload),
                    payload=payload
                )

                # Décoder payload si possible
                try:
                    capture.decoded_payload = payload.decode('utf-8', errors='ignore')
                except:
                    pass

                self.captured_packets.append(capture)

                # Détecter patterns API
                self._detect_api_patterns(capture)

                # Limiter taille du cache
                if len(self.captured_packets) > 10000:
                    self.captured_packets = self.captured_packets[-5000:]

        except Exception as e:
            print(f"❌ Erreur traitement paquet: {e}")

    def _detect_api_patterns(self, capture: NetworkCapture):
        """Détecte les patterns d'API dans le trafic"""
        if not capture.decoded_payload:
            return

        payload = capture.decoded_payload

        # Détecter requêtes HTTP
        if any(method in payload for method in ["GET ", "POST ", "PUT ", "DELETE "]):
            self._analyze_http_request(capture, payload)

        # Détecter réponses JSON
        if payload.strip().startswith('{') and payload.strip().endswith('}'):
            self._analyze_json_response(capture, payload)

    def _analyze_http_request(self, capture: NetworkCapture, payload: str):
        """Analyse une requête HTTP"""
        try:
            lines = payload.split('\n')
            if not lines:
                return

            # Parse première ligne (méthode + URL)
            request_line = lines[0].strip()
            parts = request_line.split(' ')

            if len(parts) >= 3:
                method = parts[0]
                url = parts[1]
                host = capture.dest_ip

                # Extraire host des headers si possible
                for line in lines[1:]:
                    if line.lower().startswith('host:'):
                        host = line.split(':', 1)[1].strip()
                        break

                full_url = f"http://{host}{url}"

                # Créer ou mettre à jour endpoint
                endpoint_key = f"{method}:{full_url}"

                if endpoint_key not in self.api_endpoints:
                    self.api_endpoints[endpoint_key] = APIEndpoint(
                        url=full_url,
                        method=method,
                        parameters={},
                        response_format="unknown",
                        success_rate=1.0,
                        last_seen=capture.timestamp,
                        example_request=payload[:500]  # Première partie seulement
                    )
                else:
                    self.api_endpoints[endpoint_key].last_seen = capture.timestamp

        except Exception as e:
            print(f"❌ Erreur analyse requête HTTP: {e}")

    def _analyze_json_response(self, capture: NetworkCapture, payload: str):
        """Analyse une réponse JSON"""
        try:
            # Essayer de parser le JSON
            json_data = json.loads(payload.strip())

            # Chercher l'endpoint correspondant (requête récente)
            recent_endpoints = [
                ep for ep in self.api_endpoints.values()
                if (capture.timestamp - ep.last_seen) < 5.0  # 5 secondes
            ]

            if recent_endpoints:
                # Associer à l'endpoint le plus récent
                endpoint = recent_endpoints[-1]
                endpoint.response_format = "json"
                if not endpoint.example_response:
                    endpoint.example_response = payload[:500]

        except json.JSONDecodeError:
            pass  # Pas du JSON valide
        except Exception as e:
            print(f"❌ Erreur analyse JSON: {e}")

    def stop_network_capture(self):
        """Arrête la capture réseau"""
        if not self.is_capturing:
            return

        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5)

        print(f"⏹️ Capture arrêtée - {len(self.captured_packets)} paquets capturés")

    def get_detected_apis(self) -> Dict[str, APIEndpoint]:
        """Retourne les APIs détectées"""
        return self.api_endpoints

    def export_capture_data(self, output_dir: str) -> str:
        """Exporte les données de capture"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Exporter paquets
        packets_file = output_path / f"network_capture_{timestamp}.json"
        packets_data = [asdict(packet) for packet in self.captured_packets[-1000:]]  # 1000 derniers

        # Encoder bytes en base64 pour JSON
        for packet_data in packets_data:
            if packet_data['payload']:
                import base64
                packet_data['payload'] = base64.b64encode(packet_data['payload']).decode('ascii')

        with open(packets_file, 'w', encoding='utf-8') as f:
            json.dump(packets_data, f, indent=2, ensure_ascii=False)

        # Exporter APIs
        apis_file = output_path / f"detected_apis_{timestamp}.json"
        apis_data = {key: asdict(endpoint) for key, endpoint in self.api_endpoints.items()}

        with open(apis_file, 'w', encoding='utf-8') as f:
            json.dump(apis_data, f, indent=2, ensure_ascii=False)

        print(f"📁 Données exportées vers {output_path}")
        return str(output_path)

class ReverseEngineeringOrchestrator:
    """Orchestrateur principal du reverse engineering"""

    def __init__(self, output_dir: str = "reverse_engineering_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.process_analyzer = WindowsProcessAnalyzer()
        self.network_analyzer = NetworkTrafficAnalyzer()

        # Résultats
        self.discovered_processes: List[ApplicationProcess] = []
        self.analysis_sessions: Dict[str, Dict[str, Any]] = {}

    def discover_target_applications(self) -> List[ApplicationProcess]:
        """Découvre les applications cibles (Ganymede, DOFUS Guide, etc.)"""
        print("🔍 Recherche d'applications DOFUS...")

        processes = self.process_analyzer.find_dofus_related_processes()
        self.discovered_processes = processes

        if processes:
            print(f"✅ {len(processes)} processus trouvés:")
            for proc in processes:
                print(f"  - {proc.name} (PID: {proc.pid})")
                print(f"    Path: {proc.executable_path}")
                print(f"    Connexions: {len(proc.network_connections)}")
        else:
            print("❌ Aucun processus DOFUS trouvé")

        return processes

    def analyze_application(self, process: ApplicationProcess,
                          capture_duration: int = 120) -> Dict[str, Any]:
        """Analyse complète d'une application"""
        session_id = f"{process.name}_{process.pid}_{int(time.time())}"

        print(f"🔬 Analyse de {process.name} (PID: {process.pid})...")

        analysis_data = {
            "session_id": session_id,
            "process_info": asdict(process),
            "start_time": time.time(),
            "behavior_analysis": {},
            "network_capture": {},
            "discovered_apis": {},
            "recommendations": []
        }

        try:
            # Démarrer capture réseau
            print("📡 Démarrage capture réseau...")
            self.network_analyzer.start_network_capture()

            # Analyser comportement processus
            print("📊 Analyse comportement processus...")
            behavior = self.process_analyzer.analyze_process_behavior(
                process.pid, capture_duration
            )
            analysis_data["behavior_analysis"] = behavior

            # Attendre fin capture
            time.sleep(capture_duration)

            # Arrêter capture réseau
            self.network_analyzer.stop_network_capture()

            # Récupérer APIs découvertes
            discovered_apis = self.network_analyzer.get_detected_apis()
            analysis_data["discovered_apis"] = {
                key: asdict(endpoint) for key, endpoint in discovered_apis.items()
            }

            analysis_data["network_capture"] = {
                "total_packets": len(self.network_analyzer.captured_packets),
                "apis_found": len(discovered_apis),
                "capture_duration": capture_duration
            }

            # Générer recommandations
            analysis_data["recommendations"] = self._generate_integration_recommendations(
                process, discovered_apis
            )

            # Sauvegarder session
            self.analysis_sessions[session_id] = analysis_data
            self._save_analysis_session(session_id, analysis_data)

            print(f"✅ Analyse terminée - Session: {session_id}")
            print(f"📊 APIs découvertes: {len(discovered_apis)}")

            return analysis_data

        except Exception as e:
            print(f"❌ Erreur analyse: {e}")
            analysis_data["error"] = str(e)
            return analysis_data

    def _generate_integration_recommendations(self, process: ApplicationProcess,
                                            apis: Dict[str, APIEndpoint]) -> List[str]:
        """Génère des recommandations d'intégration"""
        recommendations = []

        # Analyser les APIs trouvées
        if apis:
            recommendations.append(f"🔗 {len(apis)} endpoints API détectés - intégration possible")

            # Recommandations spécifiques par type d'endpoint
            for endpoint in apis.values():
                if 'dofus' in endpoint.url.lower():
                    recommendations.append(f"📊 Endpoint DOFUS détecté: {endpoint.url}")

                if endpoint.response_format == 'json':
                    recommendations.append(f"📄 API JSON: {endpoint.method} {endpoint.url}")

        # Analyser les connexions réseau
        if process.network_connections:
            external_connections = [
                conn for conn in process.network_connections
                if not conn['remote_address'].startswith('127.0.0.1')
            ]

            if external_connections:
                recommendations.append(
                    f"🌐 {len(external_connections)} connexions externes détectées"
                )

        # Analyser les fichiers ouverts
        if process.open_files:
            config_files = [f for f in process.open_files if any(ext in f.lower() for ext in ['.json', '.xml', '.config', '.ini'])]

            if config_files:
                recommendations.append(
                    f"⚙️ {len(config_files)} fichiers de configuration détectés"
                )

        if not recommendations:
            recommendations.append("❓ Aucune intégration évidente détectée - analyse manuelle recommandée")

        return recommendations

    def _save_analysis_session(self, session_id: str, data: Dict[str, Any]):
        """Sauvegarde une session d'analyse"""
        try:
            session_dir = self.output_dir / session_id
            session_dir.mkdir(exist_ok=True)

            # Sauvegarder métadonnées
            metadata_file = session_dir / "analysis_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            # Exporter données de capture réseau
            if hasattr(self.network_analyzer, 'captured_packets'):
                self.network_analyzer.export_capture_data(str(session_dir))

            print(f"💾 Session sauvegardée: {session_dir}")

        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")

    def generate_integration_guide(self, session_id: str) -> str:
        """Génère un guide d'intégration basé sur l'analyse"""
        if session_id not in self.analysis_sessions:
            return "Session introuvable"

        data = self.analysis_sessions[session_id]
        process_info = data["process_info"]

        guide = f"""
# Guide d'Intégration - {process_info['name']}

## Informations Processus
- **Nom**: {process_info['name']}
- **PID**: {process_info['pid']}
- **Chemin**: {process_info['executable_path']}
- **Ligne de commande**: {process_info['command_line']}

## APIs Découvertes
"""

        apis = data.get("discovered_apis", {})
        if apis:
            guide += f"**{len(apis)} endpoints détectés:**\n\n"
            for key, endpoint in apis.items():
                guide += f"- **{endpoint['method']}** `{endpoint['url']}`\n"
                guide += f"  - Format: {endpoint['response_format']}\n"
                guide += f"  - Dernière activité: {datetime.fromtimestamp(endpoint['last_seen']).strftime('%H:%M:%S')}\n\n"
        else:
            guide += "Aucune API détectée.\n\n"

        guide += "## Recommandations d'Intégration\n\n"
        for recommendation in data.get("recommendations", []):
            guide += f"- {recommendation}\n"

        guide += f"""

## Métriques d'Analyse
- **Durée**: {data.get('behavior_analysis', {}).get('duration', 0)}s
- **Paquets capturés**: {data.get('network_capture', {}).get('total_packets', 0)}
- **Connexions réseau**: {len(process_info.get('network_connections', []))}
- **Fichiers ouverts**: {len(process_info.get('open_files', []))}

## Prochaines Étapes
1. Analyser les endpoints découverts manuellement
2. Tester les APIs avec des outils comme Postman
3. Implémenter des connecteurs spécifiques
4. Mettre en place la surveillance des changements
"""

        # Sauvegarder le guide
        try:
            guide_file = self.output_dir / session_id / "integration_guide.md"
            with open(guide_file, 'w', encoding='utf-8') as f:
                f.write(guide)
            print(f"📖 Guide généré: {guide_file}")
        except Exception as e:
            print(f"❌ Erreur génération guide: {e}")

        return guide

# Factory function
def get_reverse_engineering_framework(output_dir: str = "reverse_engineering_data") -> ReverseEngineeringOrchestrator:
    """Factory pour le framework de reverse engineering"""
    return ReverseEngineeringOrchestrator(output_dir)

if __name__ == "__main__":
    # Test du framework
    print("🔧 Framework de Reverse Engineering DOFUS")
    print("=" * 50)

    orchestrator = get_reverse_engineering_framework()

    # Découvrir applications
    processes = orchestrator.discover_target_applications()

    if processes:
        print(f"\\nAnalyse du premier processus trouvé: {processes[0].name}")
        print("⚠️ Ceci va capturer le trafic réseau pendant 30 secondes...")

        try:
            analysis = orchestrator.analyze_application(processes[0], capture_duration=30)
            print(f"✅ Analyse terminée - Session: {analysis['session_id']}")

            # Générer guide
            guide = orchestrator.generate_integration_guide(analysis['session_id'])
            print("\\n📖 Guide d'intégration généré")

        except KeyboardInterrupt:
            print("\\n⏹️ Analyse interrompue")
        except Exception as e:
            print(f"❌ Erreur: {e}")

    else:
        print("\\n⚠️ Aucun processus cible trouvé")
        print("Assurez-vous qu'une application DOFUS/Ganymede est en cours d'exécution")