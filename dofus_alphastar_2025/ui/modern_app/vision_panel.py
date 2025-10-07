"""
Vision Panel - Système de visualisation et de gestion des screenshots
Affichage temps réel, historique, détections, calibration
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import os
import queue
import threading
from pathlib import Path


@dataclass
class ScreenshotData:
    """Données d'un screenshot avec métadonnées"""
    timestamp: datetime
    image_path: str
    thumbnail_path: Optional[str]
    detections: List[Dict[str, Any]]  # Liste des détections (bbox, label, confidence)
    screenshot_type: str  # 'game', 'map', 'inventory', 'combat', etc.
    game_state: Dict[str, Any]  # État du jeu au moment du screenshot
    notes: Optional[str] = None


class VisionPanel:
    """
    Panel de visualisation avancée pour le système de vision

    Fonctionnalités:
    - Affichage temps réel des screenshots
    - Historique avec miniatures
    - Overlays de détection
    - Mode debug avec toutes les couches
    - Interface de calibration
    - Export de screenshots annotés
    - Comparaison avant/après
    """

    def __init__(self, parent):
        self.parent = parent
        self.screenshot_queue = queue.Queue()
        self.screenshots_history: List[ScreenshotData] = []
        self.current_screenshot: Optional[ScreenshotData] = None
        self.current_image: Optional[Image.Image] = None
        self.show_detections = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_confidence = tk.BooleanVar(value=True)
        self.debug_mode = tk.BooleanVar(value=False)
        self.auto_refresh = tk.BooleanVar(value=True)

        # Créer les dossiers si nécessaire
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        self.thumbnails_dir = self.screenshots_dir / "thumbnails"
        self.thumbnails_dir.mkdir(exist_ok=True)

        self._setup_ui()
        self._start_screenshot_processing()

    def _setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principal
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === Toolbar supérieure ===
        toolbar = ttk.Frame(self.main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))

        # Boutons de contrôle
        ttk.Button(toolbar, text="📸 Capture Manuel",
                  command=self.manual_capture).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔄 Rafraîchir",
                  command=self.refresh_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Sauvegarder",
                  command=self.save_screenshot).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📂 Ouvrir dossier",
                  command=self.open_screenshots_folder).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Checkboxes
        ttk.Checkbutton(toolbar, text="🎯 Détections",
                       variable=self.show_detections,
                       command=self.update_display).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="🏷️ Labels",
                       variable=self.show_labels,
                       command=self.update_display).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="📊 Confiance",
                       variable=self.show_confidence,
                       command=self.update_display).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="🔍 Debug Mode",
                       variable=self.debug_mode,
                       command=self.update_display).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="⚡ Auto-refresh",
                       variable=self.auto_refresh).pack(side=tk.LEFT, padx=2)

        # === PanedWindow pour layout flexible ===
        paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # === Panneau gauche: Historique ===
        left_frame = ttk.LabelFrame(paned, text="📜 Historique", padding=5)
        paned.add(left_frame, weight=1)

        # Liste des screenshots avec scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.screenshots_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 9),
            height=20
        )
        self.screenshots_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.screenshots_listbox.yview)
        self.screenshots_listbox.bind("<<ListboxSelect>>", self.on_screenshot_select)

        # Filtres
        filter_frame = ttk.LabelFrame(left_frame, text="🔍 Filtres", padding=5)
        filter_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(filter_frame, text="Type:").grid(row=0, column=0, sticky=tk.W)
        self.filter_type = ttk.Combobox(filter_frame, values=[
            "Tous", "game", "map", "inventory", "combat", "menu", "calibration"
        ], state="readonly", width=15)
        self.filter_type.set("Tous")
        self.filter_type.grid(row=0, column=1, padx=(5, 0))
        self.filter_type.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        ttk.Button(filter_frame, text="🗑️ Effacer historique",
                  command=self.clear_history).grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky=tk.EW)

        # === Panneau central: Affichage principal ===
        center_frame = ttk.LabelFrame(paned, text="🖼️ Vue Principale", padding=5)
        paned.add(center_frame, weight=3)

        # Canvas pour affichage d'image avec zoom/pan
        self.canvas_frame = ttk.Frame(center_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg='#2b2b2b',
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)

        # Bind events pour zoom/pan
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        self.zoom_level = 1.0
        self.pan_start = None

        # Infos en bas du canvas
        info_frame = ttk.Frame(center_frame)
        info_frame.pack(fill=tk.X, pady=(5, 0))

        self.info_label = ttk.Label(info_frame, text="Aucun screenshot chargé",
                                    font=("Segoe UI", 9))
        self.info_label.pack(side=tk.LEFT)

        # Contrôles zoom
        zoom_frame = ttk.Frame(info_frame)
        zoom_frame.pack(side=tk.RIGHT)

        ttk.Button(zoom_frame, text="➖", width=3,
                  command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="➕", width=3,
                  command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔄 Reset", width=8,
                  command=self.reset_zoom).pack(side=tk.LEFT, padx=2)

        # === Panneau droit: Détails et outils ===
        right_frame = ttk.LabelFrame(paned, text="ℹ️ Détails & Outils", padding=5)
        paned.add(right_frame, weight=1)

        # Notebook pour organiser les outils
        tools_notebook = ttk.Notebook(right_frame)
        tools_notebook.pack(fill=tk.BOTH, expand=True)

        # --- Onglet Détections ---
        detections_tab = ttk.Frame(tools_notebook, padding=5)
        tools_notebook.add(detections_tab, text="🎯 Détections")

        ttk.Label(detections_tab, text="Objets détectés:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)

        det_list_frame = ttk.Frame(detections_tab)
        det_list_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        det_scroll = ttk.Scrollbar(det_list_frame)
        det_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.detections_tree = ttk.Treeview(
            det_list_frame,
            columns=("label", "confidence", "bbox"),
            show="tree headings",
            yscrollcommand=det_scroll.set,
            height=10
        )
        self.detections_tree.heading("label", text="Label")
        self.detections_tree.heading("confidence", text="Conf.")
        self.detections_tree.heading("bbox", text="BBox")
        self.detections_tree.column("#0", width=30)
        self.detections_tree.column("label", width=100)
        self.detections_tree.column("confidence", width=60)
        self.detections_tree.column("bbox", width=120)
        self.detections_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        det_scroll.config(command=self.detections_tree.yview)

        # Stats détections
        stats_frame = ttk.LabelFrame(detections_tab, text="📊 Statistiques", padding=5)
        stats_frame.pack(fill=tk.X, pady=(5, 0))

        self.stats_text = tk.Text(stats_frame, height=4, font=("Consolas", 8),
                                 wrap=tk.WORD, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # --- Onglet Métadonnées ---
        metadata_tab = ttk.Frame(tools_notebook, padding=5)
        tools_notebook.add(metadata_tab, text="📋 Métadonnées")

        self.metadata_text = tk.Text(metadata_tab, font=("Consolas", 8),
                                     wrap=tk.WORD, state=tk.DISABLED)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)

        # --- Onglet Calibration ---
        calibration_tab = ttk.Frame(tools_notebook, padding=5)
        tools_notebook.add(calibration_tab, text="🎚️ Calibration")

        ttk.Label(calibration_tab, text="Outils de calibration:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        ttk.Button(calibration_tab, text="📐 Calibrer zones UI",
                  command=self.calibrate_ui_zones).pack(fill=tk.X, pady=2)
        ttk.Button(calibration_tab, text="🎨 Calibrer couleurs",
                  command=self.calibrate_colors).pack(fill=tk.X, pady=2)
        ttk.Button(calibration_tab, text="🔢 Calibrer OCR",
                  command=self.calibrate_ocr).pack(fill=tk.X, pady=2)
        ttk.Button(calibration_tab, text="🗺️ Calibrer minimap",
                  command=self.calibrate_minimap).pack(fill=tk.X, pady=2)

        ttk.Separator(calibration_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(calibration_tab, text="Tests:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        ttk.Button(calibration_tab, text="🧪 Test détection HP/PM",
                  command=self.test_hp_pm_detection).pack(fill=tk.X, pady=2)
        ttk.Button(calibration_tab, text="🧪 Test OCR position",
                  command=self.test_ocr_position).pack(fill=tk.X, pady=2)
        ttk.Button(calibration_tab, text="🧪 Test détection combat",
                  command=self.test_combat_detection).pack(fill=tk.X, pady=2)

        # --- Onglet Export ---
        export_tab = ttk.Frame(tools_notebook, padding=5)
        tools_notebook.add(export_tab, text="💾 Export")

        ttk.Label(export_tab, text="Options d'export:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.export_with_detections = tk.BooleanVar(value=True)
        self.export_with_labels = tk.BooleanVar(value=True)
        self.export_with_metadata = tk.BooleanVar(value=False)

        ttk.Checkbutton(export_tab, text="Inclure détections",
                       variable=self.export_with_detections).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(export_tab, text="Inclure labels",
                       variable=self.export_with_labels).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(export_tab, text="Inclure métadonnées (JSON)",
                       variable=self.export_with_metadata).pack(anchor=tk.W, pady=2)

        ttk.Separator(export_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(export_tab, text="💾 Exporter screenshot actuel",
                  command=self.export_current).pack(fill=tk.X, pady=2)
        ttk.Button(export_tab, text="📦 Exporter sélection",
                  command=self.export_selection).pack(fill=tk.X, pady=2)
        ttk.Button(export_tab, text="📦 Exporter tout l'historique",
                  command=self.export_all).pack(fill=tk.X, pady=2)

        # --- Onglet Notes ---
        notes_tab = ttk.Frame(tools_notebook, padding=5)
        tools_notebook.add(notes_tab, text="📝 Notes")

        ttk.Label(notes_tab, text="Notes sur ce screenshot:",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.notes_text = tk.Text(notes_tab, font=("Segoe UI", 9),
                                 wrap=tk.WORD, height=10)
        self.notes_text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(notes_tab, text="💾 Sauvegarder notes",
                  command=self.save_notes).pack(fill=tk.X, pady=(5, 0))

    def _start_screenshot_processing(self):
        """Démarre le thread de traitement des screenshots"""
        self.processing_thread = threading.Thread(
            target=self._process_screenshots,
            daemon=True
        )
        self.processing_thread.start()

    def _process_screenshots(self):
        """Traite les screenshots de la queue"""
        while True:
            try:
                screenshot_data = self.screenshot_queue.get(timeout=0.1)
                self.screenshots_history.append(screenshot_data)

                # Mettre à jour l'affichage si auto-refresh
                if self.auto_refresh.get():
                    self.parent.after(0, self._update_screenshot_list)
                    self.parent.after(0, lambda: self.display_screenshot(screenshot_data))

            except queue.Empty:
                continue

    def add_screenshot(self, image_path: str, detections: List[Dict[str, Any]] = None,
                      screenshot_type: str = "game", game_state: Dict[str, Any] = None,
                      notes: str = None):
        """Ajoute un screenshot à la queue (thread-safe)"""
        screenshot_data = ScreenshotData(
            timestamp=datetime.now(),
            image_path=image_path,
            thumbnail_path=None,  # TODO: générer thumbnail
            detections=detections or [],
            screenshot_type=screenshot_type,
            game_state=game_state or {},
            notes=notes
        )
        self.screenshot_queue.put(screenshot_data)

    def _update_screenshot_list(self):
        """Met à jour la liste des screenshots"""
        self.screenshots_listbox.delete(0, tk.END)

        filtered = self._get_filtered_screenshots()

        for screenshot in reversed(filtered):  # Plus récent en haut
            time_str = screenshot.timestamp.strftime("%H:%M:%S")
            type_emoji = self._get_type_emoji(screenshot.screenshot_type)
            det_count = len(screenshot.detections)

            display = f"{time_str} {type_emoji} [{det_count} dét.]"
            self.screenshots_listbox.insert(tk.END, display)

    def _get_filtered_screenshots(self) -> List[ScreenshotData]:
        """Retourne les screenshots filtrés"""
        filter_type = self.filter_type.get()

        if filter_type == "Tous":
            return self.screenshots_history
        else:
            return [s for s in self.screenshots_history if s.screenshot_type == filter_type]

    def _get_type_emoji(self, screenshot_type: str) -> str:
        """Retourne l'emoji correspondant au type"""
        emojis = {
            "game": "🎮",
            "map": "🗺️",
            "inventory": "🎒",
            "combat": "⚔️",
            "menu": "📋",
            "calibration": "🎚️"
        }
        return emojis.get(screenshot_type, "📸")

    def on_screenshot_select(self, event):
        """Gère la sélection d'un screenshot"""
        selection = self.screenshots_listbox.curselection()
        if not selection:
            return

        filtered = self._get_filtered_screenshots()
        reversed_filtered = list(reversed(filtered))

        if selection[0] < len(reversed_filtered):
            screenshot = reversed_filtered[selection[0]]
            self.display_screenshot(screenshot)

    def display_screenshot(self, screenshot: ScreenshotData):
        """Affiche un screenshot sur le canvas"""
        self.current_screenshot = screenshot

        try:
            # Charger l'image
            if not os.path.exists(screenshot.image_path):
                messagebox.showerror("Erreur", f"Image introuvable: {screenshot.image_path}")
                return

            self.current_image = Image.open(screenshot.image_path)
            self.update_display()
            self.update_detections_tree()
            self.update_metadata()
            self.load_notes()

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image: {e}")

    def update_display(self):
        """Met à jour l'affichage du canvas avec overlays"""
        if not self.current_image or not self.current_screenshot:
            return

        # Copier l'image pour ne pas modifier l'originale
        display_image = self.current_image.copy()

        # Ajouter les overlays si activés
        if self.show_detections.get() and self.current_screenshot.detections:
            display_image = self._draw_detections(display_image)

        # Mode debug: ajouter grilles, zones, etc.
        if self.debug_mode.get():
            display_image = self._draw_debug_overlays(display_image)

        # Appliquer le zoom
        width = int(display_image.width * self.zoom_level)
        height = int(display_image.height * self.zoom_level)
        display_image = display_image.resize((width, height), Image.Resampling.LANCZOS)

        # Convertir pour Tkinter
        self.photo = ImageTk.PhotoImage(display_image)

        # Afficher sur le canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Mettre à jour les infos
        self._update_info_label()

    def _draw_detections(self, image: Image.Image) -> Image.Image:
        """Dessine les détections sur l'image"""
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        for det in self.current_screenshot.detections:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                # Rectangle de détection
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)

                # Label et confiance
                if self.show_labels.get():
                    label = det.get("label", "Unknown")
                    confidence = det.get("confidence", 0.0)

                    if self.show_confidence.get():
                        text = f"{label} ({confidence:.2f})"
                    else:
                        text = label

                    # Background pour le texte
                    text_bbox = draw.textbbox((x1, y1 - 20), text, font=font)
                    draw.rectangle(text_bbox, fill="lime")
                    draw.text((x1, y1 - 20), text, fill="black", font=font)

        return image

    def _draw_debug_overlays(self, image: Image.Image) -> Image.Image:
        """Dessine les overlays de debug"""
        draw = ImageDraw.Draw(image)

        # Grille
        width, height = image.size
        grid_size = 50

        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill="gray", width=1)
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill="gray", width=1)

        # Zones UI communes (exemple pour DOFUS)
        # Minimap (en haut à droite)
        minimap_x = width - 200
        minimap_y = 0
        draw.rectangle([minimap_x, minimap_y, width, 200], outline="cyan", width=2)
        draw.text((minimap_x + 5, minimap_y + 5), "Minimap", fill="cyan")

        # Zone de vie/PM (en bas)
        draw.rectangle([width // 2 - 100, height - 50, width // 2 + 100, height],
                      outline="red", width=2)
        draw.text((width // 2 - 90, height - 45), "HP/PM", fill="red")

        return image

    def update_detections_tree(self):
        """Met à jour l'arbre des détections"""
        self.detections_tree.delete(*self.detections_tree.get_children())

        if not self.current_screenshot:
            return

        for i, det in enumerate(self.current_screenshot.detections):
            label = det.get("label", "Unknown")
            confidence = det.get("confidence", 0.0)
            bbox = det.get("bbox", [])
            bbox_str = f"[{','.join(map(str, bbox))}]" if bbox else "N/A"

            self.detections_tree.insert("", tk.END, text=str(i+1),
                                       values=(label, f"{confidence:.2f}", bbox_str))

        # Mettre à jour les stats
        self._update_detection_stats()

    def _update_detection_stats(self):
        """Met à jour les statistiques de détection"""
        if not self.current_screenshot:
            return

        detections = self.current_screenshot.detections
        total = len(detections)

        if total == 0:
            stats_text = "Aucune détection"
        else:
            # Compter par type
            types_count = {}
            avg_confidence = 0.0

            for det in detections:
                label = det.get("label", "Unknown")
                types_count[label] = types_count.get(label, 0) + 1
                avg_confidence += det.get("confidence", 0.0)

            avg_confidence /= total

            stats_text = f"Total: {total} détections\n"
            stats_text += f"Confiance moy.: {avg_confidence:.2f}\n\n"
            stats_text += "Par type:\n"
            for label, count in sorted(types_count.items(), key=lambda x: -x[1]):
                stats_text += f"  {label}: {count}\n"

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", stats_text)
        self.stats_text.config(state=tk.DISABLED)

    def update_metadata(self):
        """Met à jour l'affichage des métadonnées"""
        if not self.current_screenshot:
            return

        metadata = f"Timestamp: {self.current_screenshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        metadata += f"Type: {self.current_screenshot.screenshot_type}\n"
        metadata += f"Chemin: {self.current_screenshot.image_path}\n"
        metadata += f"\nÉtat du jeu:\n"

        for key, value in self.current_screenshot.game_state.items():
            metadata += f"  {key}: {value}\n"

        self.metadata_text.config(state=tk.NORMAL)
        self.metadata_text.delete("1.0", tk.END)
        self.metadata_text.insert("1.0", metadata)
        self.metadata_text.config(state=tk.DISABLED)

    def load_notes(self):
        """Charge les notes du screenshot"""
        self.notes_text.delete("1.0", tk.END)
        if self.current_screenshot and self.current_screenshot.notes:
            self.notes_text.insert("1.0", self.current_screenshot.notes)

    def save_notes(self):
        """Sauvegarde les notes du screenshot"""
        if not self.current_screenshot:
            return

        notes = self.notes_text.get("1.0", tk.END).strip()
        self.current_screenshot.notes = notes
        messagebox.showinfo("Sauvegardé", "Notes sauvegardées")

    def _update_info_label(self):
        """Met à jour le label d'information"""
        if not self.current_screenshot or not self.current_image:
            self.info_label.config(text="Aucun screenshot chargé")
            return

        width, height = self.current_image.size
        det_count = len(self.current_screenshot.detections)
        time_str = self.current_screenshot.timestamp.strftime("%H:%M:%S")

        info = f"📸 {time_str} | 📐 {width}x{height} | 🎯 {det_count} détections"
        self.info_label.config(text=info)

        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")

    # === Contrôles de zoom/pan ===

    def zoom_in(self):
        self.zoom_level *= 1.2
        self.update_display()

    def zoom_out(self):
        self.zoom_level /= 1.2
        self.update_display()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.update_display()

    def on_mouse_wheel(self, event):
        """Zoom avec la molette"""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def on_canvas_click(self, event):
        """Début du drag"""
        self.pan_start = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def on_canvas_drag(self, event):
        """Pan/déplacement"""
        if self.pan_start:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    # === Actions ===

    def manual_capture(self):
        """Capture manuelle d'un screenshot"""
        messagebox.showinfo("Capture", "Fonctionnalité à implémenter: capture via vision engine")

    def refresh_view(self):
        """Rafraîchit l'affichage"""
        if self.current_screenshot:
            self.display_screenshot(self.current_screenshot)

    def save_screenshot(self):
        """Sauvegarde le screenshot actuel"""
        if not self.current_screenshot:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")]
        )

        if file_path:
            self.current_image.save(file_path)
            messagebox.showinfo("Sauvegardé", f"Screenshot sauvegardé: {file_path}")

    def open_screenshots_folder(self):
        """Ouvre le dossier des screenshots"""
        import subprocess
        subprocess.Popen(f'explorer "{os.path.abspath(self.screenshots_dir)}"')

    def apply_filters(self):
        """Applique les filtres à la liste"""
        self._update_screenshot_list()

    def clear_history(self):
        """Efface l'historique"""
        if messagebox.askyesno("Confirmer", "Effacer tout l'historique?"):
            self.screenshots_history.clear()
            self.screenshots_listbox.delete(0, tk.END)
            self.canvas.delete("all")
            self.current_screenshot = None
            self.current_image = None

    # === Calibration (stubs) ===

    def calibrate_ui_zones(self):
        messagebox.showinfo("Calibration", "Fonctionnalité à implémenter: calibration des zones UI")

    def calibrate_colors(self):
        messagebox.showinfo("Calibration", "Fonctionnalité à implémenter: calibration des couleurs")

    def calibrate_ocr(self):
        messagebox.showinfo("Calibration", "Fonctionnalité à implémenter: calibration OCR")

    def calibrate_minimap(self):
        messagebox.showinfo("Calibration", "Fonctionnalité à implémenter: calibration minimap")

    def test_hp_pm_detection(self):
        messagebox.showinfo("Test", "Fonctionnalité à implémenter: test détection HP/PM")

    def test_ocr_position(self):
        messagebox.showinfo("Test", "Fonctionnalité à implémenter: test OCR position")

    def test_combat_detection(self):
        messagebox.showinfo("Test", "Fonctionnalité à implémenter: test détection combat")

    # === Export ===

    def export_current(self):
        """Exporte le screenshot actuel"""
        if not self.current_screenshot:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")]
        )

        if file_path:
            # TODO: exporter avec options sélectionnées
            self.current_image.save(file_path)
            messagebox.showinfo("Exporté", f"Screenshot exporté: {file_path}")

    def export_selection(self):
        messagebox.showinfo("Export", "Fonctionnalité à implémenter: export de la sélection")

    def export_all(self):
        messagebox.showinfo("Export", "Fonctionnalité à implémenter: export de tout l'historique")

    def get_panel(self) -> ttk.Frame:
        """Retourne le frame principal"""
        return self.main_frame
