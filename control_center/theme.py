"""Original neon-industrial visual language for the Jarvis desktop app."""
from __future__ import annotations

BACKGROUND = "#07090f"
SURFACE = "#0d111a"
SURFACE_ALT = "#121824"
YELLOW = "#f7e928"
CYAN = "#19e6f2"
MAGENTA = "#ff3b78"
TEXT = "#edf4f8"
MUTED = "#8493a6"
GREEN = "#4cff9a"


STYLE_SHEET = f"""
* {{
    font-family: "Segoe UI Variable", "Segoe UI";
    font-size: 10.5pt;
    color: {TEXT};
}}
QMainWindow, QDialog {{ background: {BACKGROUND}; }}
QWidget#root {{ background: {BACKGROUND}; }}
QFrame#sidebar {{
    background: #090c13;
    border-right: 1px solid #273143;
}}
QFrame#topbar {{
    background: #090c13;
    border-bottom: 1px solid #273143;
}}
QFrame#navIndicator {{
    background: {YELLOW};
    border: none;
    border-radius: 1px;
}}
QFrame#card {{
    background: {SURFACE};
    border: 1px solid #273143;
    border-radius: 10px;
}}
QFrame#accentCard {{
    background: {SURFACE};
    border: 1px solid {CYAN};
    border-radius: 10px;
}}
QFrame#heroCard {{
    background: rgba(11, 16, 25, 235);
    border: 1px solid #31435a;
    border-left: 3px solid {CYAN};
    border-radius: 12px;
}}
QFrame#quickCard {{
    background: rgba(13, 17, 26, 245);
    border: 1px solid #2b3749;
    border-radius: 9px;
}}
QFrame#quickCard:hover {{
    background: #101722;
    border-color: #4b5d75;
}}
QFrame#metric {{
    background: #090d14;
    border: 1px solid #263246;
    border-radius: 5px;
}}
QFrame#gestureVideoCard {{
    background: #090d14;
    border: 1px solid #2b3b50;
    border-radius: 10px;
}}
QFrame#gestureTelemetry {{
    background: {SURFACE};
    border: 1px solid #31435a;
    border-top: 2px solid {CYAN};
    border-radius: 10px;
}}
QFrame#chatListCard, QFrame#chatMessageCard,
QFrame#workspaceListCard, QFrame#workspaceEditorCard {{
    background: {SURFACE};
    border: 1px solid #2b3749;
    border-radius: 9px;
}}
QFrame#divider {{ color: #273143; background: #273143; border: none; max-height: 1px; }}
QLabel#brand {{
    color: {YELLOW};
    font-size: 18pt;
    font-weight: 800;
    letter-spacing: 2px;
}}
QLabel#eyebrow {{
    color: {CYAN};
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 2px;
}}
QLabel#eyebrowYellow {{
    color: {YELLOW};
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 2px;
}}
QLabel#eyebrowMagenta {{
    color: {MAGENTA};
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 2px;
}}
QLabel#title {{ font-size: 25pt; font-weight: 750; }}
QLabel#sectionTitle {{ font-size: 15pt; font-weight: 700; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#mutedSmall {{ color: #657286; font-size: 8pt; letter-spacing: 1px; }}
QLabel#heroTitle {{
    color: {TEXT};
    font-size: 29pt;
    font-weight: 800;
    letter-spacing: 3px;
}}
QLabel#heroStateOnline {{ color: {GREEN}; font-size: 12pt; font-weight: 800; letter-spacing: 1px; }}
QLabel#heroStateOffline {{ color: {MAGENTA}; font-size: 12pt; font-weight: 800; letter-spacing: 1px; }}
QLabel#privacyBadge {{
    color: {GREEN};
    background: rgba(76, 255, 154, 14);
    border: 1px solid rgba(76, 255, 154, 75);
    border-radius: 11px;
    padding: 5px 12px;
    font-size: 8pt;
    font-weight: 750;
    letter-spacing: 1px;
}}
QLabel#metricValue {{ color: {TEXT}; font-size: 11pt; font-weight: 800; }}
QLabel#metricCaption {{ color: {MUTED}; font-size: 7pt; letter-spacing: 1px; }}
QLabel#commandHint {{
    color: #65768a;
    font-family: "Cascadia Mono", Consolas;
    font-size: 7.5pt;
    padding-top: 2px;
}}
QLabel#gestureVideo {{
    color: #536176;
    background: #05070b;
    border: 1px solid #1c2736;
    border-radius: 6px;
    font-family: "Cascadia Mono", Consolas;
    font-size: 11pt;
    letter-spacing: 1px;
}}
QLabel#gestureBadgeOnline {{
    color: {GREEN};
    background: rgba(76, 255, 154, 14);
    border: 1px solid rgba(76, 255, 154, 75);
    border-radius: 11px;
    padding: 5px 12px;
    font-size: 8pt;
    font-weight: 750;
    letter-spacing: 1px;
}}
QLabel#gestureBadgeOffline {{
    color: {MUTED};
    background: rgba(132, 147, 166, 10);
    border: 1px solid #38465a;
    border-radius: 11px;
    padding: 5px 12px;
    font-size: 8pt;
    font-weight: 750;
    letter-spacing: 1px;
}}
QLabel#gestureLabel {{ color: {YELLOW}; font-size: 32pt; font-weight: 850; letter-spacing: 3px; }}
QLabel#gestureRank {{
    color: {TEXT};
    background: #090d14;
    border: 1px solid #263246;
    border-radius: 4px;
    padding: 8px 10px;
    font-family: "Cascadia Mono", Consolas;
}}
QLabel#gesturePerformance {{
    color: {CYAN};
    font-family: "Cascadia Mono", Consolas;
    font-size: 9pt;
    font-weight: 700;
}}
QLabel#memorySummary {{
    color: {MUTED};
    background: #090d14;
    border: 1px solid #263246;
    border-left: 2px solid {MAGENTA};
    border-radius: 5px;
    padding: 9px 12px;
    font-size: 8.5pt;
}}
QLabel#workspaceStatus {{
    color: {CYAN};
    font-family: "Cascadia Mono", Consolas;
    font-size: 8pt;
    font-weight: 700;
}}
QLabel#statusOnline {{ color: {GREEN}; font-weight: 700; }}
QLabel#statusOffline {{ color: {MAGENTA}; font-weight: 700; }}
QPushButton {{
    min-height: 38px;
    padding: 0 16px;
    background: {SURFACE_ALT};
    border: 1px solid #334157;
    border-radius: 5px;
    font-weight: 650;
}}
QPushButton:hover {{ border-color: {CYAN}; color: {CYAN}; background: #101b27; }}
QPushButton:pressed {{ background: #172739; }}
QPushButton:disabled {{ color: #566071; border-color: #252d39; }}
QPushButton#primary {{
    color: #080a0e;
    background: {YELLOW};
    border: 1px solid {YELLOW};
    font-weight: 800;
}}
QPushButton#primary:hover {{ background: #fff56a; border-color: #fff56a; }}
QPushButton#primary:disabled {{
    color: rgba(247, 233, 40, 90);
    background: rgba(247, 233, 40, 28);
    border-color: rgba(247, 233, 40, 55);
}}
QPushButton#danger {{ border-color: {MAGENTA}; color: {MAGENTA}; }}
QPushButton#danger:hover {{ background: #31111e; }}
QPushButton#danger:disabled {{
    color: rgba(255, 59, 120, 80);
    background: rgba(255, 59, 120, 12);
    border-color: rgba(255, 59, 120, 45);
}}
QPushButton#nav {{
    min-height: 44px;
    text-align: left;
    padding-left: 18px;
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
    color: {MUTED};
}}
QPushButton#nav:hover {{ color: {TEXT}; background: #10151f; border-color: #3c4b61; }}
QPushButton#nav:checked {{ color: {YELLOW}; background: #171a18; border-color: transparent; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    min-height: 36px;
    padding: 0 10px;
    background: #080b12;
    border: 1px solid #2d394c;
    border-radius: 5px;
    selection-background-color: {CYAN};
    selection-color: #05070b;
}}
QPlainTextEdit {{ padding: 10px; font-family: "Cascadia Mono", Consolas; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus {{ border-color: {CYAN}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QCheckBox {{ spacing: 9px; }}
QCheckBox::indicator {{ width: 35px; height: 18px; border-radius: 9px;
    background: #262f3d; border: 1px solid #3d4b61; }}
QCheckBox::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QTabWidget::pane {{ border: 1px solid #273143; border-radius: 6px; top: -1px; }}
QTabBar::tab {{ padding: 10px 18px; color: {MUTED}; background: #0b0f17; }}
QTabBar::tab:selected {{ color: {YELLOW}; border-bottom: 2px solid {YELLOW}; }}
QTableWidget {{
    background: #080b12;
    alternate-background-color: #0d121b;
    gridline-color: #202a39;
    border: 1px solid #273143;
}}
QListWidget#chatList {{
    background: #080b12;
    border: none;
    outline: none;
}}
QListWidget#chatList::item {{
    color: {MUTED};
    padding: 11px 10px;
    margin-bottom: 4px;
    border-left: 2px solid transparent;
}}
QListWidget#chatList::item:hover {{ color: {TEXT}; background: #101722; }}
QListWidget#chatList::item:selected {{
    color: {YELLOW};
    background: #171a18;
    border-left-color: {YELLOW};
}}
QListWidget#workspaceList, QListWidget#workspaceResources {{
    background: #080b12;
    border: 1px solid #202a39;
    border-radius: 5px;
    outline: none;
}}
QListWidget#workspaceList::item {{
    color: {MUTED};
    padding: 12px 10px;
    margin: 3px;
    border-left: 2px solid transparent;
}}
QListWidget#workspaceList::item:hover {{ color: {TEXT}; background: #101722; }}
QListWidget#workspaceList::item:selected {{
    color: {YELLOW};
    background: #171a18;
    border-left-color: {YELLOW};
}}
QListWidget#workspaceResources::item {{ color: {MUTED}; padding: 7px; }}
QWidget#workspaceCanvas {{
    background: #080c13;
    border: 1px solid #2b3b50;
    border-radius: 9px;
}}
QTextBrowser#chatMessages {{
    background: #080b12;
    border: 1px solid #202a39;
    border-radius: 5px;
    padding: 8px;
}}
QHeaderView::section {{ background: #111722; color: {CYAN}; padding: 8px;
    border: none; border-right: 1px solid #273143; font-weight: 700; }}
QScrollBar:vertical {{ background: #080b12; width: 10px; }}
QScrollBar::handle:vertical {{ background: #34445b; min-height: 28px; border-radius: 5px; }}
QProgressBar {{ background: #080b12; border: 1px solid #2d394c; border-radius: 4px;
    text-align: center; min-height: 14px; }}
QProgressBar::chunk {{ background: {CYAN}; }}
QProgressBar#gestureConfidence {{
    min-height: 22px;
    background: #080b12;
    border: 1px solid #2d394c;
    border-radius: 4px;
    color: {TEXT};
    font-family: "Cascadia Mono", Consolas;
    font-size: 8pt;
    text-align: center;
}}
QProgressBar#gestureConfidence::chunk {{ background: {YELLOW}; border-radius: 3px; }}
QToolTip {{ color: {TEXT}; background: #111722; border: 1px solid {CYAN}; padding: 6px; }}
"""
