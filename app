"""
WasteNet AI - Main Application GUI
Author: Yahav
Date: May 2026

Description: Real-time waste classification application using PyQt5, OpenCV, and the trained ResNet18 model. Includes a live camera feed, confidence distribution UI, and a local SQLite database for scan history.
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import sqlite3
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QProgressBar, QTableWidget, 
    QTableWidgetItem, QHeaderView, QAbstractItemView, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# ─── Constants & Colors ────────────────────────────────────────────────────────

LABELS = ["Blue", "Brown", "Cardboards", "Electronics", "Green", "Orange", "Purple", "Clothes"]
COLORS = {
    "Blue": "#3b82f6", "Brown": "#8b5a2b", "Cardboards": "#d2b48c",
    "Electronics": "#ec4899", "Green": "#10b981", "Orange": "#f97316",
    "Purple": "#a855f7", "Clothes": "#ef4444"
}

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
DIM = "#8b949e"
TEXT_LIGHT = "#c9d1d9"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Database Manager (CRUD) ──────────────────────────────────────────────────
class DBManager:
    def __init__(self, db_name="waste_history.db"):
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_name)
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS scans
                                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                  timestamp TEXT,
                                  predicted_class TEXT,
                                  confidence REAL)''')

    def insert_scan(self, predicted_class, confidence):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.conn:
            self.conn.execute("INSERT INTO scans (timestamp, predicted_class, confidence) VALUES (?, ?, ?)",
                              (timestamp, predicted_class, confidence))

    def get_recent_scans(self, limit=5):
        cursor = self.conn.cursor()
        cursor.execute("SELECT timestamp, predicted_class, confidence FROM scans ORDER BY id DESC LIMIT ?", (limit,))
        return cursor.fetchall()

    def clear_history(self):
        with self.conn:
            self.conn.execute("DELETE FROM scans")


# ─── Preprocessing ────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─── Widgets ───────────────────────────────────────────────────────────────
class CameraWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1500, 1100)
        self.setStyleSheet(f"background:{BG}; border:3px solid {BORDER}; border-radius: 15px;")
        self._raw = None
        self.current_border_color = (136, 255, 0)

        # גודל הריבוע שבו רואים את המצלמה
        self.roi_size = 500

    def set_frame(self, frame, border_hex=None):
        self._raw = frame
        f = cv2.flip(cv2.resize(frame, (1500, 1100)), 1)

        if border_hex:
            h = border_hex.lstrip('#')
            rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            self.current_border_color = (rgb[2], rgb[1], rgb[0])

        H, W = f.shape[:2]
        cx, cy = W // 2, H // 2

        # חישוב הגבולות של הריבוע המרכזי
        x1, y1 = cx - self.roi_size // 2, cy - self.roi_size // 2
        x2, y2 = cx + self.roi_size // 2, cy + self.roi_size // 2

        # 1. יצירת קנבס יעיל ומהיר שכולו לבן
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)

        # 2. הדבקת המצלמה אך ורק בתוך הריבוע המרכזי (פעולה שלוקחת 0 אלפיות השנייה)
        canvas[y1:y2, x1:x2] = f[y1:y2, x1:x2]

        # 3. ציור מסגרת סביב החלון להכוונת המשתמש
        c = self.current_border_color
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 6)

        # כיתוב מעל הריבוע
        cv2.putText(canvas, "FILL SQUARE WITH WASTE", (cx - 210, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 4)

        rgb_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], rgb_img.shape[1] * 3, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img))

    def get_roi(self):
        if self._raw is None: return None
        f = cv2.flip(cv2.resize(self._raw, (1500, 1100)), 1)
        H, W = f.shape[:2]
        cx, cy = W // 2, H // 2

        # גוזרים למודל בדיוק את מה שיש בתוך הריבוע
        x1, y1 = cx - self.roi_size // 2, cy - self.roi_size // 2
        x2, y2 = cx + self.roi_size // 2, cy + self.roi_size // 2
        return f[y1:y2, x1:x2]


class ClassRow(QWidget):
    def __init__(self, cls):
        super().__init__()
        self.setStyleSheet(f"background:transparent;")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 5, 0, 5)
        row.setSpacing(15)
        self.dot = QLabel()
        self.dot.setFixedSize(20, 20)
        self.dot.setStyleSheet(f"background:{COLORS[cls]}; border-radius:10px;")
        row.addWidget(self.dot)
        self.nm = QLabel(cls)
        self.nm.setFixedWidth(180)
        self.nm.setStyleSheet(f"color:{DIM}; font-family:Segoe UI, sans-serif; font-size:18px; font-weight:bold;")
        row.addWidget(self.nm)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(12)
        self.bar.setStyleSheet(
            f"QProgressBar{{background:{BORDER}; border-radius:6px; border:none;}} QProgressBar::chunk{{background:{COLORS[cls]}; border-radius:6px; opacity:0.3;}}")
        row.addWidget(self.bar)
        self.val = QLabel("0%")
        self.val.setFixedWidth(80)
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val.setStyleSheet(f"color:{DIM}; font-family:Segoe UI; font-size:18px; font-weight:bold;")
        row.addWidget(self.val)

    def update(self, pct, is_top, color_hex):
        self.bar.setValue(int(pct))
        current_color = color_hex if is_top else DIM
        if is_top:
            self.nm.setStyleSheet(f"color:{TEXT_LIGHT}; font-family:Segoe UI; font-size:20px; font-weight:bold;")
            self.bar.setStyleSheet(
                f"QProgressBar{{background:{BORDER}; border-radius:6px; border:none;}} QProgressBar::chunk{{background:{current_color}; border-radius:6px;}}")
        else:
            self.nm.setStyleSheet(f"color:{DIM}; font-family:Segoe UI; font-size:18px; font-weight:bold;")
            self.bar.setStyleSheet(
                f"QProgressBar{{background:{BORDER}; border-radius:6px; border:none;}} QProgressBar::chunk{{background:{DIM}; border-radius:6px;}}")
        self.val.setStyleSheet(f"color:{current_color}; font-family:Segoe UI; font-size:18px; font-weight:bold;")
        self.val.setText(f"{pct:.1f}%")


# ─── Main Window ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.db = DBManager()
        self.setWindowTitle("WasteNet AI Dashboard")
        self.setStyleSheet(f"background:{BG};")
        self.showMaximized()

        c = QWidget()
        self.setCentralWidget(c)
        root = QHBoxLayout(c)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(40)

        # Left Column - Camera & Button
        left = QVBoxLayout()
        left.setSpacing(30)
        self.cam = CameraWidget()
        left.addWidget(self.cam, alignment=Qt.AlignCenter)

        self.btn = QPushButton("SCAN WASTE (SPACE)")
        self.btn.setFixedHeight(90)
        self.btn.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.btn.setStyleSheet(f"""
            QPushButton {{
                background: {PANEL}; color: {TEXT_LIGHT}; 
                border: 3px solid {DIM}; border-radius: 15px;
            }}
            QPushButton:hover {{ background: {BORDER}; }}
            QPushButton:pressed {{ background: {BG}; }}
        """)
        self.btn.clicked.connect(self._classify)
        left.addWidget(self.btn)
        root.addLayout(left, 2)

        # Right Panel - Dashboard
        rf = QFrame()
        rf.setFixedWidth(700)
        rf.setStyleSheet(f"background:{PANEL}; border: 1px solid {BORDER}; border-radius:20px;")
        rv = QVBoxLayout(rf)
        rv.setContentsMargins(30, 30, 30, 30)
        rv.setSpacing(20)

        title = QLabel("AI CLASSIFICATION")
        title.setStyleSheet(f"color:{TEXT_LIGHT}; font-family:Segoe UI; font-size:28px; font-weight:900; letter-spacing:2px;")
        rv.addWidget(title)

        self.lbl_cls = QLabel("READY")
        self.lbl_cls.setFont(QFont("Segoe UI Black", 48))
        self.lbl_cls.setStyleSheet(f"color:{DIM};")
        rv.addWidget(self.lbl_cls)

        self.lbl_conf = QLabel("Waiting for scan...")
        self.lbl_conf.setStyleSheet(f"color:{DIM}; font-family:Segoe UI; font-size:24px;")
        rv.addWidget(self.lbl_conf)

        self.bin_visual = QLabel()
        self.bin_visual.setFixedSize(600, 250)
        self.bin_visual.setAlignment(Qt.AlignCenter)
        self.bin_visual.setStyleSheet(f"background:{BG}; border-radius:15px; border: 2px dashed {DIM}; color:{DIM}; font-size:20px;")
        self.bin_visual.setText("BIN IMAGE")
        rv.addWidget(self.bin_visual, alignment=Qt.AlignCenter)

        dist_lbl = QLabel("CONFIDENCE DISTRIBUTION")
        dist_lbl.setStyleSheet(f"color:{TEXT_LIGHT}; font-family:Segoe UI; font-size:20px; font-weight:bold; margin-top:20px;")
        rv.addWidget(dist_lbl)

        self.rows = {}
        for cls in LABELS:
            row = ClassRow(cls)
            rv.addWidget(row)
            self.rows[cls] = row

        hist_layout = QHBoxLayout()
        hist_lbl = QLabel("RECENT SCANS")
        hist_lbl.setStyleSheet(f"color:{TEXT_LIGHT}; font-family:Segoe UI; font-size:20px; font-weight:bold; margin-top:20px;")
        hist_layout.addWidget(hist_lbl)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(80, 30)
        clear_btn.setStyleSheet(f"background:{BG}; color:{DIM}; border:1px solid {DIM}; border-radius:5px;")
        clear_btn.clicked.connect(self._clear_history)
        hist_layout.addWidget(clear_btn, alignment=Qt.AlignRight | Qt.AlignBottom)

        rv.addLayout(hist_layout)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["Time", "Class", "Confidence"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.history_table.setFixedHeight(180)
        self.history_table.setStyleSheet(f"""
            QTableWidget {{ background: {BG}; color: {TEXT_LIGHT}; border: 1px solid {BORDER}; border-radius: 10px; gridline-color: {BORDER}; }}
            QHeaderView::section {{ background: {BORDER}; color: {TEXT_LIGHT}; font-weight: bold; border: none; padding: 5px; }}
        """)
        rv.addWidget(self.history_table)
        self._update_history_table()

        root.addWidget(rf, 1)

        self.cap = cv2.VideoCapture(0)
        self.current_win_color = "#ffffff"
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

    def _tick(self):
        ret, frame = self.cap.read()
        if ret: self.cam.set_frame(frame, self.current_win_color)

    def _classify(self):
        crop = self.cam.get_roi()
        if crop is None: return

        self.btn.setText("PROCESSING AI...")
        QApplication.processEvents()

        try:
            # מעבירים את הפריים הגזור (הריבוע נטו) ישר ל-ResNet18 שלנו
            t = preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(self.model(t), dim=1)[0].cpu().tolist()

            ti = int(np.argmax(probs))
            tc = LABELS[ti]
            conf = probs[ti] * 100
            target_color = COLORS[tc]
            self.current_win_color = target_color

            # עדכון הממשק
            self.lbl_cls.setText(tc.upper())
            self.lbl_cls.setStyleSheet(f"color:{target_color}; font-size: {'36pt' if len(tc) > 10 else '48pt'};")
            self.lbl_conf.setText(f"Confidence: {conf:.1f}%")
            self.btn.setStyleSheet(f"QPushButton{{background:{PANEL}; color:{target_color}; border:3px solid {target_color}; border-radius:15px;}}")

            base_path = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(base_path, "assets", f"{tc}.png")
            if os.path.exists(img_path):
                pix = QPixmap(img_path)
                self.bin_visual.setPixmap(pix.scaled(600, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.bin_visual.setStyleSheet(f"background:{BG}; border: 2px solid {target_color}; border-radius:15px;")
            else:
                self.bin_visual.setText(f"Asset missing:\n{tc}.png\n(Put it in 'assets' folder)")
                self.bin_visual.setStyleSheet(f"background:{BG}; border: 2px dashed {DIM}; color: {DIM}; border-radius:15px;")

            for i, cls in enumerate(LABELS):
                self.rows[cls].update(probs[i] * 100, cls == tc, target_color)

            # שמירה במסד נתונים
            self.db.insert_scan(tc, conf)
            self._update_history_table()

        except Exception as e:
            print(f"Error during classification: {e}")
        finally:
            self.btn.setText("SCAN WASTE (SPACE)")

    def _update_history_table(self):
        records = self.db.get_recent_scans(limit=5)
        self.history_table.setRowCount(len(records))
        for row_idx, row_data in enumerate(records):
            time_str = row_data[0].split(" ")[1]
            cls_str = row_data[1]
            conf_str = f"{row_data[2]:.1f}%"

            self.history_table.setItem(row_idx, 0, QTableWidgetItem(time_str))

            cls_item = QTableWidgetItem(cls_str)
            cls_item.setForeground(QColor(COLORS.get(cls_str, "#ffffff")))
            self.history_table.setItem(row_idx, 1, cls_item)

            self.history_table.setItem(row_idx, 2, QTableWidgetItem(conf_str))

    def _clear_history(self):
        self.db.clear_history()
        self._update_history_table()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space: self._classify()

    def closeEvent(self, e):
        self.timer.stop()
        self.cap.release()
        e.accept()


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 1. טעינת מודל ResNet18
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))

    # 2. נתיב למודל
    ckpt_path = os.path.join(BASE_DIR, "models", "Run5_ResNet18_Augs_best.pth")

    try:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"קובץ המודל חסר!\nוודא ש:\n1. קיימת תיקייה בשם 'models'\n2. בתוכה הקובץ '{os.path.basename(ckpt_path)}'\n\nנתיב שחיפשתי בו:\n{ckpt_path}"
            )

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt)
        model.eval().to(DEVICE)

        win = MainWindow(model)
        win.show()
        sys.exit(app.exec_())

    except Exception as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("שגיאה בטעינת המערכת")
        msg.setText("המערכת נתקלה בבעיה ולכן לא יכולה להיפתח.")
        msg.setInformativeText(str(e))
        msg.exec_()
        sys.exit(1)
