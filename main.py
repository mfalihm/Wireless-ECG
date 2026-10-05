import sys
import os
import serial
import random
import numpy as np
import tensorflow as tf
from serial.tools import list_ports 
from datetime import datetime
from collections import deque
from pathlib import Path

# --- Signal Processing & Math ---
from scipy.signal import find_peaks, butter, lfilter, lfilter_zi, filtfilt

# --- GUI (PyQt6) ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QComboBox, 
                             QTextEdit, QGridLayout, QFrame, QScrollArea, QMessageBox,
                             QDialog, QFormLayout, QLineEdit, QDateEdit, QDialogButtonBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QTime, QRect, QDate 
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QIcon, QPixmap
import pyqtgraph as pg

# --- Reporting (Word & Matplotlib) ---
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# GLOBAL CONFIGURATION & PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = BASE_DIR / "resources"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models" 

# AI Configuration
MODEL_FILENAME = 'best_apnea_model_clean_60s.keras' 
AI_SAMPLE_RATE = 100
AI_WINDOW_SECONDS = 60
AI_UPDATE_INTERVAL_SECONDS = 60
AI_INPUT_SHAPE = AI_SAMPLE_RATE * AI_WINDOW_SECONDS 

# Plot Configuration
pg.setConfigOption('background', 'k') 
pg.setConfigOption('foreground', 'w') 

# =============================================================================
# THREAD: AI INFERENCE WORKER
# =============================================================================
class ApneaWorker(QThread):
    prediction_ready = pyqtSignal(str, float) 
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.data_segment = None
        self.is_model_loaded = False

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.is_model_loaded = True
                print("AI Worker: Model loaded successfully.")
            else:
                self.error_occurred.emit(f"Model not found at {self.model_path}")
        except Exception as e:
            self.error_occurred.emit(f"Failed to load model: {e}")

    def set_data(self, data):
        self.data_segment = np.array(data)
    
    def run(self):
        if not self.is_model_loaded:
            self.load_model()
            if not self.is_model_loaded:
                return

        if self.data_segment is None or len(self.data_segment) != AI_INPUT_SHAPE:
            return

        try:
            # 1. Preprocessing (Bandpass + Z-Score Normalization)
            def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
                nyquist = 0.5 * fs
                low = lowcut / nyquist
                high = highcut / nyquist
                b, a = butter(order, [low, high], btype='band')
                return filtfilt(b, a, data)

            filtered_segment = butter_bandpass_filter(self.data_segment, 0.5, 40, AI_SAMPLE_RATE)
            
            mean = np.mean(filtered_segment)
            std = np.std(filtered_segment)
            if std < 1e-6:
                normalized_segment = filtered_segment - mean
            else:
                normalized_segment = (filtered_segment - mean) / std

            input_data = normalized_segment.reshape(1, AI_INPUT_SHAPE, 1).astype(np.float32)

            # 2. Prediction
            probability = self.model.predict(input_data, verbose=0)[0][0]
            status = "APNEA" if probability >= 0.5 else "NORMAL"
            self.prediction_ready.emit(status, probability)

        except Exception as e:
            self.error_occurred.emit(f"Prediction Error: {e}")

# =============================================================================
# THREAD: SERIAL COMMUNICATION
# =============================================================================
class SerialThread(QThread):
    data_received = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    
    def __init__(self, port='COM9', baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.is_running = True
        self.ser = None
        
    def run(self):
        try:
            self.status_changed.emit(f"Connecting to {self.port}...")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.status_changed.emit(f"Connected: {self.port}")
            
            while self.is_running:
                if not self.ser.is_open:
                    break 
                
                try:
                    if self.ser.in_waiting:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            self.parse_ecg_data(line)
                    else:
                        self.msleep(10)
                except OSError:
                    break

        except serial.SerialException as e:
            if self.is_running: 
                self.status_changed.emit(f"Serial Error: {e}")
        except Exception as ex:
            if self.is_running:
                self.status_changed.emit(f"Unexpected Thread Error: {ex}")
        finally:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except:
                    pass
            
            if self.is_running: 
                self.is_running = False 
                self.status_changed.emit("Disconnected")
    
    def parse_ecg_data(self, line):
        try:
            data = {}
            parts = line.split(',')
            
            if len(parts) >= 3: 
                data['Lead_I'] = int(parts[0])
                data['CH1_LA'] = int(parts[1])
                data['CH2_RA'] = int(parts[2])
                
                if len(parts) == 4:
                    data['BATT'] = int(parts[3])
            
            if data:
                self.data_received.emit(data)
                
        except ValueError:
            pass 
        except Exception as e:
            self.status_changed.emit(f"Parse Error: {e}")
    
    def stop(self):
        self.is_running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except:
                pass

# =============================================================================
# LOGIC: ECG SIGNAL PROCESSING (HEART RATE)
# =============================================================================
class ECGProcessor:
    def __init__(self, sample_rate=100):
        self.sample_rate = sample_rate
        self.buffer_size = 5 * self.sample_rate 
        self.ecg_buffer = deque(maxlen=self.buffer_size)
        self.heart_rate = 0
        self.cardiac_status = "---" 
        self.last_peaks = [] 
        
    def add_data(self, ecg_value):
        if ecg_value == 0:
            self.ecg_buffer.clear() 
            return
            
        self.ecg_buffer.append(ecg_value)
        
        if len(self.ecg_buffer) > (self.sample_rate * 2):
            self.calculate_heart_rate()
    
    def set_lead_off_status(self):
        self.heart_rate = 0
        self.cardiac_status = "LEAD OFF" 
        self.last_peaks = []
        self.ecg_buffer.clear() 

    def calculate_heart_rate(self):
        if len(self.ecg_buffer) < (self.sample_rate * 2):
            self.heart_rate = 0
            self.cardiac_status = "Detecting..."
            return

        data = np.array(list(self.ecg_buffer))
        min_distance = int(self.sample_rate * 0.4) 
        max_height = np.max(data)
        
        if max_height < 50: 
            self.heart_rate = 0
            self.cardiac_status = "Flat/Noise"
            self.last_peaks = []
            return
            
        dynamic_threshold = max_height * 0.5
        peaks, _ = find_peaks(data, height=dynamic_threshold, distance=min_distance)
        self.last_peaks = peaks 
        
        if len(peaks) > 1:
            intervals_samples = np.diff(peaks)
            avg_interval_samples = np.mean(intervals_samples)
            
            if avg_interval_samples > 0:
                self.heart_rate = int(60 * self.sample_rate / avg_interval_samples)
                
                if 60 <= self.heart_rate <= 100:
                    self.cardiac_status = "Normal"
                elif self.heart_rate < 60:
                    self.cardiac_status = "Bradycardia"
                else:
                    self.cardiac_status = "Tachycardia"
        else:
            self.heart_rate = 0
            self.cardiac_status = "Detecting..."

# =============================================================================
# WIDGETS: CUSTOM UI COMPONENTS
# =============================================================================
class BatteryIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 100  
        self.setFixedSize(60, 30) 

    def set_level(self, level):
        self.level = max(0, min(100, int(level))) 
        self.update() 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        outline_color = QColor("#34495e") 
        bg_color = QColor("#ecf0f1")       

        if self.level > 50: fill_color = QColor("#2ecc71") 
        elif self.level > 20: fill_color = QColor("#f1c40f") 
        else: fill_color = QColor("#e74c3c") 

        margin = 4
        body_x, body_y = margin, margin
        body_w, body_h = w - 12 - margin, h - 2 * margin
        tip_w, tip_h = 4, body_h // 3
        tip_x, tip_y = body_x + body_w, body_y + (body_h - tip_h) // 2
        
        painter.setBrush(QBrush(outline_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(int(tip_x), int(tip_y), int(tip_w), int(tip_h), 2, 2)
        
        pen = QPen(outline_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(bg_color)) 
        painter.drawRoundedRect(int(body_x), int(body_y), int(body_w), int(body_h), 4, 4)

        fill_margin = 3 
        max_fill_w = body_w - (2 * fill_margin)
        current_fill_w = int((self.level / 100.0) * max_fill_w)
        fill_h = body_h - (2 * fill_margin)

        if current_fill_w > 0:
            painter.setBrush(QBrush(fill_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(int(body_x + fill_margin), int(body_y + fill_margin), current_fill_w, int(fill_h), 2, 2)
            
        painter.setPen(QColor("#2c3e50"))
        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        painter.drawText(QRect(int(body_x), int(body_y), int(body_w), int(body_h)), Qt.AlignmentFlag.AlignCenter, f"{self.level}%")
        painter.end()

class PatientDataDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Pasien")
        self.setFixedSize(400, 280)
        self.setStyleSheet("""
            QDialog { background-color: #ecf0f1; }
            QFrame#MainCard { background-color: white; border: 1px solid #bdc3c7; border-radius: 12px; }
            QLabel#headerLabel { background-color: #34495e; color: white; font-size: 14px; font-weight: bold; padding: 10px; border-top-left-radius: 12px; border-top-right-radius: 12px; qproperty-alignment: AlignCenter; }
            QLabel.fieldLabel { background-color: #eaeded; border: 1px solid #d5dbdb; border-radius: 4px; padding: 5px 10px; color: #2c3e50; font-weight: bold; font-size: 12px; min-width: 100px; max-width: 100px; }
            QLineEdit, QDateEdit { background-color: #fff; border: 1px solid #bdc3c7; border-radius: 4px; padding: 6px; color: #2d3436; font-size: 13px; }
            QLineEdit:focus, QDateEdit:focus { border: 2px solid #3498db; }
            QDateEdit::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 25px; border-left-width: 1px; border-left-color: #bdc3c7; border-left-style: solid; border-top-right-radius: 4px; border-bottom-right-radius: 4px; background-color: #ecf0f1; }
            QDateEdit::down-arrow { width: 10px; height: 10px; }
            QCalendarWidget QWidget { background-color: white; color: black; }
            QCalendarWidget QToolButton { color: black; background-color: transparent; icon-size: 20px; }
            QCalendarWidget QMenu { background-color: white; color: black; }
            QCalendarWidget QSpinBox { background-color: white; color: black; selection-background-color: #3498db; }
            QCalendarWidget QAbstractItemView:enabled { color: black; background-color: white; selection-background-color: #3498db; selection-color: white; }
            QPushButton { padding: 8px 15px; border-radius: 5px; font-weight: bold; font-size: 12px; border: none; }
            QPushButton[text="Mulai Rekam"] { background-color: #2ecc71; color: white; }
            QPushButton[text="Mulai Rekam"]:hover { background-color: #27ae60; }
            QPushButton[text="Batal"] { background-color: #e74c3c; color: white; }
            QPushButton[text="Batal"]:hover { background-color: #c0392b; }
        """)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        self.card_frame = QFrame()
        self.card_frame.setObjectName("MainCard")
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(0, 0, 0, 15)
        card_layout.setSpacing(15)
        
        header_label = QLabel("DETAIL PASIEN")
        header_label.setObjectName("headerLabel")
        card_layout.addWidget(header_label)
        
        form_container = QVBoxLayout()
        form_container.setContentsMargins(15, 0, 15, 0)
        form_container.setSpacing(10)
        
        row_nama = QHBoxLayout()
        lbl_nama = QLabel("Nama Lengkap")
        lbl_nama.setProperty("class", "fieldLabel") 
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Masukkan nama...")
        row_nama.addWidget(lbl_nama)
        row_nama.addWidget(self.name_input)
        form_container.addLayout(row_nama)
        
        row_dob = QHBoxLayout()
        lbl_dob = QLabel("Tanggal Lahir")
        lbl_dob.setProperty("class", "fieldLabel") 
        self.dob_input = QDateEdit()
        self.dob_input.setCalendarPopup(True)
        self.dob_input.setDate(QDate.currentDate())
        self.dob_input.setDisplayFormat("dd/MM/yyyy")
        row_dob.addWidget(lbl_dob)
        row_dob.addWidget(self.dob_input)
        form_container.addLayout(row_dob)
        
        card_layout.addLayout(form_container)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #ecf0f1; margin-left: 15px; margin-right: 15px;")
        card_layout.addWidget(line)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Mulai Rekam")
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("Batal")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        card_layout.addWidget(self.button_box, 0, Qt.AlignmentFlag.AlignCenter)
        self.card_frame.setLayout(card_layout)
        main_layout.addWidget(self.card_frame)
        self.setLayout(main_layout)

    def get_data(self):
        return self.name_input.text(), self.dob_input.date().toString("dd/MM/yyyy")

# =============================================================================
# MAIN APPLICATION WINDOW
# =============================================================================
class ECGWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        RESOURCE_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        MODELS_DIR.mkdir(exist_ok=True)
        
        self.setWindowTitle("ECG Wireless - Politeknik Negeri Batam")
        try:
             self.setWindowIcon(QIcon(str(RESOURCE_DIR / "icon hati.png")))
        except:
             pass
        
        self.resize(1280, 720) 
        self.setStyleSheet("background-color: #f0f0f0;")
                        
        self.buffer_size = 300 
        self.plot_pointer = 0 
        self.blank_width = 2 
    
        self.ecg_data1 = np.zeros(self.buffer_size, dtype=float)
        self.ecg_data2 = np.zeros(self.buffer_size, dtype=float)
        self.ecg_data3 = np.zeros(self.buffer_size, dtype=float)
        
        self.ecg_data1.fill(np.nan)
        self.ecg_data2.fill(np.nan)
        self.ecg_data3.fill(np.nan)
        
        self.x_axis = np.arange(self.buffer_size)

        self.new_data_ch1 = deque(maxlen=200)
        self.new_data_ch2 = deque(maxlen=200)
        self.new_data_ch3 = deque(maxlen=200)
        
        self.serial_thread = None 
        self.current_connection_is_bluetooth = False
        self.sample_rate_hz = 100 
        
        # Filter Setup
        self.b = None
        self.a = None
        self.z1 = None 
        self.z2 = None 
        self.z3 = None 
        self._create_filter() 
        
        self.ecg_processor = ECGProcessor(sample_rate=self.sample_rate_hz) 
        self.is_recording = False
        
        self.is_saving_report_data = False
        self.recorded_data = [] # Buffer RAW
        self.recorded_processed_data = [] # Buffer for Bandpass Filtered
        
        self.start_time = None
        
        self.patient_name = "-"
        self.patient_dob = "-"
        
        # AI & Buffer Config
        self.ai_buffer = [] 
        self.ai_worker = ApneaWorker(str(MODELS_DIR / MODEL_FILENAME))
        self.ai_worker.prediction_ready.connect(self.update_ai_ui)
        self.ai_worker.error_occurred.connect(self.on_ai_error)
        
        self.ai_prediction_made = False 
        self.ai_results_buffer = []
        self.recording_started = False
        self.showMaximized()
        
        self.latest_battery_percent = 0 
        self.incoming_sample_count = 0
        self.frame_count = 0
        self.incoming_sps = 0
        self.plot_fps = 0
        
        self.init_ui()
        
        self.battery_update_timer = QTimer(self)
        self.battery_update_timer.timeout.connect(self.update_battery_display)
        
        self.sps_calc_timer = QTimer(self)
        self.sps_calc_timer.timeout.connect(self.update_rate_indicators)
        self.sps_calc_timer.start(1000)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setGeometry(10, 10, 1366, 768)
        main_layoutv = QVBoxLayout()
        main_layoutv.setContentsMargins(5, 5, 5, 5) 
        main_layout = QHBoxLayout()
        top_bar_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        title = QLabel("GoSleepAssistentAPP(GoSipAPP): AI Enabled Wireless ECG Monitoring System")
        title_font = QFont()
        title_font.setPointSize(16) 
        title_font.setBold(True) 
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("padding: 5px; color: #2c3e50;") 
        
        main_layoutv.addLayout(top_bar_layout)
        main_layoutv.addLayout(main_layout)
        
        # Plot Widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        
        self.patient_plot_label = self.plot_widget.addLabel(
            text="<span style='color: #7f8c8d; font-size: 10pt;'>No Patient Data</span>", 
            row=0, col=0, 
            justify='left'
        )
        self.patient_plot_label.anchor(itemPos=(0,0), parentPos=(0,0), offset=(10, 0)) 
        
        self.plot1 = self.plot_widget.addPlot(row=1, col=0)
        self.plot2 = self.plot_widget.addPlot(row=2, col=0)
        self.plot3 = self.plot_widget.addPlot(row=3, col=0)

        font_label = QFont("Arial", 9, QFont.Weight.Bold)
        color_label = "#2ecc71" 

        self.label1 = pg.TextItem("Lead I (LA - RA)", color=color_label, anchor=(0,0))
        self.label1.setFont(font_label)
        self.plot1.addItem(self.label1)

        self.label2 = pg.TextItem("Lead II (CH1 - RA)", color=color_label, anchor=(0,0))
        self.label2.setFont(font_label)
        self.plot2.addItem(self.label2)

        self.label3 = pg.TextItem("Lead III (CH2 - RA)", color=color_label, anchor=(0,0))
        self.label3.setFont(font_label)
        self.plot3.addItem(self.label3)
        
        for p in [self.plot1, self.plot2, self.plot3]:
            p.showGrid(x=False, y=False)
            p.hideAxis('bottom')
            p.hideAxis('left')
            p.setMenuEnabled(False)
            p.enableAutoRange(axis='y', enable=False)
            p.enableAutoRange(axis='x', enable=False)
            p.setXRange(0, self.buffer_size - 1, padding=0)

        self.line1 = self.plot1.plot(x=self.x_axis, y=self.ecg_data1, pen=pg.mkPen(color='#e74c3c', width=2), connect='finite')
        self.line2 = self.plot2.plot(x=self.x_axis, y=self.ecg_data2, pen=pg.mkPen(color='#3498db', width=2), connect='finite')
        self.line3 = self.plot3.plot(x=self.x_axis, y=self.ecg_data3, pen=pg.mkPen(color='#27ae60', width=2), connect='finite')
        
        self.plot2.setXLink(self.plot1)
        self.plot3.setXLink(self.plot1)
        
        left_layout.addWidget(self.plot_widget, 1)
        main_layout.addLayout(left_layout, 1) 
        
        # Top Bar
        self.time_display = QLabel(datetime.now().strftime("%d/%m/%Y"))
        self.time_display.setFont(QFont("Arial", 11, QFont.Weight.Bold)) 
        self.time_display.setStyleSheet("color: black;")
        top_bar_layout.addWidget(self.time_display, 2)
        top_bar_layout.addWidget(title, 40)
        
        self.bt_icon = QLabel()
        bt_on_path = str(RESOURCE_DIR / "bt_biru.png")
        bt_off_path = str(RESOURCE_DIR / "bt_abu.png")
        self.pixmap_bt_on = QPixmap(bt_on_path).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation) 
        self.pixmap_bt_off = QPixmap(bt_off_path).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.bt_icon.setPixmap(self.pixmap_bt_off)
        
        self.battery_icon = BatteryIcon()
        self.battery_icon.setFixedSize(60, 30) 
        self.battery_icon.set_level(100) 
        
        top_bar_layout.addWidget(self.bt_icon, 1, Qt.AlignmentFlag.AlignRight)
        top_bar_layout.addWidget(self.battery_icon, 0, Qt.AlignmentFlag.AlignRight) 
        top_bar_layout.addSpacing(10)
        
        # Right Panel
        right_layout = QVBoxLayout()
        right_layout.setSpacing(5) 
        
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: white; border-radius: 10px; padding: 5px; border: 2px solid #ecf0f1;")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5) 
        info_layout.setContentsMargins(2, 2, 2, 2)
        
        header_font = QFont("Arial", 12, QFont.Weight.Bold)
        value_font = QFont("Arial", 14, QFont.Weight.Bold) 

        # HR Box
        hr_box = QFrame()
        hr_box.setStyleSheet("background-color: #fef5f5; border-radius: 8px; border: 1px solid #fadbd8;")
        hr_box_layout = QVBoxLayout()
        hr_box_layout.setContentsMargins(2, 2, 2, 2) 
        hr_label = QLabel("Heart Rate")
        hr_label.setFont(header_font)
        hr_label.setStyleSheet("color: #c0392b;")
        hr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hr_box_layout.addWidget(hr_label)
        self.hr_display = QLabel("BPM")
        self.hr_display.setFont(value_font)
        self.hr_display.setStyleSheet("color: #e74c3c;")
        self.hr_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hr_box_layout.addWidget(self.hr_display)
        hr_box.setLayout(hr_box_layout)
        info_layout.addWidget(hr_box)
        
        # Cardiac Status Box
        cs_box = QFrame()
        cs_box.setStyleSheet("background-color: #f0fef5; border-radius: 8px; border: 1px solid #d5f4e6;")
        cs_box_layout = QVBoxLayout()
        cs_box_layout.setContentsMargins(2, 2, 2, 2)
        cs_label = QLabel("Cardiac Status")
        cs_label.setFont(header_font)
        cs_label.setStyleSheet("color: #196f3d;")
        cs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cs_box_layout.addWidget(cs_label)
        self.cs_display = QLabel("Normal")
        self.cs_display.setFont(value_font)
        self.cs_display.setStyleSheet("color: #27ae60;")
        self.cs_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cs_box_layout.addWidget(self.cs_display)
        cs_box.setLayout(cs_box_layout)
        info_layout.addWidget(cs_box)

        # AI Prediction Box
        ai_box = QFrame()
        ai_box.setStyleSheet("background-color: #e8f8f5; border-radius: 8px; border: 1px solid #a3e4d7;")
        ai_box_layout = QVBoxLayout()
        ai_box_layout.setContentsMargins(2, 2, 2, 2)
        
        ai_label = QLabel("Apnea AI Status")
        ai_label.setFont(header_font)
        ai_label.setStyleSheet("color: #0e6655;")
        ai_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ai_box_layout.addWidget(ai_label)
        
        ai_result_layout = QHBoxLayout()
        self.ai_prob_display = QLabel("Prob: -")
        self.ai_prob_display.setFont(QFont("Arial", 10))
        self.ai_prob_display.setStyleSheet("color: #7f8c8d;")
        self.ai_prob_display.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        ai_result_layout.addWidget(self.ai_prob_display)
        
        self.ai_status_display = QLabel("60s") 
        self.ai_status_display.setFont(value_font)
        self.ai_status_display.setStyleSheet("color: #7f8c8d;")
        self.ai_status_display.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        ai_result_layout.addWidget(self.ai_status_display)
        
        ai_box_layout.addLayout(ai_result_layout)
        ai_box.setLayout(ai_box_layout)
        info_layout.addWidget(ai_box)
        
        info_frame.setLayout(info_layout)
        right_layout.addWidget(info_frame)
        
        # Device Frame
        device_frame = QFrame()
        device_frame.setStyleSheet("background-color: white; border-radius: 8px; padding: 5px; border: 1px solid #ecf0f1;")
        device_layout = QGridLayout()
        device_layout.setSpacing(5) 
        device_layout.setContentsMargins(2, 2, 2, 2)
        
        device_label = QLabel("Device Select")
        device_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        device_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        device_label.setStyleSheet("color: #2c3e50; margin-bottom: 2px;")
        device_layout.addWidget(device_label, 0, 0, 1, 3) 
        
        self.device_combo = QComboBox()
        self.device_combo.setFont(QFont("Arial", 9))
        self.device_combo.setStyleSheet("QComboBox { background-color: white; border: 1px solid #bdc3c7; border-radius: 4px; padding: 3px; color: black; } QComboBox::drop-down { border: 0px; } QComboBox QAbstractItemView { background-color: white; color: black; selection-background-color: #3498db; }")
        device_layout.addWidget(self.device_combo, 1, 0, 1, 3)
        
        btn_style = "QPushButton { color: white; padding: 5px; border-radius: 4px; border: none; font-size: 11px; font-weight: bold; } QPushButton:disabled { background-color: #95a5a6; }"
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet("QPushButton { background-color: #3498db; }" + btn_style)
        self.refresh_btn.clicked.connect(self.populate_ports)
        device_layout.addWidget(self.refresh_btn, 2, 0)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet("QPushButton { background-color: #2ecc71; }" + btn_style)
        self.connect_btn.clicked.connect(self.start_connection)
        device_layout.addWidget(self.connect_btn, 2, 1)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setStyleSheet("QPushButton { background-color: #e74c3c; }" + btn_style)
        self.disconnect_btn.clicked.connect(self.stop_connection)
        self.disconnect_btn.setEnabled(False) 
        device_layout.addWidget(self.disconnect_btn, 2, 2)
        
        self.connection_status_label = QLabel("Status: Disconnected")
        self.connection_status_label.setFont(QFont("Arial", 8))
        self.connection_status_label.setStyleSheet("color: #7f8c8d; margin-top: 2px;")
        self.connection_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        device_layout.addWidget(self.connection_status_label, 3, 0, 1, 3)
        device_frame.setLayout(device_layout)
        right_layout.addWidget(device_frame)
        self.populate_ports()
        
        # Control Frame
        control_frame = QFrame()
        control_frame.setStyleSheet("background-color: white; border-radius: 8px; padding: 5px; border: 1px solid #ecf0f1;")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(5)
        control_layout.setContentsMargins(2, 2, 2, 2)
        
        control_label = QLabel("Plot Control")
        control_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        control_label.setStyleSheet("color: #2c3e50;")
        control_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(control_label)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setStyleSheet("QPushButton { background-color: #e74c3c; }" + btn_style)
        self.pause_btn.clicked.connect(self.pause_recording)
        control_layout.addWidget(self.pause_btn)
        
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setStyleSheet("QPushButton { background-color: #27ae60; }" + btn_style)
        self.resume_btn.clicked.connect(self.resume_recording)
        control_layout.addWidget(self.resume_btn)
        
        self.time_btn = QPushButton("Recording Time")
        self.time_btn.setStyleSheet("QPushButton { background-color: #f39c12; }" + btn_style)
        self.time_btn.clicked.connect(self.start_recording_time)
        control_layout.addWidget(self.time_btn)
        
        self.recording_time_label = QLabel("Recording Time: 00:00:00")
        self.recording_time_label.setFont(QFont("Arial", 9))
        self.recording_time_label.setStyleSheet("color: #7f8c8d;")
        self.recording_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        control_frame.setLayout(control_layout)
        right_layout.addWidget(control_frame)
        right_layout.addWidget(self.recording_time_label)
        
        # Right Wrapper Frame
        right_frame = QFrame()
        right_frame.setLayout(right_layout)
        right_frame.setStyleSheet("padding: 2px;") 
        right_frame.setMaximumWidth(320) 
        
        main_layout.addWidget(right_frame, 0) 
        central_widget.setLayout(main_layoutv)
        
        close_button = QPushButton("CLOSE")
        close_button.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-weight: bold; font-size: 11px; border-radius: 5px; padding: 8px 15px; } QPushButton:hover { background-color: #c0392b; } QPushButton:pressed { background-color: #a93226; }")
        close_button.clicked.connect(self.close)
        
        right_layout.addStretch()

        self.incoming_sps_label = QLabel("Serial: -- SPS")
        self.plot_fps_label = QLabel("Plotter: -- FPS")
        
        sps_font = QFont("Arial", 7)
        sps_style = "color: #7f8c8d; margin: 2px;" 
        
        self.incoming_sps_label.setFont(sps_font)
        self.incoming_sps_label.setStyleSheet(sps_style)
        self.incoming_sps_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
        
        self.plot_fps_label.setFont(sps_font)
        self.plot_fps_label.setStyleSheet(sps_style)
        self.plot_fps_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)

        bottom_bar_layout = QHBoxLayout()
        bottom_bar_layout.setContentsMargins(0, 0, 0, 0)
        
        bottom_bar_layout.addWidget(self.incoming_sps_label)
        bottom_bar_layout.addWidget(self.plot_fps_label)
        bottom_bar_layout.addWidget(close_button) 

        right_layout.addLayout(bottom_bar_layout)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display_sweep) 
        self.update_timer.start(50) 
        
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_recording_time)
        
    def create_ecg_plot_image(self, filename_img):
        """
        Generate ECG plot image for the report.
        Handles data segmentation and auto-scaling for clean visualization.
        """
        if not self.recorded_processed_data:
            return False
            
        # Convert list to numpy array
        proc_data_list = self.recorded_processed_data
        full_signal_data = np.array([row[1:] for row in proc_data_list], dtype=float)
        
        total_samples = len(full_signal_data)
        target_duration = 10 # seconds
        required_samples = int(target_duration * self.sample_rate_hz)
        
        # Logic: Random 10-second segment or full record
        if total_samples > required_samples:
            max_start_index = total_samples - required_samples
            start_index = random.randint(0, max_start_index)
            end_index = start_index + required_samples
            
            signal_data = full_signal_data[start_index:end_index]
            t = np.linspace(0, target_duration, required_samples)
            
            start_time_sec = start_index / self.sample_rate_hz
            end_time_sec = end_index / self.sample_rate_hz
            time_info = f"(Segmen {start_time_sec:.1f}s - {end_time_sec:.1f}s)"
        else:
            signal_data = full_signal_data
            duration_sec = total_samples / self.sample_rate_hz
            t = np.linspace(0, duration_sec, total_samples)
            time_info = "(Full Record)"

        if len(t) != len(signal_data):
            min_len = min(len(t), len(signal_data))
            t = t[:min_len]
            signal_data = signal_data[:min_len]

        # Plotting
        plt.close('all') 
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, dpi=100)
        fig.subplots_adjust(hspace=0.4)
        
        leads_label = ["Lead I", "Lead II", "Lead III"]
        colors = ['black', 'black', 'black'] 
        
        for i, ax in enumerate(axes):
            ax.plot(t, signal_data[:, i], color=colors[i], linewidth=0.8)
            
            title_text = f"{leads_label[i]} {time_info if i==0 else ''}"
            ax.set_title(title_text, loc='left', fontsize=10, fontweight='bold')
            
            # Grid Configuration
            ax.set_facecolor('white')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5)) 
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            
            # Use MaxNLocator for cleaner Y-axis limits
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
            
            ax.grid(which='major', axis='x', color='#ff9999', linestyle='-', linewidth=0.8)
            ax.grid(which='minor', axis='x', color='#ffcccc', linestyle='-', linewidth=0.4)
            ax.grid(which='major', axis='y', color='#ff9999', linestyle='-', linewidth=0.8)
            
            ax.tick_params(which='both', colors='black', labelbottom=False, labelleft=True)
            ax.ticklabel_format(useOffset=False, style='plain', axis='y')

            if len(signal_data[:, i]) > 0:
                y_min, y_max = np.min(signal_data[:, i]), np.max(signal_data[:, i])
                margin = 1.0 if y_min == y_max else (y_max - y_min) * 0.1
                ax.set_ylim(y_min - margin, y_max + margin)

        axes[-1].tick_params(labelbottom=True)
        axes[-1].set_xlabel("Time (seconds)")
        
        try:
            plt.savefig(filename_img, bbox_inches='tight', pad_inches=0.1)
        except Exception as e:
            print(f"Plotting Error: {e}")
            plt.close(fig)
            return False
            
        plt.close(fig) 
        return True

    def save_word_report(self, filename_docx):
        document = Document()
        
        style = document.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)
        
        # Setup Document Margins (Narrow)
        section = document.sections[0]
        section.left_margin = Cm(1.27)
        section.right_margin = Cm(1.27)
        section.top_margin = Cm(1.27)
        section.bottom_margin = Cm(1.27)

        heading = document.add_heading('ECG MEDICAL REPORT', 0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        document.add_paragraph() 

        # Patient Info Table
        table = document.add_table(rows=4, cols=4)
        table.autofit = False 
        table.alignment = WD_ALIGN_PARAGRAPH.CENTER 
        
        for row in table.rows:
            row.cells[0].width = Cm(3.0) 
            row.cells[1].width = Cm(5.0) 
            row.cells[2].width = Cm(3.2) 
            row.cells[3].width = Cm(7.0) 

        record_date_str = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        duration_str = self.recording_time_label.text().replace('Recording Time: ', '')

        total_ai_segments = len(self.ai_results_buffer)
        jumlah_apnea = sum(1 for item in self.ai_results_buffer if item['status'] == "APNEA")
        kalimat_ai = f"{jumlah_apnea} dari {total_ai_segments} terdeteksi apnea"

        # Fill Table
        row0 = table.rows[0]
        row0.cells[0].text = "Patient Name"
        row0.cells[0].paragraphs[0].runs[0].bold = True
        row0.cells[1].text = f": {self.patient_name}"
        
        row0.cells[2].text = "Avg Heart Rate"
        row0.cells[2].paragraphs[0].runs[0].bold = True
        row0.cells[3].text = f": {self.ecg_processor.heart_rate} BPM"

        row1 = table.rows[1]
        row1.cells[0].text = "Date of Birth"
        row1.cells[0].paragraphs[0].runs[0].bold = True
        row1.cells[1].text = f": {self.patient_dob}"
        
        row1.cells[2].text = "Cardiac Status"
        row1.cells[2].paragraphs[0].runs[0].bold = True
        row1.cells[3].text = f": {self.ecg_processor.cardiac_status}"

        row2 = table.rows[2]
        row2.cells[0].text = "Record Date"
        row2.cells[0].paragraphs[0].runs[0].bold = True
        row2.cells[1].text = f": {record_date_str}"
        
        row2.cells[2].text = "AI Result"
        row2.cells[2].paragraphs[0].runs[0].bold = True
        row2.cells[3].text = f": {kalimat_ai}" 

        row3 = table.rows[3]
        row3.cells[0].text = "Duration"
        row3.cells[0].paragraphs[0].runs[0].bold = True
        row3.cells[1].text = f": {duration_str}"
        
        row3.cells[2].text = "Total Raw Samples"
        row3.cells[2].paragraphs[0].runs[0].bold = True
        row3.cells[3].text = f": {len(self.recorded_data)}"

        p_line = document.add_paragraph("_" * 95) 
        p_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
        document.add_paragraph().paragraph_format.space_after = Pt(12) 

        # Insert Plot Image
        temp_img_path = str(REPORTS_DIR / "temp_ecg_plot.png")
        
        if self.create_ecg_plot_image(temp_img_path):
            h2 = document.add_heading('ECG Signal Visualization', level=2)
            h2.alignment = WD_ALIGN_PARAGRAPH.LEFT
            
            document.add_picture(temp_img_path, width=Cm(18.0)) 
            last_p = document.paragraphs[-1] 
            last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            try:
                os.remove(temp_img_path)
            except:
                pass
        else:
            document.add_paragraph("[No Signal Data Recorded]")

        try:
            document.save(filename_docx)
        except Exception as e:
            print(f"Word save error: {e}")
            raise e
        
    def _create_filter(self):
        lowcut_hz = 0.5 
        highcut_hz = 30.0
        filter_order = 3
        
        nyq = 0.5 * self.sample_rate_hz 
        low = lowcut_hz / nyq
        high = highcut_hz / nyq
        
        low = max(0.01, low)
        high = min(high, 0.99)
        
        if low >= high:
            low = high - 0.01
            
        try:
            self.b, self.a = butter(filter_order, [low, high], btype='band')
            self.z1 = lfilter_zi(self.b, self.a) * 0
            self.z2 = lfilter_zi(self.b, self.a) * 0
            self.z3 = lfilter_zi(self.b, self.a) * 0
            print(f"Filter {lowcut_hz}-{highcut_hz}Hz (fs={self.sample_rate_hz}Hz) initialized.")

        except Exception as e:
            print(f"Filter creation failed: {e}")
            self.b, self.a = np.array([1.0]), np.array([1.0])
            self.z1, self.z2, self.z3 = np.array([0.0]), np.array([0.0]), np.array([0.0])
        
    def populate_ports(self):
        self.device_combo.clear()
        ports = list_ports.comports()
        
        if not ports:
            self.device_combo.addItem("No devices found")
            self.device_combo.setEnabled(False)
            self.connect_btn.setEnabled(False)
        else:
            self.device_combo.setEnabled(True)
            self.connect_btn.setEnabled(True)
            
            for port in ports:
                display_text = f"{port.description} ({port.device})"
                
                check_desc = "bluetooth" in port.description.lower()
                check_dev = "rfcomm" in port.device.lower()
                
                is_bt = check_desc or check_dev
                
                port_data = {
                    "device": port.device,
                    "is_bluetooth": is_bt
                }
                self.device_combo.addItem(display_text, port_data)
    
    def cleanup_thread(self):
        if self.serial_thread is not None:
            try:
                self.serial_thread.data_received.disconnect()
                self.serial_thread.status_changed.disconnect()
            except:
                pass
            
            if self.serial_thread.isRunning():
                self.serial_thread.stop()
                self.serial_thread.wait(1000) 
            
            self.serial_thread = None

    def start_connection(self):
        self.cleanup_thread()
        
        selected_data = self.device_combo.currentData()
        if selected_data is None:
            self.on_status_change("Error: No device selected.")
            return
        
        selected_port_name = selected_data["device"]
        self.current_connection_is_bluetooth = selected_data["is_bluetooth"]

        baudrate = 115200 
        
        self.plot_pointer = 0
        self.ecg_data1.fill(np.nan)
        self.ecg_data2.fill(np.nan)
        self.ecg_data3.fill(np.nan)
        self.new_data_ch1.clear()
        self.new_data_ch2.clear()
        self.new_data_ch3.clear()
        self.ai_buffer = [] 
        
        self.ai_prediction_made = False
        self.ai_status_display.setText("60s")
        self.ai_status_display.setStyleSheet("color: #7f8c8d;")
        self.ai_prob_display.setText("Prob: -")
        
        self._create_filter()

        self.serial_thread = SerialThread(selected_port_name, baudrate)
        self.serial_thread.data_received.connect(self.on_ecg_data)
        self.serial_thread.status_changed.connect(self.on_status_change)
        self.serial_thread.start()
        
        self.is_recording = True 
        
        self.latest_battery_percent = 0
        self.update_battery_display() 
        self.battery_update_timer.start(10000) 
        
        self.incoming_sample_count = 0
        self.frame_count = 0
        self.incoming_sps_label.setText("Serial: -- SPS")
        self.plot_fps_label.setText("Plotter: -- FPS")

        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.device_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
    
    def stop_connection(self):
        self.is_recording = False
        self.cleanup_thread()
        self.current_connection_is_bluetooth = False
        self.battery_update_timer.stop()
        self.incoming_sps_label.setText("Serial: -- SPS")
        self.plot_fps_label.setText("Plotter: -- FPS")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.device_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.on_status_change("Disconnected")
    
    def on_ecg_data(self, data):
        if 'BATT' in data:
            self.latest_battery_percent = data['BATT']
        
        if 'Lead_I' not in data:
            return 
            
        self.incoming_sample_count += 1

        if not self.is_recording:
            return
            
        lead_i = data.get('Lead_I', 0)
        ch2_la = data.get('CH1_LA', 0) 
        ch3_ra = data.get('CH2_RA', 0)

        # AI Buffer Logic
        self.ai_buffer.append(lead_i)
        
        if len(self.ai_buffer) >= AI_INPUT_SHAPE:
            print("Buffer Full (60s). Sending to AI Worker...")
            data_to_process = list(self.ai_buffer)
            slide_samples = int(AI_UPDATE_INTERVAL_SECONDS * AI_SAMPLE_RATE)
            
            if slide_samples >= len(self.ai_buffer):
                self.ai_buffer = [] 
            else:
                self.ai_buffer = self.ai_buffer[slide_samples:]
            
            if not self.ai_worker.isRunning():
                self.ai_worker.set_data(data_to_process)
                self.ai_worker.start()
            else:
                print("Warning: AI Worker is still busy. Dropping this window.")

        # Filter Logic
        LEAD_OFF_THRESHOLD = 8000000 
        is_lead_off = (abs(lead_i) > LEAD_OFF_THRESHOLD or 
                       abs(ch2_la) > LEAD_OFF_THRESHOLD or 
                       abs(ch3_ra) > LEAD_OFF_THRESHOLD)
        
        filtered_lead_i = 0
        filtered_ch2_la = 0
        filtered_ch3_ra = 0

        if is_lead_off:
            self.z1 = lfilter_zi(self.b, self.a) * 0
            self.z2 = lfilter_zi(self.b, self.a) * 0
            self.z3 = lfilter_zi(self.b, self.a) * 0
            self.ecg_processor.set_lead_off_status()
        else:
            try:
                f_val_1, self.z1 = lfilter(self.b, self.a, [lead_i], zi=self.z1)
                f_val_2, self.z2 = lfilter(self.b, self.a, [ch2_la], zi=self.z2)
                f_val_3, self.z3 = lfilter(self.b, self.a, [ch3_ra], zi=self.z3)
                
                filtered_lead_i = f_val_1[0]
                filtered_ch2_la = f_val_2[0]
                filtered_ch3_ra = f_val_3[0]
            except Exception as e:
                print(f"Filter error: {e}")
            
            self.ecg_processor.add_data(filtered_lead_i) 

        # Recording Logic
        if self.is_saving_report_data:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # 1. Save RAW
            self.recorded_data.append(f"{timestamp},{lead_i},{ch2_la},{ch3_ra}")
            
            # 2. Save Filtered (for Z-Score later)
            self.recorded_processed_data.append([timestamp, filtered_lead_i, filtered_ch2_la, filtered_ch3_ra])

        # Update Display Buffers
        self.new_data_ch1.append(filtered_lead_i)
        self.new_data_ch2.append(filtered_ch2_la)
        self.new_data_ch3.append(filtered_ch3_ra)
        
    def update_ai_ui(self, status, probability):
        self.ai_prediction_made = True 
        self.ai_status_display.setText(status)
        self.ai_prob_display.setText(f"Prob: {probability*100:.1f}%")
        
        if status == "APNEA":
            self.ai_status_display.setStyleSheet("color: #e74c3c; font-size: 11px; font-weight: bold;") 
        else:
            self.ai_status_display.setStyleSheet("color: #27ae60; font-size: 11px; font-weight: bold;") 
        
        if self.is_saving_report_data:
            if self.start_time:
                elapsed = datetime.now() - self.start_time
                elapsed_seconds = int(elapsed.total_seconds())
                timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                self.ai_results_buffer.append({
                    "timestamp": timestamp_str,
                    "elapsed_sec": elapsed_seconds,
                    "status": status,
                    "probability": f"{probability:.4f}"
                })
                print(f"AI Result Recorded: {status} at {timestamp_str}")

    def on_ai_error(self, message):
        print(f"AI Error: {message}")
        self.ai_status_display.setText("Error")
        self.ai_status_display.setStyleSheet("color: #c0392b;")

    def update_battery_display(self):
        percent = self.latest_battery_percent
        if isinstance(self.battery_icon, BatteryIcon):
            self.battery_icon.set_level(percent)

    def update_rate_indicators(self):
        self.incoming_sps = self.incoming_sample_count
        self.plot_fps = self.frame_count
        self.incoming_sample_count = 0
        self.frame_count = 0
        
        self.incoming_sps_label.setText(f"Serial: {self.incoming_sps} SPS")
        self.plot_fps_label.setText(f"Plotter: {self.plot_fps} FPS")
        self.time_display.setText(datetime.now().strftime("%d/%m/%Y"))
        
        if self.is_recording:
            filled_sec = len(self.ai_buffer) // AI_SAMPLE_RATE
            remaining_sec = max(0, AI_WINDOW_SECONDS - filled_sec)
            
            if not self.ai_prediction_made:
                self.ai_status_display.setText(f"{remaining_sec}s")
                self.ai_status_display.setStyleSheet("color: #7f8c8d; font-size: 14px; font-weight: bold;")
                self.ai_prob_display.setText("Buffering...")
            else:
                self.ai_prob_display.setText(f"Next update: {remaining_sec}s")
        else:
            if not self.ai_prediction_made:
                self.ai_status_display.setText("Stopped")
                self.ai_prob_display.setText("-")

    def update_display_sweep(self):
        did_wrap_around = False
        while len(self.new_data_ch1) > 0:
            d1 = self.new_data_ch1.popleft()
            d2 = self.new_data_ch2.popleft()
            d3 = self.new_data_ch3.popleft()
            
            self.ecg_data1[self.plot_pointer] = d1
            self.ecg_data2[self.plot_pointer] = d2
            self.ecg_data3[self.plot_pointer] = d3
            
            for i in range(1, self.blank_width + 1):
                blank_index = (self.plot_pointer + i) % self.buffer_size
                self.ecg_data1[blank_index] = np.nan
                self.ecg_data2[blank_index] = np.nan
                self.ecg_data3[blank_index] = np.nan
            
            new_pointer = self.plot_pointer + 1
            if new_pointer >= self.buffer_size:
                new_pointer = 0
                did_wrap_around = True 
            self.plot_pointer = new_pointer

        self.line1.setData(x=self.x_axis, y=self.ecg_data1)
        self.line2.setData(x=self.x_axis, y=self.ecg_data2)
        self.line3.setData(x=self.x_axis, y=self.ecg_data3)
        
        if did_wrap_around:
            def set_manual_range(plot, data_array, label_item):
                if np.all(np.isnan(data_array)):
                    min_val, max_val = -100, 100
                else:
                    min_val = np.nanmin(data_array)
                    max_val = np.nanmax(data_array)
                
                data_range = max_val - min_val
                padding = 25 if data_range < 50 else data_range * 0.1

                if np.isnan(min_val) or np.isnan(max_val):
                    min_val, max_val, padding = -100, 100, 0

                top_limit = max_val + padding
                plot.setYRange(min_val - padding, top_limit)
                label_item.setPos(0, top_limit)

            set_manual_range(self.plot1, self.ecg_data1, self.label1)
            set_manual_range(self.plot2, self.ecg_data2, self.label2)
            set_manual_range(self.plot3, self.ecg_data3, self.label3)

        self.hr_display.setText(f"{self.ecg_processor.heart_rate} BPM")
        self.cs_display.setText(self.ecg_processor.cardiac_status)
        
        if self.ecg_processor.cardiac_status == "Normal":
            self.cs_display.setStyleSheet("color: #27ae60;")
        elif self.ecg_processor.cardiac_status == "Bradycardia":
            self.cs_display.setStyleSheet("color: #f39c12;")
        elif self.ecg_processor.cardiac_status == "Tachycardia":
            self.cs_display.setStyleSheet("color: #e74c3c;")
        elif self.ecg_processor.cardiac_status == "LEAD OFF":
            self.cs_display.setStyleSheet("color: #95a5a6;") 
        else: 
            self.cs_display.setStyleSheet("color: #7f8c8d;")
        
        self.frame_count += 1
            
    def start_recording_time(self):
        base_style_update = "color: white; padding: 5px; border-radius: 4px; border: none; font-size: 11px; font-weight: bold;"

        if not self.recording_started:
            dialog = PatientDataDialog(self)
            if dialog.exec(): 
                self.patient_name, self.patient_dob = dialog.get_data()
                
                if not self.patient_name.strip():
                    self.patient_name = "Tanpa Nama"

                info_text = (f"<span style='color: #2ecc71; font-weight: bold; font-size: 12pt;'>"
                             f"Pasien: {self.patient_name}  |  Tanggal Lahir: {self.patient_dob}</span>")
                
                self.patient_plot_label.setText(info_text)

                # Start Recording
                self.recorded_data.clear() 
                self.recorded_processed_data.clear()
                
                self.ai_results_buffer.clear()
                self.is_saving_report_data = True
                self.start_time = datetime.now()
                self.time_timer.start(1000)
                self.recording_started = True
                self.time_btn.setText("Stop Recording & Save Report")
                
                self.time_btn.setStyleSheet(f"QPushButton {{ background-color: #e74c3c; {base_style_update} }} QPushButton:hover {{ background-color: #c0392b; }} QPushButton:pressed {{ background-color: #a93226; }}")
            else:
                return 

        else:
            self.is_saving_report_data = False
            self.start_time = None
            self.time_timer.stop()
            self.recording_started = False
            self.time_btn.setText("Recording Time")

            self.time_btn.setStyleSheet(f"QPushButton {{ background-color: #f39c12; {base_style_update} }} QPushButton:hover {{ background-color: #d68910; }} QPushButton:pressed {{ background-color: #b9770e; }}")

            self.print_report()
            self.recording_time_label.setText("Recording Time: 00:00:00")

            self.patient_plot_label.setText("<span style='color: #7f8c8d; font-size: 10pt;'>No Patient Data</span>")

            self.patient_name = "-"
            self.patient_dob = "-"
    
    def update_recording_time(self):
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.recording_time_label.setText(f"Recording Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def pause_recording(self):
        self.is_recording = False 
        self.pause_btn.setStyleSheet("QPushButton { background-color: #95a5a6; color: white; padding: 10px; border-radius: 5px; border: none; }")
        self.resume_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; border: none; } QPushButton:hover{ background-color: #229954; } QPushButton:pressed { background-color: #1e8449; }")        
    
    def resume_recording(self):
        self.is_recording = True
        self.pause_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px; border: none; } QPushButton:hover { background-color: #c0392b; } QPushButton:pressed { background-color: #a93226; }")
        self.resume_btn.setStyleSheet("QPushButton { background-color: #95a5a6; color: white; padding: 10px; border-radius: 5px; border: none; }")     
        
    def print_report(self):
        # Set cursor to wait
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        # Force UI update before heavy processing
        QApplication.processEvents()

        if self.recorded_data:
            safe_name = "".join([c if c.isalnum() else "_" for c in self.patient_name])
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            folder_name = f"{safe_name}_{time_str}"
            folder_pasien = REPORTS_DIR / folder_name
            
            try:
                folder_pasien.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Failed to create folder, using default: {e}")
                folder_pasien = REPORTS_DIR
            
            # File Paths
            filename_raw = folder_pasien / f"ECG_RAW_{safe_name}_{time_str}.csv"
            filename_proc = folder_pasien / f"ECG_PROCESSED_{safe_name}_{time_str}.csv"
            filename_anno = folder_pasien / f"ECG_AI_ANNOTATION_{safe_name}_{time_str}.csv"
            filename_docx = folder_pasien / f"ECG_REPORT_{safe_name}_{time_str}.docx"
            
            try:
                # 1. Save RAW CSV
                print(f"Saving RAW report...")
                with open(filename_raw, 'w') as f:
                    f.write("Timestamp,Lead_I_RAW,Lead_II_RAW,Lead_III_RAW\n")
                    f.write('\n'.join(self.recorded_data))
                
                QApplication.processEvents() 
                
                # 2. Save Processed CSV (Norm + Z-Score)
                if self.recorded_processed_data:
                    proc_data_list = self.recorded_processed_data
                    timestamps = [row[0] for row in proc_data_list]
                    
                    signal_data = np.array([row[1:] for row in proc_data_list], dtype=float)
                    
                    means = np.mean(signal_data, axis=0)
                    stds = np.std(signal_data, axis=0)
                    stds[stds == 0] = 1.0 
                    
                    z_scored_data = (signal_data - means) / stds
                    
                    with open(filename_proc, 'w') as f:
                        f.write("Timestamp,Lead_I_Norm,Lead_II_Norm,Lead_III_Norm\n")
                        for i in range(len(timestamps)):
                            line = f"{timestamps[i]},{z_scored_data[i][0]:.4f},{z_scored_data[i][1]:.4f},{z_scored_data[i][2]:.4f}\n"
                            f.write(line)

                QApplication.processEvents()

                # 3. Save AI Annotation CSV
                print("Saving AI Annotation report...")
                with open(filename_anno, 'w') as f:
                    f.write("Timestamp,Elapsed_Seconds,Prediction_Status,Probability\n")
                    
                    if self.ai_results_buffer:
                        for item in self.ai_results_buffer:
                            line = f"{item['timestamp']},{item['elapsed_sec']},{item['status']},{item['probability']}\n"
                            f.write(line)

                # 4. Generate Word Report
                print(f"Generating Word report...")
                self.save_word_report(filename_docx)

                self.on_status_change(f"Saved: CSVs (Clean) & Word Report")
                
                QApplication.restoreOverrideCursor()
                
                self.show_styled_message(
                    "Success", 
                    f"Reports saved successfully.\nDocx: {filename_docx.name}", 
                    "information"
                )
                
            except Exception as e:
                QApplication.restoreOverrideCursor()
                self.on_status_change(f"Error saving report: {e}")
                print(f"Detailed Error: {e}")
                self.show_styled_message("Error", f"Failed to save report:\n{e}", "critical")
        else:
            QApplication.restoreOverrideCursor()
            self.on_status_change("No data recorded to save.")
            self.show_styled_message("No Data", "No data recorded.", "warning")
    
    def on_status_change(self, status):
        self.connection_status_label.setText(f"Status: {status}")

        if "Connected: " in status:
            if self.current_connection_is_bluetooth:
                self.bt_icon.setPixmap(self.pixmap_bt_on)
            else:
                self.bt_icon.setPixmap(self.pixmap_bt_off) 
            
        elif "Disconnected" in status or "Error" in status or "Failed" in status:
            self.bt_icon.setPixmap(self.pixmap_bt_off)
            self.current_connection_is_bluetooth = False 
        
    def show_styled_message(self, title, message, icon_type="warning"):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        
        if icon_type == "warning":
            msg.setIcon(QMessageBox.Icon.Warning)
        elif icon_type == "information":
            msg.setIcon(QMessageBox.Icon.Information)
        elif icon_type == "critical":
            msg.setIcon(QMessageBox.Icon.Critical)

        msg.setStyleSheet("""
            QMessageBox { background-color: #ecf0f1; }
            QLabel { color: #2c3e50; font-size: 12px; font-weight: bold; }
            QPushButton { background-color: #3498db; color: white; border-radius: 4px; padding: 6px 15px; font-weight: bold; }
            QPushButton:hover { background-color: #2980b9; }
        """)
        msg.exec()
    
    def closeEvent(self, event):
        self.time_timer.stop()
        self.update_timer.stop() 
        self.battery_update_timer.stop() 
        self.sps_calc_timer.stop()        
        self.stop_connection() 
        event.accept()

if __name__ == '__main__':
    try:
        RESOURCE_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        MODELS_DIR.mkdir(exist_ok=True)
        print(f"Resources directory: {RESOURCE_DIR}")
        print(f"Reports directory: {REPORTS_DIR}")
        print(f"Models directory: {MODELS_DIR}")
    except Exception as e:
        print(f"Warning: Could not create directories. {e}")
    
    app = QApplication(sys.argv)
    window = ECGWindow()
    window.showMaximized()
    sys.exit(app.exec())