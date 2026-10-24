"""
pattern_gui.py - رابط گرافیکی Pattern Engine
همه‌ی تنظیمات قابل تغییر است و می‌توان آن‌ها را در فایل JSON ذخیره یا بارگذاری کرد.
"""
import dataclasses
import json
import os
import sys

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

import pattern_engine as pe
from ai_providers import PRESETS, AIProvider, ProviderConfig, preset_config

OUT_EXT = {"tiff": ".tif", "png": ".png", "jpeg": ".jpg", "pdf": ".pdf", "svg": ".svg"}
PROVIDER_FIELDS = ["base_url", "endpoint", "api_key", "auth_header", "auth_prefix",
                   "model", "upload_field", "scale_field", "model_field",
                   "response_format", "response_key", "timeout"]
SETTING_FIELDS = [f.name for f in dataclasses.fields(pe.Settings)]


def default_settings_path():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "PatternEngine")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "settings.json")


def combo(items):
    c = QComboBox()
    for label, val in items:
        c.addItem(label, val)
    return c


def dspin(lo, hi, step=1.0, dec=1):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(dec)
    return s


def ispin(lo, hi):
    s = QSpinBox()
    s.setRange(lo, hi)
    return s


def _get(w):
    if isinstance(w, QComboBox):
        return w.currentData()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        return w.value()
    return w.text().strip()


def _set(w, v):
    if isinstance(w, QComboBox):
        w.setCurrentIndex(max(w.findData(v), 0))
    elif isinstance(w, QCheckBox):
        w.setChecked(bool(v))
    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
        w.setValue(v)
    else:
        w.setText(str(v))


def test_connection(provider, log=print):
    log("ارسال تصویر آزمایشی به سرویس ...")
    out = provider.upscale(np.full((64, 64, 3), 128, np.uint8), 2)
    return f"اتصال برقرار است. پاسخ: {out.shape[1]}×{out.shape[0]} پیکسل"


class Task(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, fn, **kwargs):
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs

    def run(self):
        try:
            self.done.emit(self.fn(log=self.log.emit, **self.kwargs))
        except Exception as ex:
            self.failed.emit(str(ex))


class CornerView(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 420)
        self.pix = None
        self.img_size = None
        self.corners = None
        self.drag = None

    def set_image(self, bgr):
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.pix = QPixmap.fromImage(qimg)
        self.img_size = (w, h)
        self.corners = None
        self.update()

    def _scale(self):
        w, h = self.img_size
        s = min(self.width() / w, self.height() / h)
        return s, (self.width() - w * s) / 2, (self.height() - h * s) / 2

    def _to_img(self, x, y):
        s, ox, oy = self._scale()
        return np.array([(x - ox) / s, (y - oy) / s], dtype=np.float64)

    def _to_widget(self, q):
        s, ox, oy = self._scale()
        return QPointF(q[0] * s + ox, q[1] * s + oy)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 40, 40))
        if self.pix is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "عکس را باز کنید")
            p.end()
            return
        s, ox, oy = self._scale()
        w, h = self.img_size
        p.drawPixmap(QRectF(ox, oy, w * s, h * s), self.pix, QRectF(0, 0, w, h))
        if self.corners is not None:
            pts = [self._to_widget(c) for c in self.corners]
            p.setPen(QPen(QColor(255, 200, 0), 2))
            for i in range(4):
                p.drawLine(pts[i], pts[(i + 1) % 4])
            p.setPen(QPen(QColor(255, 0, 0), 2))
            p.setBrush(QBrush(QColor(255, 0, 0, 120)))
            for pt in pts:
                p.drawEllipse(pt, 8, 8)
        p.end()

    def mousePressEvent(self, e):
        if self.corners is None:
            return
        pos = e.position()
        d = [(self._to_widget(c).x() - pos.x()) ** 2 + (self._to_widget(c).y() - pos.y()) ** 2
             for c in self.corners]
        i = int(np.argmin(d))
        if d[i] < 400:
            self.drag = i

    def mouseMoveEvent(self, e):
        if self.drag is None or self.img_size is None:
            return
        q = self._to_img(e.position().x(), e.position().y())
        w, h = self.img_size
        q[0] = np.clip(q[0], 0, w - 1)
        q[1] = np.clip(q[1], 0, h - 1)
        self.corners[self.drag] = q
        self.update()

    def mouseReleaseEvent(self, e):
        self.drag = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pattern Engine")
        self.img = None
        self.result = None
        self.last_settings = None
        self.task = None
        self.task_kind = None
        self.w = {}

        self.view = CornerView()
        self.preview = QLabel("نتیجه اینجا نمایش داده می‌شود")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(240)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(140)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.addWidget(self._buttons())
        pl.addWidget(self._group_output())
        pl.addWidget(self._group_process())
        pl.addWidget(self._group_upscale())
        pl.addWidget(self._group_provider())
        pl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(440)

        left = QVBoxLayout()
        left.addWidget(self.view, 3)
        left.addWidget(self.preview, 2)
        top = QHBoxLayout()
        top.addLayout(left, 1)
        top.addWidget(scroll)
        root = QVBoxLayout()
        root.addLayout(top, 1)
        root.addWidget(self.log_box)
        c = QWidget()
        c.setLayout(root)
        self.setCentralWidget(c)
        self.resize(1500, 950)

        self._apply_provider(preset_config(_get(self.w["preset"])))
        self._load_settings(default_settings_path(), silent=True)
        self.w["preset"].currentIndexChanged.connect(self.on_preset_changed)

    # ---------- ساخت رابط ----------

    def _field(self, key, widget, form, label):
        self.w[key] = widget
        form.addRow(label, widget)

    def _buttons(self):
        box = QGroupBox("عملیات")
        g = QGridLayout(box)
        actions = [
            ("باز کردن عکس", self.open_image), ("تشخیص خودکار گوشه‌ها", self.auto_detect),
            ("پردازش", self.process), ("ذخیره خروجی", self.save_output),
            ("ذخیره تنظیمات", self.save_settings_dialog), ("بارگذاری تنظیمات", self.load_settings_dialog),
        ]
        for i, (text, fn) in enumerate(actions):
            b = QPushButton(text)
            b.clicked.connect(fn)
            g.addWidget(b, i // 2, i % 2)
        return box

    def _group_output(self):
        g = QGroupBox("ابعاد و خروجی")
        f = QFormLayout(g)
        self._field("unit", combo([("سانتی‌متر", "cm"), ("اینچ", "inch")]), f, "واحد")
        self._field("width", dspin(1, 2000, 1, 1), f, "عرض")
        self._field("height", dspin(1, 2000, 1, 1), f, "ارتفاع")
        self._field("dpi", ispin(72, 1200), f, "DPI")
        self._field("bleed", dspin(0, 20, 0.1, 1), f, "Bleed (حاشیه‌ی برش)")
        self._field("out_format", combo([
            ("TIFF (توصیه‌شده برای چاپ)", "tiff"), ("PNG", "png"), ("JPEG", "jpeg"),
            ("PDF", "pdf"), ("SVG (وکتور، فقط طرح ساده)", "svg"),
        ]), f, "فرمت خروجی")
        self._field("color_mode", combo([("RGB (نمایش)", "RGB"), ("CMYK (چاپ)", "CMYK")]),
                    f, "حالت رنگ")
        icc = QLineEdit()
        icc.setPlaceholderText("اختیاری: پروفایل ICC پرینتر")
        btn = QPushButton("...")
        btn.setFixedWidth(40)
        btn.clicked.connect(self._browse_icc)
        row = QHBoxLayout()
        row.addWidget(icc)
        row.addWidget(btn)
        self.w["icc_path"] = icc
        f.addRow("پروفایل ICC", row)
        self._field("jpeg_quality", ispin(60, 100), f, "کیفیت JPEG")
        self._field("tiff_compression", combo([
            ("بدون فشرده‌سازی", "none"), ("LZW", "lzw"), ("ZIP", "zip")]), f, "فشرده‌سازی TIFF")
        return g

    def _group_process(self):
        g = QGroupBox("پردازش تصویر")
        f = QFormLayout(g)
        self._field("normalize_light", QCheckBox("یکنواخت کردن نور و حذف سایه"), f, "نور")
        self._field("repeat_mode", combo([("خودکار (تشخیص تکرار)", "auto"),
                                          ("خاموش", "off")]), f, "تکرار طرح")
        self._field("min_period_px", ispin(3, 500), f, "کوچک‌ترین تناوب (px)")
        self._field("min_strength", dspin(0, 1, 0.05, 2), f, "آستانه‌ی تشخیص تکرار")
        self._field("denoise_k", ispin(0, 3), f, "قدرت حذف چین (۰ خاموش)")
        return g

    def _group_upscale(self):
        g = QGroupBox("بزرگ‌نمایی")
        f = QFormLayout(g)
        self._field("upscale_mode", combo([
            ("ساده (Lanczos، داخلی)", "lanczos"), ("هوش مصنوعی (API)", "ai")]), f, "روش")
        self._field("ai_scale", combo([("×2", 2), ("×4", 4)]), f, "مقیاس AI")
        return g

    def _group_provider(self):
        g = QGroupBox("سرویس هوش مصنوعی")
        f = QFormLayout(g)
        preset = combo([(k, k) for k in PRESETS])
        self._field("preset", preset, f, "قالب")
        self._field("p_base_url", QLineEdit(), f, "آدرس پایه")
        self._field("p_endpoint", QLineEdit(), f, "مسیر endpoint")
        key = QLineEdit()
        key.setEchoMode(QLineEdit.EchoMode.Password)
        key.setPlaceholderText(f"کلید API (یا متغیر محیطی {'PATTERN_ENGINE_API_KEY'})")
        self._field("p_api_key", key, f, "کلید API")
        self._field("p_auth_header", QLineEdit(), f, "نام هدر احراز هویت")
        self._field("p_auth_prefix", QLineEdit(), f, "پیشوند کلید")
        self._field("p_model", QLineEdit(), f, "نام مدل (اختیاری)")
        self._field("p_upload_field", QLineEdit(), f, "نام فیلد تصویر")
        self._field("p_scale_field", QLineEdit(), f, "نام فیلد مقیاس")
        self._field("p_model_field", QLineEdit(), f, "نام فیلد مدل")
        self._field("p_response_format", combo([
            ("تصویر باینری", "binary"), ("JSON با base64", "base64_json"),
            ("JSON با آدرس تصویر", "url_json")]), f, "فرمت پاسخ")
        self._field("p_response_key", QLineEdit(), f, "کلید تصویر در JSON")
        self._field("p_timeout", ispin(10, 900), f, "مهلت (ثانیه)")
        test = QPushButton("تست اتصال به سرویس")
        test.clicked.connect(self.test_api)
        f.addRow("", test)
        note = QLabel("آدرس‌ها و نام فیلدها باید دقیقاً با مستندات سرویس یکی باشد.")
        note.setWordWrap(True)
        f.addRow("", note)
        return g

    # ---------- تنظیمات ----------

    def read_settings(self):
        return pe.Settings(**{k: _get(self.w[k]) for k in SETTING_FIELDS})

    def read_provider(self):
        vals = {k: _get(self.w["p_" + k]) for k in PROVIDER_FIELDS}
        return ProviderConfig(preset=_get(self.w["preset"]), **vals)

    def _apply_settings(self, s):
        for k in SETTING_FIELDS:
            _set(self.w[k], getattr(s, k))

    def _apply_provider(self, cfg):
        for k in PROVIDER_FIELDS:
            if k == "api_key" and not cfg.api_key:
                continue
            _set(self.w["p_" + k], getattr(cfg, k))

    def on_preset_changed(self, *_):
        self._apply_provider(preset_config(_get(self.w["preset"])))

    def save_settings_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "ذخیره تنظیمات", default_settings_path(),
                                              "JSON (*.json)")
        if path:
            self._save_settings(path)

    def load_settings_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "بارگذاری تنظیمات", "", "JSON (*.json)")
        if path:
            self._load_settings(path)

    def _save_settings(self, path):
        provider = dataclasses.asdict(self.read_provider())
        provider["api_key"] = ""  # کلید API در فایل ذخیره نمی‌شود
        data = {"settings": dataclasses.asdict(self.read_settings()), "provider": provider}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.write(f"تنظیمات ذخیره شد: {path}")

    def _load_settings(self, path, silent=False):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_settings(pe.Settings(**data.get("settings", {})))
            prov = data.get("provider", {})
            self.w["preset"].blockSignals(True)
            _set(self.w["preset"], prov.get("preset", "سفارشی"))
            self.w["preset"].blockSignals(False)
            self._apply_provider(ProviderConfig(**{k: v for k, v in prov.items()
                                                   if k in ProviderConfig.__dataclass_fields__}))
            if not silent:
                self.write(f"تنظیمات بارگذاری شد: {path}")
        except Exception as ex:
            if not silent:
                QMessageBox.warning(self, "خطا", f"بارگذاری تنظیمات ناموفق بود: {ex}")

    # ---------- عملیات ----------

    def write(self, msg):
        self.log_box.append(msg)

    def _browse_icc(self, *_):
        path, _ = QFileDialog.getOpenFileName(self, "پروفایل ICC", "", "ICC (*.icc *.icm)")
        if path:
            self.w["icc_path"].setText(path)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "انتخاب عکس", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp)")
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "خطا", "خواندن عکس ناموفق بود")
            return
        self.img = img
        self.view.set_image(img)
        self.write(f"عکس باز شد: {path}")
        self.auto_detect()

    def auto_detect(self):
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        quad = pe.detect_quad(self.img)
        if quad is None:
            self.view.corners = np.array([[w * 0.1, h * 0.1], [w * 0.9, h * 0.1],
                                          [w * 0.9, h * 0.9], [w * 0.1, h * 0.9]], dtype=np.float64)
            self.write("گوشه‌ها خودکار پیدا نشدند؛ گوشه‌ها را با ماوس تنظیم کنید")
        else:
            self.view.corners = quad.astype(np.float64)
            self.write("گوشه‌ها خودکار پیدا شدند؛ در صورت نیاز تنظیم کنید")
        self.view.update()

    def start_task(self, kind, fn, **kwargs):
        self.task_kind = kind
        self.task = Task(fn, **kwargs)
        self.task.log.connect(self.write)
        self.task.done.connect(self.on_done)
        self.task.failed.connect(self.on_failed)
        self.task.start()

    def process(self):
        if self.img is None or self.view.corners is None:
            QMessageBox.information(self, "توجه", "ابتدا عکس را باز کنید")
            return
        try:
            s = self.read_settings()
            s.validate()
            provider = AIProvider(self.read_provider()) if s.upscale_mode == "ai" else None
        except Exception as ex:
            QMessageBox.warning(self, "خطا", str(ex))
            return
        self.last_settings = s
        self.write(f"شروع پردازش: {s.width} × {s.height} {s.unit} با {s.dpi} DPI")
        self.start_task("process", pe.run_pipeline, img=self.img,
                        corners=self.view.corners.copy(), s=s, provider=provider)

    def test_api(self):
        try:
            provider = AIProvider(self.read_provider())
        except Exception as ex:
            QMessageBox.warning(self, "خطا", str(ex))
            return
        self.start_task("test", test_connection, provider=provider)

    def on_done(self, result):
        if self.task_kind == "test":
            self.write(result)
            QMessageBox.information(self, "نتیجه", result)
            return
        self.result = result
        h, w = result.shape[:2]
        pw = min(900, w)
        ph = int(pw * h / w)
        small = cv2.resize(result, (pw, ph), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, pw, ph, 3 * pw, QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.write(f"پردازش تمام شد: {w}×{h} پیکسل")

    def on_failed(self, msg):
        self.write(f"خطا: {msg}")
        QMessageBox.warning(self, "خطا", msg)

    def save_output(self):
        if self.result is None or self.last_settings is None:
            QMessageBox.information(self, "توجه", "ابتدا پردازش را اجرا کنید")
            return
        fmt = self.last_settings.out_format
        path, _ = QFileDialog.getSaveFileName(self, "ذخیره خروجی", "print" + OUT_EXT[fmt],
                                              f"*{OUT_EXT[fmt]}")
        if not path:
            return
        try:
            pe.save_output(self.result, path, self.last_settings, log=self.write)
            self.write(f"ذخیره شد: {path}")
        except Exception as ex:
            QMessageBox.warning(self, "خطا", str(ex))

    def closeEvent(self, e):
        self._save_settings_quiet()
        super().closeEvent(e)

    def _save_settings_quiet(self):
        try:
            self._save_settings(default_settings_path())
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
