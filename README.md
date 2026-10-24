# Pattern Engine

## ساخت فایل نصبی ویندوز
- روش خودکار: این پوشه را در یک مخزن GitHub بگذارید و workflow با نام «Build PatternEngine installer» را از تب Actions اجرا کنید. فایل نصبی در بخش Artifacts همان اجرا قرار می‌گیرد.
- روش دستی: Python 3.11 (64 بیتی) و Inno Setup 6 را نصب کنید و `build.bat` را اجرا کنید. خروجی در `installer_output` ساخته می‌شود.

## اجرای بدون نصب (برای تست)
```
pip install -r requirements.txt
python pattern_gui.py
```

## سرور بزرگ‌نمایی شخصی (اختیاری)
اگر سرویس AI در دسترس نیست، `server.py` را روی ماشین دارای GPU اجرا کنید و قالب «سرور شخصی (Real-ESRGAN)» را در برنامه انتخاب کنید.

## نکته‌ی مهم
کد این پروژه اجرا و آزمایش نشده است. اولین اجرا را روی یک عکس واقعی انجام دهید.
