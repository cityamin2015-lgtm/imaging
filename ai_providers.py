"""
ai_providers.py
اتصال به سرویس بزرگ‌نمایی تصویر. هر سرویسی که این قرارداد را داشته باشد کار می‌کند:

  درخواست:  POST {base_url}{endpoint}
            multipart شامل: فیلد تصویر (PNG)، فیلد مقیاس، و فیلد مدل (اختیاری)
            هدر احراز هویت: auth_header با پیشوند auth_prefix
  پاسخ:     تصویر باینری، یا JSON حاوی تصویر به‌صورت base64، یا JSON حاوی آدرس تصویر
"""
import base64
import os
from dataclasses import dataclass

import cv2
import numpy as np
import requests

ENV_KEY = "PATTERN_ENGINE_API_KEY"


@dataclass
class ProviderConfig:
    preset: str = "سفارشی"
    base_url: str = ""
    endpoint: str = "/upscale"
    api_key: str = ""
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    model: str = ""
    upload_field: str = "image"
    scale_field: str = "scale"
    model_field: str = "model"
    response_format: str = "binary"   # binary | base64_json | url_json
    response_key: str = "image"       # کلید تصویر در پاسخ JSON
    timeout: int = 180


# مقادیر پیش‌فرض قالب‌ها. آدرس‌های سرویس‌های ایرانی را با مستندات خودشان تطبیق دهید.
PRESETS = {
    "سرور شخصی (Real-ESRGAN)": {
        "base_url": "http://127.0.0.1:8000", "endpoint": "/upscale",
        "auth_header": "X-API-Key", "auth_prefix": "",
        "upload_field": "image", "scale_field": "scale", "response_format": "binary",
    },
    "آوال‌AI (تنظیم دستی)": {
        "base_url": "https://api.avalai.ir/v1", "endpoint": "",
        "auth_header": "Authorization", "auth_prefix": "Bearer ",
    },
    "آروان کلود (تنظیم دستی)": {
        "base_url": "", "endpoint": "",
        "auth_header": "Authorization", "auth_prefix": "Bearer ",
    },
    "سفارشی": {},
}


def preset_config(name):
    return ProviderConfig(preset=name, **PRESETS.get(name, {}))


class AIProvider:
    def __init__(self, cfg):
        if not cfg.base_url.strip() or not cfg.endpoint.strip():
            raise ValueError("آدرس سرویس و مسیر endpoint را وارد کنید")
        self.cfg = cfg

    def _headers(self):
        key = self.cfg.api_key or os.environ.get(ENV_KEY, "")
        if key and self.cfg.auth_header:
            return {self.cfg.auth_header: self.cfg.auth_prefix + key}
        return {}

    def _extract(self, r):
        fmt = self.cfg.response_format
        if fmt == "binary":
            return r.content
        data = r.json()
        val = data[self.cfg.response_key]
        if fmt == "base64_json":
            if val.startswith("data:") and "," in val:
                val = val.split(",", 1)[1]
            return base64.b64decode(val)
        if fmt == "url_json":
            rr = requests.get(val, timeout=self.cfg.timeout)
            rr.raise_for_status()
            return rr.content
        raise ValueError(f"فرمت پاسخ نامعتبر: {fmt}")

    def upscale(self, img_bgr, scale):
        ok, buf = cv2.imencode(".png", np.ascontiguousarray(img_bgr))
        if not ok:
            raise RuntimeError("تبدیل تصویر به PNG ناموفق بود")
        url = self.cfg.base_url.rstrip("/") + self.cfg.endpoint
        data = {self.cfg.scale_field: str(scale)}
        if self.cfg.model:
            data[self.cfg.model_field] = self.cfg.model
        files = {self.cfg.upload_field: ("input.png", buf.tobytes(), "image/png")}

        r = requests.post(url, headers=self._headers(), data=data, files=files,
                          timeout=self.cfg.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"خطای API ({r.status_code}): {r.text[:300]}")

        arr = cv2.imdecode(np.frombuffer(self._extract(r), np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError("پاسخ API تصویر معتبر نیست")

        h, w = img_bgr.shape[:2]
        if arr.shape[:2] != (h * scale, w * scale):
            arr = cv2.resize(arr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        return arr
