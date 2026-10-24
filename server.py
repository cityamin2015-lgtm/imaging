"""
server.py - سرور بزرگ‌نمایی Real-ESRGAN با همان قرارداد ai_providers.py
اجرا:  pip install fastapi uvicorn python-multipart torch basicsr realesrgan opencv-python-headless
       uvicorn server:app --host 0.0.0.0 --port 8000
وزن مدل RealESRGAN_x4plus.pth را در کنار این فایل قرار دهید.
"""
import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from realesrgan import RealESRGANer

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path="RealESRGAN_x4plus.pth", model=model,
                         tile=512, tile_pad=10, pre_pad=0,
                         half=torch.cuda.is_available())
app = FastAPI()


@app.post("/upscale")
async def upscale(image: UploadFile = File(...), scale: int = Form(4)):
    img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "invalid image")
    out, _ = upsampler.enhance(img, outscale=scale)
    ok, buf = cv2.imencode(".png", out)
    return Response(content=buf.tobytes(), media_type="image/png")
