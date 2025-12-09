# =========================
# app.py  (kök dizinde)
# =========================
from __future__ import annotations

import os
import io
import re
import base64
import traceback
import mimetypes
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal, Optional

import requests
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw

# --- Savunmalı import: google-generativeai ---
try:
    import google.generativeai as genai  # type: ignore
    _GENAI_AVAILABLE = True
except ModuleNotFoundError:
    genai = None  # type: ignore
    _GENAI_AVAILABLE = False

# OpenAI SDK (resmi)
from openai import OpenAI

# =========================
# 🔐 API KEYS & CONFIG
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY", None)

GPT_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o")
GEMINI_TEXT_MODEL = "gemini-1.5-pro"
GEMINI_VISION_MODEL = "gemini-1.5-flash"
DEFAULT_CITY = "Ankara"  # Hava durumu için varsayılan

if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY eksik. GPT modları çalışmayacak.")
if not GEMINI_API_KEY:
    st.warning("⚠️ GEMINI_API_KEY eksik. Gemini modları çalışmayacak.")
if not _GENAI_AVAILABLE:
    st.error("❌ `google-generativeai` paketi kurulu değil. `requirements.txt` içine ekleyip yeniden dağıtın.")

GPT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if _GENAI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =========================
# 🎨 LOGO & FAVICON
# =========================
LOGO_LIGHT = "QelyonAIblack.png"
LOGO_DARK = "QelyonAIwhite.png"
FAVICON = "favicn.png"

st.set_page_config(
    page_title="Qelyon AI Stüdyo",
    page_icon=FAVICON if os.path.exists(FAVICON) else None,
    layout="wide",
)

# =========================
# 🎨 THEME ENGINE
# =========================
def get_theme(is_dark: bool):
    accent = "#6C47FF"
    if is_dark:
        return {
            "bg": "#050509",
            "text": "#FFFFFF",
            "sub": "#A8A8A8",
            "input": "#111111",
            "card": "rgba(255,255,255,0.05)",
            "border": "rgba(255,255,255,0.1)",
            "accent": accent,
        }
    return {
        "bg": "#F5F5FB",
        "text": "#0F172A",
        "sub": "#444444",
        "input": "#FFFFFF",
        "card": "rgba(255,255,255,0.85)",
        "border": "rgba(0,0,0,0.1)",
        "accent": accent,
    }

def apply_theme_css(t):
    st.markdown(
        f"""
        <style>
        body, .stApp {{
            background: {t['bg']} !important;
            color: {t['text']} !important;
        }}
        .stTextInput>div>div>input,
        textarea {{
            background: {t['input']} !important;
            color: {t['text']} !important;
            border-radius: 12px !important;
            border: 1px solid {t['border']} !important;
        }}
        [data-testid="stChatMessage"] {{
            background: {t['card']};
            border: 1px solid {t['border']};
            border-radius: 14px;
            padding: 10px 14px;
            margin-bottom: 10px;
        }}
        .stButton>button {{
            background: {t['accent']} !important;
            border-radius: 999px !important;
            color: white !important;
            font-weight: 600 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# 🌙 TEMA TOGGLE
# =========================
col_a, col_b = st.columns([10, 1])
with col_b:
    dark = st.toggle("🌙 / ☀️", value=True)

THEME = get_theme(dark)
apply_theme_css(THEME)

# =========================
# 🧠 SESSION
# =========================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "📸 Stüdyo Modu"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_image" not in st.session_state:
    st.session_state.chat_image = None
if "chat_filename" not in st.session_state:
    st.session_state.chat_filename = "dosya"
if "studio_result" not in st.session_state:
    st.session_state.studio_result = None

# =========================
# A2 — API CLIENTS • GEMINI + GPT
# =========================
def _gemini_ready() -> bool:
    """Gemini kullanılabilir mi."""
    return bool(_GENAI_AVAILABLE and GEMINI_API_KEY)

# ---- Gemini Text ----
def gemini_text(prompt: str):
    """Gemini 1.5 Pro ile metin üretir; uygun değilse bilgi döner."""
    if not _gemini_ready():
        return "Gemini kullanılamıyor. (Paket veya API anahtarı eksik)"
    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)  # type: ignore
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or "Metin üretilemedi."
    except Exception as e:
        return f"Gemini hata: {e}"

# ---- Gemini Vision ----
def gemini_vision(prompt: str, image_bytes: bytes):
    """Gemini Vision (Flash) ile görsel analizi yapar."""
    if not _gemini_ready():
        return "Gemini kullanılamıyor. (Paket veya API anahtarı eksik)"
    try:
        model = genai.GenerativeModel(GEMINI_VISION_MODEL)  # type: ignore
        img_data = {"mime_type": "image/png", "data": image_bytes}
        resp = model.generate_content([prompt, img_data])
        return getattr(resp, "text", "") or "Yanıt üretilemedi."
    except Exception as e:
        return f"Görsel analiz hatası: {e}"

# ---- Gemini Image Generate (SDK sürümlerinde olmayabilir) ----
def gemini_generate_image(prompt: str, size: str = "1024x1024"):
    """SDK bu özelliği desteklemeyebilir; yoksa None döner."""
    if not _gemini_ready():
        return None
    try:
        # Çoğu sürümde doğrudan image generate API yoktur.
        # Burada güvenli şekilde yok sayıyoruz.
        if not hasattr(genai, "GenerativeModel"):
            return None
        model = genai.GenerativeModel("imagegeneration")  # type: ignore
        out = model.generate_content(prompt)
        return getattr(out, "_image", None) or getattr(out, "image", None)
    except Exception:
        return None

# ---- Gemini Image Edit (SDK sürümlerinde olmayabilir) ----
def gemini_edit_scene(prompt: str, product_image_bytes: bytes):
    """Arka planı AI ile yeniden tasarlamaya çalışır; destek yoksa None."""
    if not _gemini_ready():
        return None
    try:
        model = genai.GenerativeModel(GEMINI_VISION_MODEL)  # type: ignore
        img_dict = {"mime_type": "image/png", "data": product_image_bytes}
        full_prompt = (
            "You are a professional commercial product photographer. "
            "Replace ONLY the background. Do NOT modify the product. "
            f"Background style: {prompt}"
        )
        # Bazı sürümlerde `generate_image` yoktur; try/except ile koru.
        if hasattr(model, "generate_image"):
            result = model.generate_image(prompt=full_prompt, image=img_dict, size="1024x1024")  # type: ignore
            return getattr(result, "_image", None)
        # Alternatif: multimodal prompt ile öneri üretilir; görsel dönmeyebilir.
        _ = model.generate_content([full_prompt, img_dict])
        return None
    except Exception:
        return None

# ---- GPT-4o Chat ----
def gpt_chat(messages: list[dict], model: str = GPT_MODEL):
    """OpenAI GPT-4o sohbet motoru."""
    if not GPT:
        return "GPT API Anahtarı eksik."
    try:
        res = GPT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"GPT hatası: {e}"

def model_router(mode: str):
    """Moda göre motor."""
    if mode == "GENERAL_CHAT":
        return "gemini"
    if mode in ["ECOM", "CONSULT"]:
        return "gpt"
    return "gemini"

# =========================
# 📅 ZAMAN / TARİH
# =========================
def get_tr_time():
    """Türkiye saati."""
    try:
        r = requests.get("http://worldtimeapi.org/api/timezone/Europe/Istanbul", timeout=6)
        dt = r.json().get("datetime")
        return datetime.fromisoformat(dt)
    except Exception:
        return datetime.now(ZoneInfo("Europe/Istanbul"))

def time_answer():
    now = get_tr_time()
    return f"Bugün {now.strftime('%d.%m.%Y')} — Saat {now.strftime('%H:%M')}"

# =========================
# 🌦 HAVA DURUMU
# =========================
def get_coords(city: str):
    """OpenWeather geocoding."""
    if not WEATHER_API_KEY:
        return None
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},TR&limit=1&appid={WEATHER_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if not data:
            return None
        return data[0]["lat"], data[0]["lon"]
    except Exception:
        return None

def get_weather(city: str):
    """Kısa hava özeti."""
    if not WEATHER_API_KEY:
        return "Hava durumu API Anahtarı eksik."
    coords = get_coords(city)
    if not coords:
        return f"{city} için konum bulunamadı."
    lat, lon = coords
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=tr"
        r = requests.get(url, timeout=10).json()
        desc = r["weather"][0]["description"].capitalize()
        temp = r["main"]["temp"]
        hum = r["main"]["humidity"]
        wind = r["wind"]["speed"]
        return (
            f"📍 **{city.title()}**\n"
            f"🌡️ Sıcaklık: **{temp:.1f}°C**\n"
            f"☁️ Hava: **{desc}**\n"
            f"💧 Nem: **%{hum}**\n"
            f"🍃 Rüzgar: **{wind} m/s**"
        )
    except Exception:
        return "Hava durumu alınamadı."

# =========================
# 🛡 GÜVENLİK FİLTRESİ
# =========================
BAD_WORDS = [
    r"(?i)orospu", r"(?i)siktir", r"(?i)amk",
    r"(?i)tecavüz", r"(?i)intihar", r"(?i)bomba yap",
]
def moderate_text(msg: str) -> str | None:
    """Uygunsuzluk filtreler."""
    for pat in BAD_WORDS:
        if re.search(pat, msg):
            return "Bu isteğe güvenlik nedeniyle yanıt veremiyorum. 🙏"
    return None

# =========================
# A3 — STÜDYO (LOKAL İŞLEME)
# =========================
def remove_bg_local(image: Image.Image) -> Image.Image:
    """Basit eşik ile arka planı kaldırır (parlak zeminlerde etkili)."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    gray = image.convert("L")
    mask = gray.point(lambda p: 255 if p > 240 else 0)
    result = Image.new("RGBA", image.size)
    result.paste(image, (0, 0), mask)
    return result

def center_on_canvas(img: Image.Image, size=1024) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    obj = img.copy()
    obj.thumbnail((size * 0.84, size * 0.84), Image.Resampling.LANCZOS)
    x = (size - obj.width) // 2
    y = (size - obj.height) // 2
    canvas.paste(obj, (x, y), obj)
    return canvas

def make_contact_shadow(alpha: Image.Image, intensity=150):
    """Yumuşak temas gölgesi maskesi üretir."""
    a = alpha.convert("L")
    box = a.getbbox()
    if not box:
        return Image.new("L", a.size, 0)
    w = box[2] - box[0]
    h = int((box[3] - box[1]) * 0.22)
    shadow = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(shadow)
    draw.ellipse((0, 0, w, h), fill=intensity)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=int(h * 0.45)))
    mask = Image.new("L", a.size, 0)
    mask.paste(shadow, (box[0], box[3] - h // 2))
    return mask

def make_reflection(img: Image.Image, fade=230):
    """Alt refleksiyon efekti."""
    a = img.split()[3]
    box = a.getbbox()
    if not box:
        return Image.new("RGBA", img.size, (0, 0, 0, 0))
    crop = img.crop(box)
    flip = ImageOps.flip(crop)
    grad = Image.linear_gradient("L").resize((1, flip.height))
    grad = grad.point(lambda p: int(p * (fade / 255)))
    grad = grad.resize(flip.size)
    flip.putalpha(grad)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(flip, (box[0], box[3] + 6), flip)
    return out

def compose_scene(cut: Image.Image, bg_color: str, reflection=True, shadow=True):
    """Kare sahne kompoziti oluşturur."""
    size = 1024
    obj = center_on_canvas(cut, size)
    alpha = obj.split()[3]
    colors = {
        "white": (255, 255, 255, 255),
        "black": (0, 0, 0, 255),
        "beige": (245, 240, 222, 255),
    }
    bg = Image.new("RGBA", (size, size), colors.get(bg_color, (255, 255, 255, 255)))
    final = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    final.alpha_composite(bg)
    if shadow:
        sh_mask = make_contact_shadow(alpha)
        sh = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        sh.putalpha(sh_mask)
        final.alpha_composite(sh)
    if reflection:
        ref = make_reflection(obj)
        final.alpha_composite(ref)
    final.alpha_composite(obj)
    return final

def gemini_analyze_document(
    file_bytes: bytes,
    filename: str,
    user_instruction: str = "Bu dosyayı profesyonelce özetle ve önemli maddeleri çıkar.",
) -> str:
    """PDF/Resim belgeleri Gemini 1.5 Pro ile (destek varsa) analiz eder."""
    if not _gemini_ready():
        return "Gemini kullanılamıyor. (Paket veya API anahtarı eksik)"
    if not file_bytes:
        return "Dosya içeriği boş görünüyor."
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_part = {"mime_type": mime_type, "data": file_bytes}
    prompt = (
        "Sen Qelyon AI doküman analiz uzmanısın. "
        "Önemli kısımları net ve anlaşılır özetle, aksiyon maddelerini çıkar.\n\n"
        f"Kullanıcı talimatı: {user_instruction}"
    )
    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)  # type: ignore
        response = model.generate_content([prompt, file_part])
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        return "Dosya analiz edildi fakat metin cevap üretilemedi."
    except Exception as e:
        return f"Belge analiz hatası: {e}"

# =========================
# A4 — GENEL CHAT (Gemini)
# =========================
IMAGE_TRIGGER_WORDS = [
    "görsel oluştur", "resim oluştur", "foto üret",
    "bir görsel çiz", "image create", "generate image",
    "bana bir tasarım yap", "logo yap", "arka plan üret",
]
def is_image_generation_request(msg: str) -> bool:
    return any(t in msg.lower() for t in IMAGE_TRIGGER_WORDS)

def gemini_general_chat(user_message: str, user_image: bytes | None):
    """Gemini 1.5 Pro tabanlı genel sohbet (destek kontrolü içerir)."""
    if not _gemini_ready():
        return "Gemini kullanılamıyor. (Paket veya API anahtarı eksik)"
    try:
        history = []
        for msg in st.session_state.chat_history[-25:]:
            if msg["role"] == "user":
                history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant" and msg["content"] != "(Görsel üretildi)":
                history.append({"role": "model", "parts": [msg["content"]]})
        new_parts: list[dict] = [{"text": user_message}]
        if user_image and not st.session_state.chat_filename.lower().endswith(".pdf"):
            new_parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(user_image).decode("utf-8"),
                }
            })
        user_turn = {"role": "user", "parts": new_parts}
        full_prompt = history + [user_turn]
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)  # type: ignore
        response = model.generate_content(full_prompt)
        if hasattr(response, "text"):
            return response.text
        return "Bir yanıt üretemedim."
    except Exception as e:
        return f"💥 Genel chat hatası: {e}"

def analyze_uploaded_file_in_chat(user_message: str) -> str:
    """Chat içinde dosya analizi tetikleyici."""
    if st.session_state.chat_image is None:
        return ""
    triggers = [
        "pdfi özetle", "pdf'i özetle", "pdf özetle",
        "bu dosyayı özetle", "bu dosyayı analiz et", "belgeyi analiz et",
        "dokümanı analiz et", "bu görseli analiz et", "bu resmi analiz et",
        "dosyayı incele",
    ]
    if not any(t in user_message.lower() for t in triggers):
        return ""
    file_bytes = st.session_state.chat_image
    filename = st.session_state.chat_filename
    user_instruction = user_message
    return gemini_analyze_document(file_bytes, filename, user_instruction)

def handle_general_chat(user_message: str):
    """Genel Chat bağlayıcı."""
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)
    if is_image_generation_request(user_message):
        with st.chat_message("assistant"):
            st.write("🎨 Görsel oluşturuluyor...")
            img_bytes = gemini_generate_image(user_message)
            if img_bytes:
                st.image(img_bytes, caption="✨ Gemini 1.5 Flash tarafından üretildi", width=350)
                ai_answer = "(Görsel üretildi)"
            else:
                ai_answer = "⚠️ Görsel oluşturulamadı veya desteklenmiyor."
            if ai_answer != "(Görsel üretildi)":
                st.write(ai_answer)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Qelyon AI düşünüyor..."):
                ai_answer = gemini_general_chat(user_message, st.session_state.chat_image)
                st.write(ai_answer)
    st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})

# =========================
# A5 — GPT PERSONA
# =========================
def build_system_talimati(profile: Literal["ecom", "consult"]) -> str:
    if profile == "ecom":
        return """
Sen Qelyon AI'nın E-Ticaret Uzmanı modundasın.
Görevlerin:
1) Ürün açıklaması (SEO)
2) 5 fayda
3) Kutu içeriği
4) Hedef kitle
5) Kullanım önerileri
6) CTA
7) Görsel/PDF analizi
8) A/B başlık
9) Trendyol etiket
10) Fiyat psikolojisi
11) Varyantlar
12) Yorum analizi
13) Sosyal reklam metinleri
14) Marka hikâyesi
15) Tümü Türkçe, ticari ve net.
"""
    if profile == "consult":
        return """
Sen Qelyon AI'nın Danışmanlık Uzmanı modundasın.
Görevler: İş modeli analizi, strateji, KPI, SWOT, plan, gelir modeli, validasyon, PDF/görsel analizi.
Tarz: Kesin, analitik, stratejik, gereksiz hikâye yok.
"""
    return "Qelyon AI sistem talimatı uygulanamadı."

def custom_identity_interceptor(msg: str) -> Optional[str]:
    m = msg.lower()
    if any(x in m for x in ["kimsin", "sen neysin", "kim yapt", "kim geliştirdi"]):
        return "Ben Qelyon AI'yım. Hibrit: Gemini Vision + GPT-4o."
    if "openai" in m or "gpt" in m:
        return "Qelyon AI: GPT-4o teknolojisi ve özel yeteneklerle genişletilmiş hibrit sistem."
    if "ne iş yaparsın" in m or "görevin ne" in m:
        return "Profesyonel danışmanlık ve veri destekli içgörüler sunarım."
    return None

def custom_utility_interceptor(msg: str) -> Optional[str]:
    m = msg.lower()
    if "saat" in m and ("kaç" in m or "?" in m):
        return time_answer()
    if "hava" in m or "hava durumu" in m:
        return get_weather(DEFAULT_CITY)
    return None

def gpt_assistant(profile: Literal["ecom", "consult"], user_message: str) -> str:
    if not GPT:
        return "GPT API Anahtarı eksik."
    try:
        system_msg = build_system_talimati(profile)
        user_content = [{"type": "text", "text": user_message}]
        if st.session_state.chat_image:
            if not st.session_state.chat_filename.lower().endswith(".pdf"):
                encoded_image = base64.b64encode(st.session_state.chat_image).decode("utf-8")
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}})
            else:
                pdf_analysis = gemini_analyze_document(
                    st.session_state.chat_image,
                    st.session_state.chat_filename,
                    "Bu PDF/doküman içeriğini özetle."
                )
                system_msg += f"\n\n[EK DOSYA ANALİZİ ({st.session_state.chat_filename})]:\n{pdf_analysis}"
        msgs = [{"role": "system", "content": system_msg}]
        for m in st.session_state.chat_history[-10:]:
            if m["role"] == "user" and m["content"] == user_message:
                continue
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_content})
        res = GPT.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            temperature=0.3,
            max_tokens=1800,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"GPT-4o hata: {e}"

def handle_gpt_assistant(profile: Literal["ecom", "consult"], user_message: str):
    if not user_message:
        return
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)
    ident = custom_identity_interceptor(user_message)
    util = custom_utility_interceptor(user_message)
    if ident:
        answer = ident
    elif util:
        answer = util
    else:
        with st.chat_message("assistant"):
            with st.spinner("Qelyon AI düşünüyor..."):
                answer = gpt_assistant(profile, user_message)
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# =========================
# A6 — UİLER
# =========================
def general_chat_ui():
    st.markdown("### 💬 Qelyon AI — Genel Chat (Gemini)")
    st.caption("Gemini 1.5 Pro & Flash.")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["content"] != "(Görsel üretildi)":
                st.write(msg["content"])
    upload = st.file_uploader(
        "Görsel / PDF yükle (opsiyonel)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        key="general_upload",
    )
    if upload is not None:
        file_bytes = upload.read()
        st.session_state.chat_image = file_bytes
        st.session_state.chat_filename = upload.name
        st.success(f"📎 Dosya yüklendi: {upload.name}! Mesajında bu dosyadan bahsedebilirsin.")
    elif "general_upload" in st.session_state and st.session_state.general_upload is None:
        st.session_state.chat_image = None
        st.session_state.chat_filename = "dosya"
    user_msg = st.chat_input("Mesajını yaz...")
    if user_msg:
        mod = moderate_text(user_msg)
        if mod:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(mod)
            st.session_state.chat_history.append({"role": "assistant", "content": mod})
            return
        doc_answer = analyze_uploaded_file_in_chat(user_msg)
        if doc_answer:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(doc_answer)
            st.session_state.chat_history.append({"role": "assistant", "content": doc_answer})
            return
        handle_general_chat(user_msg)

def ecom_chat_ui():
    st.markdown("### 🛒 Qelyon AI — E-Ticaret Asistanı (GPT-4o)")
    st.caption("Ürün açıklamaları, SEO başlıklar, etiketler.")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    upload = st.file_uploader(
        "Ürün görseli veya PDF yükle (opsiyonel)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        key="ecom_upload",
    )
    if upload is not None:
        st.session_state.chat_image = upload.read()
        st.session_state.chat_filename = upload.name
        st.success(f"📎 Dosya yüklendi: {upload.name}!")
    elif "ecom_upload" in st.session_state and st.session_state.ecom_upload is None:
        st.session_state.chat_image = None
        st.session_state.chat_filename = "dosya"
    user_msg = st.chat_input("Ürün veya ihtiyacını anlat...")
    if user_msg:
        handle_gpt_assistant("ecom", user_message=user_msg)

def consult_chat_ui():
    st.markdown("### 💼 Qelyon AI — Danışmanlık Asistanı (GPT-4o)")
    st.caption("İş modeli, büyüme stratejisi, KPI/OKR.")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    upload = st.file_uploader(
        "Rapor, PDF veya görsel yükle (opsiyonel)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        key="consult_upload",
    )
    if upload is not None:
        st.session_state.chat_image = upload.read()
        st.session_state.chat_filename = upload.name
        st.success(f"📎 Dosya yüklendi: {upload.name}!")
    elif "consult_upload" in st.session_state and st.session_state.consult_upload is None:
        st.session_state.chat_image = None
        st.session_state.chat_filename = "dosya"
    user_msg = st.chat_input("İşini veya sorunu anlat...")
    if user_msg:
        handle_gpt_assistant("consult", user_message=user_msg)

# =========================
# A7 — STÜDYO MODU
# =========================
PRESETS = {
    "🧹 Şeffaf Arka Plan": "transparent",
    "⬜ Beyaz Arka Plan": "white",
    "⬛ Siyah Arka Plan": "black",
    "🍦 Bej Arka Plan": "beige",
    "✨ Profesyonel Stüdyo": "pro",
}

def apply_preset(img: Image.Image, preset: str):
    """Hazır tema uygular."""
    cut = remove_bg_local(img)
    if preset == "transparent":
        return cut
    if preset == "white":
        return compose_scene(cut, "white", reflection=False)
    if preset == "black":
        return compose_scene(cut, "black", reflection=False)
    if preset == "beige":
        return compose_scene(cut, "beige", reflection=False)
    if preset == "pro":
        return compose_scene(cut, "white", reflection=True)
    return cut

def render_studio_mode():
    st.markdown("## 📸 Qelyon AI — Stüdyo Modu")
    st.caption("Profesyonel arka plan, ışık, gölge ve sahne oluşturma.")
    uploaded = st.file_uploader(
        "🎨 Ürün fotoğrafını yükle",
        type=["png", "jpg", "jpeg", "webp"],
        key="studio_upload",
    )
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGBA")
        st.image(img, caption="Yüklenen Görsel", width=350)
        st.session_state.studio_source = img
    if "studio_source" not in st.session_state or st.session_state.studio_source is None:
        st.info("Başlamak için bir ürün görseli yükle.")
        return
    img = st.session_state.studio_source
    col_presets, col_ai = st.columns(2)
    with col_presets:
        st.markdown("### 🎛 Hazır Temalar")
        preset_name = st.selectbox("Bir tema seç:", list(PRESETS.keys()), index=0, key="studio_preset_select")
        apply_preset_btn = st.button("🎨 Temayı Uygula", use_container_width=True)
    with col_ai:
        st.markdown("### ✨ AI Sahne Oluşturma (Opsiyonel)")
        ai_prompt = st.text_area(
            "Profesyonel sahne",
            placeholder="ör: lüks stüdyo ışığı, soft shadow, minimal set",
            key="ai_prompt_text",
            height=100,
        )
        generate_ai_scene = st.button("✨ AI Sahne Oluştur (Gemini Vision)", type="primary", use_container_width=True)
    result = None
    if apply_preset_btn:
        with st.spinner("Temanız işleniyor..."):
            result = apply_preset(img, PRESETS[preset_name])
            st.session_state.studio_result = result
    if generate_ai_scene and ai_prompt.strip():
        with st.spinner("AI sahne oluşturuluyor... (Gemini Vision)"):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            bytes_img = buffered.getvalue()
            ai_img_bytes = gemini_edit_scene(ai_prompt, bytes_img)
            if ai_img_bytes:
                result = Image.open(io.BytesIO(ai_img_bytes)).convert("RGBA")
                st.session_state.studio_result = result
            else:
                st.error("AI sahne oluşturulamadı veya bu SDK sürümü desteklemiyor.")
    if st.session_state.studio_result is not None:
        st.divider()
        st.markdown("### 📤 Çıktı")
        st.image(st.session_state.studio_result, width=512)
        output_buffer = io.BytesIO()
        if st.session_state.studio_result.mode == "P":
            st.session_state.studio_result.convert("RGB").save(output_buffer, format="PNG")
        else:
            st.session_state.studio_result.save(output_buffer, format="PNG")
        st.download_button(
            "📥 Çıktıyı İndir (PNG)",
            data=output_buffer.getvalue(),
            file_name="qelyon_studio_output.png",
            mime="image/png",
            use_container_width=True,
        )

# =========================
# 🖼️ ANA UYGULAMA
# =========================
def render_main_logo(dark_mode: bool):
    """Logo + mod butonları."""
    logo_path = LOGO_DARK if dark_mode else LOGO_LIGHT
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        if os.path.exists(logo_path):
            st.markdown(
                f'<img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" '
                f'style="height: 50px; margin-top: 10px;">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<h1 style='color: {THEME['accent']}; margin-top: 10px;'>QALYON</h1>",
                unsafe_allow_html=True,
            )
    with col_title:
        st.markdown(
            f"<h1 style='color: {THEME['accent']}; margin-top: 10px;'>Qelyon AI Stüdyo</h1>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    mode_cols = st.columns(4)
    modes = {
        "📸 Stüdyo Modu": "📸 Stüdyo (Gemini Vision)",
        "GENERAL_CHAT": "💬 Genel Chat (Gemini 1.5 Pro)",
        "ECOM": "🛒 E-Ticaret Asistanı (GPT-4o)",
        "CONSULT": "💼 Danışmanlık Asistanı (GPT-4o)",
    }
    for i, (key, label) in enumerate(modes.items()):
        with mode_cols[i]:
            if st.button(
                label,
                use_container_width=True,
                type="primary" if st.session_state.app_mode == key else "secondary",
                key=f"mode_btn_{i}",
            ):
                if key != st.session_state.app_mode:
                    st.session_state.chat_history = []
                    st.session_state.chat_image = None
                    st.session_state.chat_filename = "dosya"
                st.session_state.app_mode = key
                st.rerun()
    st.divider()

def render_footer():
    """Sabit footer."""
    footer_html = f"""
    <style>
    .footer {{
        position: fixed;
        left: 0; bottom: 0; width: 100%;
        background-color: {THEME['bg']}; color: {THEME['sub']};
        text-align: center; padding: 10px; font-size: 14px;
        border-top: 1px solid {THEME['border']}; z-index: 100;
    }}
    </style>
    <div class="footer">Qelyon AI © 2025 — Developed by Alper</div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def main_app_router():
    """Mod yönlendirici."""
    render_main_logo(dark)
    if st.session_state.app_mode == "📸 Stüdyo Modu":
        render_studio_mode()
    elif st.session_state.app_mode == "GENERAL_CHAT":
        general_chat_ui()
    elif st.session_state.app_mode == "ECOM":
        ecom_chat_ui()
    elif st.session_state.app_mode == "CONSULT":
        consult_chat_ui()
    render_footer()

if __name__ == "__main__":
    main_app_router()

