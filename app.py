# ==========================================================
# QELYON AI STÜDYO — FINAL v8
# Gemini Vision • Gemini Flash • Gemini 1.5 Pro • GPT-4o Hibrit Sistem
# ==========================================================

from __future__ import annotations

import os
import io
import re
import base64
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal, Optional

import requests
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw
import google.generativeai as genai
from openai import OpenAI
import mimetypes

import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_image(prompt: str) -> bytes:
    """GPT Image ile yeni görsel üret."""
    result = client.images.generate(
        model="gpt-image-1",          # veya gpt-image-1-mini
        prompt=prompt,
        size="1024x1024",             # kare ürün görseli için ideal
        n=1,
        # output_format="png",        # gpt-image-1 için opsiyonel, varsayılan png :contentReference[oaicite:2]{index=2}
    )
    b64 = result.data[0].b64_json
    return base64.b64decode(b64)

st.title("Qelyon AI — Görsel Stüdyo")

user_prompt = st.text_input(
    "Görsel isteğin",
    value="Beyaz fonda, stüdyo ışığında, parlayan gerçekçi yeşil elma, e-ticaret ürün görseli"
)

if st.button("Görsel oluştur"):
    img_bytes = generate_image(user_prompt)
    st.image(img_bytes, caption=user_prompt)


# ==========================================================
# 🔐 API KEYS & CONFIG
# ==========================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY", None)

GPT_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o")
GEMINI_TEXT_MODEL = "gemini-1.5-pro"
GEMINI_VISION_MODEL = "gemini-1.5-flash"
DEFAULT_CITY = "Ankara" # Hava durumu için varsayılan şehir

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY eksik. GPT modları çalışmaz.")
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY eksik. Gemini modları çalışmaz.")

# GPT istemcisini sadece anahtar varsa başlat
GPT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Gemini istemcisini konfigüre et
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    # Boş da olsa initialize eder, hataları yakalarız
    genai.configure(api_key="")

# ==========================================================
# 🎨 LOGO & FAVICON
# ==========================================================
# Bu dosyaların Streamlit uygulamasının dizininde olması gereklidir.
LOGO_LIGHT = "QelyonAIblack.png"
LOGO_DARK = "QelyonAIwhite.png"
FAVICON = "favicn.png"

st.set_page_config(
    page_title="Qelyon AI Stüdyo",
    page_icon=FAVICON,
    layout="wide",
)

# ==========================================================
# 🎨 THEME ENGINE
# ==========================================================
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
    else:
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

# ==========================================================
# 🌙 TEMA TOGGLE & UYGULAMA
# ==========================================================
col_a, col_b = st.columns([10,1])
with col_b:
    dark = st.toggle("🌙 / ☀️", value=True)

THEME = get_theme(dark)
apply_theme_css(THEME)

# ==========================================================
# 🧠 GLOBAL SESSION SETUP
# ==========================================================
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

# ==========================================================
# A2 — API CLIENTS • GEMINI + GPT • UTILITY FONKSİYONLARI
# ==========================================================

# ---------------------------
# 🔥 Gemini Client (Google AI)
# ---------------------------
def gemini_text(prompt: str):
    """Gemini 1.5 Pro ile metin üretimi"""
    if not GEMINI_API_KEY: return "Gemini API Anahtarı eksik."
    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print("Gemini text error:", e)
        return "Gemini şu anda yanıt veremiyor."

def gemini_vision(prompt: str, image_bytes: bytes):
    """Gemini Vision (Flash) ile görsel analiz"""
    if not GEMINI_API_KEY: return "Gemini API Anahtarı eksik."
    try:
        model = genai.GenerativeModel(GEMINI_VISION_MODEL)
        img_data = {"mime_type": "image/png", "data": image_bytes}
        resp = model.generate_content([prompt, img_data])
        return resp.text
    except Exception as e:
        print("Gemini vision error:", e)
        return "Görsel analizinde bir hata oluştu."

def gemini_generate_image(prompt: str, size="1024x1024"):
    """Gemini Flash Image ile görsel üretimi"""
    if not GEMINI_API_KEY: return None
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_image(prompt=prompt, size=size)
        return result._image  # bytes
    except Exception as e:
        print("Gemini image error:", e)
        return None

# ---------------------------
# 🤖 GPT-4o Client
# ---------------------------
def gpt_chat(messages: list[dict], model: str = GPT_MODEL):
    """
    GPT-4o tabanlı sohbet motoru (E-ticaret & Danışmanlık için)
    """
    if not GPT: return "GPT API Anahtarı eksik."
    try:
        res = GPT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        return res.choices[0].message.content
    except Exception as e:
        print("GPT error:", e)
        return "GPT sistemi şu anda cevap veremiyor."


# ---------------------------
# ⚡ MODEL ROUTER (Mod seçimine göre motor)
# ---------------------------
def model_router(mode: str):
    """Moda göre kullanılacak ana motoru belirler."""
    if mode == "GENERAL_CHAT":
        return "gemini"
    if mode in ["ECOM", "CONSULT"]:
        return "gpt"
    return "gemini"


# ==========================================================
# 📅 ZAMAN / TARİH SERVİSLERİ
# ==========================================================
def get_tr_time():
    """Türkiye yerel saatini döner."""
    try:
        r = requests.get("http://worldtimeapi.org/api/timezone/Europe/Istanbul")
        dt = r.json().get("datetime")
        return datetime.fromisoformat(dt)
    except:
        return datetime.now(ZoneInfo("Europe/Istanbul"))

def time_answer():
    now = get_tr_time()
    return f"Bugün {now.strftime('%d.%m.%Y')} — Saat {now.strftime('%H:%M')}"


# ==========================================================
# 🌦 HAVA DURUMU SERVİSİ
# ==========================================================
def get_coords(city: str):
    """Şehir adına göre enlem/boylam bulur."""
    if not WEATHER_API_KEY: return None
    try:
        url = (
            f"http://api.openweathermap.org/geo/1.0/direct?"
            f"q={city},TR&limit=1&appid={WEATHER_API_KEY}"
        )
        r = requests.get(url)
        data = r.json()
        if not data:
            return None
        return data[0]["lat"], data[0]["lon"]
    except:
        return None

def get_weather(city: str):
    """Şehir için hava durumu bilgisini döner."""
    if not WEATHER_API_KEY: return "Hava durumu API Anahtarı eksik."

    coords = get_coords(city)
    if not coords:
        return f"{city} için konum bulunamadı."

    lat, lon = coords
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=tr"
        )
        r = requests.get(url).json()

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
    except:
        return "Hava durumu alınamadı."


# ==========================================================
# 🛡 GÜVENLİK FİLTRESİ
# ==========================================================
BAD_WORDS = [
    r"(?i)orospu", r"(?i)siktir", r"(?i)amk",
    r"(?i)tecavüz", r"(?i)intihar", r"(?i)bomba yap",
]

def moderate_text(msg: str) -> str | None:
    """Mesaj uygunsuzsa engelle."""
    for pat in BAD_WORDS:
        if re.search(pat, msg):
            return "Bu isteğe güvenlik nedeniyle yanıt veremiyorum. 🙏"
    return None
    
# ==========================================================
# A3 — STÜDYO MODU • GÖRSEL İŞLEME (GEMINI + LOCAL)
# ==========================================================

# ---------------------------------------
# 🧼 1) LOKAL ARKA PLAN KALDIRMA (HQ MASKING)
# ---------------------------------------
def remove_bg_local(image: Image.Image) -> Image.Image:
    """
    Ürünü fotoğraftan lokal threshold + mask algoritması ile ayırır.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    gray = image.convert("L")

    # Yüksek threshold → parlak arka plan silinir
    mask = gray.point(lambda p: 255 if p > 240 else 0)

    result = Image.new("RGBA", image.size)
    result.paste(image, (0, 0), mask)
    return result


# ---------------------------------------
# 🎛 2) ÜRÜNÜ KARE TUVALE MERKEZE YERLEŞTİRME
# ---------------------------------------
def center_on_canvas(img: Image.Image, size=1024) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    obj = img.copy()
    obj.thumbnail((size * 0.84, size * 0.84), Image.Resampling.LANCZOS)

    x = (size - obj.width) // 2
    y = (size - obj.height) // 2

    canvas.paste(obj, (x, y), obj)
    return canvas


# ---------------------------------------
# 🌓 3) PROFESYONEL TEMAS GÖLGESİ
# ---------------------------------------
def make_contact_shadow(alpha: Image.Image, intensity=150):
    """Ürünün altına ticari stüdyo tarzı soft shadow üretir."""
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


# ---------------------------------------
# 🌫 4) STÜDYO REFLECTION (Soft Reflection)
# ---------------------------------------
def make_reflection(img: Image.Image, fade=230):
    """Alt kısımda premium stüdyo refleks efekti üretir."""
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


# ---------------------------------------
# 🎨 5) TEMA KOMPOZİT MOTORU
# ---------------------------------------
def compose_scene(cut: Image.Image, bg_color: str, reflection=True, shadow=True):
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


# ---------------------------------------
# ✨ 6) GEMINI — AI SAHNE OLUŞTURMA
# ---------------------------------------
def gemini_edit_scene(prompt: str, product_image_bytes: bytes):
    """
    Ürünü bozmadan; sadece arka planı AI ile profesyonel olarak yeniden tasarlar.
    Gemini Flash Image modeli ile çalışır.
    """
    if not GEMINI_API_KEY: return None
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        img_dict = {
            "mime_type": "image/png",
            "data": product_image_bytes,
        }

        full_prompt = (
            "You are a professional commercial product photographer. "
            "Replace ONLY the background. "
            "Do NOT modify the product (color, texture, geometry). "
            "Generate a clean, studio-grade, elegant scene.\n"
            f"Background style: {prompt}"
        )

        result = model.generate_image(
            prompt=full_prompt,
            image=img_dict,
            size="1024x1024",
        )

        return result._image  # PNG bytes

    except Exception as e:
        print("Gemini Edit Scene Error:", e)
        return None


# ---------------------------------------
# 🏗 7) HAZIR TEMA PRESETLERİ
# ---------------------------------------
PRESETS = {
    "🧹 Şeffaf Arka Plan": "transparent",
    "⬜ Beyaz Arka Plan": "white",
    "⬛ Siyah Arka Plan": "black",
    "🍦 Bej Arka Plan": "beige",
    "✨ Profesyonel Stüdyo": "pro",
}

def apply_preset(img: Image.Image, preset: str):
    """
    Kullanıcının seçtiği hazır temayı uygular.
    """
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
    
# ==========================================================
# A8 — 📄 PDF & 🖼 Görsel OCR + Belge Analiz Motoru (Gemini 1.5 Pro)
# ==========================================================

def guess_mime_type(filename: str, default: str = "application/octet-stream") -> str:
    """Dosya adına göre MIME type tahmini."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or default


def gemini_analyze_document(
    file_bytes: bytes,
    filename: str,
    user_instruction: str = "Bu dosyayı profesyonelce özetle ve önemli maddeleri çıkar.",
) -> str:
    """PDF, PNG, JPG gibi dosyaları Gemini 1.5 Pro ile okur (OCR + anlamlandırma)."""

    if not GEMINI_API_KEY: return "Gemini API Anahtarı eksik."
    if not file_bytes: return "Dosya içeriği boş görünüyor."

    mime_type = guess_mime_type(filename)

    file_part = {
        "mime_type": mime_type,
        "data": file_bytes,
    }

    prompt = (
        "Sen Qelyon AI doküman analiz uzmanısın. "
        "PDF, resim veya taranmış belge içeriğini dikkatlice okur, "
        "önemli kısımları net ve anlaşılır bir şekilde özetlersin. "
        "Maddeler halinde kritik başlıkları ve aksiyon alınabilir önerileri çıkar.\n\n"
        f"Kullanıcı talimatı: {user_instruction}"
    )

    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        response = model.generate_content([prompt, file_part])

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "Dosya analiz edildi fakat metin cevap üretilemedi."
    except Exception as e:
        print("Gemini Document Analyze Error:", e)
        return "Belge analizinde bir hata oluştu. Lütfen daha sonra tekrar dene."


def analyze_uploaded_file_in_chat(user_message: str) -> str:
    """
    General Chat içinde:
      - Kullanıcı PDF / görsel yüklediyse
      - 'bu pdfi özetle', 'bu görseli analiz et', 'bu dosyayı incele' gibi bir şey yazdıysa
    → Bu fonksiyon çağrılıp Gemini belge analiz çalıştırılabilir.
    """
    if st.session_state.chat_image is None:
        return ""

    triggers = [
        "pdfi özetle", "pdf'i özetle", "pdf özetle",
        "bu dosyayı özetle", "bu dosyayı analiz et", "belgeyi analiz et",
        "dokümanı analiz et", "bu görseli analiz et", "bu resmi analiz et",
        "dosyayı incele",
    ]

    if not any(t in user_message.lower() for t in triggers):
        # Kullanıcının isteği doğrudan doküman analizi değilse, normal chat akışı devam eder.
        return ""

    # Belge analizi isteniyor
    file_bytes = st.session_state.chat_image
    filename = st.session_state.chat_filename
    user_instruction = user_message # Kullanıcının mesajını talimat olarak kullan

    result = gemini_analyze_document(file_bytes, filename, user_instruction)
    return result

# ==========================================================
# A4 — GENEL CHAT MOTORU (GEMINI 1.5 PRO)
# ==========================================================
IMAGE_TRIGGER_WORDS = [
    "görsel oluştur", "resim oluştur", "foto üret",
    "bir görsel çiz", "image create", "generate image",
    "bana bir tasarım yap", "logo yap", "arka plan üret",
]

def is_image_generation_request(msg: str) -> bool:
    """Kullanıcının görsel üretim isteği yapıp yapmadığını kontrol eder."""
    msg = msg.lower()
    return any(t in msg for t in IMAGE_TRIGGER_WORDS)


def gemini_general_chat(user_message: str, user_image: bytes | None):
    """
    Genel Chat (💬) için tam sohbet motoru: Gemini 1.5 Pro metin ve vision.
    """
    if not GEMINI_API_KEY: return "Gemini API Anahtarı eksik."

    try:
        # --- 1) Sohbet geçmişini Gemini formatına dönüştür ---
        history = []
        for msg in st.session_state.chat_history[-25:]:
            if msg["role"] == "user":
                history.append({
                    "role": "user",
                    "parts": [msg["content"]]
                })
            elif msg["role"] == "assistant" and msg["content"] != "(Görsel üretildi)":
                history.append({
                    "role": "model",
                    "parts": [msg["content"]]
                })

        # --- 2) Kullanıcının yeni mesajı ---
        new_parts = [{"text": user_message}]

        # --- 3) Eğer görsel yüklüyse ekle ---
        if user_image:
            # Burası sadece görsel yüklenmişse tetiklenir (Aynı anda PDF yüklenmişse de)
            # Analiz için sadece görsel part'ı eklenir
            if not st.session_state.chat_filename.lower().endswith('.pdf'):
                new_parts.append({
                    "inline_data": {
                        "mime_type": "image/png", # Varsayılan olarak png kabul edilir
                        "data": base64.b64encode(user_image).decode("utf-8")
                    }
                })
        
        user_turn = {
            "role": "user",
            "parts": new_parts
        }

        full_prompt = history + [user_turn]

        # --- 4) Gemini model ---
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(full_prompt)

        if hasattr(response, "text"):
            return response.text

        return "Bir yanıt üretemedim."

    except Exception as e:
        print("General Chat Error:", e)
        return "💥 Üzgünüm, şu anda genel chat yanıt veremiyor."


def handle_general_chat(user_message: str):
    """Genel Chat UI → Motor bağlayıcı."""

    # 1) Kullanıcı mesajını geçmişe kaydet
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_message
    })

    with st.chat_message("user"):
        st.write(user_message)

    # 2) Görsel üretim isteği
    if is_image_generation_request(user_message):
        with st.chat_message("assistant"):
            st.write("🎨 Görsel oluşturuluyor...")
            img_bytes = gemini_generate_image(user_message)

            if img_bytes:
                st.image(img_bytes, caption="✨ Gemini 1.5 Flash tarafından üretildi", width=350)
                ai_answer = "(Görsel üretildi)" # Geçmişe kısa not
            else:
                ai_answer = "⚠️ Görsel oluşturulamadı, lütfen tekrar deneyin."
            
            if ai_answer != "(Görsel üretildi)":
                st.write(ai_answer)

    # 3) Normal metin + görsel analizi sohbeti (Görsel üretim istenmediyse)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Qelyon AI düşünüyor..."):
                ai_answer = gemini_general_chat(
                    user_message,
                    st.session_state.chat_image
                )
                st.write(ai_answer)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": ai_answer
    })


# ==========================================================
# A5 — GPT SYSTEM TALİMATI (E-Ticaret + Danışmanlık Persona)
# ==========================================================

def build_system_talimati(profile: Literal["ecom", "consult"]) -> str:

    if profile == "ecom":
        return """
Sen Qelyon AI'nın E-Ticaret Uzmanı modundasın.
Görevlerin:
1) Ürün açıklaması (SEO uyumlu, profesyonel, ikna edici)
2) Ürünün öne çıkan 5 faydasını yaz
3) Kutu içeriği oluştur
4) Hedef kitle analizi yap
5) Kullanım önerileri üret
6) Ürüne özel CTA (satın almaya yönlendiren)
7) Ürün görseli/PDF varsa analiz et, metne entegre et
8) Ürün için A/B testli başlık varyantları oluştur
9) Trendyol için akıllı etiket algoritması çalıştır
10) Fiyat psikolojisi optimizasyonu önerileri ver
11) Ürün varyantlarını belirle (renk, beden, kapasite, model)
12) Müşteri yorumlarını analiz edip memnuniyet/şikayet temalarını çıkar
13) Sosyal medya reklam metinleri üret (Meta, TikTok, Instagram)
14) Marka hikâyesi yaz
15) İçerikleri Türkçe ve profesyonel bir tonda oluştur

DİKKAT:
- Gereksiz uzunluk yok, doğrudan ticari fayda odaklı yaz.
- Ürün görseli/PDF varsa mutlaka analiz ederek davran.
- Qelyon AI kimliğinden sapma: YASAK.
"""
        
    # ------------------------------------------------------

    if profile == "consult":
        return """
Sen Qelyon AI'nın Danışmanlık Uzmanı modundasın.
Uzmanlık alanların:
- İş geliştirme
- Marka konumlandırma
- Finansal iyileştirme
- Operasyonel verimlilik
- Pazarlama stratejisi
- Dijital dönüşüm
- SWOT + rakip analizi
- KPI çıkarımı
- Yol haritası oluşturma

Görevlerin:
1) Kullanıcının iş modelini analiz et
2) Sektöre özel strateji öner
3) KPI ve hedef sistemi çıkar
4) SWOT analizi yap
5) Adım adım gelişim planı oluştur
6) Gerektiğinde gelir modeli öner
7) İş fikri validasyonu yap
8) PDF veya doküman varsa analiz et, çıktıya dahil et
9) Görsel varsa içgörü üret (ör: mağaza fotoğrafı, ürün, afiş)

Kimlik:
“Qelyon AI olarak, profesyonel danışmanlık ve veri destekli içgörülerle iş hedeflerine ulaşmanı hızlandırıyorum.”

Tarz:
- Kesin
- Analitik
- Stratejik
- Gereksiz hikâye yok, tamamen iş odaklı.
"""

    return "Qelyon AI sistem talimatı uygulanamadı."

# ==========================================================
# A5 — IDENTITY INTERCEPTOR (Qelyon AI KİMLİK SİSTEMİ)
# ==========================================================

def custom_identity_interceptor(msg: str) -> Optional[str]:
    """Kimlik veya tanışma sorularına cevap verir."""
    msg_low = msg.lower()

    if any(x in msg_low for x in ["kimsin", "sen neysin", "kim yapt", "kim geliştirdi"]):
        return "Ben Qelyon AI'yım. Hibrit bir mimari kullanıyorum: Gemini Vision + GPT-4o. Qelyon AI ekibi tarafından geliştirildim."

    if "openai" in msg_low or "gpt" in msg_low:
        return "Ben Qelyon AI'yım. GPT-4o teknolojisini kullanıyorum ancak Qelyon AI'ya özel yeteneklerle genişletildim. Hibrit bir sistemim."

    if "ne iş yaparsın" in msg_low or "görevin ne" in msg_low:
        return "Qelyon AI olarak, profesyonel danışmanlık ve veri destekli içgörülerle iş hedeflerine ulaşmanı hızlandırıyorum."

    return None

# ==========================================================
# A5 — UTILITY INTERCEPTOR (Zaman + Hava)
# ==========================================================

def custom_utility_interceptor(msg: str) -> Optional[str]:
    """Saat ve hava durumu gibi genel bilgilere cevap verir."""
    m = msg.lower()

    # Saat
    if "saat" in m and ("kaç" in m or "?" in m):
        return time_answer()

    # Hava durumu
    if "hava" in m or "hava durumu" in m:
        return get_weather(DEFAULT_CITY)

    return None

# ==========================================================
# A5 — GPT-4o ANA ASİSTAN MOTORU
# ==========================================================

def gpt_assistant(profile: Literal["ecom", "consult"], user_message: str) -> str:
    if not GPT: return "GPT API Anahtarı eksik."

    try:
        system_msg = build_system_talimati(profile)
        
        # Kullanıcının mesajını ve yüklü dosya bilgisini system'e ekle
        user_content = [{"type": "text", "text": user_message}]
        
        # Eğer chat_image yüklü ise
        if st.session_state.chat_image:
            # Görsel için
            if not st.session_state.chat_filename.lower().endswith('.pdf'):
                encoded_image = base64.b64encode(st.session_state.chat_image).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                })
            # PDF için metinsel olarak referans ver
            else:
                pdf_analysis = gemini_analyze_document(
                    st.session_state.chat_image, 
                    st.session_state.chat_filename, 
                    "Bu PDF/doküman içeriğini özetle."
                )
                system_msg += f"\n\n[EK DOSYA ANALİZİ ({st.session_state.chat_filename})]:\n{pdf_analysis}"


        msgs = [{"role": "system", "content": system_msg}]
        
        # Geçmişteki metin mesajlarını ekle (Çoklu-modal mesajlar GPT-4o'da farklı ele alınır)
        for m in st.session_state.chat_history[-10:]: # Sadece metin mesajlarını ekle
             if m["role"] == "user" and m["content"] == user_message:
                 continue # Şu anki mesajı eklememek için

             msgs.append({
                 "role": m["role"],
                 "content": m["content"]
             })

        # En son kullanıcı mesajını ekle (içinde görsel/pdf bilgisi de var)
        msgs.append({
            "role": "user", 
            "content": user_content
        })


        res = GPT.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            temperature=0.3,
            max_tokens=1800,
        )

        return res.choices[0].message.content

    except Exception as e:
        print("GPT Assist Error:", e)
        return "Şu anda GPT-4o yanıt veremiyor. Birkaç dakika sonra tekrar deneyin."

# ==========================================================
# A5 — GPT ASİSTANI UI ROUTER
# ==========================================================

def handle_gpt_assistant(profile: Literal["ecom", "consult"], user_message: str):
    """GPT Asistanları için ana işleyici (ECOM ve CONSULT)."""
    if not user_message:
        return

    # 1) Kullanıcı mesajını geçmişe kaydet
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    # 2) Interceptor (Kimlik/Yardımcı) kontrolü
    ident = custom_identity_interceptor(user_message)
    util = custom_utility_interceptor(user_message)

    if ident:
        answer = ident
    elif util:
        answer = util
    else:
        # 3) Normal GPT-4o cevabı
        with st.chat_message("assistant"):
            with st.spinner("Qelyon AI düşünüyor..."):
                answer = gpt_assistant(profile, user_message)
    
    with st.chat_message("assistant"): st.write(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ==========================================================
# A6 — MOD SEÇİMİ + GENEL CHAT (Gemini) + GPT ASİSTAN UI
# ==========================================================

# ---------------------------------------------
# 💬 2) GENEL CHAT UI (Gemini 1.5 Pro + Flash)
# ---------------------------------------------
def general_chat_ui():
    st.markdown("### 💬 Qelyon AI — Genel Chat (Gemini)")
    st.caption("Gemini 1.5 Pro & Flash ile metin, görsel analizi ve görsel oluşturma yapabilirsin.")

    # --- Mesaj geçmişi ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["content"] != "(Görsel üretildi)": # Görsel üretim notunu gösterme
                st.write(msg["content"])

    # --- Dosya yükleme (görsel / pdf) ---
    upload = st.file_uploader(
        "Görsel / PDF yükle (isteğe bağlı)",
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

    # --- Kullanıcı mesajı ---
    user_msg = st.chat_input("Mesajını yaz...")

    if user_msg:
        # Güvenlik filtresi
        mod = moderate_text(user_msg)
        if mod:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"): st.write(user_msg)
            with st.chat_message("assistant"): st.write(mod)
            st.session_state.chat_history.append({"role": "assistant", "content": mod})
            return

        # Doküman analizi tetikleniyor mu?
        doc_answer = analyze_uploaded_file_in_chat(user_msg)
        if doc_answer:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"): st.write(user_msg)
            with st.chat_message("assistant"): st.write(doc_answer)
            st.session_state.chat_history.append({"role": "assistant", "content": doc_answer})
            return

        # Normal Gemini general chat akışı
        handle_general_chat(user_msg)


# ---------------------------------------------
# 🛒 3) E-TİCARET ASİSTANI UI (GPT-4o)
# ---------------------------------------------
def ecom_chat_ui():
    st.markdown("### 🛒 Qelyon AI — E-Ticaret Asistanı (GPT-4o)")
    st.caption("Ürün açıklamaları, SEO başlıklar, etiketler ve kampanya metinleri için kullan.")

    # Mesaj geçmişi
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # İsteğe bağlı görsel / pdf yükleme
    upload = st.file_uploader(
        "Ürün görseli veya PDF yükle (opsiyonel)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        key="ecom_upload",
    )
    if upload is not None:
        st.session_state.chat_image = upload.read()
        st.session_state.chat_filename = upload.name
        st.success(f"📎 Dosya yüklendi: {upload.name}! Ürün açıklamasında bu dosyaya referans verebilirsin.")
    elif "ecom_upload" in st.session_state and st.session_state.ecom_upload is None:
         st.session_state.chat_image = None
         st.session_state.chat_filename = "dosya"

    user_msg = st.chat_input("Ürün veya ihtiyacını anlat...")

    if user_msg:
        handle_gpt_assistant("ecom", user_message=user_msg)


# ---------------------------------------------
# 💼 4) DANIŞMANLIK ASİSTANI UI (GPT-4o)
# ---------------------------------------------
def consult_chat_ui():
    st.markdown("### 💼 Qelyon AI — Danışmanlık Asistanı (GPT-4o)")
    st.caption("İş modeli, büyüme stratejisi, KPI/OKR ve operasyonel verimlilik için kullan.")

    # Mesaj geçmişi
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Opsiyonel doküman/görsel yükleme
    upload = st.file_uploader(
        "Rapor, PDF veya görsel yükle (opsiyonel)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        key="consult_upload",
    )
    if upload is not None:
        st.session_state.chat_image = upload.read()
        st.session_state.chat_filename = upload.name
        st.success(f"📎 Dosya yüklendi: {upload.name}! Analiz yaparken bu dosyadan bahsedebilirsin.")
    elif "consult_upload" in st.session_state and st.session_state.consult_upload is None:
         st.session_state.chat_image = None
         st.session_state.chat_filename = "dosya"

    user_msg = st.chat_input("İşini veya sorunu anlat...")

    if user_msg:
        handle_gpt_assistant("consult", user_message=user_msg)


# ==========================================================
# A7 — 📸 QELYON AI STÜDYO MODU (FINAL v8)
# ==========================================================

def render_studio_mode():
    st.markdown("## 📸 Qelyon AI — Stüdyo Modu")
    st.caption("Ürünlerin için profesyonel arka plan, ışık, gölge ve sahne oluşturma modu.")

    # ------------------------
    # 1) Görsel yükleme alanı
    # ------------------------
    uploaded = st.file_uploader(
        "🎨 Ürün fotoğrafını yükle",
        type=["png", "jpg", "jpeg", "webp"],
        key="studio_upload",
    )

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGBA")
        st.image(img, caption="Yüklenen Görsel", width=350)
        st.session_state.studio_source = img
    
    # Eğer görsel yoksa ve önceki yükleme durumunda görsel yoksa devam etme
    if "studio_source" not in st.session_state or st.session_state.studio_source is None:
        st.info("Başlamak için bir ürün görseli yükle.")
        return

    img = st.session_state.studio_source

    col_presets, col_ai = st.columns(2)

    with col_presets:
        # ------------------------
        # 2) Preset seçimleri
        # ------------------------
        st.markdown("### 🎛 Hazır Temalar")

        preset_name = st.selectbox(
            "Bir tema seç:",
            list(PRESETS.keys()),
            index=0,
            key="studio_preset_select"
        )
        
        # 4) İşlem butonu
        apply_preset_btn = st.button("🎨 Temayı Uygula", use_container_width=True)


    with col_ai:
        # ------------------------
        # 3) Profesyonel AI sahne oluşturma
        # ------------------------
        st.markdown("### ✨ AI Sahne Oluşturma (Opsiyonel)")

        ai_prompt = st.text_area(
            "Profesyonel sahne (örn: 'lüx stüdyo ışığı, soft shadow, minimal set')",
            placeholder="Buraya yazarsan Gemini Vision özel sahne oluşturur.",
            key="ai_prompt_text",
            height=100
        )

        generate_ai_scene = st.button("✨ AI Sahne Oluştur (Gemini Vision)", type="primary", use_container_width=True)

    
    result = None

    # ------------------------
    # 5) Temayı uygula (lokal render)
    # ------------------------
    if apply_preset_btn:
        with st.spinner("Temanız işleniyor..."):
            result = apply_preset(img, PRESETS[preset_name])
            st.session_state.studio_result = result

    # ------------------------
    # 6) AI sahne oluştur
    # ------------------------
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
                st.error("AI sahne oluşturulamadı. Lütfen yeniden deneyin.")

    # ------------------------
    # 7) Sonuç görüntüleme
    # ------------------------
    if st.session_state.studio_result is not None:
        st.divider()
        st.markdown("### 📤 Çıktı")

        st.image(st.session_state.studio_result, width=512)

        # İndirilebilir link
        output_buffer = io.BytesIO()
        # st.session_state.studio_result.save(output_buffer, format="PNG")
        
        # Eğer sonuç AI'dan geldiyse (JPG/WebP olabilir), PNG olarak kaydetmeyi deneyin
        if st.session_state.studio_result.mode == 'P':
             st.session_state.studio_result.convert('RGB').save(output_buffer, format="PNG")
        else:
             st.session_state.studio_result.save(output_buffer, format="PNG")
             
        st.download_button(
            "📥 Çıktıyı İndir (PNG)",
            data=output_buffer.getvalue(),
            file_name="qelyon_studio_output.png",
            mime="image/png",
            use_container_width=True
        )


# ==========================================================
# 🖼️ B1 — ANA UYGULAMA YAPISI (MAIN APP)
# ==========================================================

def render_main_logo(dark_mode: bool):
    """Koyu/açık moda göre logo ve başlık hizalaması ve mod butonları."""
    logo_path = LOGO_DARK if dark_mode else LOGO_LIGHT
    
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        # Logo dosyasının varlığını kontrol et
        if os.path.exists(logo_path):
             # Base64 ile küçük bir logo render et
            st.markdown(f'<img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" style="height: 50px; margin-top: 10px;">', unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: {THEME['accent']}; margin-top: 10px; font-size: 30px;'>QALYON</h1>", unsafe_allow_html=True)

    with col_title:
        st.markdown(f"<h1 style='color: {THEME['accent']}; margin-top: 10px;'>Qelyon AI Stüdyo</h1>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='margin-bottom: 20px;'></div>",
        unsafe_allow_html=True
    )

    # 4 Modun butonları
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
                key=f"mode_btn_{i}"
            ):
                # Mod değişimi yapıldığında chat geçmişini temizle
                if key != st.session_state.app_mode:
                    st.session_state.chat_history = []
                    st.session_state.chat_image = None
                    st.session_state.chat_filename = "dosya"
                    
                st.session_state.app_mode = key
                st.rerun()

    st.divider()

def render_footer():
    """İstenilen footer bilgisini sayfanın en altına sabitleyen HTML/CSS."""
    footer_html = f"""
    <style>
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: {THEME['bg']};
        color: {THEME['sub']};
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid {THEME['border']};
        z-index: 100;
    }}
    </style>
    <div class="footer">
        Qelyon AI © 2025 — Developed by Alper
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def main_app_router():
    """Ana akışı yöneten router."""
    
    render_main_logo(dark) # Tema toggl'ını kullan

    # Mod seçimine göre içeriği yönlendir
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
