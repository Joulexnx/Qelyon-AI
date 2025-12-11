# ==========================================================
# QELYON AI STÜDYO — FINAL v14 (Prompt Optimizasyonlu)
# ==========================================================

from __future__ import annotations

import os
import io
import re
import base64
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal, Optional, Any

import requests
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw
from openai import OpenAI
import mimetypes
from tempfile import NamedTemporaryFile

import base64
from io import BytesIO
from PIL import Image
# client tanımı ve generate_image fonksiyonu korunmuştur
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_image(prompt: str) -> bytes:
    """GPT Image ile yeni görsel üretir (Kullanıcının özel konfigürasyonu korunmuştur)."""
    # NOT: DALL-E/GPT Image API'si 1080x1350'yi desteklemez. En yakın yüksek çözünürlük 1024x1024'e düşürüldü.
    # Ancak orijinal kodunuzdaki model ve boyutlar istek üzerine korundu, API'nin hata verebileceğini unutmayın.
    result = client.images.generate(
        model="gpt-image-1", # veya "dall-e-3" kullanılması önerilir
        prompt=prompt,
        size="1024x1024", # 1080x1350 yerine desteklenen standart boyut kullanıldı.
        n=1,
    )
    b64 = result.data[0].b64_json
    return base64.b64decode(b64)

# Eski başlangıç bloğu kaldırıldı.

# ==========================================================
# 🔐 API KEYS & CONFIG
# ==========================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
# GPT_MODEL korundu
GPT_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY eksik. Uygulama çalışmaz.")

# GPT istemcisini sadece anahtar varsa başlat
GPT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==========================================================
# 🎨 LOGO & FAVICON
# ==========================================================
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
# Modlar sadece Stüdyo ve Sohbet olarak ayarlandı
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "🎨 Stüdyo" # Varsayılan mod değiştirildi

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_image" not in st.session_state:
    st.session_state.chat_image = None

if "chat_filename" not in st.session_state:
    st.session_state.chat_filename = "dosya"

if "studio_result" not in st.session_state:
    st.session_state.studio_result = None

# Yeni session state'ler eklendi
if "studio_last_image_bytes" not in st.session_state:
    st.session_state.studio_last_image_bytes = None
    
if "studio_base_prompt" not in st.session_state:
    st.session_state.studio_base_prompt = ""


# ==========================================================
# A2 — API CLIENTS • UTILITY FONKSİYONLARI (Sadece OpenAI)
# ==========================================================

# ---------------------------
# 🤖 GPT-4o Client (Metin)
# ---------------------------
def gpt_chat_only(messages: list[dict], model: str = GPT_MODEL) -> str:
    """GPT-4o tabanlı sadece metin sohbet motoru."""
    if not GPT: return "OpenAI API Anahtarı eksik."
    try:
        res = GPT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        return res.choices[0].message.content
    except Exception as e:
        print("GPT chat error:", e)
        return "OpenAI sistemi şu anda cevap veremiyor."


# ---------------------------
# 🖼️ GÖRSEL DÜZENLEME (OPTIMİZE EDİLDİ) - GÜNCELLENMİŞ VERSİYON
# ---------------------------
def get_dalle_regenerative_prompt(base_image_bytes: bytes, user_command: str) -> str | None:
    """
    GPT-4o Vision'ı kullanarak mevcut bir görseli analiz eder ve 
    kullanıcının isteği doğrultusunda yeniden oluşturulmuş, güçlü bir DALL-E 3 prompt'u üretir.
    (Görselin kompozisyonunu ve ürünlerin yerleşimini korumaya odaklanılmıştır.)
    """
    if not GPT: return None
    
    # Görseli Base64'e çevir
    base64_image = base64.b64encode(base_image_bytes).decode('utf-8')
    
    # GÜNCELLENMİŞ PROMPT BURADA
    analysis_prompt = (
        "Sen üst düzey bir DALL-E 3 prompt mühendisisin. Görevin, verilen görselin kompozisyonunu, "
        "ürün yerleşimini (dikey sıra, grup, tekil, oran), stilini (peluş, kumaş, el yapımı vb.), "
        "ışığını (stüdyo, doğal) ve tüm estetik detaylarını **mükemmel doğrulukla** analiz etmektir. "
        "Bu analize dayanarak ve KULLANICININ İSTEDİĞİ DEĞİŞİKLİĞİ (Arka planı kaldır/değiştir/renk değiştir vb.) **EN YÜKSEK KALİTEDE** uygulayan, "
        "orijinal görselin **kompozisyonunu ve açısını birebir koruyan**, yepyeni bir DALL-E 3 prompt'u oluştur. "
        "Prompt'un en önemli kısmı, ürünlerin orijinal görseldeki **AYNI DÜZENDE, AYNI SAYIDA** ve **AYNI POZİSYONDA** olmasını sağlamaktır. "
        "Sadece **yeni prompt'u** döndür, başka hiçbir metin veya açıklama ekleme. "
        "Kullanıcının Düzenleme İsteği: " + user_command
    )
    # GÜNCELLENMİŞ PROMPT SONU
    
    try:
        response = GPT.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": analysis_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.8,
            max_tokens=300
        )
        new_prompt = response.choices[0].message.content.strip()
        
        # Sonuçta sadece prompt metninin döndüğünden emin olmak için temizlik
        new_prompt = new_prompt.replace('"', '').replace("'", '').strip()
        
        return new_prompt
    except Exception as e:
        print(f"Prompt Üretme Hatası (GPT-4o Vision): {e}")
        return None

def optimized_dalle_edit(image_bytes: bytes, user_command: str) -> bytes | None:
    """
    GPT-4o Vision ile analiz edilen ve yeniden oluşturulan prompt'u kullanarak
    GPT Image 1 ile GERÇEK EDIT yapar (orijinal ürünü olabildiğince korur).
    """
    if not client:
        return None

    # 1) Vision'dan base prompt'u al
    new_prompt = get_dalle_regenerative_prompt(image_bytes, user_command)
    if not new_prompt:
        st.error("Görseli analiz edip yeni komut oluşturulamadı.")
        return None

    # 2) Edit için daha güvenli, ürün odaklı final prompt
    # (İngilizce tutmak, görsel modeller için daha stabil oluyor)
    full_prompt = (
        "Edit this product photo. Keep the original product exactly the same "
        "(shape, size, logo, colors, camera angle). "
        "Only apply the following change to the background or environment: "
        f"{user_command}. "
        "Do not add new products or remove existing ones.\n\n"
        f"Base layout description:\n{new_prompt}"
    )

    st.info(f"🎨 Oluşturulan edit komutu: {full_prompt[:160]}...")

    # 3) image_bytes → PNG → geçici dosya (images.edit dosya objesi bekliyor)
    tmp_path: Optional[str] = None
    try:
        # Bytes'tan resmi aç
        img = Image.open(io.BytesIO(image_bytes))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # Geçici PNG dosyası oluştur
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(buf.getvalue())
            tmp_path = tmp.name

        # 4) GPT Image edit endpoint'i ile gerçek düzenleme
        with open(tmp_path, "rb") as f:
            result = client.images.edit(
                model="gpt-image-1",
                image=f,
                prompt=full_prompt,
                size="1024x1024",
                input_fidelity="high",   # Ürünü olabildiğince koru :contentReference[oaicite:2]{index=2}
            )

        if result.data and result.data[0].b64_json:
            img_bytes = base64.b64decode(result.data[0].b64_json)
            st.session_state.studio_last_image_bytes = img_bytes
            st.session_state.studio_base_prompt = full_prompt
            return img_bytes

        return None

    except Exception as e:
        st.error(f"Görsel Düzenleme Hatası (GPT Image edit): {e}")
        return None

    finally:
        # Geçici dosyayı temizle
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass



# ---------------------------
# 🛡 GÜVENLİK FİLTRESİ
# ---------------------------
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
    
# ---------------------------
# 🖼️ GÖRSEL İSTEĞİ TESPİTİ (SOHBET MODU İÇİN)
# ---------------------------
VISUAL_EDIT_TRIGGERS = [
    "renk değiştir", "nesne ekle", "stil değiştir", 
    "arka planı değiştir", "çizgisel yap", "çıkar",
    " yap", "yeşil yap", "mavi yap",
    "kaldır", "yerine koy", "yapıştır", "olsun"
]



def is_visual_edit_request(msg: str) -> bool:
    """Kullanıcının görsel üzerinde düzenleme isteği yapıp yapmadığını kontrol eder."""
    msg = msg.lower()
    return any(t in msg for t in VISUAL_EDIT_TRIGGERS)

# ---------------------------
# 🛍️ ÜRÜN METNİ İSTEĞİ TESPİTİ
# ---------------------------
PRODUCT_TEXT_TRIGGERS = [
    "ürün ismi", "ürün adı", "ürüne isim", "ürüne ad",
    "isim ve açıklama", "isim açıklama", "ürün açıklaması",
    "ürün için açıklama", "cta yaz", "satış metni yaz",
    "ürün metni yaz", "ürün için isim"
]

def is_product_text_request(msg: str) -> bool:
    """Kullanıcı ürün ismi/açıklaması istiyor mu?"""
    msg = msg.lower()
    return any(t in msg for t in PRODUCT_TEXT_TRIGGERS)


def product_copy_from_image(image_bytes: bytes, user_instruction: str) -> str:
    """Yüklenen ürün fotoğrafına bakarak isim + açıklama + CTA üretir."""
    if not GPT:
        return "OpenAI API anahtarı olmadığı için ürün metni üretemiyorum."

    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": (
                    "Sen Türkçe yazan, e-ticaret odaklı bir metin yazarı asistansın. "
                    "Kullanıcının gönderdiği ürün fotoğrafını analiz et ve sadece şu formatta cevap ver:\n\n"
                    "1) Ürün adı: ...\n"
                    "2) Kısa açıklama: 2-3 cümle\n"
                    "3) CTA: Satın almaya teşvik eden kısa bir cümle\n\n"
                    "Sade, profesyonel ve akılda kalıcı bir ton kullan. Emoji kullanma."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            },
        ]

        res = GPT.chat.completions.create(
            model="gpt-4o",   # Vision destekli model
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        return res.choices[0].message.content.strip()

    except Exception as e:
        print("product_copy_from_image error:", e)
        return "Ürün ismi ve açıklaması oluşturulurken bir hata oluştu."


    
# ==========================================================
# 💬 SOHBET MODU (CHAT)
# ==========================================================

def handle_chat_visual_request(user_message: str, image_bytes: bytes) -> tuple[str, bytes | None]:
    """Sohbet modundan gelen görsel düzenleme isteğini optimize Stüdyo motoruna yönlendirir."""
    
    if not image_bytes:
        return "Görsel düzenleme isteği aldım, ancak düzenlenecek bir görsel bulamadım. Lütfen görseli yüklediğinden emin ol.", None

    # Optimize edilmiş DALL-E Edit fonksiyonunu kullan
    edited_img_bytes = optimized_dalle_edit(image_bytes, user_message)
    
    if edited_img_bytes:
        return f"Görsel düzenleme isteğin başarıyla tamamlandı: **'{user_message}'**. Yeni görsel aşağıdadır.", edited_img_bytes
    else:
        return "Üzgünüm, görsel düzenleme sırasında bir hata oluştu veya isteğin gerçekleştirilemedi.", None

def render_chat_mode():
    st.markdown("### 💬 Sohbet")
    st.caption("Genel bilgi ve diyalog için kullanın. Görsel yükleyip düzenleme de talep edebilirsin.")

    # --- Dosya yükleme (görsel / pdf) ---
    upload = st.file_uploader(
        "Görsel / PDF yükle (isteğe bağlı)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        key="general_chat_upload",
    )

    if upload is not None:
        file_bytes = upload.read()
        st.session_state.chat_image = file_bytes
        st.session_state.chat_filename = upload.name

        # Yüklenen görseli üstte göster (sadece image ise)
        if upload.type and upload.type.startswith("image/"):
            try:
                st.image(file_bytes, caption="Yüklenen Görsel", width=300)
            except Exception:
                pass

        # Aynı dosya için sohbet geçmişine sadece 1 kez ekle
        if st.session_state.get("last_chat_upload_name") != upload.name:
            st.session_state.last_chat_upload_name = upload.name
            st.session_state.chat_history.append({
                "role": "user",
                "content": {
                    "text": f"📎 Görsel yüklendi: {upload.name}",
                    "image": file_bytes,
                },
            })

        st.success(
            f"📎 Dosya yüklendi: {upload.name}! "
            "Mesajında bu dosyadan bahsedebilir, ürün ismi/açıklaması isteyebilir veya düzenleme talep edebilirsin."
        )

    elif "general_chat_upload" in st.session_state and st.session_state.general_chat_upload is None:
        st.session_state.chat_image = None
        st.session_state.chat_filename = "dosya"

    # --- Mesaj geçmişi ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], str):
                st.write(msg["content"])
            elif isinstance(msg["content"], dict):
                if "text" in msg["content"]:
                    st.write(msg["content"]["text"])
                if "image" in msg["content"]:
                    caption = "Yüklenen Görsel" if msg["role"] == "user" else "İşlem Görmüş Görsel"
                    st.image(msg["content"]["image"], caption=caption, width=350)

    # --- Kullanıcı mesajı ---
    user_msg = st.chat_input("Mesajını yaz...")

    if not user_msg:
        return

    # 1) Kullanıcı mesajını geçmişe kaydet
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # 2) Güvenlik filtresi
    mod = moderate_text(user_msg)
    if mod:
        with st.chat_message("assistant"):
            st.write(mod)
        st.session_state.chat_history.append({"role": "assistant", "content": mod})
        return

    # 3) Eğer görsel düzenleme isteği varsa (ve görsel yüklüyse) -> Stüdyo motoru
    if st.session_state.chat_image and is_visual_edit_request(user_msg):
        with st.chat_message("assistant"):
            with st.spinner("🎨 Görsel düzenleniyor (Stüdyo Motoru)..."):
                ai_answer_text, edited_bytes = handle_chat_visual_request(
                    user_msg,
                    st.session_state.chat_image
                )

                if edited_bytes:
                    st.image(edited_bytes, caption="Düzenlenmiş Görsel", width=350)
                    ai_answer_content = {"text": ai_answer_text, "image": edited_bytes}
                else:
                    ai_answer_content = ai_answer_text
                    st.write(ai_answer_text)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ai_answer_content}
                )
                return

    # 4) Eğer görsel yüklü ve mesaj ürün ismi/açıklaması istiyorsa -> Vision ile ürün metni
    if st.session_state.chat_image and is_product_text_request(user_msg):
        with st.chat_message("assistant"):
            with st.spinner("🛍️ Ürün ismi ve açıklaması hazırlanıyor..."):
                answer = product_copy_from_image(st.session_state.chat_image, user_msg)
                st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        return

    # 5) Normal metin sohbet akışı
    with st.chat_message("assistant"):
        with st.spinner("Qelyon AI düşünüyor..."):
            ai_answer = gpt_chat_only(
                [
                    {
                        "role": "system",
                        "content": (
                            "Sen Qelyon AI'nın genel sohbet asistanısın. "
                            "Kısa, net ve genel bilgiler sun."
                        ),
                    },
                    {"role": "user", "content": user_msg},
                ]
            )
            st.write(ai_answer)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})


# ==========================================================
# 🎨 STÜDYO MODU (GÖRSEL OLUŞTURMA VE ARDIŞIK DÜZENLEME)
# ==========================================================

def render_studio_mode():
    st.markdown("## 🎨 Stüdyo")
    st.caption("Sıfırdan görsel oluştur, yüklenen görseli düzenle ve ardışık düzenleme akışını kullan. Tüm işlemler GPT Image ile yapılır.")

    tab1, tab2 = st.tabs(["🖼️ Görsel Oluşturma (Yeni/Ardışık)", "✏️ Görsel Düzenleme (Yükle)"])
    
    if 'current_studio_tab' not in st.session_state:
        st.session_state.current_studio_tab = 1
    

    # ---------------------------------------------
    # TAB 1: GÖRSEL OLUŞTURMA & ARDŞIK DÜZENLEME
    # ---------------------------------------------
    with tab1:
        st.session_state.current_studio_tab = 1
        st.markdown("### ✍️ Yeni Görsel Oluştur veya Son Görseli Düzenle")
        
        last_bytes = st.session_state.studio_last_image_bytes
        
        if last_bytes:
            st.image(last_bytes, caption="Son Görseliniz", width=250)
            st.info(f"📝 Son temel prompt: {st.session_state.studio_base_prompt[:100]}... Yeni komut sadece istediğin değişikliği belirtmelidir.")
        else:
            st.info("Bu alanda metin girerek sıfırdan görsel oluşturabilirsin.")
            
        user_prompt = st.text_area(
            "Görsel İsteği / Düzenleme Komutu",
            placeholder="Örn: 'Lüks stüdyo ışığı altında, beyaz fonda uçan  spor ayakkabı' (Ardışık düzenleme için ' ayakkabıyı mavi yap' gibi komutlar kullanın)",
            key="studio_prompt_text",
            height=100
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        if last_bytes:
            process_label = "✏️ Görseli Düzenle (Ardışık)"
            process_key = "studio_edit_btn"
        else:
            process_label = "🖼️ Yeni Görsel Oluştur"
            process_key = "studio_create_btn"
            
        if col_btn1.button(process_label, use_container_width=True, key=process_key, type="primary"):
            if not user_prompt.strip():
                st.error("Lütfen bir komut girin.")
                return
            
            with st.spinner(f"Görseliniz işleniyor... ({process_label})"):
                if last_bytes:
                    # Düzenleme (Optimize Edit)
                    result_bytes = optimized_dalle_edit(last_bytes, user_prompt)
                else:
                    # Oluşturma (Create)
                    # generate_image fonksiyonu kullanıcının özel konfigürasyonunu (gpt-image-1) kullanır.
                    result_bytes = generate_image(user_prompt) 
                
                if result_bytes:
                    st.session_state.studio_last_image_bytes = result_bytes
                    st.session_state.studio_result = Image.open(io.BytesIO(result_bytes))
                else:
                    st.error("Görsel işlenirken bir hata oluştu. Lütfen tekrar deneyin.")

    # ---------------------------------------------
    # TAB 2: YÜKLENEN GÖRSELİ DÜZENLEME
    # ---------------------------------------------
    with tab2:
        st.session_state.current_studio_tab = 2
        st.markdown("### 📸 Mevcut Görseli Yükle ve Düzenle")
        
        uploaded = st.file_uploader(
            "Düzenlemek istediğiniz fotoğrafı yükle",
            type=["png", "jpg", "jpeg", "webp"],
            key="studio_upload_edit",
        )
        
        if uploaded:
            uploaded_bytes = uploaded.read()
            st.image(uploaded_bytes, caption="Yüklenen Görsel", width=300)
            
            edit_prompt = st.text_area(
                "Düzenleme Komutu (Yüklenen Görsel İçin)",
                placeholder="Örn: 'Bu siyah arabayı parlak kırmızı yap'",
                key="studio_upload_prompt",
                height=100
            )
            
            if st.button("✏️ Yüklenen Görseli Düzenle", use_container_width=True, key="upload_edit_btn", type="primary"):
                if not edit_prompt.strip():
                    st.error("Lütfen düzenleme için bir komut girin.")
                    return
                
                with st.spinner("Görseliniz düzenleniyor (Optimize GPT Image Edit)..."):
                    # Optimize edilmiş DALL-E Edit fonksiyonunu kullan
                    result_bytes = optimized_dalle_edit(uploaded_bytes, edit_prompt)
                    
                    if result_bytes:
                        st.session_state.studio_last_image_bytes = result_bytes 
                        st.session_state.studio_result = Image.open(io.BytesIO(result_bytes))
                    else:
                        st.error("Görsel işlenirken bir hata oluştu.")
        
    # ---------------------------------------------
    # ÇIKTI BÖLÜMÜ (Tüm sekmeler için ortaktır)
    # ---------------------------------------------
    if st.session_state.studio_result is not None:
        st.divider()
        st.markdown("### 📤 Sonuç")
        
        st.image(st.session_state.studio_result, width=512)

        output_buffer = io.BytesIO()
        try:
            st.session_state.studio_result.convert('RGB').save(output_buffer, format="PNG")
        except:
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
        if os.path.exists(logo_path):
            st.markdown(f'<img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" style="height: 50px; margin-top: 10px;">', unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: {THEME['accent']}; margin-top: 10px; font-size: 30px;'>QALYON</h1>", unsafe_allow_html=True)

    with col_title:
        st.markdown(f"<h1 style='color: {THEME['accent']}; margin-top: 10px;'>Qelyon AI Stüdyo</h1>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # 2 Modun butonları
    mode_cols = st.columns(2)
    modes = {
        "💬 Sohbet": "💬 Sohbet",
        "🎨 Stüdyo": "🎨 Stüdyo",
    }
    
    for i, (key, label) in enumerate(modes.items()):
        with mode_cols[i]:
            # Stüdyo Modu adını key olarak "🎨 Stüdyo" olarak kabul et.
            actual_key = "🎨 Stüdyo" if key == "🎨 Stüdyo" else "💬 Sohbet"
            
            if st.button(
                label,
                use_container_width=True,
                type="primary" if st.session_state.app_mode == actual_key else "secondary",
                key=f"mode_btn_{i}"
            ):
                if actual_key != st.session_state.app_mode:
                    st.session_state.chat_history = []
                    st.session_state.chat_image = None
                    st.session_state.chat_filename = "dosya"
                    st.session_state.studio_result = None
                    st.session_state.studio_last_image_bytes = None
                    st.session_state.studio_base_prompt = ""
                    
                st.session_state.app_mode = actual_key
                st.rerun()

    st.divider()

def render_footer():
    """İstenilen footer bilgisini sayfanın en altına sabitleyen HTML/CSS."""
    footer_html = f"""
    <style>
    .footer {{
        position: fixed; left: 0; bottom: 0; width: 100%; 
        background-color: {THEME['bg']}; color: {THEME['sub']}; 
        text-align: center; padding: 10px; font-size: 14px; 
        border-top: 1px solid {THEME['border']}; z-index: 100;
    }}
    </style>
    <div class="footer">
        Qelyon AI © 2025 — Developed by Alper
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def main_app_router():
    """Ana akışı yöneten router."""
    
    render_main_logo(dark)

    if st.session_state.app_mode == "🎨 Stüdyo":
        render_studio_mode()
    elif st.session_state.app_mode == "💬 Sohbet":
        render_chat_mode()
    
    render_footer()

if __name__ == "__main__":
    main_app_router()



