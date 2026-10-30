# Phase 2, Step 6 + Phase 3, Step 10: Streamlit dashboard. Run: streamlit run frontend.py
import json
import os
import time

import cv2
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components

API = os.getenv("VISIONPATH_API", "http://127.0.0.1:8000")
VOICE = {"English": "en-IN", "Hindi": "hi-IN"}

st.set_page_config(page_title="VisionPath AI", page_icon="🦯")
st.title("🦯 VisionPath AI")
st.caption("An AI guide for visually impaired users")

language = st.sidebar.selectbox("Auto-speak language", ["English", "Hindi"])
speak_on = st.sidebar.checkbox("Speak instructions aloud", value=True)
question = st.text_input("Ask something (e.g. 'Where is my medicine?')")

def speak(items):
    """items = [(text, 'English'|'Hindi'), ...]"""
    data = [[t, VOICE[l]] for t, l in items]
    components.html(
        f"""<script>
        // run {time.time()}
        var items = {json.dumps(data)};
        function go() {{
            var voices = window.speechSynthesis.getVoices();
            window.speechSynthesis.cancel();
            items.forEach(function(it) {{
                var u = new SpeechSynthesisUtterance(it[0]);
                u.lang = it[1];
                var same = voices.filter(function(v) {{
                    return v.lang.replace('_', '-') === it[1];
                }});
                var v = same.find(function(x) {{ return x.localService; }}) || same[0];
                if (v) u.voice = v;
                window.speechSynthesis.speak(u);
            }});
        }}
        if (window.speechSynthesis.getVoices().length) {{ go(); }}
        else {{ window.speechSynthesis.onvoiceschanged = go; }}
        </script>""",
        height=0,
    )

photo = st.camera_input("Turn camera on and take a snapshot")

if photo:
    img_bytes = photo.getvalue()
    if st.session_state.get("last_img") != img_bytes:
        try:
            resp = requests.post(
                f"{API}/guide",
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
                data={"language": language, "question": question},
                timeout=60,
            )
            st.session_state["result"] = resp.json()
            st.session_state["last_img"] = img_bytes
            st.session_state["auto_speak"] = True
        except requests.exceptions.ConnectionError:
            st.error("Backend is not running. Start uvicorn first (Terminal 1).")

result = st.session_state.get("result")
if photo and result:
    frame = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    for o in result["objects"]:
        x, y, w, h = o["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        dist = f" {o['distance_m']}m" if o["distance_m"] else ""
        cv2.putText(frame, f"{o['label']}{dist}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="What VisionPath sees")

    for w in result["quality"]["warnings"]:
        st.warning({"low_light": "Low light - please be careful.",
                    "blurry": "Image is blurry."}[w])

    ins = result["instructions"]
    st.success(f"{ins['English']}")
    st.success(f"{ins['Hindi']}")
    st.subheader(f"Detected objects: {len(result['objects'])}")
    st.dataframe(
        [
            {
                "Object": o["label"],
                "Distance (m)": o["distance_m"] if o["distance_m"] else "unknown",
                "Direction": o["direction"],
                "Confidence": o["confidence"],
            }
            for o in result["objects"]
        ],
        use_container_width=True,
    )
    if result["text"]:
        st.info(f"Text found: {result['text']}")

    c1, c2, c3 = st.columns(3)
    rep_en = c1.button("🔁 English")
    rep_hi = c2.button("🔁 हिन्दी")
    rep_both = c3.button("🔁 Both")

    auto = st.session_state.pop("auto_speak", False)
    if speak_on:
        if rep_both:
            speak([(ins["English"], "English"), (ins["Hindi"], "Hindi")])
        elif rep_en:
            speak([(ins["English"], "English")])
        elif rep_hi:
            speak([(ins["Hindi"], "Hindi")])
        elif auto:
            speak([(ins[language], language)])

    with st.expander("Technical details"):
        st.json(result["objects"])