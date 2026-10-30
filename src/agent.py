"""The 'thinking' agent. Returns instructions in English AND Hindi."""
import os

SYSTEM = (
    "You are VisionPath, a navigation assistant for a visually impaired person. "
    "You receive what the camera sees. Reply with ONE or TWO short sentences, "
    "in {lang}, telling the user what to do right now. Prioritise safety: "
    "stairs, vehicles, obstacles closer than 1.5 m, then useful text (signs, medicine). "
    "Say direction (left/right/ahead) and distance. Never invent objects not listed."
)

LABELS_HI = {
    "person": "इंसान", "chair": "कुर्सी", "car": "कार", "bus": "बस", "truck": "ट्रक",
    "bicycle": "साइकिल", "motorcycle": "मोटरसाइकिल", "dining table": "मेज़",
    "couch": "सोफ़ा", "bed": "बिस्तर", "tv": "टीवी", "bottle": "बोतल", "cup": "कप",
    "laptop": "लैपटॉप", "refrigerator": "फ्रिज", "bench": "बेंच", "dog": "कुत्ता",
    "cat": "बिल्ली", "stop sign": "स्टॉप साइन", "traffic light": "ट्रैफिक लाइट",
    "book": "किताब", "cell phone": "फ़ोन", "backpack": "बैग", "handbag": "हैंडबैग",
    "umbrella": "छाता", "potted plant": "पौधा", "clock": "घड़ी", "scissors": "कैंची",
    "keyboard": "कीबोर्ड", "mouse": "माउस", "remote": "रिमोट", "bowl": "कटोरा",
}
DIR_HI = {"left": "बाईं ओर", "right": "दाईं ओर", "ahead": "सामने"}


def build_context(result, user_question=None):
    parts = []
    for o in result["objects"][:8]:
        dist = f"{o['distance_m']}m" if o["distance_m"] else "unknown distance"
        parts.append(f"{o['label']} ({dist}, {o['direction']})")
    ctx = "Objects: " + (", ".join(parts) if parts else "none detected")
    if result["text"]:
        ctx += f"\nText seen: {result['text'][:200]}"
    if result["quality"]["warnings"]:
        ctx += f"\nCamera warnings: {', '.join(result['quality']['warnings'])}"
    if user_question:
        ctx += f"\nUser asks: {user_question}"
    return ctx


def rule_based(result, lang="English"):
    hindi = lang.lower().startswith("hin")
    w = result["quality"]["warnings"]
    if "low_light" in w:
        return "रोशनी कम है, कृपया संभल कर चलें।" if hindi else "Low light, please walk carefully."
    if "blurry" in w:
        return "कैमरा धुंधला है, कृपया रुकें।" if hindi else "Camera is blurry, please stop."

    objs = result["objects"]
    if not objs:
        return "रास्ता साफ है।" if hindi else "The path looks clear."

    o = objs[0]
    d = o["distance_m"]
    near = d is not None and d < 1.5
    if hindi:
        label = LABELS_HI.get(o["label"], o["label"])
        where = DIR_HI[o["direction"]]
        if near:
            return f"सावधान! {where} {label} सिर्फ {d} मीटर दूर है। रुकिए।"
        extra = f", लगभग {d} मीटर दूर" if d else ""
        return f"{where} {label} है{extra}।"
    extra = f", about {d} metres" if d else ""
    if near:
        return f"Careful! {o['label']} {o['direction']}, only {d} metres. Please stop."
    return f"{o['label']} {o['direction']}{extra}."


def decide(result, lang="English", user_question=None):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return rule_based(result, lang)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=120,
            system=SYSTEM.format(lang=lang),
            messages=[{"role": "user", "content": build_context(result, user_question)}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return rule_based(result, lang)


def decide_both(result, user_question=None):
    return {
        "English": decide(result, "English", user_question),
        "Hindi": decide(result, "Hindi", user_question),
    }