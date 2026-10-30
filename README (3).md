# VisionPath AI - Complete Build Guide

An AI navigation assistant for visually impaired users: camera -> object detection ->
distance -> OCR -> LLM agent -> voice.

```
visionpath/
  requirements.txt      export_model.py     vision.py (core CV)
  phase1_webcam.py      agent.py            backend.py (FastAPI)
  frontend.py (Streamlit)
```

---------------------------------------------------------------
## PHASE 1 - Foundation (Steps 1-4)
---------------------------------------------------------------

### Step 1: Environment setup
```bash
python --version                     # need 3.10+
python -m venv venv
source venv/bin/activate             # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Install the Tesseract program too (needed for OCR in Step 7):
- Windows: https://github.com/UB-Mannheim/tesseract/wiki (add to PATH)
- Ubuntu:  `sudo apt install tesseract-ocr tesseract-ocr-hin`
- Mac:     `brew install tesseract`

### Step 2: Get the model (one time)
```bash
python export_model.py     # creates yolov8n.onnx
```

### Step 3: Webcam + detection
```bash
python phase1_webcam.py
```
You should see green boxes with labels. Read `vision.py` -> `Detector.detect()`:
1. `blobFromImage` resizes the frame to 640x640 and scales pixels to 0-1.
2. `net.forward()` returns 8400 candidate boxes, each with 4 box numbers + 80 class scores.
3. We keep scores above `conf`, then `NMSBoxes` removes duplicate overlapping boxes.

### Step 4: Distance estimation
Formula (pinhole camera): `distance = real_height x focal_length_px / box_height_px`

**Calibrate FOCAL_PX** (this makes your demo accurate):
1. Stand a person (1.7 m tall) exactly 2 m from the webcam.
2. Print the box height in pixels (`d["box"][3]`), e.g. 480.
3. `FOCAL_PX = box_height x 2.0 / 1.7`. Put that value in `vision.py`.

---------------------------------------------------------------
## PHASE 2 - UI and Backend (Steps 5-7)
---------------------------------------------------------------

### Step 5: Backend
```bash
uvicorn backend:app --reload --port 8000
```
Open http://localhost:8000/docs and test `/analyze` by uploading a photo.

### Step 6: Frontend (second terminal)
```bash
streamlit run frontend.py
```
Take a snapshot -> boxes, warnings, JSON and the spoken instruction appear.

### Step 7: OCR
Already in `vision.read_text()`. Test: hold a medicine strip or a printed "EXIT" sign
in front of the camera; the text appears in the blue box.
Tip: for Hindi text use `pytesseract.image_to_string(gray, lang="eng+hin")`.

---------------------------------------------------------------
## PHASE 3 - Agentic Intelligence (Steps 8-10)
---------------------------------------------------------------

### Step 8: Context-aware prompt
`agent.build_context()` turns detections into:
`Objects: chair (1.2m, left), person (3.0m, ahead)  Text seen: EXIT`

### Step 9: Autonomous decisions
`agent.decide()` sends that to an LLM with a system prompt that demands a safety-first
decision ("stop", "move left"...). Enable it:
```bash
export ANTHROPIC_API_KEY=your_key        # Windows: set ANTHROPIC_API_KEY=your_key
```
Without a key the rule-based fallback runs, so the demo never breaks.
Check the model name in the Anthropic docs (https://docs.claude.com) if the call errors.

### Step 10: Voice
`frontend.py` uses the browser's speech engine, so it works on phones too.
Offline alternative on your own laptop:
```python
import pyttsx3
e = pyttsx3.init(); e.say("Stop, chair ahead"); e.runAndWait()
```

---------------------------------------------------------------
## PHASE 4 - Cloud and Optimization (Steps 11-13)
---------------------------------------------------------------

### Step 11: Deploy on AWS Graviton
1. EC2 -> Launch instance -> Ubuntu 22.04 **ARM64**, type `t4g.medium` (Graviton).
2. Security group: open ports 22, 8000, 8501.
3. SSH in, then:
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip tesseract-ocr tesseract-ocr-hin libgl1
# copy the project (scp -r visionpath ubuntu@<ip>:~)
cd visionpath && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && python export_model.py
# On the server you can then remove ultralytics to save space.
nohup uvicorn backend:app --host 0.0.0.0 --port 8000 &
VISIONPATH_API=http://localhost:8000 nohup streamlit run frontend.py --server.port 8501 &
```
4. Open `http://<ec2-public-ip>:8501`.
5. For the competition write a short note: *what runs on Graviton, speed measured
   (ms/frame on Arm), and cost vs x86* (t4g is cheaper per hour).
   Also verify the current special-award rules on the official competition page.

### Step 12: Latency
Already done: frames resized to 640px (`backend.decode`), small model (YOLOv8n).
More ideas:
- Frontend: resize image before upload (`cv2.resize` + JPEG quality 70).
- Run OCR only every few seconds (`ocr=false` on most calls).
- Measure: add `time.time()` around `analyze()` and report ms/frame in your demo.

### Step 13: Edge cases and Responsible AI
Already in `vision.image_quality()` (dark / blur warnings -> spoken alert).
Add to your report:
- This is an **assistive aid, not a replacement** for a cane or guide dog.
- Distances are estimates; the system says "about".
- No images are stored; mention privacy.
- Test with real visually impaired users or an NGO if possible.

---------------------------------------------------------------
## Make it unique
1. Hindi: choose "Hindi" in the sidebar (the LLM and fallback both support it).
2. "Where is my medicine?" -> type or speak it in the question box; the agent
   gets it in the context and answers using object positions and OCR text.
3. Next upgrades: stairs/door detection with a custom-trained YOLO model,
   continuous streaming with `streamlit-webrtc`, speech input.

## Troubleshooting
- `cv2.error ... readNetFromONNX`: re-run `export_model.py`, keep `opset=12`.
- Webcam not opening: try `VideoCapture(1)`, close Zoom/Teams.
- OCR returns empty: Tesseract not installed or not on PATH.
- No detections: lower `conf` to 0.3 in `Detector(...)`.
