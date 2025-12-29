from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "my_model.pt")
model = YOLO(MODEL_PATH)

app = Flask(__name__)

@app.post("/detect")
def detect():
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

    f = request.files["image"]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        f.save(tmp.name)
        path = tmp.name

    try:
        results = model.predict(path, imgsz=640, conf=float(request.form.get("conf", 0.25)), verbose=False)
        r = results[0]

        dets = []
        # boxes: xyxy, conf, cls
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names
            for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
                dets.append({
                    "cls": names.get(int(k), str(int(k))),
                    "cls_id": int(k),
                    "conf": float(c),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)]
                })

        return jsonify({"count": len(dets), "detections": dets})
    finally:
        try: os.remove(path)
        except: pass

@app.get("/")
def health():
    return {"ok": True}
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)