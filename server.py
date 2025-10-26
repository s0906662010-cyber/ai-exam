from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import mediapipe as mp

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands()

# 做弊率分析的邏輯（暫時簡單版）
def analyze_frame(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(img_rgb)
    hand_results = hands.process(img_rgb)
    faces = face_results.multi_face_landmarks or []
    hands_list = hand_results.multi_hand_landmarks or []

    cheat_score = 0.05
    reasons = []

    # 偵測臉和手
    if len(faces) == 0:
        cheat_score = 0
        reasons.append("沒偵測到臉")
    else:
        cheat_score += 0.1 * len(faces)
        if len(hands_list) > 0:
            cheat_score += 0.3
            reasons.append("手靠近臉")

    cheat_score = min(1.0, cheat_score)
    return {"score": cheat_score, "reasons": reasons}

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "沒有圖片"}), 400

    image_bytes = base64.b64decode(image_b64.split(",")[1])
    img_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    result = analyze_frame(frame)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
