from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import mediapipe as mp

# 初始化 Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ 開放所有來源（可改為指定 GitHub 網址）

@app.route("/")
def home():
    return "✅ Flask server is running on Render!"

# 初始化 MediaPipe 模組
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 分析影像幀的函式
def analyze_frame(img_bgr):
    """
    分析影像幀，計算行為風險分數。
    將「作弊」概念轉為「異常行為」，以符合金融科技應用。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    face_results = face_mesh.process(img_rgb)
    hand_results = hands.process(img_rgb)
    
    risk_score = 0.0
    risk_factors = []

    # 1. 偵測多個臉部
    if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 1:
        risk_score += 0.8
        risk_factors.append("偵測到多個臉部，可能有旁人")

    # 2. 沒有臉部
    elif not face_results.multi_face_landmarks:
        risk_score += 0.9
        risk_factors.append("未偵測到臉部，使用者可能離開")

    # 3. 手部靠近臉部
    if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        
        face_center_y = sum([lm.y for lm in face_landmarks.landmark]) / len(face_landmarks.landmark)
        hand_center_y = sum([lm.y for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
        
        if abs(hand_center_y - face_center_y) < 0.2:
            risk_score += 0.5
            risk_factors.append("手部靠近臉部，可能在操作其他裝置或遮擋")

    # 4. 視線偏離
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        nose_tip = face_landmarks.landmark[mp_face_mesh.FaceLandmark.NOSE_TIP]
        left_ear = face_landmarks.landmark[mp_face_mesh.FaceLandmark.LEFT_EAR]
        right_ear = face_landmarks.landmark[mp_face_mesh.FaceLandmark.RIGHT_EAR]
        if abs(nose_tip.x - (left_ear.x + right_ear.x) / 2) > 0.1:
            risk_score += 0.3
            risk_factors.append("視線明顯偏離螢幕")

    final_score = min(1.0, risk_score)
    return {"score": final_score, "reasons": risk_factors}

# API 端點
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    image_bytes = base64.b64decode(image_b64.split(",")[1])
    img_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    result = analyze_frame(frame)
    return jsonify(result)

# 啟動伺服器
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
