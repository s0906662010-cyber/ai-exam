from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import mediapipe as mp

# 修正 Flask 語法，確保能正確運行
app = Flask(__name__)

# 初始化 MediaPipe 模組
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 新的行為風險分析邏輯
def analyze_frame(img_bgr):
    """
    分析影像幀，計算行為風險分數。
    將「作弊」概念轉為「異常行為」，以符合金融科技應用。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 處理臉部和手部偵測
    face_results = face_mesh.process(img_rgb)
    hand_results = hands.process(img_rgb)
    
    # 初始化風險分數和原因
    risk_score = 0.0
    risk_factors = []

    # 1. 偵測多個臉部：潛在的旁人干擾
    if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 1:
        risk_score += 0.8  # 高風險行為
        risk_factors.append("偵測到多個臉部，可能有旁人")
    
    # 2. 偵測沒有臉部：使用者離開鏡頭
    elif not face_results.multi_face_landmarks:
        risk_score += 0.9  # 高風險行為
        risk_factors.append("未偵測到臉部，使用者可能離開")
    
    # 3. 偵測手部靠近臉部：可能在遮擋或與第三方互動
    if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
        # 簡單的判斷邏輯：判斷手部是否在臉部附近
        # 這裡可以透過座標比較來實現更精準的判斷
        # 範例：檢查手部關鍵點是否與臉部關鍵點重疊
        face_landmarks = face_results.multi_face_landmarks[0]
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        
        # 獲取手部和臉部標誌點的平均y座標，進行簡化判斷
        face_center_y = sum([lm.y for lm in face_landmarks.landmark]) / len(face_landmarks.landmark)
        hand_center_y = sum([lm.y for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
        
        # 如果手部y座標與臉部y座標接近，則視為「手靠近臉部」
        if abs(hand_center_y - face_center_y) < 0.2: # 調整此閾值來控制靈敏度
            risk_score += 0.5  # 中高風險行為
            risk_factors.append("手部靠近臉部，可能在操作其他裝置或遮擋")

    # 4. 偵測視線偏離：使用者注意力分散
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        # 這裡可以透過計算頭部或眼球姿勢來實現
        # 簡化範例：檢查頭部是否明顯側向
        nose_tip = face_landmarks.landmark[mp_face_mesh.FaceLandmark.NOSE_TIP]
        left_ear = face_landmarks.landmark[mp_face_mesh.FaceLandmark.LEFT_EAR]
        right_ear = face_landmarks.landmark[mp_face_mesh.FaceLandmark.RIGHT_EAR]
        if abs(nose_tip.x - (left_ear.x + right_ear.x) / 2) > 0.1: # 調整此閾值
             risk_score += 0.3 # 中風險行為
             risk_factors.append("視線明顯偏離螢幕")

    # 將最終分數限制在 0.0 到 1.0 之間
    final_score = min(1.0, risk_score)
    
    return {"score": final_score, "reasons": risk_factors}

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

# 修正 main 語法，確保能正確運行
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)