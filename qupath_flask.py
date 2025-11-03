# app.py (Flask 서버 예시)
import torch
from flask import Flask, request, jsonify
from Model.CellViT_ViT256_Custom import CellViTCustom
from PIL import Image
import numpy as np
import io

# 모델 로드
model = CellViTCustom(
    num_nuclei_classes=5,  # 예시로 5개 클래스
    num_tissue_classes=0,
    img_size=256,
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0
)
model.load_state_dict(torch.load('Checkpoints/CellViT/cellvit_vit256_consep_merged_best.pth'))
model.eval()

# Flask 앱 설정
app = Flask(__name__)

# 이미지 전처리 함수
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))  # 모델에 맞게 리사이즈
    img = np.array(img) / 255.0  # [0,1]로 정규화
    img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    return torch.tensor(img).unsqueeze(0).float()  # (1, C, H, W) 텐서로 변환

# 추론 API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    # 이미지 요청 받기
    image_file = request.files['image']
    image_bytes = image_file.read()

    # 이미지 전처리
    img_tensor = preprocess_image(image_bytes)

    # 모델에 추론 요청
    with torch.no_grad():
        output = model(img_tensor)

    # 출력 처리 (예: 타입 맵)
    type_map = output['nuclei_type_map'].squeeze().cpu().numpy()  # 예시로 nuclei_type_map 반환

    # JSON 응답
    return jsonify({
        'type_map': type_map.tolist()  # 리스트로 변환해서 반환
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # 서버 실행
