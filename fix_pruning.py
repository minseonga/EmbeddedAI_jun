import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO
import os

# [수정됨] 모델 파일이 있는 경로 설정
TARGET_DIR = 'assets/models'

# 해당 경로가 존재하는지 확인
if not os.path.exists(TARGET_DIR):
    print(f"❌ 경로를 찾을 수 없습니다: {TARGET_DIR}")
    # 혹시 몰라 현재 경로로 다시 설정
    TARGET_DIR = '.'
    print(f"🔄 현재 폴더({os.getcwd()})에서 다시 찾아봅니다...")

# 경로 내의 pruned 파일 찾기 (전체 경로 포함)
files = [
    os.path.join(TARGET_DIR, f) 
    for f in os.listdir(TARGET_DIR) 
    if f.endswith('.pt') and 'pruned' in f and 'fixed' not in f
]

if not files:
    print(f"❌ '{TARGET_DIR}' 폴더에서 Pruned 된 .pt 파일을 찾을 수 없습니다.")
else:
    print(f"📂 '{TARGET_DIR}' 폴더에서 {len(files)}개의 파일을 발견했습니다.")
    
    for f_path in files:
        print(f"🔧 처리 중: {f_path}")
        try:
            # 모델 로드
            model = YOLO(f_path)
            
            # 마스크 제거 및 가중치 0 확정
            count = 0
            for name, m in model.model.named_modules():
                if hasattr(m, "weight_mask"):
                    prune.remove(m, "weight")
                    count += 1
            
            # 새 파일명 생성 (_fixed 추가)
            new_path = f_path.replace(".pt", "_fixed.pt")
            model.save(new_path)
            
            # 용량 비교
            old_size = os.path.getsize(f_path) / (1024*1024)
            new_size = os.path.getsize(new_path) / (1024*1024)
            print(f"✅ 완료! {count}개 레이어 고정됨.")
            print(f"📉 용량 변화: {old_size:.2f}MB -> {new_size:.2f}MB\n")
            
        except Exception as e:
            print(f"❌ 에러 발생 ({f_path}): {e}")