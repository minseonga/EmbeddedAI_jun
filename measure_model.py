import sys
import os
import csv
import time
import torch
import torch.nn as nn
from pathlib import Path
from thop import profile

# --- 필수 경로 설정 (기존과 동일) ---
ROOT = Path(__file__).resolve().parents[0] 
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# HandTrackingPipeline 불러오기 (경로 문제 해결 후 사용)
try:
    from hand_tracking.pipeline import HandTrackingPipeline # 파이프라인이 파일 이름이므로 수정
except ImportError:
    print("❌ 에러: src/hand_tracking/pipeline.py에서 HandTrackingPipeline을 찾을 수 없습니다. 경로 확인 필수!")
    sys.exit(1)

# --- 유틸리티 함수 ---

def get_model_info(model: nn.Module, input_size=(1, 3, 256, 192)):
    """
    PyTorch 모델의 FLOPs와 Params를 계산합니다.
    """
    device = next(model.parameters()).device 
    dummy_input = torch.randn(input_size).to(device)
    
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    
    flops_g = macs / 1e9       
    params_m = params / 1e6    
    
    return params_m, flops_g

def save_result_to_csv(filename, precision, prune_rate, params, flops, avg_fps=0.0, avg_latency=0.0):
    # CSV 저장 로직 (이전과 동일)
    file_exists = os.path.isfile(filename)
    # ... (생략) ...
    # CSV 저장 로직 (이전과 동일)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Precision', 'Prune Rate', 'Params(M)', 'FLOPs(G)', 'FPS_App', 'Latency(ms)_App'])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            precision,
            prune_rate,
            f"{params:.3f}",
            f"{flops:.3f}",
            f"{avg_fps:.2f}",
            f"{avg_latency:.2f}"
        ])

def measure_performance(prune_rate):
    """
    특정 prune rate에서 모델을 로드하고 FLOPs 및 Params를 측정합니다.
    """
    print(f"\n--- [ Prune Rate: {prune_rate} 측정 중 ] ---")
    
    try:
        # FP32 모드로 초기화하여 PyTorch 모델 구조를 가져옵니다.
        # pruning 테스트를 위해서는 FP32 모드로 로드해야 합니다.
        pipeline = HandTrackingPipeline(precision='fp32', prune_rate=prune_rate)
        print("✅ Pipeline 객체 생성 성공 (FP32 PyTorch 모드)")
    except Exception as e:
        print(f"❌ Pipeline 로드 실패: {e}")
        return 0.0, 0.0 
        
    # 1. PyTorch 모델 객체 추출 (확정된 변수명 사용)
    #    self.hand_model.model은 YOLO 객체 내부의 PyTorch 모델입니다.
    model_object = pipeline.hand_model.model if hasattr(pipeline.hand_model, 'model') else None

    if not isinstance(model_object, nn.Module):
        print("❌ 경고: 로드된 객체가 PyTorch 모델(nn.Module)이 아닙니다. (TensorRT 엔진일 수 있음)")
        print("        => FLOPs/Params 계산을 건너뜁니다.")
        # fp32로 로드했는데도 nn.Module이 아니라면 파일 문제이므로 0 반환
        return 0.0, 0.0

    # 2. 모델 정보 측정
    try:
        model_object.to('cpu') # RAM 절약을 위해 CPU로 이동
        model_object.eval()
        
        params_m, flops_g = get_model_info(model_object) 
        
        # [CLEANUP 코드]: thop이 생성한 임시 속성을 제거하여 다음 반복에서 오류 방지
        for module in model_object.modules():
            for attr in ['total_ops', 'total_params', 'n_macs', 'n_params']:
                if hasattr(module, attr):
                    delattr(module, attr)
        
        print(f"✅ 측정 완료: {params_m:.3f}M Params, {flops_g:.3f}G FLOPs")
        
        # CSV 파일에 Params와 FLOPs 저장 
        save_result_to_csv(
            "experiment_results.csv",
            'fp32_Base',
            prune_rate,
            params_m,
            flops_g
        )

        return params_m, flops_g
    
    except Exception as e:
        print(f"❌ FLOPs 계산 중 오류 발생: {e}")
        return 0.0, 0.0

# --- 메인 실행 ---

if __name__ == "__main__":
    print("🖥️ FLOPs 및 파라미터 측정 모드 (CPU Only)")
    print("⚠️ 이 코드는 FP32(PyTorch) 모델의 이론적 복잡도만 측정합니다.")
    
    test_prune_rates = [0.0, 0.3, 0.5, 0.7] 
    
    # 이전에 실행한 결과를 삭제하고 시작합니다.
    if os.path.exists("experiment_results.csv"):
        os.remove("experiment_results.csv")
        
    for rate in test_prune_rates:
        p, f = measure_performance(rate)
    
    print("\n\n📊 [이론적 복잡도 측정 완료] - experiment_results.csv 파일을 확인하세요.")