import torch
import torch.nn as nn

# 크기가 (N, C)인 입력과 (N,)인 레이블을 예로 들겠습니다.
N, C = 5, 3
inputs = torch.randn(N, C, requires_grad=True)  # 예측 로짓
targets = torch.randint(0, C, (N,))  # 실제 클래스 레이블
print("inputs:", inputs)
print("targets:", targets)
# CrossEntropyLoss 인스턴스 생성
criterion = nn.CrossEntropyLoss()

# 손실 계산
loss = criterion(inputs, targets)

# 그라디언트 계산
loss.backward()

# 결과 출력
print("Loss:", loss.item())
