import torch

# 텐서 t를 생성합니다.
t = torch.randn(1, 256, 30, 28, 28)  # 예시 텐서

# torch.split을 사용하여 텐서를 분할합니다.
split_result = torch.split(t, 1, dim=2)

# 슬라이싱을 사용하여 텐서를 분할합니다.
slicing_result = [t[:, :, i:i+1, :, :] for i in range(t.shape[2])]

# 결과 비교
print("비교 결과:")
for split_tensor, slice_tensor in zip(split_result, slicing_result):
    # 두 텐서가 동일한지 확인
    if torch.equal(split_tensor, slice_tensor):
        print("일치")
    else:
        print("불일치")

# 두 방법의 결과 텐서 크기 출력
print("torch.split 결과 크기:", [x.size() for x in split_result])
print("슬라이싱 결과 크기:", [x.size() for x in slicing_result])