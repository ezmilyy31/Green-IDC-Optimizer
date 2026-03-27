# lstm_model.py
"""
LSTM 기반 시계열 예측 모델 로직.

이 파일에서 구현할 것
- PyTorch LSTM 모델 정의
- sequence 데이터 학습 루프
- validation / loss 계산
- multi-step forecast inference
- 모델 저장/로드 인터페이스

이 파일의 책임
- 시계열 sequence 모델 정의와 학습
- 입력 shape / hidden state 처리
- horizon 단위 예측 로직 제공
"""

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1): ...
    def forward(self, x): ...

class LSTMTrainer:
    def train(self, model, train_loader, val_loader, epochs): ...
    def evaluate(self, model, data_loader): ...
    def predict(self, model, x_seq): ...
    def forecast_recursive(self, model, recent_sequence, horizon): ...