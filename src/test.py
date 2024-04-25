import numpy as np
import torch  # PyTorchライブラリのインポート
import torch.nn as nn  # PyTorchのニューラルネットワークモジュール
import torch.optim as optim  # PyTorchの最適化モジュール

# データセットをロードし、入力（X）と出力（y）に分割
dataset = np.loadtxt('C:\Users\加古匠\PycharmProjects\pythonProject1\datasets\pima-indians-diabetes_2.csv', delimiter=',')  # CSVからデータを読み込む
X = dataset[:, 0:8]  # 入力特徴量（最初の8列）
y = dataset[:, 8]  # 出力ラベル（最後の1列）

# PyTorchテンソルに変換し、浮動小数点データ型にキャスト
X = torch.tensor(X, dtype=torch.float32)  # Xをテンソルに変換
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # yをテンソルに変換し、形状を修正

# モデルの定義
model = nn.Sequential(  # 順次層のシーケンスとして定義
  nn.Linear(8, 12),  # 8入力、12出力の完全連結層
  nn.ReLU(),  # ReLU活性化関数
  nn.Linear(12, 8),  # 12入力、8出力の完全連結層
  nn.ReLU(),  # もう一度ReLU活性化関数
  nn.Linear(8, 1),  # 8入力、1出力の完全連結層
  nn.Sigmoid()  # シグモイド活性化関数
)
print(model)  # モデルの構造を表示

# モデルのトレーニング
loss_fn = nn.BCELoss()  # バイナリ交差エントロピー損失関数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam最適化手法と学習率0.001

n_epochs = 100  # エポック数
batch_size = 10  # バッチサイズ

# エポックごとのトレーニングループ
for epoch in range(n_epochs):  # エポックを繰り返す
  for i in range(0, len(X), batch_size):  # バッチごとにトレーニング
    Xbatch = X[i:i + batch_size]  # バッチの入力
    y_pred = model(Xbatch)  # モデルによる予測
    ybatch = y[i:i + batch_size]  # バッチの正解ラベル
    loss = loss_fn(y_pred, ybatch)  # 損失を計算
    optimizer.zero_grad()  # 勾配のリセット
    loss.backward()  # 勾配の計算
    optimizer.step()  # 重みの更新
  print(f'Finished epoch {epoch}, latest loss {loss}')  # エポック終了時の損失を表示

# 正確性の計算（no_gradで勾配を無効化）
with torch.no_grad():  # 勾配計算を無効化して予測
  y_pred = model(X)  # 全データに対して予測
accuracy = (y_pred.round() == y).float().mean()  # 正確性を計算
print(f"Accuracy {accuracy}")  # 正確性を表示

# モデルを使ってクラス予測を行う
predictions = (model(X) > 0.5).int()  # モデルの出力を使って予測し、0.5を閾値として整数に変換

# 最初の5つの予測を表示
for i in range(5):  # 最初の5つのデータポイントをループ
  print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))