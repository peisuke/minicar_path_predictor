# minicar_path_predictor

LiDAR データから走行方向を予測する ML ベースのナビゲーション ROS2 パッケージ。

コース画像から訓練データを生成し、ニューラルネットワークを学習させ、推論ノードでリアルタイムに走行制御を行う。

## アーキテクチャ

```
コース画像 (PNG)
    |
    v
データ生成 (data_generator.py)
    |  LiDAR シミュレーション + ターゲット角度/ウェイポイント
    v
訓練 (scripts/train_multitask_*.py)
    |  BEV CNN / 1D CNN / MLP
    v
モデル (.pt)
    |
    v
推論ノード (ml_nav_node / inference_node)
    |  LiDAR → モデル → PD制御 → Twist
    v
ロボット制御
```

## ノード

### ml_nav_node

BEV (Bird's Eye View) 画像に変換した LiDAR データから走行角度を分類予測し、PD 制御で走行する。

- **入力**: `/{ns}/scan` (sensor_msgs/LaserScan)
- **出力**: `/{ns}/diff_drive_controller/cmd_vel_unstamped` or `/{ns}/ackermann_steering_controller/reference_unstamped` (geometry_msgs/Twist)
- **モデル**: MultiTaskBEVCNN (走行可否 + 9クラス角度分類)

### inference_node

LiDAR からウェイポイント列を回帰予測し、Pure Pursuit で走行する。

- **入力**: `/{ns}/scan` (sensor_msgs/LaserScan)
- **出力**: Twist + `/predicted_path` (nav_msgs/Path)
- **モデル**: PathPredictorMLP / CNN / Transformer

## セットアップ

```bash
cd ~/ros2_ws
colcon build --packages-select minicar_path_predictor --symlink-install
source install/setup.bash
```

### 依存関係

- ROS2 (rclpy, sensor_msgs, geometry_msgs, nav_msgs)
- PyTorch
- NumPy, OpenCV, SciPy

## 使い方

### シミュレーションで実行

```bash
# ターミナル 1: Gazebo 起動
ros2 launch minicar_simulation road_env_minicar.launch.py seed:=42 gui:=true

# ターミナル 2 (10秒後): ML ナビゲーション起動
ros2 launch minicar_path_predictor ml_nav.launch.py \
  use_sim_time:=true \
  robot_type:=diff
```

### パラメータ調整 (ランタイム)

```bash
# 速度変更
ros2 param set /ml_nav_node common.target_velocity 0.5

# PD ゲイン変更
ros2 param set /ml_nav_node controllers.pd.kp_angular 3.0
```

## 訓練

### 1. データ生成

コース画像 (`data/images/course.png`) から LiDAR + ターゲット角度のデータを生成:

```bash
python3 scripts/generate_angle_data.py
```

### 2. モデル訓練

```bash
# BEV マルチタスクモデル (推奨)
python3 scripts/train_multitask_bev_model.py

# 1D CNN マルチタスクモデル
python3 scripts/train_multitask_model.py
```

訓練済みモデルは `data/models/` に保存される。

### 3. モデル切替

`data/models/angle_predictor.pt` のシンボリックリンクを張り替える:

```bash
cd data/models
ln -sf <model_dir>/model_final.pt angle_predictor.pt
```

## ディレクトリ構成

```
minicar_path_predictor/
  minicar_path_predictor/
    ml_nav_node.py          # 角度分類 + PD 制御ノード
    inference_node.py       # ウェイポイント回帰 + Pure Pursuit ノード
    model.py                # PathPredictor モデル定義
    data_generator.py       # コース画像からの訓練データ生成
    dataset.py              # PyTorch Dataset
    trainer.py              # 訓練ループ + 損失関数
    utils.py                # 座標変換、画像処理ユーティリティ
  config/
    ml_nav.yaml             # ml_nav_node パラメータ
    default.yaml            # データ生成・訓練設定
  launch/
    ml_nav.launch.py        # ML ナビゲーション起動
    path_predictor.launch.py # ウェイポイント予測ノード起動
  scripts/
    generate_angle_data.py          # データ生成
    train_multitask_bev_model.py    # BEV マルチタスク訓練
    train_multitask_model.py        # 1D マルチタスク訓練
    train_multitask_bev_regression.py # BEV 回帰訓練
    experiments/                    # 可視化・実験スクリプト
    archive/                        # 旧スクリプト
  data/
    images/course.png       # コース画像 (黒=壁, 赤=理想経路)
    models/                 # 訓練済みモデル
```

## パラメータ

### ml_nav.yaml

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `model_path` | `data/models/angle_predictor.pt` | モデルファイルパス |
| `common.target_velocity` | 3.0 m/s | 目標速度 |
| `common.max_angular_velocity` | 8.0 rad/s | 最大角速度 |
| `common.lookahead_distance` | 0.5 m | ルックアヘッド距離 |
| `controllers.pd.kp_angular` | 4.0 | 角速度 P ゲイン |
| `controllers.pd.kd_angular` | 0.5 | 角速度 D ゲイン |
| `controllers.pd.min_velocity` | 0.181 m/s | 最低速度 |

## 安全機構

- **緊急停止**: 前方 ±45° コーン内の 30% 以上が 20cm 以内に障害物 → 即停止
- **速度制御**: 旋回角度が大きいほど速度を低下
