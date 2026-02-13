# Path Attraction ベース経路生成仕様

## 概要

minicar_navigationの`PathPipeline`ロジックをベースに、距離場の作り方のみを変更して
理想レーシングライン（赤線）に沿った経路を生成する。

## オリジナルとの差分

### 1. 距離場の変更

**オリジナル (wall distance)**
```python
dist_inside = 壁からの距離（壁から離れるほど値が大きい）
```

**変更後 (path_attraction)**
```python
# 理想経路からの距離場を取得
local_dist_to_path = cv2.remap(generator.path_distance_field, ...)

# path_attraction = max(wall_dist_max - dist_to_path, 0)
wall_dist_max = dist_inside_orig.max()
path_attraction = np.maximum(wall_dist_max - local_dist_to_path, 0)

# 可視領域のみに適用
dist_inside = path_attraction * (mask > 0).astype(np.float32)
```

- `dist_to_path`: 理想経路からの距離（理想経路上で0、離れるほど大きい）
- `path_attraction`: 理想経路に近いほど値が大きい（峰が理想経路上に立つ）

### 2. ルックアヘッド刈り込みの無効化

**理由**: path_attraction場では、ルックアヘッド地点が可視領域外(mask=0)に
落ちた場合`la_val=0`になり、正常なパスも誤って棄却される。

**変更箇所**: `GraphPathSearcher.build_and_deduplicate_paths`相当の処理で
`_trim_paths_by_lookahead`をスキップ

```python
# オリジナル
coordinate_paths = path_searcher.build_and_deduplicate_paths(...)

# 変更後
result_paths = path_searcher.build_valid_paths_coords(...)
deduplicated_paths_ids = path_searcher.deduplicator.deduplicate_paths(...)
# SKIP: _trim_paths_by_lookahead
coordinate_paths = path_searcher._paths_to_coords(deduplicated_paths_ids, ...)
```

**補足**: 経路は必ず参考経路上を通るため、壁方向チェックは不要。

## 3ステージパイプライン（変更なし）

| Stage | 処理 | 出力 |
|-------|------|------|
| 1 | Peak detection | `centered_points` (峰点座標) |
| 2 | Path building + Dedup | `coordinate_paths` (粗い経路) |
| 3 | Smoothing | `smoothed_paths` (滑らかな経路) |

## パラメータ（変更なし）

```python
local_size = 200          # ローカルマップサイズ (pixels)
resolution = 0.02         # m/pixel
R_in = [25, 50, 75, 100, 125, 150]  # リング半径 (pixels)
dist_thresh = 15.0        # 峰検出閾値
front_deg = 90.0          # 前方角度範囲
```

## スムージングの特性

- 始点: ロボット現在位置 (0, 0)
- 終点: coordinate_pathsの終点
- 補間: 3次ベジェ曲線 + target_pathとのブレンド
- 結果: coordinate_pathsより若干ショートカット気味の滑らかな曲線

## 座標系

```
ロボット座標系 (meters):
  X: 前方 (positive)
  Y: 左方 (positive)
  原点: ロボット位置

画像座標系 (pixels):
  X: 右方 (positive)
  Y: 下方 (positive)
  中心: (center_x, center_y) = (100, 100)

変換:
  x_robot = (px - center_x) * resolution
  y_robot = -(py - center_y) * resolution
```

## 検証スクリプト

- `scripts/visualize_path_attraction_stages.py`: 3ステージ可視化
- `scripts/compare_3stage.py`: オリジナルとの比較

## TODO

- [ ] `data_generator.py`への統合
- [ ] ML学習データ生成への適用
