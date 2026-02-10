#!/usr/bin/env python3
"""
ローカルパス計画システム

機能:
1. 地図読み込み（PNG画像から occupancy grid）
2. 初期位置設定（地図座標系）
3. LiDAR計測シミュレーション
4. ローカルパス生成（ロボット座標系で出力）
5. 可視化（描画時に地図座標系に変換）

座標系:
- 地図座標系（Map Coordinates）: 画像座標、左上原点、Y軸下向き
- ロボット座標系（Robot Coordinates）: ロボット中心、前方X正、左方向Y正
- 世界座標系（World Coordinates）: 地図中心、Y軸上向き（可視化用）
"""

import numpy as np
import cv2
import heapq
import inspect
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass


# =============================
# ユーティリティメソッド  
# =============================
def _safe_divide(numerator, denominator, default=0.0, threshold=None):
    """ゼロ除算を回避した安全な割り算"""
    if threshold is None:
        threshold = 1e-10
    if abs(denominator) < threshold:
        return default
    return numerator / denominator

def _safe_normalize(vector, default_vector=None, threshold=None):
    """ゼロベクトルを回避した安全な正規化"""
    if threshold is None:
        threshold = 1e-10
    norm = np.linalg.norm(vector)
    if norm < threshold:
        if default_vector is not None:
            return default_vector
        if len(vector) == 2:
            return np.array([1.0, 0.0])
        elif len(vector) == 3:
            return np.array([1.0, 0.0, 0.0])
        else:
            return np.ones_like(vector) / np.sqrt(len(vector))
    return vector / norm

def _validate_array_bounds(x, y, height, width):
    """配列の境界を検証"""
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid shape dimensions: {height}x{width}")
    
    x_safe = np.clip(x, 0, width - 1)
    y_safe = np.clip(y, 0, height - 1)
    
    return x_safe, y_safe


def _validate_numpy_array(arr, name="array", min_dims=1, max_dims=None, 
                         shape=None, dtype=None, allow_empty=False):
    """NumPy配列の検証"""
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be numpy.ndarray, got {type(arr)}")
    
    if arr.size == 0 and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    
    if arr.ndim < min_dims:
        raise ValueError(f"{name} must have at least {min_dims} dimensions, got {arr.ndim}")
    
    if max_dims is not None and arr.ndim > max_dims:
        raise ValueError(f"{name} must have at most {max_dims} dimensions, got {arr.ndim}")
    
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {arr.shape}")
    
    if dtype is not None and arr.dtype != dtype:
        raise ValueError(f"{name} dtype mismatch: expected {dtype}, got {arr.dtype}")

def _validate_positive_number(value, name="value", allow_zero=False):
    """正の数値検証"""
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a number, got {type(value)}")
    
    if allow_zero and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    elif not allow_zero and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

def _validate_range(value, min_val=None, max_val=None, name="value"):
    """値の範囲検証"""
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")


def validate_inputs(validator_func):
    """デコレータ: 入力パラメータの検証を分離（単一検証関数版）"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 引数名と値のマッピング
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # self を除いた引数を取得
            method_args = list(bound.arguments.values())[1:]  # self を除く
            
            # 検証実行
            try:
                validator_func(*method_args)
            except Exception as e:
                raise ValueError(f"Validation failed for {func.__name__}: {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_multi_inputs(**validators):
    """デコレータ: 複数パラメータの個別検証"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 引数名と値のマッピング
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # 各パラメータの検証
            for param_name, validator_func in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    try:
                        validator_func(value)
                    except Exception as e:
                        raise ValueError(f"Parameter '{param_name}' validation failed: {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_distance_field_params(params):
    """DistanceFieldParams専用の検証"""
    if params is None:
        raise ValueError("params cannot be None")
    _validate_positive_number(params.H, "params.H")
    _validate_positive_number(params.W, "params.W")
    _validate_numpy_array(params.R_in, "params.R_in", min_dims=1, max_dims=1, allow_empty=False)
    _validate_positive_number(params.ring_thickness, "params.ring_thickness")
    _validate_range(params.front_deg, min_val=0, max_val=360, name="params.front_deg")
    _validate_positive_number(params.dist_thresh, "params.dist_thresh", allow_zero=True)
    _validate_positive_number(params.border_thickness, "params.border_thickness")

def validate_view_points(view_points):
    """LiDARビューポイントの検証"""
    _validate_numpy_array(view_points, "view_points", min_dims=2, max_dims=2)
    if view_points.shape[1] != 2:
        raise ValueError(f"view_points must have shape (N, 2), got {view_points.shape}")

def validate_grid_dimensions(H, W):
    """グリッド寸法の検証"""
    _validate_positive_number(H, "H")
    _validate_positive_number(W, "W")

def validate_ring_params(radius_squared, R_in, thickness):
    """リングパラメータの検証"""
    _validate_numpy_array(radius_squared, "radius_squared", min_dims=2, max_dims=2)
    _validate_numpy_array(R_in, "R_in", min_dims=1, max_dims=1, allow_empty=False)
    _validate_positive_number(thickness, "thickness")

def validate_front_mask_params(delta_x, delta_y, deg):
    """前方マスクパラメータの検証"""
    _validate_numpy_array(delta_x, "delta_x")
    _validate_numpy_array(delta_y, "delta_y")
    _validate_range(deg, min_val=0, max_val=360, name="deg")

def validate_pixel_selection_params(dist_inside, ring_mask, front_mask, thresh):
    """ピクセル選択パラメータの検証"""
    _validate_numpy_array(dist_inside, "dist_inside", min_dims=2, max_dims=2)
    _validate_numpy_array(ring_mask, "ring_mask")
    _validate_numpy_array(front_mask, "front_mask")
    _validate_positive_number(thresh, "thresh", allow_zero=True)

def validate_region_connection_params(selected_pixels, kernel_sz):
    """領域結合処理パラメータの検証"""
    _validate_numpy_array(selected_pixels, "selected_pixels", min_dims=2, max_dims=2)
    _validate_positive_number(kernel_sz, "kernel_sz")



# =============================
# 設定
# ============================
@dataclass(frozen=True)
class PeakParams:
    """ピーク検出パラメータ"""
    min_area: int = 10
    kernel_sz: int = 3
    smooth: bool = True
    peak_offs: Tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class DistanceFieldParams:
    """距離場生成パラメータ"""
    H: int = 480
    W: int = 640
    R_in: np.ndarray = None
    ring_thickness: int = 1
    front_deg: float = 90.0
    dist_thresh: float = 15.0
    border_thickness: int = 1
    
    def __post_init__(self):
        if self.R_in is None:
            object.__setattr__(self, 'R_in', np.array([25, 50, 75, 100, 125, 150], dtype=np.int32))

@dataclass(frozen=True)
class PeakDetectorParams:
    """ピーク検出処理の統合パラメータ（距離場生成含む）"""
    # 距離場生成パラメータ
    H: int = 480
    W: int = 640
    R_in: np.ndarray = None
    ring_thickness: int = 1
    front_deg: float = 90.0
    dist_thresh: float = 15.0
    border_thickness: int = 1
    
    # ピーク検出パラメータ
    min_area: int = 10
    kernel_sz: int = 3
    smooth: bool = True
    peak_offs: Tuple[int, ...] = (1, 3, 5)
    
    def __post_init__(self):
        if self.R_in is None:
            object.__setattr__(self, 'R_in', np.array([25, 50, 75, 100, 125, 150], dtype=np.int32))
    
    def to_distance_field_params(self) -> DistanceFieldParams:
        """DistanceFieldParamsに変換"""
        return DistanceFieldParams(
            H=self.H, W=self.W, R_in=self.R_in,
            ring_thickness=self.ring_thickness, front_deg=self.front_deg,
            dist_thresh=self.dist_thresh, border_thickness=self.border_thickness
        )
    

@dataclass(frozen=True) 
class PathPlannerConfig:
    """ローカルパスプランナーの統合設定"""
    # 画像サイズ
    HEIGHT: int = 480
    WIDTH: int = 640
    
    # 地図解像度
    MAP_RESOLUTION: float = 0.015  # meter/pixel (実環境用: 0.005の3倍)
    
    # リングパラメータ
    RING_RADII: Tuple[int, ...] = (25, 50, 75, 100, 125, 150)
    RING_THICKNESS: int = 1
    
    # 角度パラメータ（度）
    FRONT_DEG: float = 90.0          # ピーク検出用の前方角度
    PATH_FRONT_DEG: float = 90.0     # パス構築用の前方角度
    
    # 距離パラメータ
    DIST_THRESH: float = 5.0         # 壁からの最小安全距離(px) - 狭い道対応で15.0→5.0
    BORDER_THICKNESS: int = 1
    LOCAL_DIST_THRESH: float = 4.0   # ローカル計画の距離閾値(px) - 狭い道対応で10.0→4.0
    MAX_GRADIENT_DROP: float = 24.0  # エッジの距離場勾配の最大許容下降量(px) - 前方パス安定性のため緩和
    LOOKAHEAD_ALPHA: float = 1.0     # パス末端ルックアヘッド係数（末端エッジ長の何倍先を見るか）
    MIN_PATH_NODES: int = 3          # ルックアヘッド刈り込み後の最小ノード数（未満のパスは棄却）
    MIN_PEAK_DIST_VALUE: float = 15.0  # ピーク検出時の最低距離場値(0-255) - 狭い道対応で30.0→15.0

    # パス生成パラメータ
    START_IDS: Tuple[int, ...] = (0,)

    # 方向バイアス（度）: 正=反時計回り（左寄り）、負=時計回り（右寄り）
    # 例: -30.0 → 右に30度傾いた方向を「直進」として扱う
    DIRECTION_BIAS_DEG: float = 0.0

    # Belief-based path selection パラメータ
    BELIEF_EMA_ALPHA: float = 0.3             # 信念更新の指数移動平均係数 (0=更新なし, 1=即時置換)
    BELIEF_CONFIDENCE_LENGTH_SCALE: float = 2.0  # 信頼度算出のパス長スケール(m)
    BELIEF_MIN_CONFIDENCE: float = 0.1         # 最低信頼度閾値（未満のパスは選択候補外）
    BELIEF_CONSISTENCY_WEIGHT: float = 2.0     # スコア中の一貫性重み（加算式）

    # パス平滑化パラメータ
    CONTROL_STRENGTH_FACTOR: float = 3.0  # エルミート補間の制御点強度係数
    AUTO_WEIGHT_DISTANCE_FACTOR: float = 15.0  # 自動ブレンドの距離重み係数
    
    # 画像処理パラメータ
    CONNECTIVITY: int = 8
    DIST_MASK_SIZE: int = 5
    ROI_MARGIN: int = 10
    
    # ラベリングパラメータ
    LABEL_BIN_SIZE: float = 5.0
    
    # 補間パラメータ
    INTERPOLATION_POINTS: int = 15


# =============================
# 距離場生成クラス
# =============================
class DistanceFieldGenerator:
    """距離場生成処理クラス
    
    LiDARデータから障害物までの距離場を生成し、
    パス計画に適した領域を選定するクラス。
    """
    
    def __init__(self, config: PathPlannerConfig = None):
        """
        Args:
            config: パスプランナー設定
        """
        self.config = config or PathPlannerConfig()
        self._center_grid_cache = {}
        self._ring_cache = {}
    
    @validate_multi_inputs(
        view_points=validate_view_points,
        params=validate_distance_field_params
    )
    def generate_distance_field(self, view_points: np.ndarray, params: DistanceFieldParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
        """距離場生成と適切な領域選定（メインエントリポイント）
        
        Args:
            view_points: LiDARビューポイント
            params: 距離場生成パラメータ
            
        Returns:
            mask, dist_inside, dist_norm, center_x, center_y, path_candidate_pixels
        """
        # 1. ポリゴン距離場生成
        mask, dist_inside, dist_norm = self._create_polygon_distance_inside(
            view_points, params.H, params.W, params.border_thickness
        )
        
        # 2. 中心グリッド計算（ロボット中心からの距離）
        center_x, center_y, delta_x, delta_y, radius_squared = self._compute_center_grid(params.H, params.W)
        
        # 3. リングマスクと前方マスク生成（探索範囲限定）
        ring_mask = self._create_ring_mask(radius_squared, params.R_in, thickness=params.ring_thickness)
        front_mask = self._create_front_mask(delta_x, delta_y, params.front_deg)
        
        # 4. パス計画に適したピクセル選定
        path_candidate_pixels = self._select_pixels(dist_inside, ring_mask, front_mask, params.dist_thresh)
        
        return mask, dist_inside, dist_norm, center_x, center_y, path_candidate_pixels
    
    def _create_polygon_distance_inside(self, view_points: np.ndarray, H: int, W: int, 
                                     border_thickness: int = 1, dist_mask_size: int = None, margin: int = None):
        """ポリゴン距離変換処理のROI最適化版"""
        # デフォルト値の設定
        if dist_mask_size is None:
            dist_mask_size = self.config.DIST_MASK_SIZE
        if margin is None:
            margin = self.config.ROI_MARGIN
            
        # 1. ROI（関心領域）の計算
        roi_bounds = self._calculate_roi_bounds(view_points, H, W, margin)
        if roi_bounds is None:
            return self._create_empty_results(H, W)
        
        x0, y0, x1, y1, roi_width, roi_height = roi_bounds
        
        # 2. ポリゴンマスクの生成
        polygon_mask = self._create_polygon_mask(view_points, x0, y0, roi_height, roi_width, border_thickness)
        
        # 3. 距離変換と正規化
        distance_map, normalized_distance = self._compute_distance_transform(polygon_mask, dist_mask_size)
        
        # 4. 結果をフル画像サイズに展開
        return self._expand_to_full_size(polygon_mask, distance_map, normalized_distance, 
                                       H, W, x0, y0, x1, y1)
    
    @validate_inputs(validate_grid_dimensions)
    def _compute_center_grid(self, H: int, W: int):
        """中心グリッド計算（キャッシュ付き）"""
        key = (H, W)
        if key in self._center_grid_cache:
            return self._center_grid_cache[key]

        center_x, center_y = W // 2, H // 2
        yy, xx = np.ogrid[:H, :W]
        delta_x = xx - center_x
        delta_y = yy - center_y
        radius_squared = delta_x * delta_x + delta_y * delta_y
        
        result = (center_x, center_y, delta_x, delta_y, radius_squared)
        self._center_grid_cache[key] = result
        return result
    
    @validate_inputs(validate_ring_params)
    def _create_ring_mask(self, radius_squared, R_in, thickness=1):
        """リングマスク生成（キャッシュ付き）"""
        R_in_tuple = tuple(np.asarray(R_in, dtype=np.int32).tolist())
        key = (R_in_tuple, int(thickness), radius_squared.shape)
        
        if key in self._ring_cache:
            return self._ring_cache[key]
            
        result = self._ring_mask_fast(radius_squared, R_in, thickness)
        self._ring_cache[key] = result
        return result
    
    def _ring_mask_fast(self, radius_squared: np.ndarray, R_in: np.ndarray, thickness: int = 1) -> np.ndarray:
        """高速リングマスク実装"""
        R_in = np.asarray(R_in, dtype=np.int32)
        t = int(thickness)
        output = np.zeros(radius_squared.shape, dtype=bool)
        for Rin in R_in:
            rin2 = Rin * Rin
            Rout = Rin + t
            rout2 = Rout * Rout
            output |= (radius_squared >= rin2) & (radius_squared <= rout2)
        return output
    
    @validate_inputs(validate_front_mask_params)
    def _create_front_mask(self, delta_x: np.ndarray, delta_y: np.ndarray, deg: float) -> np.ndarray:
        """前方領域マスク生成"""
        if deg >= 90:
            return (delta_x >= 0)
        t = np.tan(np.deg2rad(deg))
        return (delta_x > 0) & (np.abs(delta_y) <= t * delta_x)
    
    @validate_inputs(validate_pixel_selection_params)
    def _select_pixels(self, dist_inside: np.ndarray, ring_mask: np.ndarray, 
                     front_mask: np.ndarray, thresh: float) -> np.ndarray:
        """最終ピクセル選択"""
        return ring_mask & front_mask & (dist_inside >= thresh)
    
    @validate_inputs(validate_region_connection_params)
    def connect_regions_and_label(self, selected_pixels: np.ndarray, kernel_sz: int):
        """領域結合とラベリング
        
        パス候補ピクセルの小さな際間を埋めて連結し、
        意味のある領域としてラベリングする。
        """
        selected_uint8 = (selected_pixels.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz))
        selected_uint8 = cv2.morphologyEx(selected_uint8, cv2.MORPH_CLOSE, kernel)
        num, labels = cv2.connectedComponents(selected_uint8, connectivity=self.config.CONNECTIVITY)
        return num, labels, selected_uint8
    
    # Helper methods (private)
    def _calculate_roi_bounds(self, view_points: np.ndarray, H: int, W: int, margin: int):
        """ROI境界の計算"""
        x_coords = view_points[:, 0]
        y_coords = view_points[:, 1]

        x0 = max(int(x_coords.min()) - margin, 0)
        y0 = max(int(y_coords.min()) - margin, 0)
        x1 = min(int(x_coords.max()) + margin + 1, W)
        y1 = min(int(y_coords.max()) + margin + 1, H)

        roi_width = x1 - x0
        roi_height = y1 - y0
        
        if roi_width <= 0 or roi_height <= 0:
            return None
            
        return x0, y0, x1, y1, roi_width, roi_height
    
    def _create_empty_results(self, H: int, W: int):
        """空の結果を作成"""
        mask = np.zeros((H, W), np.uint8)
        dist_inside = np.zeros((H, W), np.float32)
        dist_norm = np.zeros((H, W), np.uint8)
        return mask, dist_inside, dist_norm
    
    def _create_polygon_mask(self, view_points: np.ndarray, x0: int, y0: int, 
                           roi_height: int, roi_width: int, border_thickness: int):
        """ポリゴンマスクの生成"""
        roi_view_points = view_points.copy()
        roi_view_points[:, 0] -= x0
        roi_view_points[:, 1] -= y0

        mask_roi = np.zeros((roi_height, roi_width), np.uint8)
        polygon = [roi_view_points.reshape(-1, 1, 2)]
        cv2.fillPoly(mask_roi, polygon, 255)
        cv2.polylines(mask_roi, polygon, isClosed=True, color=0, thickness=border_thickness)
        
        return mask_roi
    
    def _compute_distance_transform(self, polygon_mask: np.ndarray, dist_mask_size: int):
        """距離変換と正規化"""
        distance_map = cv2.distanceTransform(polygon_mask, cv2.DIST_L2, dist_mask_size)
        distance_map[polygon_mask == 0] = 0
        
        normalized_distance = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return distance_map, normalized_distance
    
    def _expand_to_full_size(self, polygon_mask: np.ndarray, distance_map: np.ndarray, 
                           normalized_distance: np.ndarray, H: int, W: int, 
                           x0: int, y0: int, x1: int, y1: int):
        """結果をフル画像サイズに展開"""
        mask = np.zeros((H, W), np.uint8)
        dist_inside = np.zeros((H, W), np.float32)
        dist_norm = np.zeros((H, W), np.uint8)

        mask[y0:y1, x0:x1] = polygon_mask
        dist_inside[y0:y1, x0:x1] = distance_map
        dist_norm[y0:y1, x0:x1] = normalized_distance

        return mask, dist_inside, dist_norm


# =============================
# ピーク検出クラス
# =============================
class PeakDetector:
    """ピーク検出・候補点抽出処理クラス"""
    
    def __init__(self, params: PeakParams = None, config: PathPlannerConfig = None):
        """
        Args:
            params: ピーク検出パラメータ
            config: パスプランナー設定
        """
        self.params = params or PeakParams()
        self.config = config or PathPlannerConfig()
        self.distance_field_generator = DistanceFieldGenerator(config)
    
    def detect_from_view_points(self, view_points: np.ndarray, 
                              peak_detector_params: PeakDetectorParams) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """view_pointsから距離場生成とピーク検出を統合実行（完全独立メソッド）"""
        # 1. 距離場生成（内包されたgenerator使用）
        distance_field_params = peak_detector_params.to_distance_field_params()
        mask, dist_inside, dist_norm, center_x, center_y, path_candidate_pixels = \
            self.distance_field_generator.generate_distance_field(view_points, distance_field_params)
        
        # 2. 候補ピクセルを連結領域としてラベリング
        num, labels, _ = self.distance_field_generator.connect_regions_and_label(
            path_candidate_pixels, peak_detector_params.kernel_sz
        )
        
        # 3. ラベル付き領域からピーク検出と候補点抽出
        peaks, centered_points, sorted_labels = self._detect_path_candidates(labels, dist_norm, center_x, center_y)
        
        return peaks, centered_points, sorted_labels, dist_inside, dist_norm, center_x, center_y
    def _detect_path_candidates(self, labels: np.ndarray, dist_norm: np.ndarray, 
                              center_x: int, center_y: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """ピーク検出と候補点抽出"""
        y_coords, x_coords, component_labels, values = self._filter_components_by_area(
            labels, dist_norm, self.params.min_area
        )
        
        peaks = self._cluster_peaks(
            y_coords, x_coords, component_labels, values, center_x, center_y, self.params
        )
        centered_points, peak_labels = self._peaks_to_centered_points(peaks, center_x, center_y)
        sorted_labels = self._sort_labels(centered_points, bin_size=self.config.LABEL_BIN_SIZE)
        
        return peaks, centered_points, sorted_labels
    
    def _filter_components_by_area(self, labels: np.ndarray, dist_norm: np.ndarray, min_area: int):
        """labels から全画素を取り、面積でフィルタして返す。"""
        lab_f = labels.ravel()
        pos = np.flatnonzero(lab_f)
        
        labs = lab_f[pos].astype(np.int32)
        
        num = int(labs.max()) + 1 if labs.size else (labels.max() + 1)
        area = np.bincount(labs, minlength=num)
        
        valid = (area >= min_area)
        if valid.size:
            valid[0] = False
        
        m = valid[labs]
        pos = pos[m]
        labs = labs[m]
        
        W = labels.shape[1]
        ys = (pos // W).astype(np.int32)
        xs = (pos - ys * W).astype(np.int32)
        
        vals = dist_norm.ravel()[pos].astype(np.int32)
        return ys, xs, labs, vals
    
    def _detect_peaks_1d(self, v: np.ndarray, offs: np.ndarray) -> np.ndarray:
        """NumPy ベクトル化ピーク判定"""
        n = len(v)
        if n < 3:
            return np.empty((0,), dtype=np.int64)

        idxs = np.arange(n)[:, None]
        O = offs[None, :]

        iL = idxs - O
        iR = idxs + O

        L = iL >= 0
        R = iR < n

        iL_c = np.clip(iL, 0, n - 1)
        iR_c = np.clip(iR, 0, n - 1)

        vC = v[:, None]
        vL = v[iL_c]
        vR = v[iR_c]

        cond_both = (~L | (vC >= vL)) & (~R | (vC > vR))
        cond_left = (~L) | (vC > vL)
        cond_right = (~R) | (vC > vR)

        ok_each = np.where(
            L & R,
            cond_both,
            np.where(L, cond_left,
                     np.where(R, cond_right, True))
        )

        ok = ok_each.all(axis=1)
        return np.where(ok)[0]
    
    def _cluster_peaks(self, ys: np.ndarray, xs: np.ndarray, labs: np.ndarray, 
                      vals: np.ndarray, cx: int, cy: int, params: PeakParams):
        """クラスタごとに角度でソート → 平滑化 → ピーク検出"""
        peaks = []
        offs = np.array(params.peak_offs, dtype=int)

        for lab in np.unique(labs):
            idx = (labs == lab)

            xs_l = xs[idx]
            ys_l = ys[idx]
            vals_l = vals[idx].astype(float)

            dx = xs_l - cx
            dy = ys_l - cy
            theta = np.arctan2(dy, dx)

            order = np.argsort(theta)
            xs_l = xs_l[order]
            ys_l = ys_l[order]
            vals_l = vals_l[order]

            if params.smooth and len(vals_l) >= 3:
                v = np.convolve(vals_l, [1, 2, 1], mode="same") / 4.0
            else:
                v = vals_l

            peak_i = self._detect_peaks_1d(v.astype(float), offs)

            for i in peak_i:
                val = float(vals_l[i])
                if val >= self.config.MIN_PEAK_DIST_VALUE:
                    peaks.append((int(xs_l[i]), int(ys_l[i]), val, int(lab)))

        peaks.sort(key=lambda t: t[2], reverse=True)
        return peaks
    
    def _peaks_to_centered_points(self, peaks, cx: int, cy: int):
        """peaks [(x,y,dist,label), ...] -> (N,2) centered coords"""
        if len(peaks) == 0:
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32)

        pts = np.array(
            [[x - cx, y - cy] for x, y, _, _ in peaks],
            dtype=np.int32
        )
        labels = [lbl for _, _, _, lbl in peaks]
        return pts, labels
    
    def _sort_labels(self, points: np.ndarray, bin_size: float = None):
        """points: (N,2) center基準の座標"""
        if bin_size is None:
            bin_size = self.config.LABEL_BIN_SIZE
            
        points = np.asarray(points)
        if points.size == 0:
            return np.empty((0,), dtype=np.int32)

        r = np.sqrt(np.sum(points * points, axis=1))
        bins = np.floor(r / bin_size) * bin_size
        _, labels = np.unique(bins, return_inverse=True)
        return labels.astype(np.int32)


# =============================
# パス構築クラス
# =============================
class GraphPathSearcher:
    """グラフパス探索クラス
    
    候補点間の有効なパスをグラフ構造で探索し、
    障害物回避と接続性を検証する。
    """
    
    def __init__(self, config: PathPlannerConfig = None):
        """
        Args:
            config: パスプランナー設定
        """
        self.config = config or PathPlannerConfig()
        self.deduplicator = PathDeduplicator(self.config)
    
    def build_and_deduplicate_paths(self, labels: np.ndarray, centered_points: np.ndarray,
                                  dist_inside: np.ndarray, center_x: int, center_y: int,
                                  dist_thresh: float, path_front_deg: float, start_ids: Tuple[int, ...]) -> List[np.ndarray]:
        """パス構築と重複除去を統合実行（完全独立メソッド）"""
        # 1. パス構築
        result_paths = self.build_valid_paths_coords(
            labels=labels,
            centered_pts=centered_points,
            dist_map=dist_inside,
            cx=center_x,
            cy=center_y,
            thr=dist_thresh,
            front_deg=path_front_deg,
            start_ids=list(start_ids),
        )
        
        # 2. パス重複除去（内包されたdeduplicator使用）
        paths_ids = result_paths["paths_ids"]
        deduplicated_paths_ids = self.deduplicator.deduplicate_paths(
            paths_ids, centered_points, result_paths["flat_to_pts"]
        )

        # 3. パス末端ルックアヘッド刈り込み
        trimmed_paths_ids = self._trim_paths_by_lookahead(
            deduplicated_paths_ids, centered_points, result_paths["flat_to_pts"],
            dist_inside, center_x, center_y, dist_thresh
        )

        # 4. ID→座標変換
        deduplicated_paths_coords = self._paths_to_coords(
            trimmed_paths_ids, centered_points, result_paths["flat_to_pts"]
        )

        return deduplicated_paths_coords
    
    def _trim_paths_by_lookahead(self, paths_ids: List[List[int]], centered_pts: np.ndarray,
                                flat_to_pts: np.ndarray, dist_map: np.ndarray,
                                cx: int, cy: int, thr: float) -> List[List[int]]:
        """パス末端からルックアヘッドで壁方向ノードを刈り込む

        A->B->C->Dのパスで、C->Dの延長先が壁ならDを削除。
        削除後のA->B->Cでも、B->Cの延長先が壁ならCを削除。
        最終的にMIN_PATH_NODES未満になったパスは棄却する。
        """
        if not paths_ids:
            return []

        H, W = dist_map.shape[:2]
        alpha = self.config.LOOKAHEAD_ALPHA
        max_drop = self.config.MAX_GRADIENT_DROP
        min_nodes = self.config.MIN_PATH_NODES
        trimmed = []

        for path in paths_ids:
            path = list(path)

            while len(path) >= 2:
                # 末端エッジ: path[-2] -> path[-1]
                src_idx = flat_to_pts[path[-2]]
                dst_idx = flat_to_pts[path[-1]]
                src_pt = centered_pts[src_idx].astype(float)
                dst_pt = centered_pts[dst_idx].astype(float)
                edge_vec = dst_pt - src_pt

                # ルックアヘッド地点
                la = dst_pt + alpha * edge_vec
                la_x = np.clip(int(np.rint(la[0])) + cx, 0, W - 1)
                la_y = np.clip(int(np.rint(la[1])) + cy, 0, H - 1)
                la_val = dist_map[la_y, la_x]

                # dst地点の距離値
                dst_x = np.clip(int(np.rint(dst_pt[0])) + cx, 0, W - 1)
                dst_y = np.clip(int(np.rint(dst_pt[1])) + cy, 0, H - 1)
                dst_val = dist_map[dst_y, dst_x]

                # 勾配チェック: 延長先で距離値が大きく下がるなら末端を削除
                gradient = la_val - dst_val
                if gradient >= -max_drop:
                    break  # 壁に向かっていない→刈り込み終了
                path.pop()

            if len(path) >= min_nodes:
                trimmed.append(path)

        return trimmed

    def _paths_to_coords(self, paths_ids: List[List[int]], centered_pts: np.ndarray, flat_to_pts: np.ndarray):
        """パスIDを座標に変換するプライベートメソッド"""
        coordinate_paths = []
        for path_ids in paths_ids:
            if len(path_ids) == 0:
                continue
            original_indices = flat_to_pts[path_ids]
            coords = centered_pts[original_indices]
            coordinate_paths.append(coords)
        return coordinate_paths
    
    def _build_layer_indices(self, labels: np.ndarray):
        """ラベルから層インデックスを構築"""
        if labels.size == 0:
            return []
        L = int(labels.max()) + 1
        order = np.argsort(labels, kind="mergesort")
        labels_sorted = labels[order]
        starts = np.searchsorted(labels_sorted, np.arange(L), side="left")
        ends = np.searchsorted(labels_sorted, np.arange(L), side="right")
        layer_idx = [order[starts[i]:ends[i]] for i in range(L)]
        return layer_idx
    
    def _build_offsets(self, layer_idx):
        """層インデックスからオフセット計算"""
        if len(layer_idx) == 0:
            return np.array([0], dtype=np.int32), np.array([], dtype=np.int32)
        counts = np.array([len(ix) for ix in layer_idx], dtype=np.int32)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
        return offsets, counts
    
    def _set_edges_fast(self, offsets, counts):
        """高速エッジ設定"""
        src_all = []
        dst_all = []
        L = len(counts)
        if L <= 1:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        for i in range(L - 1):
            m = int(counts[i])
            n = int(counts[i + 1])
            if m == 0 or n == 0:
                continue
            src = offsets[i] + np.repeat(np.arange(m, dtype=np.int32), n)
            dst = offsets[i + 1] + np.tile(np.arange(n, dtype=np.int32), m)
            src_all.append(src)
            dst_all.append(dst)
        if len(src_all) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        return np.concatenate(src_all), np.concatenate(dst_all)
    
    def _set_valid_edges_midpoint_nearest(self, labels, pts, dist_map, cx, cy, thr: float, front_deg: float = 90):
        """エッジの中点で有効性判定
        
        ピーク点間を接続するエッジを生成し、各エッジが走行可能かを判定:
        - エッジの中点が障害物領域を通らないかチェック
        - 距離閾値（thr）以内のエッジのみを有効とする
        - 前方角度制限（front_deg）内のエッジのみを有効とする
        """
        if labels.size == 0:
            return self._empty_edge_result()
        
        # ピーク点をレイヤー別に整理し、グラフ構造（ノード・エッジ）を構築
        graph_data = self._build_graph_structure(labels, pts)
        
        # 各エッジが有効（障害物なし、距離・角度制約内）かを判定
        edge_validity = self._validate_edges(graph_data, dist_map, cx, cy, thr, front_deg)
        
        # 有効なエッジのみを抽出してグラフ情報を返す
        return self._apply_edge_filters(graph_data, edge_validity)
    
    def _empty_edge_result(self):
        """空のエッジ結果を返す"""
        return (np.array([], dtype=np.int32), 
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([], dtype=np.int32))
    
    def _build_graph_structure(self, labels, pts):
        """グラフ構造を構築"""
        layer_idx = self._build_layer_indices(labels)
        offsets, counts = self._build_offsets(layer_idx)
        
        flat_to_pts = np.concatenate(layer_idx)
        flat_pts = pts[flat_to_pts]
        src_id, dst_id = self._set_edges_fast(offsets, counts)
        
        return {
            'flat_to_pts': flat_to_pts,
            'flat_pts': flat_pts,
            'src_id': src_id,
            'dst_id': dst_id,
            'offsets': offsets,
            'counts': counts
        }
    
    def _validate_edges(self, graph_data, dist_map, cx, cy, thr, front_deg):
        """エッジの有効性を検証"""
        src_id, dst_id, flat_pts = graph_data['src_id'], graph_data['dst_id'], graph_data['flat_pts']

        # 距離による検証
        dist_mask = self._validate_edge_distances(flat_pts, src_id, dst_id, dist_map, cx, cy, thr)

        # 方向による検証
        direction_mask = self._validate_edge_directions(flat_pts, src_id, dst_id, front_deg)

        return (dist_mask & direction_mask)
    
    def _validate_edge_distances(self, flat_pts, src_id, dst_id, dist_map, cx, cy, thr):
        """エッジの中点距離および勾配による検証

        検証条件:
        1. エッジ中点の距離場値 >= 閾値（壁から十分離れている）
        2. 距離場の勾配（終点-始点）>= -MAX_GRADIENT_DROP（壁に向かって急降下しない）
        """
        if dist_map.size == 0:
            raise ValueError("Distance map is empty")

        H, W = dist_map.shape[:2]

        # 中点の距離値チェック
        mid = (flat_pts[src_id] + flat_pts[dst_id]) * 0.5
        mid_x = np.rint(mid[:, 0]).astype(np.int32)
        mid_y = np.rint(mid[:, 1]).astype(np.int32)
        mid_x_safe, mid_y_safe = _validate_array_bounds(mid_x + cx, mid_y + cy, H, W)
        mid_vals = dist_map[mid_y_safe, mid_x_safe]
        midpoint_mask = mid_vals >= thr

        # 始点と終点の距離値を取得
        src_pts = flat_pts[src_id]
        dst_pts = flat_pts[dst_id]

        src_x = np.rint(src_pts[:, 0]).astype(np.int32)
        src_y = np.rint(src_pts[:, 1]).astype(np.int32)
        src_x_safe, src_y_safe = _validate_array_bounds(src_x + cx, src_y + cy, H, W)
        src_vals = dist_map[src_y_safe, src_x_safe]

        dst_x = np.rint(dst_pts[:, 0]).astype(np.int32)
        dst_y = np.rint(dst_pts[:, 1]).astype(np.int32)
        dst_x_safe, dst_y_safe = _validate_array_bounds(dst_x + cx, dst_y + cy, H, W)
        dst_vals = dist_map[dst_y_safe, dst_x_safe]

        # 勾配チェック: 終点 - 始点 >= -MAX_GRADIENT_DROP
        gradient = dst_vals - src_vals
        gradient_mask = gradient >= -self.config.MAX_GRADIENT_DROP

        return midpoint_mask & gradient_mask
    
    def _validate_edge_directions(self, flat_pts, src_id, dst_id, front_deg):
        """エッジの方向による検証"""
        src_pts = flat_pts[src_id]
        dst_pts = flat_pts[dst_id]
        direction_vec = dst_pts - src_pts
        
        if front_deg >= 90:
            return direction_vec[:, 0] >= 0
        else:
            t = np.tan(np.deg2rad(front_deg))
            return (direction_vec[:, 0] > 0) & (np.abs(direction_vec[:, 1]) <= t * direction_vec[:, 0])
    
    def _apply_edge_filters(self, graph_data, edge_validity):
        """有効なエッジのみを適用"""
        src_id = graph_data['src_id'][edge_validity]
        dst_id = graph_data['dst_id'][edge_validity]
        flat_to_pts = graph_data['flat_to_pts']
        offsets = graph_data['offsets']
        counts = graph_data['counts']
        
        return src_id, dst_id, flat_to_pts, offsets, counts
    
    def _enumerate_paths_int(self, src_id: np.ndarray, dst_id: np.ndarray, start_ids=None):
        """パス列挙
        
        有効なエッジ情報からグラフを構築し、全ての可能なパスを探索:
        - 隣接リスト形式でグラフを表現
        - 指定された開始点から深さ優先探索でパスを列挙
        - 各パスはノードID（ピーク点の番号）の配列として表現
        """
        if len(src_id) == 0:
            return []
        
        # エッジ情報から隣接リスト形式のグラフを構築
        adjacency_list = self._build_adjacency_list(src_id, dst_id)
        
        # 深さ優先探索で全ての可能なパスを探索・列挙
        return self._traverse_paths(adjacency_list, start_ids)
    
    def _build_adjacency_list(self, src_id: np.ndarray, dst_id: np.ndarray):
        """隣接リストを構築"""
        n_nodes = int(max(src_id.max(), dst_id.max())) + 1
        
        order = np.argsort(src_id, kind="mergesort")
        src = src_id[order]
        dst = dst_id[order]
        
        starts = np.searchsorted(src, np.arange(n_nodes), side="left")
        ends = np.searchsorted(src, np.arange(n_nodes), side="right")
        
        return [dst[starts[i]:ends[i]] for i in range(n_nodes)]
    
    def _traverse_paths(self, adjacency_list, start_ids=None):
        """パス探索を実行"""
        n_nodes = len(adjacency_list)
        paths = []
        
        if start_ids is None:
            start_ids = list(range(n_nodes))
        
        for start_id in start_ids:
            if start_id >= n_nodes:
                continue
            
            stack = [[start_id]]
            while stack:
                path = stack.pop()
                curr = path[-1]
                
                if len(adjacency_list[curr]) == 0:
                    if len(path) > 1:
                        paths.append(path)
                else:
                    for nxt in adjacency_list[curr]:
                        if nxt not in path:
                            stack.append(path + [nxt])
        
        return paths


    def build_valid_paths_coords(self, labels: np.ndarray, centered_pts: np.ndarray, 
                                 dist_map: np.ndarray, cx: int, cy: int, thr: float, 
                                 front_deg: float = 90, start_ids=None):
        """有効パス座標を構築
        
        この関数は、LiDAR検出結果から実際に走行可能なパスを構築する処理を行う:
        1. 検出されたピーク点同士を接続するエッジを生成
        2. 各エッジが有効（障害物なし、距離閾値内）かを検証
        3. 有効なエッジを使ってパスを探索・列挙
        4. パスIDを実際の座標に変換
        """
        
        # ステップ1: 有効なエッジ（ピーク点間の接続）を構築
        # - ピーク点をレイヤー別に整理してグラフ構造を作成
        # - 各エッジが障害物を通らないか、距離制約を満たすかを検証
        src_id, dst_id, flat_to_pts, offsets, counts = self._set_valid_edges_midpoint_nearest(
            labels, centered_pts, dist_map, cx, cy, thr, front_deg
        )

        # ステップ2: パス探索の開始点を決定
        # - counts[0]: 最初のレイヤーのピーク数
        # - start_idsが指定されていない場合はデフォルトで[0]（最初のピーク）から開始
        if len(counts) > 0:
            if start_ids is not None and counts[0] > 0:
                start_ids = start_ids  # 指定された開始点を使用
            else:
                start_ids = [0]  # デフォルト: 最初のピークから開始

        # ステップ3: 有効なエッジを使ってパスを列挙
        # - 隣接リスト形式のグラフを構築
        # - 深さ優先探索でロボットから遠方への全ての可能なパスを探索
        paths_ids = self._enumerate_paths_int(src_id, dst_id, start_ids=start_ids)
        
        # パス重複除去と座標変換用に必要な情報のみ返却
        return {
            "paths_ids": paths_ids,           # パスのノードID列挙 [[0,1,2], [0,3,4], ...]
            "flat_to_pts": flat_to_pts,      # フラット化ノードID→座標点インデックス
        }


# =============================
# アルゴリズム実装レイヤー：パス生成パイプライン
# =============================
class PathPipeline:
    """パス生成パイプラインクラス（アルゴリズム実装レイヤー）
    
    責任:
    - 純粋なアルゴリズム実装
    - 画像処理、距離場生成、パス探索
    - 内部座標系（画面座標系）での処理
    - アルゴリズムパラメータの管理
    """
    
    def __init__(self, params: PeakParams = None, config: PathPlannerConfig = None):
        """
        Args:
            params: ピーク検出パラメータ
            config: パスプランナー設定
        """
        self.params = params or PeakParams()
        self.config = config or PathPlannerConfig()
        self.peak_detector = PeakDetector(params, config)
        self.path_searcher = GraphPathSearcher(config)
        self.smoother = PotentialPathSmoother(config)
    
    def _setup_pipeline_parameters(self, H, W, R_in, ring_thickness, front_deg, 
                                  path_front_deg, dist_thresh, border_thickness):
        """パイプラインパラメータの初期化とデフォルト値設定"""
        H = H or self.config.HEIGHT
        W = W or self.config.WIDTH
        R_in = R_in if R_in is not None else np.array(self.config.RING_RADII, dtype=np.int32)
        ring_thickness = ring_thickness or self.config.RING_THICKNESS
        front_deg = front_deg or self.config.FRONT_DEG
        path_front_deg = path_front_deg or self.config.PATH_FRONT_DEG
        dist_thresh = dist_thresh or self.config.DIST_THRESH
        border_thickness = border_thickness or self.config.BORDER_THICKNESS
        
        return H, W, R_in, ring_thickness, front_deg, path_front_deg, dist_thresh, border_thickness
    

    def run_pipeline(self,
                    view_points: np.ndarray,
                    H: int = None,
                    W: int = None,
                    *,
                    R_in: np.ndarray = None,
                    ring_thickness: int = None,
                    front_deg: float = None,
                    path_front_deg: float = None,
                    dist_thresh: float = None,
                    border_thickness: int = None) -> Dict[str, Any]:
        """ローカルパス生成パイプライン（オーケストレーター）"""
        
        # 1. 入力パラメータの検証
        self._validate_pipeline_inputs(view_points)
        
        # 2. パイプライン実行に必要なパラメータを準備
        pipeline_params = self._prepare_pipeline_parameters(
            H, W, R_in, ring_thickness, front_deg, path_front_deg, dist_thresh, border_thickness
        )
        
        # 3. メインの処理パイプラインを実行（画面座標系で統一処理）
        processing_results = self._execute_core_pipeline(view_points, pipeline_params)
        
        # 4. 結果をまとめて返却用の辞書形式に整形
        return self._build_pipeline_results(processing_results, pipeline_params)
    
    def _validate_pipeline_inputs(self, view_points: np.ndarray):
        """パイプライン入力の検証（画面座標系データ）"""
        if view_points is None or view_points.size == 0:
            raise ValueError("View points array is None or empty")
        if view_points.ndim != 2 or view_points.shape[1] != 2:
            raise ValueError(f"View points must have shape (N, 2), got {view_points.shape}")
    
    def _prepare_pipeline_parameters(self, H, W, R_in, ring_thickness, front_deg, 
                                   path_front_deg, dist_thresh, border_thickness):
        """パイプラインパラメータの準備と正規化"""
        H, W, R_in, ring_thickness, front_deg, path_front_deg, dist_thresh, border_thickness = \
            self._setup_pipeline_parameters(H, W, R_in, ring_thickness, front_deg, 
                                           path_front_deg, dist_thresh, border_thickness)
        
        return {
            'H': H, 'W': W, 'R_in': R_in, 'ring_thickness': ring_thickness,
            'front_deg': front_deg, 'path_front_deg': path_front_deg, 
            'dist_thresh': dist_thresh, 'border_thickness': border_thickness
        }
    
    def _execute_core_pipeline(self, view_points: np.ndarray, params: dict):
        """コアパイプライン処理の実行（独立クラス群で構成）"""
        # 1. 距離場生成＋ピーク検出（統合独立クラス）
        peak_detector_params = PeakDetectorParams(
            H=params['H'], W=params['W'], R_in=params['R_in'],
            ring_thickness=params['ring_thickness'], front_deg=params['front_deg'],
            dist_thresh=params['dist_thresh'], border_thickness=params['border_thickness']
        )
        peaks, centered_points, sorted_labels, dist_inside, dist_norm, center_x, center_y = \
            self.peak_detector.detect_from_view_points(view_points, peak_detector_params)
        
        # 2. パス構築（独立クラス統合処理）
        coordinate_paths = self.path_searcher.build_and_deduplicate_paths(
            sorted_labels, centered_points, dist_inside, center_x, center_y,
            params['dist_thresh'], params['path_front_deg'], self.config.START_IDS
        )
        
        # 4. パス平滑化（独立クラス）
        smoothed_paths = self.smoother.smooth_paths(coordinate_paths, dist_inside, params['H'], params['W'])
        
        return {
            'view_points': view_points, 'mask': None, 'dist_inside': dist_inside,
            'dist_norm': dist_norm, 'path_candidate_pixels': None, 'center_x': center_x,
            'center_y': center_y, 'peaks': peaks, 'centered_points': centered_points,
            'sorted_labels': sorted_labels, 'coordinate_paths': coordinate_paths,
            'smoothed_paths': smoothed_paths
        }
    
    def _build_pipeline_results(self, processing_results: dict, params: dict):
        """パイプライン結果の構築"""
        return {
            "view_points": processing_results['view_points'],
            "mask": processing_results['mask'],
            "dist_inside": processing_results['dist_inside'],
            "dist_norm": processing_results['dist_norm'],
            "path_candidate_pixels": processing_results['path_candidate_pixels'],
            "labels": processing_results['sorted_labels'],
            "peaks": processing_results['peaks'],
            "peaks_centered": processing_results['centered_points'],
            "rough_paths": processing_results['coordinate_paths'],
            "paths": processing_results['smoothed_paths']
        }
    
    
    
    
    
    
    


# =============================
# パス重複除去クラス
# =============================
class PathDeduplicator:
    """パス重複除去：無駄なパスの除去"""

    def __init__(self, config: PathPlannerConfig = None):
        self.config = config or PathPlannerConfig()

    def deduplicate_paths(self, paths_ids: List[List[int]], centered_pts: np.ndarray = None, flat_to_pts: np.ndarray = None) -> List[List[int]]:
        """
        パスの重複除去：冗長パスを除去 (高速版)
        
        Args:
            paths_ids: パスIDのリスト [[start, ..., end], ...]
            
        Returns:
            deduplicated_paths: 重複除去後のパス
        """
        if not paths_ids:
            return []
       
        all_shortest_paths = self._find_all_shortest_paths(paths_ids, centered_pts, flat_to_pts)
        
        
        # 直接マージポイントベースで重複除去 (O(N²) → O(N))
        paths_ids = self._filter_redundant_paths(paths_ids, all_shortest_paths, centered_pts, flat_to_pts)

        return paths_ids
    
    def _build_graph(self, paths_ids: List[List[int]], centered_pts: np.ndarray, flat_to_pts: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
        """パスIDからグラフを構築（座標ベース重み）"""
        graph = {}
        
        # 全ノードの抽出
        all_nodes = set()
        for path in paths_ids:
            all_nodes.update(path)
        
        # グラフ初期化
        for node in all_nodes:
            graph[node] = []
        
        # 全エッジを収集
        edges = []
        for path in paths_ids:
            for i in range(len(path) - 1):
                edges.append((path[i], path[i + 1]))
        
        # 重複を除去
        unique_edges = list(set(edges))
        
        if unique_edges:
            # numpy配列に変換してベクトル化計算
            edge_array = np.array(unique_edges)
            from_nodes = edge_array[:, 0]
            to_nodes = edge_array[:, 1]
            
            # 座標をまとめて取得
            from_coords = centered_pts[flat_to_pts[from_nodes]]  # (N, 2)
            to_coords = centered_pts[flat_to_pts[to_nodes]]      # (N, 2)
            
            # ベクトル化されたユークリッド距離計算
            diff = to_coords - from_coords  # (N, 2)
            weights = np.sqrt(np.sum(diff * diff, axis=1))  # (N,)
            
            # グラフに追加
            for i, (from_node, to_node) in enumerate(unique_edges):
                graph[from_node].append((to_node, float(weights[i])))
        
        return graph
    
    def _find_all_shortest_paths(self, paths_ids: List[List[int]], centered_pts: np.ndarray, flat_to_pts: np.ndarray) -> Dict[int, Dict[int, List[int]]]:
        """実際の開始ノードからの最短パス（Dijkstra法）"""
        graph = self._build_graph(paths_ids, centered_pts, flat_to_pts)
        
        if not graph:
            return {}
        
        # 実際のパス開始ノードを取得
        unique_start_nodes = list(set(path[0] for path in paths_ids))
        
        # 各実際の開始ノードからの最短パス
        all_shortest_paths = {}
        
        for start_node in unique_start_nodes:
            shortest_paths = self._dijkstra_single_source(graph, start_node)
            all_shortest_paths[start_node] = shortest_paths
        
        return all_shortest_paths
    
    def _dijkstra_single_source(self, graph: Dict[int, List[Tuple[int, float]]], start_node: int) -> Dict[int, List[int]]:
        """単一始点Dijkstra法"""
        distances = {node: float('inf') for node in graph}
        distances[start_node] = 0.0
        previous = {node: None for node in graph}
        visited = set()
        
        heap = [(0.0, start_node)]
        
        while heap:
            current_dist, current_node = heapq.heappop(heap)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            for neighbor, weight in graph[current_node]:
                new_dist = current_dist + weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(heap, (new_dist, neighbor))
        
        # パス復元
        paths = {}
        for end_node in graph:
            if distances[end_node] != float('inf'):
                path = []
                current = end_node
                while current is not None:
                    path.append(current)
                    current = previous[current]
                path.reverse()
                paths[end_node] = path
            else:
                paths[end_node] = []
        
        return paths
    
    def _filter_redundant_paths(self, paths_ids: List[List[int]], all_shortest_paths: Dict[int, Dict[int, List[int]]], centered_pts: np.ndarray, flat_to_pts: np.ndarray) -> List[List[int]]:
        """冗長パスの除去：Dijkstra最短パス結果を直接使用"""
        if len(paths_ids) <= 1:
            return paths_ids
        
        start_node = paths_ids[0][0]
        shortest_paths = all_shortest_paths.get(start_node, {})
        
        if not shortest_paths:
            return paths_ids
        
        # 途中ノード（ゴールではない）を収集
        intermediate_nodes = set()
        for path_to_node in shortest_paths.values():
            intermediate_nodes.update(path_to_node[:-1])  # 終端以外のノード
        
        # ゴールノード = shortest_pathsにあるが途中ノードではないノード
        goal_nodes = [node for node in shortest_paths.keys() if node not in intermediate_nodes]
        
        # ゴールノードへの最短パスを取得
        optimized_paths = []
        for goal_node in goal_nodes:
            if goal_node in shortest_paths:
                optimized_paths.append(shortest_paths[goal_node])
        
        # 曲率でソート（曲率が高いほど小さいindex）
        if optimized_paths:
            optimized_paths = self._sort_paths_by_turning_angle(optimized_paths, centered_pts, flat_to_pts)
        
        return optimized_paths
    
    def _sort_paths_by_turning_angle(self, paths_ids: List[List[int]], centered_pts: np.ndarray, flat_to_pts: np.ndarray) -> List[List[int]]:
        """パスを旋回角度でソート（内側に曲がっているほど小さいindex）"""
        if len(paths_ids) <= 1:
            return paths_ids
        
        path_angles = []
        for path in paths_ids:
            turning_angle = self._calculate_turning_angle(path, centered_pts, flat_to_pts)
            path_angles.append((abs(turning_angle), path))  # 絶対値で曲がり具合を評価
        
        # 旋回角度が大きい順（内側に曲がっている順）でソート
        path_angles.sort(key=lambda x: x[0], reverse=True)
        
        return [path for angle, path in path_angles]
    
    def _calculate_turning_angle(self, path_ids: List[int], centered_pts: np.ndarray, flat_to_pts: np.ndarray) -> float:
        """現在の向き（ロボット前方=X軸正方向）と現在位置から最終ノードへの方向の角度差を計算

        バイアスを考慮: DIRECTION_BIAS_DEG分だけ基準方向をずらす
        例: バイアス=-30° → 右30°方向が「直進（角度差0）」として扱われる
        """
        if len(path_ids) < 1:
            return 0.0

        # 現在の位置（ロボット位置 = 原点）
        robot_pos = np.array([0.0, 0.0])

        # 最終ノードの座標を取得
        final_node_pos = centered_pts[flat_to_pts[path_ids[-1]]]

        # 現在位置から最終ノードへの方向ベクトル
        direction_to_final = final_node_pos - robot_pos

        # パスの方向角度を計算
        angle = np.arctan2(direction_to_final[1], direction_to_final[0])

        # バイアスを適用した基準方向（度→ラジアン）
        bias_rad = np.deg2rad(self.config.DIRECTION_BIAS_DEG)

        # バイアスを考慮した角度差
        turning_angle = angle - bias_rad

        # -πからπの範囲に正規化
        while turning_angle > np.pi:
            turning_angle -= 2 * np.pi
        while turning_angle < -np.pi:
            turning_angle += 2 * np.pi

        return turning_angle
    


class PotentialPathSmoother:
    """エルミート補間ベースのパス平滑化"""
    
    def __init__(self, config: PathPlannerConfig = None):
        """初期化"""
        self.config = config or PathPlannerConfig()
    
    def smooth_paths(self, paths_coords: List[np.ndarray], dist_inside: np.ndarray, 
                    H: int, W: int) -> List[np.ndarray]:
        """
        エルミート補間を使ってパスを平滑化
        
        Args:
            paths_coords: パス座標のリスト [[x, y], ...]
            dist_inside: 距離変換結果 (H x W) - 現在未使用
            H, W: 画像のサイズ - 現在未使用
            
        Returns:
            smoothed_paths: 平滑化されたパス
        """
        if not paths_coords:
            return paths_coords
            
        smoothed_paths = []
        # ロボット現在位置（画面座標系では中心）
        start_pos = np.array([0.0, 0.0])  # centered coordinates
        # ロボットの現在の向き（画面座標系では前方=X軸正方向）
        current_direction = np.array([1.0, 0.0])  # 前方向

        for path in paths_coords:
            if len(path) > 2:
                smoothed_path = self._parameter_free_path_optimization(
                    start_pos, current_direction, path
                )
                smoothed_paths.append(smoothed_path)
            else:
                smoothed_paths.append(path)

        return smoothed_paths
    
    
    # 実際に使用される関数のみを保持
    def _parameter_free_path_optimization(self, start_pos: np.ndarray, current_direction: np.ndarray,
                                         target_path: np.ndarray) -> np.ndarray:
        """パラメータフリーのエルミート補間による経路最適化"""
        
        goal_pos = target_path[-1]
        
        # ゴール方向を計算（最後の2点から）
        if len(target_path) < 2:
            raise ValueError("Target path must have at least 2 points")
            
        final_direction = target_path[-1] - target_path[-2]
        goal_dir = _safe_normalize(final_direction, default_vector=current_direction)
        
        # エルミート補間でベースライン経路を生成
        baseline_path = self._hermite_interpolate(start_pos, current_direction, goal_pos, goal_dir)
        
        # target_pathとの自動ブレンド
        blended_path = self._automatic_blend(baseline_path, target_path)
        
        return blended_path
    
    def _hermite_interpolate(self, start_pos: np.ndarray, start_dir: np.ndarray,
                           goal_pos: np.ndarray, goal_dir: np.ndarray,
                           num_points: int = None) -> np.ndarray:
        """エルミート補間による滑らかな経路生成"""
        if num_points is None:
            num_points = self.config.INTERPOLATION_POINTS
            
        # 距離ベースで制御点の強度を決定
        distance = np.linalg.norm(goal_pos - start_pos)
        control_strength = _safe_divide(distance, self.config.CONTROL_STRENGTH_FACTOR, default=1.0)
        
        # ベジェ曲線の制御点
        p0 = start_pos
        p1 = start_pos + start_dir * control_strength
        p2 = goal_pos - goal_dir * control_strength
        p3 = goal_pos
        
        # ベジェ曲線でパスを生成（ベクトル化）
        t_values = np.linspace(0, 1, num_points)
        
        # 3次ベジェ曲線の公式をベクトル化
        t = t_values[:, np.newaxis]  # (num_points, 1)
        one_minus_t = 1 - t
        
        # ベジェ曲線の係数をベクトル化して計算
        path_points = (one_minus_t**3 * p0 +
                      3 * one_minus_t**2 * t * p1 +
                      3 * one_minus_t * t**2 * p2 +
                      t**3 * p3)
        
        return path_points
    
    def _automatic_blend(self, baseline_path: np.ndarray, target_path: np.ndarray) -> np.ndarray:
        """target_pathとの自動ブレンド（パラメータなし）"""
        
        if len(target_path) == 0:
            return baseline_path
        
        n_points = len(baseline_path)
        if n_points <= 2:
            return baseline_path
        
        # 進行度を一括計算
        progress_values = np.arange(n_points) / (n_points - 1)
        
        # target_pathの対応点を一括取得（ベクトル化）
        target_indices = progress_values * (len(target_path) - 1)
        indices_floor = np.floor(target_indices).astype(int)
        indices_ceil = np.clip(indices_floor + 1, 0, len(target_path) - 1)
        
        # 線形補間の重み
        t_values = target_indices - indices_floor
        
        # target_pathの対応点を一括計算
        target_points = ((1 - t_values[:, np.newaxis]) * target_path[indices_floor] + 
                        t_values[:, np.newaxis] * target_path[indices_ceil])
        
        # 距離ベースの重みを一括計算
        distances = np.linalg.norm(target_points - baseline_path, axis=1)
        distance_weights = _safe_divide(distances, self.config.AUTO_WEIGHT_DISTANCE_FACTOR, default=1.0)
        auto_weights = 1.0 / (1.0 + distance_weights)
        
        # ブレンドを一括実行
        blended_path = baseline_path.copy()
        blended_path[1:-1] = ((1 - auto_weights[1:-1, np.newaxis]) * baseline_path[1:-1] + 
                             auto_weights[1:-1, np.newaxis] * target_points[1:-1])
        
        return blended_path
    





# =============================
# まとめて実行する “パイプライン”
# =============================
class LocalPathPlanner:
    """ローカルパス計画システム（公開APIレイヤー）
    
    責任:
    - 外部インターフェースの提供
    - パラメータの検証とデフォルト値管理
    - 座標系変換（ロボット座標系への統一）
    - ビジネスロジック（パス選択戦略など）
    """
    
    def __init__(self, config: PathPlannerConfig = None):
        """
        Args:
            occupancy_map: 占有格子地図
            config: パスプランナー設定
        """        
        self.config = config or PathPlannerConfig()
        self.path_pipeline = PathPipeline(config=self.config)

        # Belief state for path selection (persists across frames)
        self._belief_direction = np.array([1.0, 0.0], dtype=np.float32)  # 初期: 前方
        self._belief_initialized = False

    def generate_local_paths(self, last_lidar_data: Dict[str, Any]) -> List[np.ndarray]:
        """ローカルパス生成（公開API）
        
        責任:
        - LiDARデータの検証（APIレイヤー）
        - アルゴリズムレイヤーへの委託
        - 結果の座標系変換（画面座標→ロボット座標）
            
        Returns:
            List of path arrays, each shape (N, 2) in robot coordinates
        """
        # 1. 入力座標変換（APIレイヤーの責任）：極座標 → 画面座標系
        view_points = self._polar_to_view_points(
            last_lidar_data['ranges'], 
            last_lidar_data['angles'], 
            self.config.HEIGHT, 
            self.config.WIDTH
        )
        
        # 2. アルゴリズムレイヤー：画面座標系でパイプライン全体を実行
        pipeline_result = self.path_pipeline.run_pipeline(
            view_points=view_points,
            H=self.config.HEIGHT,
            W=self.config.WIDTH,
            dist_thresh=self.config.LOCAL_DIST_THRESH
        )
        
        # パス座標を取得（centered coordinates）
        centered_paths = pipeline_result.get("paths", [])
        
        # 画面座標系のパスをロボット座標系に変換
        centered_rough_paths = pipeline_result.get("rough_paths", [])
        robot_paths = self._convert_centered_to_robot_coordinates(centered_paths)
        robot_rough_paths = self._convert_centered_to_robot_coordinates(centered_rough_paths)
 
        return robot_paths, robot_rough_paths
    
    def _convert_centered_to_robot_coordinates(self, centered_paths: List[np.ndarray]) -> List[np.ndarray]:
        """座標系変換（APIレイヤーの責任）
        
        画面座標系（アルゴリズム内部）からロボット座標系（外部API）へ変換
        
        Args:
            centered_paths: 画面座標系のパス一覧（アルゴリズム出力、ピクセル単位）
            
        Returns:
            robot_paths: ロボット座標系のパス一覧（API出力、メートル単位）
        """
        robot_paths = []
        for path in centered_paths:
            if len(path) > 0:
                # 画面座標系 → ロボット座標系変換
                # 画面座標系：中心原点、右X正、下Y正（ピクセル単位）
                # ロボット座標系：前方X正、左Y正（メートル単位）
                robot_path = np.zeros_like(path, dtype=np.float32)
                robot_path[:, 0] = path[:, 0] * self.config.MAP_RESOLUTION   # 画面X → ロボットX（ピクセル→メートル）
                robot_path[:, 1] = path[:, 1] * self.config.MAP_RESOLUTION   # 画面Y → ロボットY（ピクセル→メートル）
                robot_paths.append(robot_path)
        return robot_paths
    
    def _polar_to_view_points(self, ranges: np.ndarray, local_angles: np.ndarray, H: int, W: int) -> np.ndarray:
        """入力座標変換（APIレイヤーの責任）：極座標 → 画面座標系
        
        LiDARの極座標データ(range, angle)をアルゴリズム内部で使う画面座標系に変換
        
        Args:
            ranges: 距離配列 (メートル単位)
            local_angles: 角度配列 (ローカル座標)
            H, W: 画面サイズ
            
        Returns:
            view_points: 画面座標系のビューポイント (N, 2)
        """
        # メートルからピクセルに変換
        pixel_ranges = ranges / self.config.MAP_RESOLUTION
        
        px = pixel_ranges * np.cos(local_angles)
        py = pixel_ranges * np.sin(local_angles)
        p = np.stack([px, py], axis=1)
        vp = p + np.array([[W // 2, H // 2]], dtype=p.dtype)
        return vp.astype(np.int32)
    
    def select_outermost_path(self, paths: List[np.ndarray]) -> Tuple[np.ndarray, int]:
        """Belief-based path selection.

        各パスを信頼度×信念一貫性でスコアリングし、最高スコアを選択。
        選択後に信念方向をEMA更新する。パスが0本なら信念を維持。

        Args:
            paths: ローカルパス配列のリスト

        Returns:
            (選択されたパス, インデックス)
        """
        if not paths:
            return np.array([]), -1

        # 全パスをスコアリング
        candidates = []
        for idx, path in enumerate(paths):
            score, confidence, _ = self._score_path(path)
            if confidence >= self.config.BELIEF_MIN_CONFIDENCE:
                candidates.append((score, confidence, idx))

        if not candidates:
            return np.array([]), -1

        # 最高スコアのパスを選択
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_confidence, best_idx = candidates[0]
        best_path = paths[best_idx]

        # 信念更新
        self._update_belief(best_path, best_confidence)

        return best_path, best_idx

    def _calculate_path_confidence(self, path: np.ndarray) -> float:
        """パスの物理長から信頼度(0~1)を算出。長いパスほど高信頼。"""
        if len(path) == 0:
            return 0.0

        if len(path) > 1:
            diffs = np.diff(path, axis=0)
            physical_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        else:
            physical_length = float(np.linalg.norm(path[0]))

        return float(np.tanh(physical_length / self.config.BELIEF_CONFIDENCE_LENGTH_SCALE))

    def _extract_path_direction(self, path: np.ndarray) -> np.ndarray:
        """パス終点方向の単位ベクトルを取得。"""
        if len(path) == 0:
            return None

        endpoint = path[-1]
        dist = np.linalg.norm(endpoint)
        if dist < 1e-6:
            return None

        return (endpoint / dist).astype(np.float32)

    def _calculate_consistency(self, path_direction: np.ndarray) -> float:
        """信念方向との余弦類似度(-1~1)を算出。"""
        if path_direction is None:
            return 0.0
        return float(np.dot(self._belief_direction, path_direction))

    def _score_path(self, path: np.ndarray) -> Tuple[float, float, float]:
        """confidence + weight × (consistency + bias) で総合スコアを算出。

        - consistency: 信念方向との一致度（過去の選択との連続性）
        - bias: DIRECTION_BIAS_DEG方向との一致度（優先方向）
        """
        confidence = self._calculate_path_confidence(path)
        path_direction = self._extract_path_direction(path)

        if path_direction is None:
            return (confidence, confidence, 0.0)

        # バイアス方向との一致度
        bias_rad = np.deg2rad(self.config.DIRECTION_BIAS_DEG)
        bias_direction = np.array([np.cos(bias_rad), np.sin(bias_rad)], dtype=np.float32)
        bias_preference = float(np.dot(bias_direction, path_direction))

        if not self._belief_initialized:
            # 初期化前はバイアスのみ使用
            score = confidence + self.config.BELIEF_CONSISTENCY_WEIGHT * bias_preference
            return (score, confidence, bias_preference)

        # 信念との一致度 + バイアス
        consistency = self._calculate_consistency(path_direction)
        combined = (consistency + bias_preference) / 2.0
        score = confidence + self.config.BELIEF_CONSISTENCY_WEIGHT * combined
        return (score, confidence, combined)

    def _update_belief(self, selected_path: np.ndarray, confidence: float):
        """選択パスの方向でEMA更新。高信頼パスほど大きく信念を動かす。"""
        if len(selected_path) == 0:
            return

        path_direction = self._extract_path_direction(selected_path)
        if path_direction is None:
            return

        effective_alpha = self.config.BELIEF_EMA_ALPHA * confidence
        new_belief = (1.0 - effective_alpha) * self._belief_direction + effective_alpha * path_direction

        belief_norm = np.linalg.norm(new_belief)
        if belief_norm > 1e-6:
            self._belief_direction = (new_belief / belief_norm).astype(np.float32)

        self._belief_initialized = True


def main():
    import argparse
    from model import RobotPose
    from map_manager import MapManager
    from utils import CoordinateTransform
    from fast_lidar import FastLiDARProcessor

    parser = argparse.ArgumentParser(description="Local Path Planning System")
    parser.add_argument("--map", default="map.png", help="Map image file")
    parser.add_argument("--resolution", type=float, default=0.05, help="Map resolution [m/pixel]")
    parser.add_argument("--robot-x", type=float, default=340, help="Robot initial X (map coords)")
    parser.add_argument("--robot-y", type=float, default=320, help="Robot initial Y (map coords)")
    parser.add_argument("--robot-theta", type=float, default=180.0, help="Robot initial theta [deg]")
    parser.add_argument("--output", help="Output visualization file")
    
    args = parser.parse_args()
    
    # 地図読み込み
    map_manager = MapManager(args.resolution)
    map_manager.load_map(args.map)
    coord_transform = CoordinateTransform(map_manager.get_map_config())

    # センサの作成
    lidar_processor = FastLiDARProcessor(
        num_beams=181,
        fov_deg=180.0,
        max_range=200.0
    )

    # パスプランナー
    planner = LocalPathPlanner(
        map_manager.get_occupancy_map(),
    )
    
    # ロボット位置設定
    theta = coord_transform.normalize_angle(args.robot_theta)
    robot_pose = RobotPose(
        x=args.robot_x, 
        y=args.robot_y, 
        theta=theta
    )
 
    # 新しいLiDARProcessorクラスを使用
    local_angles, ranges, hit_points_map = lidar_processor.simulate_lidar_vectorized(
        map_manager.get_occupancy_map(),
        robot_pose
    )

    last_lidar_data = {
        'ranges': ranges,
        'angles': local_angles,
    }
    
    # パス生成
    paths, rough_paths = planner.generate_local_paths(last_lidar_data)
    
    return 0


if __name__ == "__main__":
    main()