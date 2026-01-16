# Copilot Instructions (SfM + Bundle Adjustment)

## 项目概览（先看这些入口）
- 预处理：`preprocess.py`（SIFT→BFMatcher→RANSAC→scene graph），产物写入 `predictions/<dataset>/...`
- 增量 SfM：`sfm.py`（初始化→PnP 注册→三角化增量→可选 BA），结果写入 `predictions/<dataset>/results/<split>/...`
- BA 残差：`bundle_adjustment.py` 的 `compute_ba_residuals`（给 `scipy.optimize.least_squares` 用）
- 可视化：`visualize.py`（读 `points3d.npy` 用 Open3D 展示）
- 对拍测试：`test.py`（把 `predictions/` 与 `ta-results/` 的同名文件逐项比较）

## 数据与产物目录约定（不要改名）
- 输入数据：`data/<dataset>/images/` + `data/<dataset>/intrinsics.txt`（由 `get_camera_intrinsics()` 读入 3x3 K）
- 预处理目录：
  - 关键点：`predictions/<dataset>/keypoints/<image_id>.pkl`（`encode_keypoint/decode_keypoint` 约定）
  - BF 匹配：`predictions/<dataset>/bf-match/<imgA>_<imgB>.npy`（N×2，存 keypoint 索引对）
  - RANSAC：`predictions/<dataset>/ransac-match/<imgA>_<imgB>.npy`
  - 本质矩阵：`predictions/<dataset>/ransac-fundamental/<imgA>_<imgB>.npy`（变量名叫 *ESSENTIAL*，目录名保留现状）
  - 场景图：`predictions/<dataset>/scene-graph.json`（adjacency list：image_id -> neighbors[]）
- SfM 结果目录：`predictions/<dataset>/results/{no-bundle-adjustment|bundle-adjustment}/`
  - `points3d.npy`：M×3
  - `all-extrinsic.json`：image_id -> 3×4 外参（list-of-lists）
  - `correspondences2d3d.json`：image_id -> {keypoint_idx:int -> point3d_idx:int}
  - `registration-trajectory.txt`：按注册顺序逐行写 image_id

## 关键实现/约定细节（改代码时必须遵守）
- 任何“配对文件名”一律用排序后的 image_id：`match_id = '_'.join(sorted([id1,id2]))`
  - `sfm.load_matches()` 会在需要时 flip 列顺序；新增代码保持这一约定，否则会读错匹配列。
- 初始化对：`get_init_image_ids(scene_graph)` 选 RANSAC inlier 数最多的边。
- 初始化外参：`get_init_extrinsics()` 假设第一张 `[I|0]`，第二张用 `cv2.recoverPose(E, ...)` 得到 `[R|t]`。
- 三角化：`cv2.triangulatePoints(K[E1], K[E2], ...)`，齐次归一化得到 3D。
- PnP：`solve_pnp()` 内部做 RANSAC，使用 `cv2.solvePnP(..., flags=cv2.SOLVEPNP_ITERATIVE)` + `cv2.Rodrigues`。
- BA：`sfm.bundle_adjustment()` 构造参数向量：`[rvec,tvec]*C + points3d.flatten()`；
  - `bundle_adjustment.compute_ba_residuals()` 要求 **不写 Python 循环**（向量化：索引、`np.matmul`、广播）。

## 常用命令（以 README 为准）
```bash
python preprocess.py --dataset mini-temple
python sfm.py --dataset mini-temple
python sfm.py --dataset mini-temple --ba
python visualize.py --dataset mini-temple
python test.py --dataset mini-temple
```

## 资源/限制
- `preprocess.py` 有保护：`--ba` + `--dataset temple` 会 `assert` 失败（默认不允许对大数据集做 BA）。
