# handpose3d × 3x RealSense D405 (multi-camera hand mocap)

[TemugeB/handpose3d](https://github.com/TemugeB/handpose3d) をベースに、
3台の Intel RealSense D405 でハンドポーズ(MediaPipe 21点)を多視点DLT三角測量するランナーを追加したもの。
LEAP Hand 等へのリターゲティング、および Fast-FoundationStereo による深度補正を見据えたデータ記録に対応。

## 追加ファイル

- `run_handpose3d_d405.py` — メインランナー(ライブ計測 / オフライン / 録画)
- `calib/` — カメラ校正(下記参照)。リポジトリ単体で動くよう同梱
- `camera_parameters_d405/` — 元リポジトリ形式の校正ファイル(c{i}.dat / rot_trans_c{i}.dat)
- `takes/` — 計測データ(テイク)

## セットアップ

Python 3.10 推奨:

```bash
pip install mediapipe==0.10.9 pyrealsense2 opencv-contrib-python matplotlib numpy
```

## 使い方

```bash
python run_handpose3d_d405.py --plot3d --save takes/take001   # ライブ計測+保存
# 終了: GUIで q、またはターミナルで Ctrl+C(1回押して [save] 表示を待つ)
```

主なオプション: `--no-ir`(IRステレオ記録なし) / `--no-video`(生映像なし) /
`--record out.mp4`(オーバーレイ動画) / `--reset`(カメラ復旧) /
`--min-det-conf 0.3`(検出しきい値) / `--session DIR`(オフライン1フレーム) /
`--export-dat DIR`(元リポジトリ形式の校正出力)

USB2 で誤認識された場合は起動時に自動で hardware_reset して USB3 復帰を試みる。

## 校正(calib/)

- 機材: Intel RealSense D405 ×3(serial 130322270922 = cam0 / 230422271452 = cam1 / 353322271600 = cam2)
- `optimized_3cam_extrinsics.json`: ChArUco キューブ PnP + バンドル最適化(2026-05-01 再校正、
  点群オーバーレイ誤差 平均 1.2mm)。world = cam0 color 光学フレーム、t はメートル
- `intrinsics_session/cam{i}/intrinsics.json`: color K・歪み、IR(左) K、ステレオ基線長、ir_left→color 変換
- **注意: この校正は上記シリアルの個体・配置に固有。** カメラや配置を変えたら再校正が必要

## テイクのデータ形式(takes/<name>/)

| ファイル | 内容 |
|---|---|
| `take.npz` | `t_wall`(N,) UNIX秒 / `kpts3d_world`(N,21,3) m, NaN=欠測 / `kpts2d_px`(N,3,21,2) / `det_mask`(N,3) / `handed`(N,3) -1なし,0=左,1=右 / `handed_score`(N,3) / `reproj_err_px`(N,21) / `n_views`(N,) / `T_world_palm`(N,4,4) 掌座標系 |
| `meta.json` | 校正スナップショット・座標系定義・リターゲット変換ヒント・IR/FFS用パラメータ |
| `cam{0,1,2}.avi` | color 生映像(MJPG)。**フレーム k = take.npz の行 k**、実時刻は t_wall |
| `cam{0,1,2}_ir{L,R}.avi` | レクティファイ済みIRステレオペア(Y8)。Fast-FoundationStereo 入力用(エミッタなしパッシブ) |
| `kpts_*.dat` | 元リポジトリの `show_3d_hands.py` 用 |

掌座標系: 原点=手首(kpt0)、x=手首→中指MCP、z=cross(手首→人差指MCP, 手首→薬指MCP)直交化(右手で掌側)、y=z×x。

## リターゲティング(LEAP Hand 等)

- [dex-retargeting](https://github.com/dexsuite/dex-retargeting)(LEAP対応):
  `joint_pos = (R_palm.T @ (kpts3d_world - wrist).T).T @ OPERATOR2MANO`
- [Bidex_VisionPro_Teleop](https://github.com/leap-hand/Bidex_VisionPro_Teleop) 方式(PyBullet IK):
  掌座標系での各指 PIP+指先(MediaPipe idx: 親指3,4 / 人差指6,8 / 中指10,12 / 薬指14,16 / 小指18,20)をターゲットに、人→LEAPスケール約1.35–1.5倍
- `handed` を確認のこと(LEAP は右手。左手テイクはミラーリング必要)
- `handed` は**実際の左右**(MediaPipe の生ラベルは自撮りミラー前提のため、非反転入力では逆になる — スクリプト内で補正済み)
- 幾何チェック: 掌座標系で親指MCP(kpt2)の y < 0 なら右手(take002 で全フレーム一貫を確認済み)

## 深度補正(Fast-FoundationStereo)

`cam{i}_ir{L,R}.avi` + `meta.json` の `ir_left.fx` / `stereo_baseline_m` で
`depth = fx × baseline / disparity` → `ir_left_to_color` → `world_to_cam{i}` で world 座標へ。
`kpts3d_world` と同一座標系で突き合わせて骨格を補正できる。
