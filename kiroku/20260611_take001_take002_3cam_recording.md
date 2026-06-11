# 2026-06-11 3カメラ D405 ハンドモーキャップ計測 take001 / take002

## 目的

- LEAP Hand へのリターゲティング用の**右手**モーションデータ取得(第1弾)
- 将来の Fast-FoundationStereo 深度補正に備え、IRステレオペアも同時記録
- handpose3d ベースの 3カメラ DLT 三角測量パイプライン(`run_handpose3d_d405.py`)の初実運用

## 構成・環境

| 項目 | 内容 |
|---|---|
| カメラ | Intel RealSense D405 ×3(cam0=130322270922 / cam1=230422271452 / cam2=353322271600、各 USB3 個別接続) |
| 校正 | `calib/optimized_3cam_extrinsics.json`(2026-05-01 ChArUco キューブ再校正、点群オーバーレイ誤差 平均1.2mm)。world = cam0 color 光学フレーム |
| 環境 | conda env `recog`(mediapipe 0.10.9 / pyrealsense2 / OpenCV 4.12)、ホスト assam |
| 手法 | MediaPipe Hands(各カメラ)→ 歪み補正 → 多視点DLT → 21関節 3D + 掌座標系 |

## 実行コマンド

```bash
conda activate recog
cd ~/workspace/handpose3d
python run_handpose3d_d405.py --plot3d --save takes/take001   # 終了は Ctrl+C(1回)
python run_handpose3d_d405.py --plot3d --save takes/take002
```

## 結果

| take | 開始時刻 | フレーム | 長さ | 実効fps | 3D化率(≥2視点) | 平均再投影誤差 | 左右 | 容量 |
|---|---|---|---|---|---|---|---|---|
| take001 | 06-11 07:46:37 | 125 | 28.0 s | 4.5 | 97% | 10.0 px | 右(98%) | 32 MB |
| take002 | 06-11 07:47:15 | 107 | 23.7 s | 4.5 | **100%** | **6.2 px** | 右(98%) | 26 MB |

- 手首位置は cam0 前方 約27〜28cm(D405 の共有視野ゾーン内)で安定
- take002 の方が高品質。リターゲットの最初の入力には take002 推奨
- 実効 4.5fps は MediaPipe ×3カメラの処理律速(録画フレームは欠落なし、実時刻は `t_wall` 参照)

## 保存先

- **ローカル**: `/home/initial/workspace/handpose3d/takes/take001/`, `take002/`
- **リモート(受け渡し用)**: https://github.com/MY-CODE-1981/handpose3d (`main`)
  - commit `863f3a1`(ランナー+校正同梱+テイク一式)
  - commit `24a621b`(左右判定修正、npz 再エンコード)
- 別PCでは `git clone` だけで全データ取得可(Git LFS 不使用)

## 生成ファイル(各 take ディレクトリ内)

| ファイル | 内容 |
|---|---|
| `take.npz` | リターゲット用本体: `t_wall` / `kpts3d_world`(N,21,3)m / `kpts2d_px`(N,3,21,2) / `det_mask` / `handed`(0=左,1=右) / `handed_score` / `reproj_err_px` / `n_views` / `T_world_palm`(N,4,4) |
| `meta.json` | 校正スナップショット(K/D/R/t, IR-K, 基線長, ir_left→color)・座標系定義・リターゲット変換ヒント |
| `cam{0,1,2}.avi` | color 生映像(MJPG、フレーム k = npz 行 k) |
| `cam{0,1,2}_ir{L,R}.avi` | レクティファイ済み IRステレオ(Y8、FFS 深度推定用、エミッタなし) |
| `kpts_*.dat` | 元リポジトリ `show_3d_hands.py` 互換形式 |

## トラブルと対処(本計測中に発生)

1. **`Couldn't resolve requests`**: 3台とも USB2.1 で誤認識され、IR2 の 640x480@30 プロファイルが不提供 → `hardware_reset` で USB3.2 復帰。スクリプトに USB2 検出→自動リセットを実装済み
2. **Ctrl+C で take.npz が消える**: 保存処理がループ後にあり例外で素通り → `KeyboardInterrupt` 捕捉で finalize するよう修正済み(本テイクはこの修正後に取得)
3. **左右判定の反転**: MediaPipe の handedness は自撮りミラー前提 → 非反転入力では逆になる。`detect_hands_2d` でラベル反転し、既存 npz も再エンコード(commit `24a621b`)。幾何検証: 掌座標系で親指MCP y<0 = 右手、take002 全フレーム一貫

## 次のステップ

- 受け取り側PCで `pip install dex_retargeting` → take002 の掌ローカル座標を LEAP Hand へリターゲット(変換式は `README_D405.md` / `meta.json` の `retarget_hint`)
- 必要なら `cam*_ir{L,R}.avi` + `meta.json` の IR パラメータで FFS 深度 → 骨格 z 補正
- 高 fps が必要になったら: MediaPipe `model_complexity=0` / 検出を並列化 / 録画専用モード(検出オフライン化)を検討
