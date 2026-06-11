"""
3x Intel RealSense D405 multi-camera hand pose triangulation (handpose3d-style).

MediaPipe Hands per camera -> undistort 2D keypoints -> multi-view DLT -> 3D (21 joints).

Calibration sources (PoseRigFusion_ws / multi_camera_extrinsic_calib):
  - Extrinsics: output/result/optimized_3cam_extrinsics.json
      world = cam0 color frame, t in meters (validated 2026-05-01, overlay error ~1.2 mm)
  - Intrinsics: output/d405_3cam/<session>/cam{i}/intrinsics.json (color K + distortion)
      cam index is matched by librealsense serial:
        cam0=130322270922  cam1=230422271452  cam2=353322271600

Usage (recog env has mediapipe + pyrealsense2):
  conda activate recog
  python run_handpose3d_d405.py                          # live, 3x D405 (tiled 2D window)
  python run_handpose3d_d405.py --plot3d                 # + live 3D skeleton (matplotlib)
  python run_handpose3d_d405.py --record out.mp4         # save tiled overlay video
  python run_handpose3d_d405.py --save out               # retarget-ready take.npz + meta.json
                                                         #   + raw cam{0,1,2}.avi (--no-video to skip)
                                                         #   + IR stereo cam*_ir{L,R}.avi for FFS
                                                         #     depth refinement (--no-ir to skip)
                                                         #   + legacy kpts_*.dat for show_3d_hands
  python run_handpose3d_d405.py --session <capture_dir>  # offline: one captured frame set
  python run_handpose3d_d405.py --export-dat camera_parameters_d405  # write repo-format .dat
  python run_handpose3d_d405.py --reset                  # hardware-reset D405s first (hang fix)

Triangulation needs the hand visible in >=2 cameras at once — each tile shows
HAND/no hand so you can find the shared workspace zone (~25-40 cm in front of cam0).

Saved kpts_3d.dat is compatible with show_3d_hands.py (units: meters, cam0 frame).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2 as cv

_HERE = Path(__file__).resolve().parent
CALIB_PKG = Path('/home/initial/workspace/PoseRigFusion_ws/src/multi_camera_extrinsic_calib')
DEFAULT_EXTRINSICS = CALIB_PKG / 'output/result/optimized_3cam_extrinsics.json'
DEFAULT_SESSION_ROOT = CALIB_PKG / 'output/d405_3cam'
if not DEFAULT_EXTRINSICS.exists():  # cloned repo on another PC: use bundled copy
    DEFAULT_EXTRINSICS = _HERE / 'calib/optimized_3cam_extrinsics.json'

FRAME_W, FRAME_H, FPS = 640, 480, 30
N_CAMS = 3

# MediaPipe 21-keypoint hand skeleton
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


def latest_session_dir(root=DEFAULT_SESSION_ROOT):
    root = Path(root)
    if root.exists():
        dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / 'cam0/intrinsics.json').exists())
        if dirs:
            return dirs[-1]
    fallback = _HERE / 'calib/intrinsics_session'  # bundled copy for cloned repos
    if (fallback / 'cam0/intrinsics.json').exists():
        return fallback
    raise FileNotFoundError(f'no capture session with intrinsics under {root} or {fallback}')


def load_calibration(extrinsics_path, intrinsics_session):
    """Per camera: serial, K(3x3), D(dist coeffs), R/t (world->cam, m), P=K[R|t]."""
    with open(extrinsics_path) as f:
        extr = json.load(f)
    cams = []
    for i in range(N_CAMS):
        with open(Path(intrinsics_session) / f'cam{i}/intrinsics.json') as f:
            info = json.load(f)
        c = info['color']
        K = np.array([[c['fx'], 0, c['ppx']], [0, c['fy'], c['ppy']], [0, 0, 1]], dtype=np.float64)
        D = np.array(c['coeffs'], dtype=np.float64)
        w2c = extr[f'world_to_cam{i}']
        R = np.array(w2c['R'], dtype=np.float64)
        t = np.array(w2c['t'], dtype=np.float64).reshape(3, 1)
        P = K @ np.hstack([R, t])
        cams.append(dict(idx=i, serial=info['serial'], K=K, D=D, R=R, t=t, P=P,
                         size=(c['width'], c['height']), raw=info))
    return cams


def export_repo_dat(cams, out_dir):
    """Write handpose3d-native camera_parameters files (c{i}.dat / rot_trans_c{i}.dat)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        i = cam['idx']
        with open(out / f'c{i}.dat', 'w') as f:
            f.write('intrinsic:\n')
            for row in cam['K']:
                f.write(' '.join(str(v) for v in row) + ' \n')
            f.write('distortion:\n')
            f.write(' '.join(str(v) for v in cam['D']) + ' \n')
        with open(out / f'rot_trans_c{i}.dat', 'w') as f:
            f.write('R:\n')
            for row in cam['R']:
                f.write(' '.join(str(v) for v in row) + ' \n')
            f.write('T:\n')
            for v in cam['t'].ravel():
                f.write(str(v) + ' \n')
    print(f'[export] wrote c*/rot_trans_c*.dat to {out}')


def triangulate_dlt(Ps, pts2d):
    """Multi-view DLT (>=2 views). Ps: list of 3x4, pts2d: list of (u, v). Returns (3,)."""
    A = []
    for P, (u, v) in zip(Ps, pts2d):
        A.append(v * P[2] - P[1])
        A.append(P[0] - u * P[2])
    A = np.asarray(A)
    _, _, Vh = np.linalg.svd(A.T @ A)
    X = Vh[-1]
    return X[:3] / X[3]


def undistort_px(pt, K, D):
    """Distortion-corrected pixel coords (same K), consistent with solvePnP-based calib."""
    src = np.array([[pt]], dtype=np.float64)
    dst = cv.undistortPoints(src, K, D, P=K)
    return dst[0, 0]


def detect_hands_2d(hands, frame_bgr):
    """Run MediaPipe on one BGR frame.

    Returns ((21,2) float pixels or None, landmarks, (handed_int, score)):
    handed_int: -1 no hand, 0 Left, 1 Right — ACTUAL handedness. MediaPipe labels
    assume a mirrored (selfie) image; our frames are not mirrored, so the label
    is swapped here. Chirality check: right hand => thumb MCP (kpt 2) has y < 0
    in the palm_frame() coordinates.
    """
    rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None, (-1, 0.0)
    lm = res.multi_hand_landmarks[0]
    cls = res.multi_handedness[0].classification[0]
    handed = (1 if cls.label == 'Left' else 0, float(cls.score))
    h, w = frame_bgr.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in lm.landmark], dtype=np.float64)
    return pts, lm, handed


def triangulate_frame(cams, pts_per_cam):
    """pts_per_cam: list of (21,2) or None.

    Returns (p3d (21,3) world(=cam0) meters NaN if <2 views,
             reproj_err (21,) mean px residual over used views (NaN if no 3D),
             n_views).
    """
    p3d = np.full((21, 3), np.nan)
    reproj = np.full(21, np.nan)
    views = [i for i, p in enumerate(pts_per_cam) if p is not None]
    if len(views) < 2:
        return p3d, reproj, len(views)
    und = {i: np.array([undistort_px(pt, cams[i]['K'], cams[i]['D']) for pt in pts_per_cam[i]])
           for i in views}
    for k in range(21):
        Ps = [cams[i]['P'] for i in views]
        uv = [und[i][k] for i in views]
        p3d[k] = triangulate_dlt(Ps, uv)
        errs = []
        for P, (u, v) in zip(Ps, uv):
            proj = P @ np.append(p3d[k], 1.0)
            errs.append(np.hypot(proj[0] / proj[2] - u, proj[1] / proj[2] - v))
        reproj[k] = np.mean(errs)
    return p3d, reproj, len(views)


def palm_frame(p3d):
    """T_world_palm (4,4) from triangulated keypoints, NaN if no 3D.

    origin = wrist (kpt 0); x = wrist->middle_MCP (9); z = unit cross(wrist->index_MCP(5),
    wrist->ring_MCP(13)) orthogonalized — points out of the PALM for a right hand
    (out of the back for a left hand); y = z × x.
    """
    T = np.full((4, 4), np.nan)
    if np.isnan(p3d[[0, 5, 9, 13]]).any():
        return T
    w = p3d[0]
    ex = p3d[9] - w
    ex /= np.linalg.norm(ex)
    n = np.cross(p3d[5] - w, p3d[13] - w)
    ez = n - n.dot(ex) * ex
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = ex, ey, ez, w
    T[3] = [0, 0, 0, 1]
    return T


def reproject(cams, p3d, cam_idx):
    """Project (21,3) world points into camera cam_idx pixels (no distortion)."""
    cam = cams[cam_idx]
    Xc = (cam['R'] @ p3d.T + cam['t'])
    uv = (cam['K'] @ Xc).T
    return uv[:, :2] / uv[:, 2:3]


def draw_overlay(frame, lm, p3d, cams, cam_idx, mp_drawing, mp_hands):
    if lm is not None:
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
    if not np.isnan(p3d).all():
        uv = reproject(cams, p3d, cam_idx)
        for u, v in uv:
            if np.isfinite(u) and np.isfinite(v):
                cv.circle(frame, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)
    cv.putText(frame, f'cam{cam_idx}', (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


FINGER_COLORS = {  # bone index ranges per finger for the 3D plot
    'thumb': ('tab:red', [0, 1, 2, 3]),
    'index': ('tab:blue', [4, 5, 6, 7]),
    'middle': ('tab:green', [8, 9, 10, 11]),
    'ring': ('tab:orange', [12, 13, 14, 15]),
    'pinky': ('tab:purple', [16, 17, 18, 19, 20]),
}


class LivePlot3D:
    """Live 3D hand skeleton in cam0 frame, plotted as (x, z, -y) so up is up."""

    def __init__(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig = plt.figure('handpose3d 3D (cam0 frame)', figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.lines = []
        bone_color = ['gray'] * len(HAND_BONES)
        for color, idxs in FINGER_COLORS.values():
            for bi in idxs:
                bone_color[bi] = color
        for c in bone_color:
            self.lines.append(self.ax.plot([], [], [], 'o-', ms=2.5, lw=2, color=c)[0])
        self.ax.set_xlim(-0.3, 0.3)
        self.ax.set_ylim(0.0, 0.6)
        self.ax.set_zlim(-0.3, 0.3)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('z depth [m]')
        self.ax.set_zlabel('-y up [m]')
        self.ax.set_title('wrist: -')
        self.fig.canvas.draw()
        self.plt.show(block=False)

    def update(self, p3d):
        if np.isnan(p3d).all():
            self.ax.set_title('wrist: (hand not triangulated)')
        else:
            for line, (a, b) in zip(self.lines, HAND_BONES):
                line.set_data([p3d[a, 0], p3d[b, 0]], [p3d[a, 2], p3d[b, 2]])
                line.set_3d_properties([-p3d[a, 1], -p3d[b, 1]])
            w = p3d[0]
            self.ax.set_title(f'wrist: [{w[0]: .3f} {w[1]: .3f} {w[2]: .3f}] m')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def write_kpts(path, kpts_list):
    """One line per frame, space-separated floats (read back by show_3d_hands.read_keypoints)."""
    with open(path, 'w') as f:
        for frame_kpts in kpts_list:
            flat = np.asarray(frame_kpts, dtype=float).ravel()
            f.write(' '.join(f'{v:.6f}' for v in flat) + '\n')


def _reset_d405(rs, serials=None, wait=8):
    """hardware_reset D405s (all, or only given serials) and wait for re-enumeration."""
    for d in rs.context().query_devices():
        if 'D405' not in d.get_info(rs.camera_info.name):
            continue
        if serials is None or d.get_info(rs.camera_info.serial_number) in serials:
            d.hardware_reset()
    time.sleep(wait)


def _usb_speeds(rs, serials):
    """{serial: usb_type_descriptor} for connected requested devices."""
    out = {}
    for d in rs.context().query_devices():
        ser = d.get_info(rs.camera_info.serial_number)
        if ser in serials and d.supports(rs.camera_info.usb_type_descriptor):
            out[ser] = d.get_info(rs.camera_info.usb_type_descriptor)
    return out


def open_realsense_pipelines(cams, reset=False, with_ir=False, emitter=False):
    import pyrealsense2 as rs
    serials = [c['serial'] for c in cams]
    if reset:
        print('[rs] hardware reset all D405 ...')
        _reset_d405(rs)

    # On USB 2.x the D405 hides IR2 640x480@30 -> "Couldn't resolve requests".
    # A hardware_reset re-enumerates back to USB 3.x (verified 2026-06-11).
    for attempt in range(2):
        speeds = _usb_speeds(rs, serials)
        missing = [s for s in serials if s not in speeds]
        if missing:
            raise RuntimeError(f'D405 not connected: {missing} (found: {speeds})')
        slow = [s for s, u in speeds.items() if not u.startswith('3')]
        if not slow:
            break
        print(f'[rs] USB {speeds} — {len(slow)} cam(s) on USB2, resetting to recover USB3 ...')
        _reset_d405(rs, serials=slow)
    else:
        raise RuntimeError(
            f'cameras still on USB2 after reset: {_usb_speeds(rs, serials)} — '
            'replug the USB3 cables/ports (doc: ports 4-1 / 4-5 / 4-2)')
    print(f"[rs] USB speeds: {_usb_speeds(rs, serials)}")

    ctx = rs.context()
    pipes = []
    for cam in cams:  # start one at a time (simultaneous startup is flaky, see doc 2026-05-01)
        cfg = rs.config()
        cfg.enable_device(cam['serial'])
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FPS)
        if with_ir:
            cfg.enable_stream(rs.stream.infrared, 1, FRAME_W, FRAME_H, rs.format.y8, FPS)
            cfg.enable_stream(rs.stream.infrared, 2, FRAME_W, FRAME_H, rs.format.y8, FPS)
        profile = None
        for attempt in range(2):
            try:
                pipe = rs.pipeline(ctx)
                profile = pipe.start(cfg)
                break
            except RuntimeError as e:  # documented hang fix: reset this device, retry once
                if attempt == 1:
                    raise
                print(f"[rs] cam{cam['idx']} start failed ({e}) — resetting that device ...")
                _reset_d405(rs, serials=[cam['serial']])
                ctx = rs.context()
        # guarded like capture_3cam_d405_with_ffs_depth.py (no-op on D405: passive stereo,
        # no projector — but keeps stereo pattern-free on emitter-equipped models too)
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter else 0.0)
        pipes.append(pipe)
        print(f"[rs] cam{cam['idx']} ({cam['serial']}) started"
              + (' +IR stereo' if with_ir else ''))
        time.sleep(0.5)
    for _ in range(15):  # AE warmup
        for p in pipes:
            p.wait_for_frames(5000)
    return pipes


def grab_frames(pipes, with_ir=False):
    """Returns (colors, irLs, irRs); irLs/irRs are None when with_ir is False."""
    colors, irLs, irRs = [], [], []
    for p in pipes:
        fs = p.wait_for_frames(5000)
        colors.append(np.asanyarray(fs.get_color_frame().get_data()).copy())
        if with_ir:
            irLs.append(np.asanyarray(fs.get_infrared_frame(1).get_data()).copy())
            irRs.append(np.asanyarray(fs.get_infrared_frame(2).get_data()).copy())
    return colors, (irLs if with_ir else None), (irRs if with_ir else None)


def run(cams, frame_source, args, n_frames=None):
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = [mp_hands.Hands(min_detection_confidence=args.min_det_conf, max_num_hands=1,
                            min_tracking_confidence=0.5) for _ in range(N_CAMS)]

    plot3d = LivePlot3D() if args.plot3d else None
    recorder = None
    raw_writers = None
    need_vis = (not args.no_gui) or args.record

    kpts_2d = [[] for _ in range(N_CAMS)]
    kpts_3d = []
    take = dict(t_wall=[], kpts3d_world=[], kpts2d_px=[], det_mask=[],
                handed=[], handed_score=[], reproj_err_px=[], n_views=[],
                T_world_palm=[])
    frame_idx = 0
    ir_writers = None
    try:
        while n_frames is None or frame_idx < n_frames:
            pkt = frame_source()
            if pkt is None:
                break
            frames, irLs, irRs = pkt
            t_wall = time.time()
            if args.save and not args.no_video:
                # write BEFORE draw_overlay mutates the frames; frame k <-> take.npz row k
                if raw_writers is None:
                    out = Path(args.save)
                    out.mkdir(parents=True, exist_ok=True)
                    fourcc = cv.VideoWriter_fourcc(*'MJPG')
                    raw_writers = [cv.VideoWriter(str(out / f'cam{i}.avi'), fourcc, FPS,
                                                  (f.shape[1], f.shape[0]))
                                   for i, f in enumerate(frames)]
                    if irLs is not None:
                        ir_writers = [(cv.VideoWriter(str(out / f'cam{i}_irL.avi'), fourcc,
                                                      FPS, (g.shape[1], g.shape[0]), False),
                                       cv.VideoWriter(str(out / f'cam{i}_irR.avi'), fourcc,
                                                      FPS, (g.shape[1], g.shape[0]), False))
                                      for i, g in enumerate(irLs)]
                for wtr, f in zip(raw_writers, frames):
                    wtr.write(f)
                if ir_writers is not None:
                    for (wl, wr), gl, gr in zip(ir_writers, irLs, irRs):
                        wl.write(gl)
                        wr.write(gr)
            pts_per_cam, lms, handeds = [], [], []
            for i, frame in enumerate(frames):
                pts, lm, handed = detect_hands_2d(hands[i], frame)
                pts_per_cam.append(pts)
                lms.append(lm)
                handeds.append(handed)
            p3d, reproj, n_views = triangulate_frame(cams, pts_per_cam)

            for i in range(N_CAMS):
                kpts_2d[i].append(pts_per_cam[i] if pts_per_cam[i] is not None
                                  else np.full((21, 2), -1.0))
            kpts_3d.append(np.where(np.isnan(p3d), -1.0, p3d))

            take['t_wall'].append(t_wall)
            take['kpts3d_world'].append(p3d)
            take['kpts2d_px'].append([p if p is not None else np.full((21, 2), np.nan)
                                      for p in pts_per_cam])
            take['det_mask'].append([p is not None for p in pts_per_cam])
            take['handed'].append([h[0] for h in handeds])
            take['handed_score'].append([h[1] for h in handeds])
            take['reproj_err_px'].append(reproj)
            take['n_views'].append(n_views)
            take['T_world_palm'].append(palm_frame(p3d))

            if not np.isnan(p3d).all():
                wrist = p3d[0]
                print(f'frame {frame_idx:5d}  views={n_views}  '
                      f'wrist(m)=[{wrist[0]: .3f} {wrist[1]: .3f} {wrist[2]: .3f}]  '
                      f'reproj={np.nanmean(reproj):.1f}px')
            else:
                print(f'frame {frame_idx:5d}  views={n_views}  (need >=2 views)')

            if need_vis:
                vis = []
                for i, f in enumerate(frames):
                    f = draw_overlay(f, lms[i], p3d, cams, i, mp_drawing, mp_hands)
                    ok = pts_per_cam[i] is not None
                    cv.putText(f, 'HAND' if ok else 'no hand', (8, 48),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 0) if ok else (0, 0, 255), 2)
                    vis.append(f)
                tiled = np.hstack(vis)
                status = (f'views={n_views}  3D OK' if not np.isnan(p3d).all()
                          else f'views={n_views}  need >=2 views')
                cv.putText(tiled, status, (8, FRAME_H - 12),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if args.record:
                    if recorder is None:
                        recorder = cv.VideoWriter(args.record, cv.VideoWriter_fourcc(*'mp4v'),
                                                  FPS, (tiled.shape[1], tiled.shape[0]))
                    recorder.write(tiled)
                if not args.no_gui:
                    cv.imshow('handpose3d D405 (q to quit)', tiled)
                    if cv.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
            if plot3d is not None:
                plot3d.update(p3d)
            frame_idx += 1
    except KeyboardInterrupt:
        print(f'\n[stop] Ctrl+C — finalizing take ({frame_idx} frames) ...')
    finally:
        for h in hands:
            h.close()
        if recorder is not None:
            recorder.release()
            print(f'[record] wrote {args.record}')
        if raw_writers is not None:
            for wtr in raw_writers:
                wtr.release()
            print(f'[save] raw videos -> {args.save}/cam{{0,1,2}}.avi '
                  f'(MJPG, frame k = take.npz row k, true timing in t_wall)')
        if ir_writers is not None:
            for wl, wr in ir_writers:
                wl.release()
                wr.release()
            print(f'[save] IR stereo -> {args.save}/cam{{0,1,2}}_ir{{L,R}}.avi '
                  f'(Y8 rectified pair, emitter {"ON" if args.emitter else "OFF"}; '
                  f'for Fast-FoundationStereo)')
        cv.destroyAllWindows()

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(N_CAMS):
            write_kpts(out / f'kpts_cam{i}.dat', kpts_2d[i])
        write_kpts(out / 'kpts_3d.dat', kpts_3d)
        np.savez_compressed(
            out / 'take.npz',
            t_wall=np.array(take['t_wall']),                       # (N,) unix sec
            kpts3d_world=np.array(take['kpts3d_world']),           # (N,21,3) m, NaN=missing
            kpts2d_px=np.array(take['kpts2d_px']),                 # (N,3,21,2) px, NaN=missing
            det_mask=np.array(take['det_mask'], dtype=bool),       # (N,3)
            handed=np.array(take['handed'], dtype=np.int8),        # (N,3) -1/0=L/1=R
            handed_score=np.array(take['handed_score']),           # (N,3)
            reproj_err_px=np.array(take['reproj_err_px']),         # (N,21) mean over views
            n_views=np.array(take['n_views'], dtype=np.int8),      # (N,)
            T_world_palm=np.array(take['T_world_palm']),           # (N,4,4) NaN=missing
        )
        meta = dict(
            created=time.strftime('%Y-%m-%d %H:%M:%S'),
            world_frame='cam0 color optical frame (= optimized_3cam_extrinsics.json world)',
            units='meters; 2D in pixels on 640x480 color (undistortion applied only inside DLT)',
            landmark_order='MediaPipe Hands 21 (0=wrist, 4=thumb_tip, 8=index_tip, ...)',
            palm_frame=palm_frame.__doc__,
            extrinsics=str(args.extrinsics),
            intrinsics_session=str(args.intrinsics_session_resolved),
            cameras=[dict(idx=c['idx'], serial=c['serial'], K=c['K'].tolist(),
                          D=c['D'].tolist(), R_world2cam=c['R'].tolist(),
                          t_world2cam=c['t'].ravel().tolist(),
                          # for Fast-FoundationStereo on cam{i}_ir{L,R}.avi:
                          # depth = ir_left.fx * stereo_baseline_m / disparity,
                          # then ir_left_to_color + world_to_cam to reach world frame
                          ir_left=c['raw'].get('ir_left'),
                          stereo_baseline_m=c['raw'].get('baseline_m'),
                          ir_left_to_color=c['raw'].get('ir_left_to_color')) for c in cams],
            mediapipe=dict(min_detection_confidence=args.min_det_conf,
                           min_tracking_confidence=0.5, max_num_hands=1),
            retarget_hint=('dex-retargeting: joint_pos = (R_palm.T @ (kpts3d_world - wrist).T).T '
                           'then @ OPERATOR2MANO_RIGHT; Bidex-style PyBullet IK targets: '
                           'kpts [3,4 | 6,8 | 10,12 | 14,16 | 18,20] in palm frame'),
            raw_video=(None if args.no_video else dict(
                files=[f'cam{i}.avi' for i in range(N_CAMS)], codec='MJPG',
                note='frame index k corresponds to take.npz row k; container is nominal '
                     f'{FPS} fps — true per-frame timestamps are t_wall')),
            ir_stereo=(dict(
                files=[f'cam{i}_ir{s}.avi' for i in range(N_CAMS) for s in 'LR'],
                codec='MJPG mono (Y8)', emitter='ON' if args.emitter else 'OFF',
                note='rectified stereo pair, frame k = take.npz row k; '
                     'inputs for Fast-FoundationStereo depth')
                if ir_writers is not None else None),
        )
        with open(out / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f'[save] {len(kpts_3d)} frames -> {out}/  (take.npz + meta.json + '
              f'kpts_cam*.dat / kpts_3d.dat for show_3d_hands.py)')
    return kpts_3d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extrinsics', default=str(DEFAULT_EXTRINSICS))
    ap.add_argument('--intrinsics-session', default=None,
                    help='capture dir with cam{0,1,2}/intrinsics.json (default: latest in d405_3cam)')
    ap.add_argument('--session', default=None,
                    help='offline mode: run on cam{0,1,2}/color.png of this capture dir')
    ap.add_argument('--save', default=None, help='output dir for kpts_cam*.dat / kpts_3d.dat')
    ap.add_argument('--export-dat', default=None,
                    help='write handpose3d-native camera_parameters .dat files to this dir and exit')
    ap.add_argument('--reset', action='store_true', help='hardware-reset D405s before start')
    ap.add_argument('--no-gui', action='store_true')
    ap.add_argument('--plot3d', action='store_true', help='live 3D skeleton plot (matplotlib)')
    ap.add_argument('--record', default=None, metavar='FILE.mp4',
                    help='record tiled overlay video (works with --no-gui too)')
    ap.add_argument('--min-det-conf', type=float, default=0.5,
                    help='MediaPipe min_detection_confidence (lower if hands are missed)')
    ap.add_argument('--no-video', action='store_true',
                    help='skip raw per-camera video (cam*.avi) when using --save')
    ap.add_argument('--no-ir', action='store_true',
                    help='skip IR stereo recording (cam*_ir{L,R}.avi) when using --save')
    ap.add_argument('--emitter', action='store_true',
                    help='IR emitter ON if the device has one (default OFF; D405 has none)')
    args = ap.parse_args()

    intr_session = args.intrinsics_session or args.session or latest_session_dir()
    args.intrinsics_session_resolved = intr_session
    cams = load_calibration(args.extrinsics, intr_session)
    print(f'[calib] extrinsics: {args.extrinsics}')
    print(f'[calib] intrinsics: {intr_session}')
    for cam in cams:
        print(f"  cam{cam['idx']}: serial={cam['serial']} fx={cam['K'][0,0]:.1f} "
              f"baseline_from_cam0={np.linalg.norm(cam['t']):.3f} m")

    if args.export_dat:
        export_repo_dat(cams, args.export_dat)
        return

    if args.session:
        frames = [cv.imread(str(Path(args.session) / f'cam{i}/color.png')) for i in range(N_CAMS)]
        if any(f is None for f in frames):
            sys.exit(f'missing cam*/color.png under {args.session}')
        run(cams, lambda fs=[(frames, None, None)]: fs.pop() if fs else None, args, n_frames=1)
        if not args.no_gui:
            cv.waitKey(0)
    else:
        with_ir = bool(args.save) and not args.no_video and not args.no_ir
        pipes = open_realsense_pipelines(cams, reset=args.reset, with_ir=with_ir,
                                         emitter=args.emitter)
        try:
            run(cams, lambda: grab_frames(pipes, with_ir), args)
        finally:
            for p in pipes:
                p.stop()


if __name__ == '__main__':
    main()
