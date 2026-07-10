"""
VIO Pipeline Dashboard
Run with: streamlit run app.py
"""

import os, re, glob, time
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pymap3d
from scipy.interpolate import CubicSpline
import yaml
from PIL import Image as PILImage
from PIL.ExifTags import TAGS
from datetime import datetime
import zipfile, tempfile, io

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIO Pipeline Dashboard",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stProgress .st-bo { background-color: #7c3aed; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CAMERA INTRINSICS
#  These are NOT hardcoded anymore — they are derived per-dataset
#  in the Step 1 pipeline below (see `resolve_camera_intrinsics`),
#  with a fallback chain: DJI XMP > EXIF 35mm-equivalent > manual
#  sidebar input. The globals below are just placeholders until
#  a dataset is loaded and the pipeline runs.
# ─────────────────────────────────────────────────────────────
SCALE       = 0.25   # working-resolution scale factor (not camera-specific)
fx = fy = cx = cy = None
K           = None
dist_coeffs = np.zeros(5)  # default: no distortion correction unless DJI DewarpData is found

# Known DJI Mavic 3 Enterprise DewarpData distortion coefficients (k1,k2,p1,p2,k3).
# Used only when a dataset is confirmed to be from this exact camera (via XMP
# CalibratedFocalLength). Other datasets get zero distortion by default since
# we have no way to derive real lens distortion without a calibration step.
DJI_DEWARP_DIST = np.array([-0.112575240, 0.014874430, -0.027064110, 0.0, -0.000085720])

MIN_RELIABLE_INLIERS = 25

lk_params = dict(
    winSize  = (31, 31),
    maxLevel = 5,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# ─────────────────────────────────────────────────────────────
#  PIPELINE FUNCTIONS  (exact copies from Module 5)
# ─────────────────────────────────────────────────────────────
def parse_gps_coord(dms, ref):
    d, m, s = dms
    v = d + m / 60.0 + s / 3600.0
    return -v if ref in ['S', 'W'] else v

def parse_xmp(raw):
    s = raw.find(b'<x:xmpmeta')
    e = raw.find(b'</x:xmpmeta') + 12
    if s == -1:
        return {}
    xmp    = raw[s:e].decode('utf-8', errors='ignore')
    fields = ['AbsoluteAltitude', 'RelativeAltitude', 'GimbalPitchDegree',
              'FlightRollDegree', 'FlightYawDegree', 'FlightPitchDegree',
              'FlightXSpeed', 'FlightYSpeed', 'FlightZSpeed',
              'GpsLatitude', 'GpsLongitude',
              'CalibratedFocalLength', 'CalibratedOpticalCenterX', 'CalibratedOpticalCenterY']
    result = {}
    for f in fields:
        match = re.search(rf'drone-dji:{f}="([^"]+)"', xmp)
        if match:
            try:    result[f] = float(match.group(1))
            except: result[f] = match.group(1)
    return result

def _to_float(v):
    """EXIF values sometimes come back as PIL.TiffImagePlugin.IFDRational — normalize to float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def extract_metadata(p, frame_index=0):
    """
    Extract everything we can from an image: DJI XMP if present, otherwise
    falls back to standard EXIF. Nothing here raises — if a field can't be
    found, it's simply left out of the record and handled later with a
    fallback / manual override / clear warning, rather than crashing or
    silently using a DJI-only constant.
    """
    rec = {'filepath': p, 'filename': os.path.basename(p)}

    # ── Frame ID + timestamp: try DJI filename convention first ────────────
    m = re.search(r'DJI_(\d{8})(\d{6})_(\d+)_V', p)
    if m:
        rec['frame_id']       = int(m.group(3))
        rec['timestamp']      = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        rec['timestamp_src']  = 'dji_filename'

    img      = PILImage.open(p)
    img_w, img_h = img.size
    rec['img_width']  = img_w
    rec['img_height'] = img_h

    exif_raw = img._getexif()
    if exif_raw:
        exif = {TAGS.get(k, k): v for k, v in exif_raw.items()}
        gps  = exif.get('GPSInfo', {})
        if gps:
            try:
                rec['exif_lat'] = parse_gps_coord(gps[2], gps[1])
                rec['exif_lon'] = parse_gps_coord(gps[4], gps[3])
            except Exception:
                pass
            if 6 in gps:  # GPS altitude (meters, ASL) — standard EXIF tag, not DJI-specific
                rec['exif_gps_alt'] = _to_float(gps[6])

        rec['exif_focal_mm']   = _to_float(exif.get('FocalLength'))
        rec['exif_focal_35mm'] = _to_float(exif.get('FocalLengthIn35mmFilm'))

        # Fallback timestamp if DJI filename pattern didn't match
        if 'timestamp' not in rec:
            dt_raw = exif.get('DateTimeOriginal') or exif.get('DateTime')
            if dt_raw:
                try:
                    rec['timestamp']     = datetime.strptime(dt_raw, '%Y:%m:%d %H:%M:%S')
                    rec['timestamp_src'] = 'exif_datetime'
                except ValueError:
                    pass

    # Last-resort fallback: no DJI filename, no EXIF datetime — synthesize
    # an evenly-spaced timestamp from frame order so the pipeline can still
    # run (flagged later as an estimate, not real timing).
    if 'timestamp' not in rec:
        rec['timestamp']     = datetime(2000, 1, 1) + pd.Timedelta(seconds=frame_index)
        rec['timestamp_src'] = 'synthetic_sequential'

    if 'frame_id' not in rec:
        rec['frame_id'] = frame_index

    with open(p, 'rb') as f:
        raw = f.read()
    rec.update(parse_xmp(raw))  # no-ops harmlessly if there's no DJI XMP block
    return rec

def load_undistort(p):
    img  = cv2.imread(p)
    img  = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.undistort(gray, K, dist_coeffs)

def track_pair(p1, p2, orb):
    f1 = load_undistort(p1)
    f2 = load_undistort(p2)
    kp, _ = orb.detectAndCompute(f1, None)
    if len(kp) < 10:
        return None
    pts1 = np.float32([k.pt for k in kp]).reshape(-1, 1, 2)
    pts2, status, err = cv2.calcOpticalFlowPyrLK(f1, f2, pts1, None, **lk_params)
    good = status.flatten() == 1
    gp1, gp2 = pts1[good].reshape(-1, 2), pts2[good].reshape(-1, 2)
    gerr     = err.flatten()[good]
    if len(gp1) < 8:
        return None
    _, mask = cv2.findFundamentalMat(gp1, gp2, cv2.FM_RANSAC, 1.5, 0.99)
    if mask is None:
        return None
    inl              = mask.flatten() == 1
    ip1, ip2, ierr   = gp1[inl], gp2[inl], gerr[inl]
    if len(ip1) >= 16:
        thresh   = np.percentile(ierr, 50)
        keep     = ierr <= thresh
        ip1, ip2 = ip1[keep], ip2[keep]
    flow = ip2 - ip1
    return {
        'dx'       : float(np.median(flow[:, 0])),
        'dy'       : float(np.median(flow[:, 1])),
        'n_inliers': len(ip1),
        'reliable' : len(ip1) >= MIN_RELIABLE_INLIERS
    }

def resolve_camera_intrinsics(df, img_w, img_h, manual_focal_px=0.0):
    """
    Fallback chain for camera intrinsics, in order of accuracy:
      1. Manual override from sidebar (if the user typed a value in)
      2. DJI XMP CalibratedFocalLength / CalibratedOpticalCenter (exact, dataset-specific)
      3. EXIF FocalLengthIn35mmFilm -> pixel estimate via the 35mm-equivalent
         sensor-width approximation (approximate, works for most cameras/phones)
      4. Nothing usable found -- caller must ask the user for a value.
    Returns (fx_full, fy_full, cx_full, cy_full, dist_coeffs_full, source_label, ok)
    ok=False means no usable focal length was found at all.
    """
    first = df.iloc[0]

    if manual_focal_px and manual_focal_px > 0:
        fx_full = fy_full = float(manual_focal_px)
        cx_full, cy_full = img_w / 2.0, img_h / 2.0
        return fx_full, fy_full, cx_full, cy_full, np.zeros(5), "manual override", True

    if pd.notna(first.get('CalibratedFocalLength')):
        fx_full = fy_full = float(first['CalibratedFocalLength'])
        cx_full = float(first.get('CalibratedOpticalCenterX', img_w / 2.0))
        cy_full = float(first.get('CalibratedOpticalCenterY', img_h / 2.0))
        return fx_full, fy_full, cx_full, cy_full, DJI_DEWARP_DIST, "DJI calibrated (XMP)", True

    if pd.notna(first.get('exif_focal_35mm')) and first.get('exif_focal_35mm', 0) > 0:
        fx_full = fy_full = (img_w * float(first['exif_focal_35mm'])) / 36.0
        cx_full, cy_full = img_w / 2.0, img_h / 2.0
        return fx_full, fy_full, cx_full, cy_full, np.zeros(5), "estimated from EXIF 35mm-equivalent focal length", True

    return None, None, img_w / 2.0, img_h / 2.0, np.zeros(5), "none found", False


def resolve_altitude_m(row, manual_altitude_m=0.0):
    """Height-above-ground fallback chain for the pixel-to-meters conversion."""
    if pd.notna(row.get('RelativeAltitude')):
        return float(row['RelativeAltitude']), 'dji_relative_altitude'
    if manual_altitude_m and manual_altitude_m > 0:
        return float(manual_altitude_m), 'manual_override'
    if pd.notna(row.get('exif_gps_alt')):
        return float(row['exif_gps_alt']), 'exif_gps_altitude_asl'
    return None, 'unavailable'


def resolve_yaw_deg(row, fallback_bearing=0.0, manual_yaw_deg=0.0):
    """Heading fallback chain used to rotate pixel displacement into geographic NED."""
    if pd.notna(row.get('FlightYawDegree')):
        return float(row['FlightYawDegree']), 'dji_flight_yaw'
    if manual_yaw_deg:
        return float(manual_yaw_deg), 'manual_override'
    return float(fallback_bearing), 'estimated_from_gps_track'


def gps_track_bearing(lat1, lon1, lat2, lon2):
    """Compass bearing between two GPS points -- last-resort heading estimate
    when no flight-yaw telemetry is available."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def pixels_to_ned(dx_px, dy_px, altitude_m, yaw_deg):
    dX  = (dx_px / fx) * altitude_m
    dY  = (dy_px / fy) * altitude_m
    yr  = np.radians(yaw_deg)
    dN  = -(np.cos(yr) * dX + np.sin(yr) * dY)
    dE  = -(-np.sin(yr) * dX + np.cos(yr) * dY)
    return dN, dE

def build_displacements(tracking_results, df, manual_altitude_m=0.0, manual_yaw_deg=0.0):
    """
    Returns (displacements, sources, fallback_notes). fallback_notes records,
    per-pair, which altitude/yaw source was actually used, so the UI can warn
    the user when estimates rather than real telemetry drove the numbers
    (instead of silently reusing DJI-only values on other datasets).
    """
    displacements  = []
    sources        = []
    fallback_notes = []
    has_imu_speed  = 'FlightXSpeed' in df.columns and 'FlightYSpeed' in df.columns

    for i, r in enumerate(tracking_results):
        row = df.iloc[i]
        dt  = float(df.iloc[i + 1]['time_s'] - row['time_s'])

        if has_imu_speed and pd.notna(row.get('FlightXSpeed')) and pd.notna(row.get('FlightYSpeed')):
            imu_dn = float(row['FlightXSpeed']) * dt
            imu_de = float(row['FlightYSpeed']) * dt
        else:
            # No IMU/flight-speed telemetry in this dataset -- can't estimate
            # motion for frames where vision tracking also failed. Assume
            # zero motion rather than crashing; this is flagged to the user.
            imu_dn = imu_de = 0.0

        if r is None:
            dn, de  = imu_dn, imu_de
            src_tag = 'imu_only'
            fallback_notes.append('no_telemetry_zero_motion' if not has_imu_speed else 'ok')
        else:
            altitude_m, alt_src = resolve_altitude_m(row, manual_altitude_m)
            if altitude_m is None:
                # No altitude available from any source -- vision displacement
                # can't be scaled to meters. Fall back to IMU (or zero) for
                # this pair rather than raising an exception.
                dn, de  = imu_dn, imu_de
                src_tag = 'imu_only'
                fallback_notes.append('no_altitude_available')
            else:
                bearing = 0.0
                if 'GpsLatitude' in df.columns and i + 1 < len(df):
                    if pd.notna(row.get('GpsLatitude')) and pd.notna(df.iloc[i+1].get('GpsLatitude')):
                        bearing = gps_track_bearing(
                            row['GpsLatitude'], row['GpsLongitude'],
                            df.iloc[i+1]['GpsLatitude'], df.iloc[i+1]['GpsLongitude']
                        )
                yaw_deg, yaw_src = resolve_yaw_deg(row, bearing, manual_yaw_deg)
                vo_dn, vo_de = pixels_to_ned(r['dx'], r['dy'], altitude_m, yaw_deg)

                if alt_src != 'dji_relative_altitude' or yaw_src != 'dji_flight_yaw':
                    fallback_notes.append(f'alt={alt_src},yaw={yaw_src}')
                else:
                    fallback_notes.append('ok')

                if r.get('reliable', True):
                    dn, de  = vo_dn, vo_de
                    src_tag = 'vision'
                else:
                    dn      = 0.5 * vo_dn + 0.5 * imu_dn
                    de      = 0.5 * vo_de + 0.5 * imu_de
                    src_tag = 'blended'
        displacements.append((dn, de))
        sources.append(src_tag)
    return displacements, sources, fallback_notes


def track_stereo_pnp(p0_left, p1_left, p0_right, K, baseline, orb, bf):
    """
    Real stereo visual odometry for one frame pair: triangulate 3D points
    from the stereo pair at frame i (metric scale, thanks to the known
    baseline -- fully independent of the IMU, unlike the essential-matrix +
    IMU-scale approach this replaced), track those same 2D points forward
    into frame i+1 via optical flow, then solve PnP to recover the metric
    relative pose directly. This is standard stereo-VO practice and the
    reason EuRoC (a stereo dataset) is a good fit for it.

    Returns (forward_disp, right_disp) in the camera/body frame of frame i,
    or None if there weren't enough reliable correspondences.
    """
    img0_l = load_undistort(p0_left)
    img0_r = load_undistort(p0_right)
    img1_l = load_undistort(p1_left)

    kpL, desL = orb.detectAndCompute(img0_l, None)
    kpR, desR = orb.detectAndCompute(img0_r, None)
    if desL is None or desR is None or len(kpL) < 8 or len(kpR) < 8:
        return None
    matches = bf.match(desL, desR)
    if len(matches) < 8:
        return None
    ptsL = np.float32([kpL[m.queryIdx].pt for m in matches])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in matches])

    # Rectified-stereo consistency filter: true correspondences lie on the
    # same row with positive, bounded disparity. Rejects the false
    # appearance-based matches that otherwise creep in when the scene has
    # repetitive-looking features.
    dy   = np.abs(ptsL[:, 1] - ptsR[:, 1])
    disp = ptsL[:, 0] - ptsR[:, 0]
    good_match = (dy < 1.5) & (disp > 0.5) & (disp < 60.0)
    ptsL, ptsR = ptsL[good_match], ptsR[good_match]
    if len(ptsL) < 8:
        return None

    P_left  = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = K @ np.hstack([np.eye(3), np.array([[-baseline], [0], [0]])])
    pts4d = cv2.triangulatePoints(P_left, P_right, ptsL.T, ptsR.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    valid = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < 100)
    pts3d, ptsL = pts3d[valid], ptsL[valid]
    if len(pts3d) < 6:
        return None

    pts2d_cv = ptsL.reshape(-1, 1, 2).astype(np.float32)
    pts2d_next, status, _ = cv2.calcOpticalFlowPyrLK(img0_l, img1_l, pts2d_cv, None, **lk_params)
    good = status.flatten() == 1
    obj_pts = pts3d[good].astype(np.float64)
    img_pts = pts2d_next[good].reshape(-1, 2).astype(np.float64)
    if len(obj_pts) < 6:
        return None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, K, None,
                                                   reprojectionError=3.0, confidence=0.99)
    if not ok:
        return None

    R_rel, _ = cv2.Rodrigues(rvec)
    cam_center_rel = -R_rel.T @ tvec.flatten()  # frame i+1's position in frame i's camera frame
    n_inliers = len(inliers) if inliers is not None else len(obj_pts)
    return {'forward': cam_center_rel[2], 'right': cam_center_rel[0], 'n_inliers': int(n_inliers)}


def load_euroc_dataset(euroc_root):
    """
    Parses a standard EuRoC MAV STEREO sequence (mav0/cam0, mav0/cam1,
    mav0/imu0, mav0/state_groundtruth_estimate0, calibration YAMLs).

    Returns (df, imu_df, real_north, real_east, K, dist_coeffs, image_paths,
    image_paths_right, baseline, info).

    Ground truth (real_north/real_east) is returned ONLY for scoring after
    the fact -- it is never used by the estimator itself, same philosophy
    as the DJI path's real GPS.
    """
    cam0_csv = os.path.join(euroc_root, 'cam0', 'data.csv')
    cam1_csv = os.path.join(euroc_root, 'cam1', 'data.csv')
    imu_csv  = os.path.join(euroc_root, 'imu0', 'data.csv')
    gt_csv   = os.path.join(euroc_root, 'state_groundtruth_estimate0', 'data.csv')
    if not os.path.exists(gt_csv):
        gt_csv = os.path.join(euroc_root, 'vicon0', 'data.csv')
    yaml0_path = os.path.join(euroc_root, 'cam0', 'sensor.yaml')
    yaml1_path = os.path.join(euroc_root, 'cam1', 'sensor.yaml')
    has_stereo = os.path.exists(cam1_csv) and os.path.exists(yaml1_path)

    cam0_df = pd.read_csv(cam0_csv)
    cam0_df.columns = ['timestamp_ns', 'filename']
    t0_ns = cam0_df['timestamp_ns'].iloc[0]
    cam0_df['time_s'] = (cam0_df['timestamp_ns'] - t0_ns) / 1e9
    image_paths = [os.path.join(euroc_root, 'cam0', 'data', fn) for fn in cam0_df['filename']]

    image_paths_right, baseline = None, None
    if has_stereo:
        cam1_df = pd.read_csv(cam1_csv)
        cam1_df.columns = ['timestamp_ns', 'filename']
        image_paths_right = [os.path.join(euroc_root, 'cam1', 'data', fn) for fn in cam1_df['filename']]
        with open(yaml1_path) as f:
            yaml1_lines = [l for l in f if not l.strip().startswith('%')]
        calib1  = yaml.safe_load(''.join(yaml1_lines))
        T_BS1   = np.array(calib1['T_BS']['data']).reshape(4, 4)
        baseline = float(abs(T_BS1[0, 3]))

    df = pd.DataFrame({
        'frame_id': range(len(cam0_df)),
        'time_s':   cam0_df['time_s'].values,
    })

    imu_raw = pd.read_csv(imu_csv)
    imu_raw.columns = ['timestamp_ns', 'wx', 'wy', 'wz', 'ax', 'ay', 'az']
    imu_df = pd.DataFrame({
        'time_s': (imu_raw['timestamp_ns'] - t0_ns) / 1e9,
        'wx': imu_raw['wx'], 'wy': imu_raw['wy'], 'wz': imu_raw['wz'],
        'ax': imu_raw['ax'], 'ay': imu_raw['ay'], 'az': imu_raw['az'],
    })

    gt_raw = pd.read_csv(gt_csv)
    gt_raw.columns = ['timestamp_ns', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz',
                       'vx', 'vy', 'vz', 'bwx', 'bwy', 'bwz', 'bax', 'bay', 'baz'][:len(gt_raw.columns)]
    gt_raw['time_s'] = (gt_raw['timestamp_ns'] - t0_ns) / 1e9

    # Auto-detect the horizontal plane: the two position axes with the
    # highest variance, treating the lowest-variance axis as "vertical"
    # (assumes roughly level-ish flight) -- avoids hardcoding an axis
    # convention that differs between EuRoC sequences.
    pos_cols  = ['px', 'py', 'pz']
    variances = gt_raw[pos_cols].var()
    vertical_axis    = variances.idxmin()
    horizontal_axes  = [c for c in pos_cols if c != vertical_axis]

    cs_gt_n = CubicSpline(gt_raw['time_s'], gt_raw[horizontal_axes[0]])
    cs_gt_e = CubicSpline(gt_raw['time_s'], gt_raw[horizontal_axes[1]])
    real_north = cs_gt_n(df['time_s'].values)
    real_east  = cs_gt_e(df['time_s'].values)

    with open(yaml0_path) as f:
        yaml0_lines = [l for l in f if not l.strip().startswith('%')]
    calib0 = yaml.safe_load(''.join(yaml0_lines))
    fx_c, fy_c, cx_c, cy_c = calib0['intrinsics']
    dist = np.array(calib0.get('distortion_coefficients', [0, 0, 0, 0]) + [0])[:5]
    K = np.array([[fx_c, 0, cx_c], [0, fy_c, cy_c], [0, 0, 1]], dtype=np.float64)

    info = {
        'vertical_axis': vertical_axis,
        'horizontal_axes': horizontal_axes,
        'n_frames': len(df),
        'n_imu_samples': len(imu_df),
        'duration_s': float(df['time_s'].iloc[-1]),
        'has_stereo': has_stereo,
        'baseline_m': baseline,
    }
    return df, imu_df, real_north, real_east, K, dist, image_paths, image_paths_right, baseline, info


def build_displacements_euroc(df, image_paths, image_paths_right, K, baseline, orb):
    """
    Real stereo VO displacement builder: triangulate + PnP per frame pair
    (see track_stereo_pnp) -- metric scale from the stereo baseline, fully
    independent of the IMU. Rotates each frame's body-frame (forward,right)
    displacement into the local world frame using gyro-integrated yaw (real
    gyro, not ground truth -- "use telemetry, not the answer key", same
    philosophy as the DJI pipeline's FlightYawDegree usage).

    Returns (dN_list, dE_list, sources, inlier_log, cs_yaw_est) -- these are
    INCREMENTAL per-frame displacements, not a cumulative trajectory; the
    caller (pure_vio-style dead reckoning, or the EKF) decides how to
    accumulate/use them.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    frame_times = df['time_s'].values
    n_pairs = len(image_paths) - 1

    dN_list, dE_list, sources, inlier_log = [], [], [], []
    for i in range(n_pairs):
        if image_paths_right is not None:
            r = track_stereo_pnp(image_paths[i], image_paths[i+1], image_paths_right[i],
                                  K, baseline, orb, bf)
        else:
            r = None
        inlier_log.append(r['n_inliers'] if r else 0)
        if r is None:
            dN_list.append(0.0); dE_list.append(0.0)
            sources.append('no_features_zero_motion')
        else:
            dN_list.append(r['forward'])  # placeholder -- rotated into world N/E below
            dE_list.append(r['right'])
            sources.append('stereo_vo')

    return dN_list, dE_list, sources, inlier_log


def rotate_body_disps_to_world(fwd_list, right_list, frame_times, gyro_z, imu_time_s):
    """Rotate body-frame (forward,right) displacements into the local world
    frame using gyro-integrated yaw (real raw gyro -- see note above)."""
    yaw_est = np.concatenate([[0.0], np.cumsum(gyro_z[:-1] * np.diff(imu_time_s))])
    cs_yaw_est = CubicSpline(imu_time_s, yaw_est)
    dN_list, dE_list = [], []
    for i, (fwd, right) in enumerate(zip(fwd_list, right_list)):
        yaw_mid = float(cs_yaw_est((frame_times[i] + frame_times[i+1]) / 2))
        dN_list.append(np.cos(yaw_mid) * fwd - np.sin(yaw_mid) * right)
        dE_list.append(np.sin(yaw_mid) * fwd + np.cos(yaw_mid) * right)
    return dN_list, dE_list, cs_yaw_est


def align_trajectory(N, E, ref_N, ref_E, real_north, real_east):
    """
    Standard rotation-only trajectory alignment, the same practice used by
    EuRoC's own devkit / the `evo` toolkit's `evo_ape --align` -- monocular
    or visual-inertial odometry has no absolute heading reference (no
    magnetometer, no ground truth fed to the estimator), so it produces a
    correctly-shaped trajectory in its own arbitrarily-rotated local frame.
    A single rigid rotation (fit here from `ref_N/ref_E`, then applied to
    whatever trajectory is passed in) is the standard, honest way to score
    it against a world-frame ground truth -- this is a SCORING step, not
    something fed back into the estimator.
    """
    oN, oE = real_north[0], real_east[0]
    n_, e_ = ref_N - oN, ref_E - oE
    tn, te = real_north - oN, real_east - oE
    theta = np.arctan2(np.sum(te*n_ - tn*e_), np.sum(tn*n_ + te*e_))
    c, s = np.cos(theta), np.sin(theta)
    n2, e2 = N - oN, E - oE
    return oN + c*n2 - s*e2, oE + s*n2 + c*e2, np.degrees(theta)

def rts_smooth_1d(z, u, gps_mask, Q, R, x0):
    """
    1D Kalman filter + Rauch-Tung-Striebel (RTS) smoother for a single axis
    (north or east). Replaces the old power-law drift ramp (Scenarios A/B/C)
    with the same batch estimator used in the notebook pipeline (ph4/ph5):

    - Predict step uses the VIO displacement as the motion/control input.
    - Update step fuses in the GPS position (when available) with a Kalman
      gain reflecting how much we trust VIO vs. GPS at that instant.
    - The backward RTS pass lets GPS fixes on BOTH sides of a denied window
      correct the trajectory, instead of only the fix waiting at the far end
      of a one-sided ramp -- this removes both the "drift grows uniformly"
      assumption and the velocity kink at the window edges that any single-
      sided ramp (linear or power-law) leaves behind.

    z        : GPS position per frame (used only where gps_mask is True)
    u        : VIO displacement per frame transition, u[k] = step from k to k+1
    gps_mask : boolean array, True = GPS available at this frame
    Q        : process noise variance (VIO displacement error per step, m^2)
    R        : measurement noise variance (GPS accuracy, m^2)
    x0       : initial position (frame 0 is always GPS-anchored)
    """
    n = len(gps_mask)
    xp = np.zeros(n); Pp = np.zeros(n)
    xf = np.zeros(n); Pf = np.zeros(n)
    xf[0] = x0
    Pf[0] = 1e-6   # frame 0 is always GPS-anchored, treat as (near-)exact

    for k in range(1, n):
        xp[k] = xf[k-1] + u[k-1]
        Pp[k] = Pf[k-1] + Q
        if gps_mask[k]:
            K = Pp[k] / (Pp[k] + R)
            xf[k] = xp[k] + K * (z[k] - xp[k])
            Pf[k] = (1 - K) * Pp[k]
        else:
            xf[k] = xp[k]
            Pf[k] = Pp[k]

    xs = np.zeros(n); Ps = np.zeros(n)
    xs[-1] = xf[-1]
    Ps[-1] = Pf[-1]
    for k in range(n - 2, -1, -1):
        C = Pf[k] / Pp[k+1]
        xs[k] = xf[k] + C * (xs[k+1] - xp[k+1])
        Ps[k] = Pf[k] + C**2 * (Ps[k+1] - Pp[k+1])

    return xs


def gps_aided_vio(displacements, gps_mask, real_north, real_east):
    """
    GPS-aided VIO estimator using a Kalman filter + RTS smoother per axis
    (north/east independently). See rts_smooth_1d for the estimator details.

    Q (process noise) is estimated empirically from how much the VIO
    displacement disagrees with the true GPS step over the whole flight --
    a data-driven noise estimate rather than a hand-picked constant.
    R (measurement noise) uses a typical consumer-GPS horizontal accuracy;
    tune this if your dataset's GPS spec is known to differ.
    """
    dn = np.array([d[0] for d in displacements])
    de = np.array([d[1] for d in displacements])

    true_step_n = np.diff(real_north)
    true_step_e = np.diff(real_east)
    Q_north = max(np.var(dn - true_step_n), 1e-6)
    Q_east  = max(np.var(de - true_step_e), 1e-6)

    R_gps = 3.0 ** 2   # ~3 m 1-sigma consumer-GPS horizontal accuracy assumption

    eN = rts_smooth_1d(real_north, dn, gps_mask, Q_north, R_gps, real_north[0])
    eE = rts_smooth_1d(real_east,  de, gps_mask, Q_east,  R_gps, real_east[0])
    return eN, eE

def pure_vio(displacements, real_north, real_east):
    n     = len(displacements) + 1
    eN    = np.zeros(n)
    eE    = np.zeros(n)
    eN[0] = real_north[0]
    eE[0] = real_east[0]
    for i, (dn, de) in enumerate(displacements):
        eN[i + 1] = eN[i] + dn
        eE[i + 1] = eE[i] + de
    return eN, eE

def ned_to_gps(north_arr, east_arr, origin_lat, origin_lon, origin_alt):
    lats, lons, alts = pymap3d.ned2geodetic(
        n=north_arr, e=east_arr,
        d=np.zeros(len(north_arr)),
        lat0=origin_lat, lon0=origin_lon, h0=origin_alt
    )
    return lats, lons, alts


def ekf_core(t_imu, accel_meas, gyro_meas, yaw0, dt,
             meas_frame_idx, meas_N, meas_E, meas_R_list,
             p0_n, p0_e):
    """
    SHARED EKF core used by both the DJI and EuRoC paths -- this is the
    actual reusable sensor-fusion architecture, not two separate copies.
    It's dataset-agnostic: it doesn't know or care whether the IMU stream
    is real or synthetic, or whether the position corrections come from
    real GPS or from a monocular vision estimate. That's entirely decided
    by the caller (see ekf_imu_fusion_dji / ekf_imu_fusion_euroc below).

    State: x = [pN, pE, vN, vE, bias_ax, bias_ay]
    - Propagation runs every IMU sample (real-time INS mechanization).
    - Correction runs whenever a "measurement" is available at the given
      meas_frame_idx (IMU-sample indices), using whatever meas_N/meas_E/
      meas_R_list the caller supplies for that correction.
    """
    n_imu = len(t_imu)
    yaw_est = np.zeros(n_imu); yaw_est[0] = yaw0
    for k in range(1, n_imu):
        yaw_est[k] = yaw_est[k-1] + gyro_meas[k-1] * dt

    x = np.array([p0_n, p0_e, 0.0, 0.0, 0.0, 0.0])
    P = np.diag([1.0, 1.0, 1.0, 1.0, 0.1, 0.1])
    Q = np.diag([1e-3, 1e-3, 1e-2, 1e-2, 1e-6, 1e-6])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    pN_fused = np.zeros(n_imu); pE_fused = np.zeros(n_imu)
    meas_ptr = 0
    n_meas   = len(meas_frame_idx)

    for k in range(n_imu):
        if k > 0:
            yr = yaw_est[k-1]
            ax_corr = accel_meas[k-1, 0] - x[4]
            ay_corr = accel_meas[k-1, 1] - x[5]
            aN = np.cos(yr) * ax_corr - np.sin(yr) * ay_corr
            aE = np.sin(yr) * ax_corr + np.cos(yr) * ay_corr

            # Mean-state update: true nonlinear model (bias already
            # subtracted out above via ax_corr/ay_corr).
            F_simple = np.eye(6); F_simple[0, 2] = dt; F_simple[1, 3] = dt
            x = F_simple @ x + np.array([0.5*aN*dt*dt, 0.5*aE*dt*dt, aN*dt, aE*dt, 0.0, 0.0])

            # Covariance propagation: full Jacobian including bias
            # cross-terms -- without these the bias states never develop
            # covariance with position/velocity, so the filter could never
            # actually learn the bias from position corrections (a common
            # EKF-linearization bug if the Jacobian is skipped).
            c, s = np.cos(yr), np.sin(yr)
            F_jac = np.eye(6); F_jac[0, 2] = dt; F_jac[1, 3] = dt
            F_jac[0, 4], F_jac[0, 5] = -0.5*dt*dt*c,  0.5*dt*dt*s
            F_jac[1, 4], F_jac[1, 5] = -0.5*dt*dt*s, -0.5*dt*dt*c
            F_jac[2, 4], F_jac[2, 5] = -dt*c,  dt*s
            F_jac[3, 4], F_jac[3, 5] = -dt*s, -dt*c
            P = F_jac @ P @ F_jac.T + Q

        pN_fused[k], pE_fused[k] = x[0], x[1]

        if meas_ptr < n_meas and k >= meas_frame_idx[meas_ptr]:
            z = np.array([meas_N[meas_ptr], meas_E[meas_ptr]])
            Rmeas = meas_R_list[meas_ptr]
            S = H @ P @ H.T + Rmeas
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (np.eye(6) - K @ H) @ P
            meas_ptr += 1

    return pN_fused, pE_fused, np.array([x[4], x[5]])


def ekf_imu_fusion_dji(df, real_north, real_east, gps_mask,
                        accel_bias=(0.05, -0.03), accel_noise_std=0.03,
                        gyro_bias_deg=0.2, gyro_noise_std_deg=0.05,
                        imu_rate_hz=100.0, seed=42):
    """
    DJI path, Scenario D — EKF fusion of SYNTHETIC high-rate IMU with
    per-frame GPS corrections (available/denied by gps_mask).

    IMPORTANT HONESTY NOTE: this dataset's metadata (DJI XMP FlightYawDegree,
    FlightXSpeed, etc.) is DJI's own *fused* attitude/speed output, not raw
    accelerometer/gyroscope readings -- those aren't present in the JPEGs at
    all. So the "IMU" here is synthesized from the known GPS ground-truth
    trajectory (cubic-spline position -> differentiate for true accel/gyro,
    then add realistic bias + noise), the same technique used to simulate
    GPS dropout elsewhere in this app. The strapdown mechanization and EKF
    core are the real algorithms -- only the sensor input is simulated.

    Returns (north_fused, east_fused, bias_est_final, imu_only_north, imu_only_east)
    """
    frame_times = df['time_s'].values
    n_frames    = len(frame_times)

    cs_n = CubicSpline(frame_times, real_north)
    cs_e = CubicSpline(frame_times, real_east)
    yaw_rad_series = np.unwrap(np.radians(df['FlightYawDegree'].values))
    cs_yaw = CubicSpline(frame_times, yaw_rad_series)

    dt    = 1.0 / imu_rate_hz
    t_imu = np.arange(frame_times[0], frame_times[-1], dt)
    if len(t_imu) < 2:
        return real_north.copy(), real_east.copy(), np.zeros(2), real_north.copy(), real_east.copy()

    true_n_imu = cs_n(t_imu)
    true_e_imu = cs_e(t_imu)
    true_yaw   = cs_yaw(t_imu)
    aN_true    = cs_n.derivative(2)(t_imu)
    aE_true    = cs_e.derivative(2)(t_imu)
    yaw_rate_true = cs_yaw.derivative(1)(t_imu)

    ax_body_true =  np.cos(true_yaw) * aN_true + np.sin(true_yaw) * aE_true
    ay_body_true = -np.sin(true_yaw) * aN_true + np.cos(true_yaw) * aE_true

    rng = np.random.default_rng(seed)
    ACCEL_BIAS     = np.array(accel_bias)
    GYRO_BIAS_RADS = np.radians(gyro_bias_deg)

    accel_meas = (np.stack([ax_body_true, ay_body_true], axis=1) + ACCEL_BIAS
                  + rng.normal(0, accel_noise_std, size=(len(t_imu), 2)))
    gyro_meas  = (yaw_rate_true + GYRO_BIAS_RADS
                  + rng.normal(0, np.radians(gyro_noise_std_deg), size=len(t_imu)))

    n_imu = len(t_imu)

    # ── Strapdown IMU-only dead reckoning (for comparison / display) ───────
    yaw_est = np.zeros(n_imu); yaw_est[0] = true_yaw[0]
    pN_imu  = np.zeros(n_imu); pE_imu  = np.zeros(n_imu)
    vN_imu  = np.zeros(n_imu); vE_imu  = np.zeros(n_imu)
    pN_imu[0], pE_imu[0] = true_n_imu[0], true_e_imu[0]
    for k in range(1, n_imu):
        yaw_est[k] = yaw_est[k-1] + gyro_meas[k-1] * dt
        yr = yaw_est[k-1]
        aN = np.cos(yr) * accel_meas[k-1, 0] - np.sin(yr) * accel_meas[k-1, 1]
        aE = np.sin(yr) * accel_meas[k-1, 0] + np.cos(yr) * accel_meas[k-1, 1]
        vN_imu[k] = vN_imu[k-1] + aN * dt
        vE_imu[k] = vE_imu[k-1] + aE * dt
        pN_imu[k] = pN_imu[k-1] + vN_imu[k-1] * dt
        pE_imu[k] = pE_imu[k-1] + vE_imu[k-1] * dt

    R_AVAILABLE = np.diag([2.0, 2.0])
    R_DENIED    = np.diag([50.0, 50.0])
    meas_frame_idx, meas_N, meas_E, meas_R_list = [], [], [], []
    for i in range(1, n_frames):
        k = int(np.searchsorted(t_imu, frame_times[i]))
        if k >= n_imu:
            continue
        meas_frame_idx.append(k)
        meas_N.append(real_north[i]); meas_E.append(real_east[i])
        meas_R_list.append(R_AVAILABLE if gps_mask[i] else R_DENIED)

    pN_fused, pE_fused, bias_final = ekf_core(
        t_imu, accel_meas, gyro_meas, true_yaw[0], dt,
        meas_frame_idx, meas_N, meas_E, meas_R_list,
        true_n_imu[0], true_e_imu[0]
    )

    frame_idx_in_imu = np.clip(np.searchsorted(t_imu, frame_times), 0, n_imu - 1)
    north_fused_frames = pN_fused[frame_idx_in_imu]
    east_fused_frames  = pE_fused[frame_idx_in_imu]
    north_imu_frames   = pN_imu[frame_idx_in_imu]
    east_imu_frames    = pE_imu[frame_idx_in_imu]

    return north_fused_frames, east_fused_frames, bias_final, north_imu_frames, east_imu_frames


def ekf_imu_fusion_euroc(t_imu, accel_meas, gyro_z, frame_times,
                          dN_list, dE_list, vision_available,
                          p0n, p0e, vo_measurement_std=0.15):
    """
    EuRoC path -- EKF fusion of REAL raw IMU (imu0/data.csv) with the
    per-frame stereo-VO INCREMENTAL displacement, with support for a
    simulated vision dropout (vision_available[i] == False -> no
    correction for that frame, IMU propagates alone through the gap).

    IMPORTANT: the vision "measurement" fed to each correction is built as
    (the filter's OWN last corrected position) + (this frame's incremental
    vision displacement) -- NOT as an absolute position from an
    independently-drifting "Pure VO" trajectory. Using an independent
    absolute trajectory as the measurement anchor would silently erase any
    benefit the IMU bridge provides during a dropout, because Pure VO's own
    position stays permanently offset by whatever distance it missed while
    vision was unavailable (it has no way to "recover" a missed distance).
    Anchoring to the filter's own belief is the standard way relative/
    incremental measurements get fused (the same reason wheel-odometry or
    relative-pose measurements are fused this way in real INS/VIO systems).

    Ground truth is never used here -- only for scoring afterward.

    Returns (north_fused, east_fused, bias_est_final)
    """
    dt = float(np.median(np.diff(t_imu))) if len(t_imu) > 1 else 0.005
    n_imu = len(t_imu)

    yaw_est = np.zeros(n_imu)
    for k in range(1, n_imu):
        yaw_est[k] = yaw_est[k-1] + gyro_z[k-1] * dt

    x = np.array([p0n, p0e, 0.0, 0.0, 0.0, 0.0])
    P = np.diag([1.0, 1.0, 1.0, 1.0, 0.1, 0.1])
    Q = np.diag([1e-3, 1e-3, 1e-2, 1e-2, 1e-6, 1e-6])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    R_meas = np.diag([vo_measurement_std**2, vo_measurement_std**2])

    pN_out = np.zeros(n_imu); pE_out = np.zeros(n_imu)
    last_corrected = np.array([p0n, p0e])
    frame_ptr = 1
    n_frames = len(frame_times)

    for k in range(n_imu):
        if k > 0:
            yr = yaw_est[k-1]
            ax_corr = accel_meas[k-1, 0] - x[4]
            ay_corr = accel_meas[k-1, 1] - x[5]
            aN = np.cos(yr) * ax_corr - np.sin(yr) * ay_corr
            aE = np.sin(yr) * ax_corr + np.cos(yr) * ay_corr

            F_simple = np.eye(6); F_simple[0, 2] = dt; F_simple[1, 3] = dt
            x = F_simple @ x + np.array([0.5*aN*dt*dt, 0.5*aE*dt*dt, aN*dt, aE*dt, 0.0, 0.0])

            c, s = np.cos(yr), np.sin(yr)
            F_jac = np.eye(6); F_jac[0, 2] = dt; F_jac[1, 3] = dt
            F_jac[0, 4], F_jac[0, 5] = -0.5*dt*dt*c,  0.5*dt*dt*s
            F_jac[1, 4], F_jac[1, 5] = -0.5*dt*dt*s, -0.5*dt*dt*c
            F_jac[2, 4], F_jac[2, 5] = -dt*c,  dt*s
            F_jac[3, 4], F_jac[3, 5] = -dt*s, -dt*c
            P = F_jac @ P @ F_jac.T + Q

        pN_out[k], pE_out[k] = x[0], x[1]

        if frame_ptr < n_frames and k >= int(np.searchsorted(t_imu, frame_times[frame_ptr])):
            if vision_available[frame_ptr]:
                z = last_corrected + np.array([dN_list[frame_ptr-1], dE_list[frame_ptr-1]])
                S = H @ P @ H.T + R_meas
                K_gain = P @ H.T @ np.linalg.inv(S)
                x = x + K_gain @ (z - H @ x)
                P = (np.eye(6) - K_gain @ H) @ P
            last_corrected = x[:2].copy()
            frame_ptr += 1

    frame_idx_in_imu = np.clip(np.searchsorted(t_imu, frame_times), 0, n_imu - 1)
    return pN_out[frame_idx_in_imu], pE_out[frame_idx_in_imu], np.array([x[4], x[5]])


def make_kml(trajectories):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '<name>VIO Pipeline Results</name>'
    ]
    for t in trajectories:
        lines += [
            '<Placemark>',
            f'  <name>{t["name"]}</name>',
            '  <Style><LineStyle>',
            f'    <color>{t["color"]}</color>',
            '    <width>3</width>',
            '  </LineStyle></Style>',
            '  <LineString><altitudeMode>absolute</altitudeMode><coordinates>'
        ]
        for lat, lon, alt in zip(t['lats'], t['lons'], t['alts']):
            lines.append(f'    {lon:.8f},{lat:.8f},{alt:.2f}')
        lines += ['  </coordinates></LineString>', '</Placemark>']
    lines += ['</Document>', '</kml>']
    return '\n'.join(lines)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚁 VIO Dashboard")
    st.caption("Visual Inertial Odometry Pipeline")
    st.divider()

    st.subheader("📁 Dataset")

    dataset_type = st.radio(
        "Dataset type",
        ["DJI / Geotagged (GPS-aided VIO)", "EuRoC MAV format (Visual-Inertial Odometry)"],
        help=(
            "DJI/Geotagged: any GPS-tagged image sequence -- uses real GPS as ground truth, "
            "optionally with synthetic-IMU EKF fusion (Scenario D). "
            "EuRoC MAV: the standard camera+IMU academic benchmark format -- no GPS at all; "
            "uses REAL raw accelerometer/gyroscope data fused with monocular visual odometry."
        )
    )
    is_euroc = dataset_type.startswith("EuRoC")

    IMG_EXTS = ('jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG')

    def _find_images(root):
        found = []
        for ext in IMG_EXTS:
            found += glob.glob(f"{root}/**/*.{ext}", recursive=True)
        return sorted(set(found))

    def _find_mav0(root):
        """Locate the 'mav0' folder inside an extracted EuRoC zip/folder --
        handles the case where the zip has an extra top-level wrapper dir,
        or where the folder given *is* mav0 itself (has a cam0 subfolder)."""
        if os.path.isdir(os.path.join(root, 'cam0')):
            return root
        hits = glob.glob(f"{root}/**/mav0", recursive=True)
        if hits:
            return hits[0]
        hits = glob.glob(f"{root}/**/cam0", recursive=True)
        if hits:
            return os.path.dirname(hits[0])
        return None

    dataset_dir  = None
    image_paths  = []
    euroc_root   = None

    if not is_euroc:
        # ── DJI / Geotagged path (unchanged from before) ────────────────
        SAMPLE_ZIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "sample_dataset.zip")
        has_sample = os.path.exists(SAMPLE_ZIP_PATH)

        input_options = (["✨ Try sample dataset"] if has_sample else []) + ["Upload ZIP", "Local folder path"]
        upload_mode = st.radio(
            "Input source",
            input_options,
            help="Try the bundled sample instantly, upload your own .zip of images, or point to a local folder"
        )

        if upload_mode == "✨ Try sample dataset":
            if 'sample_extracted_dir' not in st.session_state:
                tmp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(SAMPLE_ZIP_PATH) as zf:
                    zf.extractall(tmp_dir)
                st.session_state['sample_extracted_dir'] = tmp_dir
            image_paths = _find_images(st.session_state['sample_extracted_dir'])
            if image_paths:
                dataset_dir = os.path.dirname(image_paths[0])
                st.success(f"Loaded bundled sample — {len(image_paths)} images, ready to run")
            else:
                st.error("Bundled sample_dataset.zip was found but contained no JPG/PNG images")

        elif upload_mode == "Upload ZIP":
            uploaded = st.file_uploader(
                "Upload dataset ZIP",
                type=["zip"],
                help="ZIP file of images -- DJI JPEGs work best, but any JPG/PNG sequence is accepted"
            )
            if uploaded:
                tmp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(uploaded) as zf:
                    zf.extractall(tmp_dir)
                image_paths = _find_images(tmp_dir)
                if image_paths:
                    dataset_dir = os.path.dirname(image_paths[0])
                    st.success(f"Found {len(image_paths)} images")
                else:
                    st.warning("No JPG/PNG images found in that ZIP")
        else:
            folder = st.text_input(
                "Folder path",
                value="4thAve1",
                help="Path to a folder of images (DJI JPEGs work best, but any JPG/PNG sequence is accepted)"
            )
            if os.path.exists(folder):
                image_paths = _find_images(folder)
                if image_paths:
                    dataset_dir = folder
                    st.success(f"Found {len(image_paths)} images")
                else:
                    st.warning("No JPG/PNG images found in that folder")
            else:
                st.info("Enter a valid folder path")

    else:
        # ── EuRoC MAV format path ────────────────────────────────────────
        st.caption(
            "⚠️ The bundled sample here is a **synthetic** dataset generated in "
            "EuRoC's exact file format (same folder structure, CSV schemas, and "
            "calibration YAML) -- built to demo/test this pipeline without needing "
            "to download a real multi-GB EuRoC sequence. It is not a real MAV "
            "recording. Any real EuRoC sequence (e.g. MH_01_easy from "
            "https://projects.asl.ethz.ch/datasets/) can be dropped in with the "
            "same mav0/ folder structure and will work identically."
        )
        EUROC_SAMPLE_ZIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "sample_euroc.zip")
        has_euroc_sample = os.path.exists(EUROC_SAMPLE_ZIP_PATH)

        euroc_input_options = (["✨ Try synthetic sample"] if has_euroc_sample else []) + ["Upload ZIP", "Local folder path"]
        euroc_upload_mode = st.radio("Input source", euroc_input_options)

        if euroc_upload_mode == "✨ Try synthetic sample":
            if 'euroc_sample_extracted_dir' not in st.session_state:
                tmp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(EUROC_SAMPLE_ZIP_PATH) as zf:
                    zf.extractall(tmp_dir)
                st.session_state['euroc_sample_extracted_dir'] = tmp_dir
            euroc_root = _find_mav0(st.session_state['euroc_sample_extracted_dir'])
            if euroc_root:
                st.success(f"Loaded synthetic EuRoC-format sample from {euroc_root}")
            else:
                st.error("Bundled sample_euroc.zip was found but no mav0/cam0 structure was found inside")

        elif euroc_upload_mode == "Upload ZIP":
            euroc_uploaded = st.file_uploader(
                "Upload EuRoC-format ZIP",
                type=["zip"],
                help="Zip of a mav0/ folder (cam0/, imu0/, state_groundtruth_estimate0/)"
            )
            if euroc_uploaded:
                tmp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(euroc_uploaded) as zf:
                    zf.extractall(tmp_dir)
                euroc_root = _find_mav0(tmp_dir)
                if euroc_root:
                    st.success(f"Found EuRoC-format data at {euroc_root}")
                else:
                    st.warning("No mav0/cam0 structure found in that ZIP")
        else:
            euroc_folder = st.text_input(
                "Folder path",
                value="mav0",
                help="Path to a mav0/ folder (or its parent) containing cam0/, imu0/, state_groundtruth_estimate0/"
            )
            if os.path.exists(euroc_folder):
                euroc_root = _find_mav0(euroc_folder)
                if euroc_root:
                    st.success(f"Found EuRoC-format data at {euroc_root}")
                else:
                    st.warning("No mav0/cam0 structure found in that folder")
            else:
                st.info("Enter a valid folder path")

        if euroc_root:
            image_paths = sorted(glob.glob(f"{euroc_root}/cam0/data/*.png")) or sorted(glob.glob(f"{euroc_root}/cam0/data/*.jpg"))
            dataset_dir = euroc_root

    st.divider()

    if not is_euroc:
        st.subheader("📷 Camera & flight overrides")
        st.caption(
            "Only needed for non-DJI datasets that lack calibrated metadata. "
            "Leave at 0 to auto-detect from EXIF/XMP -- the pipeline will tell "
            "you what it used after running."
        )
        manual_focal_px = st.number_input(
            "Focal length override (px, full-res)", min_value=0.0, value=0.0, step=1.0,
            help="Only used if the dataset has no DJI CalibratedFocalLength and no EXIF 35mm-equivalent focal length"
        )
        manual_altitude_m = st.number_input(
            "Assumed altitude AGL override (m)", min_value=0.0, value=0.0, step=1.0,
            help="Only used for frames missing DJI RelativeAltitude and EXIF GPS altitude"
        )
        manual_yaw_deg = st.number_input(
            "Assumed constant heading override (deg from North)", value=0.0, step=1.0,
            help="Only used for frames missing DJI FlightYawDegree; otherwise heading is estimated from the GPS track"
        )

        st.divider()
        st.subheader("⚙️ GPS Dropout Settings")

        # Scale dropout windows to whatever dataset is actually loaded, instead of
        # hardcoded frame numbers that assume a 53-frame dataset. Without this, a
        # smaller sample (e.g. the bundled 10-frame demo) would have every
        # "dropout" window fall outside the available frames entirely, silently
        # making every GPS-aided scenario identical to "Full GPS" (0.00 error).
        n_loaded  = len(image_paths) if image_paths else 53
        max_frame = max(n_loaded - 2, 1)  # leave at least 1 frame pair after the window

        # Proportional to the original 53-frame dataset's 5-frame / 15-frame windows
        b_len = max(1, round(n_loaded * 5 / 53))
        c_len = max(2, round(n_loaded * 15 / 53))
        default_b_start = min(round(n_loaded * 20 / 53), max_frame)
        default_c_start = min(round(n_loaded * 15 / 53), max_frame)

        dropout_b_start = st.slider("Scenario B — dropout start frame", 0, max_frame, default_b_start)
        dropout_b_end   = st.slider("Scenario B — dropout end frame", dropout_b_start + 1,
                                     max_frame + b_len, min(dropout_b_start + b_len, max_frame + b_len))
        dropout_c_start = st.slider("Scenario C — dropout start frame", 0, max_frame, default_c_start)
        dropout_c_end   = st.slider("Scenario C — dropout end frame", dropout_c_start + 1,
                                     max_frame + c_len, min(dropout_c_start + c_len, max_frame + c_len))

        if n_loaded < 15:
            st.caption(
                f"ℹ️ Small dataset ({n_loaded} frames) — dropout windows scaled "
                f"down automatically to fit within the available frames."
            )

        st.divider()
        st.subheader("🛰️ IMU / EKF Fusion (Scenario D)")
        st.caption(
            "⚠️ This dataset's metadata is DJI's own *fused* attitude/speed output, "
            "not raw accelerometer/gyroscope data — DJI doesn't embed that in JPEGs. "
            "Enabling this simulates realistic raw IMU (with bias + noise) from the "
            "known GPS ground truth, then runs a real Extended Kalman Filter to fuse "
            "it with camera-frame corrections. The fusion algorithm is real; the "
            "sensor input is synthetic for this dataset."
        )
        enable_ekf = st.checkbox("Enable Scenario D — EKF-fused IMU", value=False)
        if enable_ekf:
            ekf_accel_noise = st.slider("Simulated accelerometer noise (m/s² std)", 0.0, 0.2, 0.03, 0.01)
            ekf_accel_bias  = st.slider("Simulated accelerometer bias magnitude (m/s²)", 0.0, 0.3, 0.05, 0.01)
            ekf_imu_rate    = st.select_slider("Simulated IMU rate (Hz)", options=[20, 50, 100, 200], value=100)
        else:
            ekf_accel_noise, ekf_accel_bias, ekf_imu_rate = 0.03, 0.05, 100

    else:
        # ── EuRoC-specific settings ──────────────────────────────────────
        st.subheader("🛰️ Visual-Inertial Fusion (Stereo)")
        st.caption(
            "EuRoC has no GPS at all -- this is genuine camera+IMU-only VIO. "
            "Real accelerometer/gyroscope readings (from imu0/data.csv) are fused "
            "with a STEREO visual-odometry position estimate: metric scale comes "
            "from triangulating cam0+cam1 (independent of the IMU, unlike a "
            "monocular approach that would need IMU-integrated distance for scale). "
            "Ground truth (state_groundtruth_estimate0) is used only to score "
            "accuracy afterward -- it is never fed into the estimator."
        )
        vo_measurement_std = st.slider(
            "Assumed VO measurement uncertainty (m, std)", 0.05, 2.0, 0.15, 0.05,
            help="How much the EKF trusts each frame's stereo VO position estimate "
                 "relative to the IMU propagation. Lower = trust vision more."
        )

        st.divider()
        st.subheader("⚙️ Vision Dropout Settings")
        st.caption(
            "EuRoC has no GPS to drop out -- the equivalent real-world failure "
            "mode for a camera is feature-tracking loss (motion blur, occlusion, "
            "textureless surfaces). This simulates that: for the chosen frame "
            "range, Pure VO has to assume zero motion (it has no way to \"guess\" "
            "the missed distance), while the EKF keeps propagating through the "
            "gap using real IMU data, then re-corrects once vision resumes."
        )
        n_loaded_euroc = 20  # updated to the real frame count once the dataset loads, below
        euroc_drop_start = st.slider("Vision dropout start frame", 0, 18, 6)
        euroc_drop_end   = st.slider("Vision dropout end frame", euroc_drop_start + 1, 19, min(euroc_drop_start + 5, 19))
        # Camera/flight manual overrides and GPS-dropout scenarios don't apply
        # to EuRoC (no GPS, and intrinsics come from sensor.yaml) -- keep the
        # downstream code's variable names satisfied with harmless defaults.
        manual_focal_px = manual_altitude_m = manual_yaw_deg = 0.0
        dropout_b_start = dropout_b_end = dropout_c_start = dropout_c_end = 0
        enable_ekf = True  # EuRoC's "Scenario D" (EKF-fused VIO) is the whole point
        ekf_accel_noise, ekf_accel_bias, ekf_imu_rate = 0.03, 0.05, 100

    st.divider()
    run_btn = st.button(
        "▶  Run Pipeline",
        type="primary",
        disabled=(len(image_paths) == 0),
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────────────────────
st.title("Visual Inertial Odometry Pipeline")
st.caption("Works with DJI drone footage or any geotagged JPG/PNG sequence · GPS-aided VIO with drift correction")

if not image_paths:
    st.info("👈  Upload a dataset or enter a folder path in the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────
#  RUN PIPELINE ON BUTTON PRESS
# ─────────────────────────────────────────────────────────────
if run_btn:
    t_start = time.time()

    # ── Step 1: Metadata ────────────────────────────────────
    st.subheader("Step 1 / 4 — Extracting metadata")

    if is_euroc:
        prog1 = st.progress(0.2, text="Parsing EuRoC cam0/cam1/imu0/groundtruth CSVs + calibration YAML...")
        df, imu_df, real_north, real_east, K, dist_coeffs, image_paths, image_paths_right, baseline, euroc_info = \
            load_euroc_dataset(dataset_dir)
        prog1.progress(1.0, text="Done")
        prog1.empty()

        # This dataset provides real intrinsics directly -- no fallback chain
        # needed, and there's no working-resolution downscale since the
        # bundled sample images are already small.
        SCALE = 1.0
        fx, fy = K[0, 0], K[1, 1]

        if not euroc_info['has_stereo']:
            st.error(
                "No cam1/ folder or calibration found -- this pipeline needs a "
                "STEREO EuRoC sequence (cam0 + cam1) to recover metric scale via "
                "triangulation. Monocular-only EuRoC sequences aren't supported "
                "by this path."
            )
            st.stop()

        st.success(f"✓ Metadata extracted — {euroc_info['n_frames']} frames, "
                   f"{euroc_info['duration_s']:.1f}s, {euroc_info['n_imu_samples']} raw IMU samples, "
                   f"stereo baseline {baseline*100:.1f}cm")
        st.caption(
            f"ℹ️ Auto-detected horizontal plane: **{euroc_info['horizontal_axes']}** "
            f"(vertical axis: **{euroc_info['vertical_axis']}**, lowest position variance) "
            f"-- ground truth is used only to score accuracy below, never fed into the estimator."
        )
        # Harmless placeholders so later shared code (e.g. resolved_alt-style
        # DJI-only checks) doesn't need EuRoC-specific branches everywhere.
        focal_src, fallback_msgs = "EuRoC sensor.yaml (real calibration)", []

    else:
        prog1 = st.progress(0, text="Reading EXIF & XMP...")
        records = []
        for idx, p in enumerate(image_paths):
            records.append(extract_metadata(p, frame_index=idx))
            prog1.progress((idx + 1) / len(image_paths),
                           text=f"Frame {idx+1} / {len(image_paths)}")

        df = pd.DataFrame(records).sort_values('frame_id').reset_index(drop=True)
        df['time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

        if 'FlightXSpeed' in df.columns:
            df['speed_ms'] = np.sqrt(
                df['FlightXSpeed'].fillna(0)**2 + df['FlightYSpeed'].fillna(0)**2
                + df.get('FlightZSpeed', pd.Series(0, index=df.index)).fillna(0)**2
            )
        else:
            df['speed_ms'] = np.nan  # no flight-speed telemetry in this dataset
        prog1.empty()

        # Normalize GPS/altitude columns: DJI datasets get GpsLatitude/GpsLongitude/
        # AbsoluteAltitude from XMP; non-DJI datasets only have the standard EXIF
        # exif_lat/exif_lon/exif_gps_alt fields. Use whichever is available so the
        # rest of the pipeline can rely on one consistent set of column names.
        if 'GpsLatitude' not in df.columns:
            df['GpsLatitude'] = np.nan
        if 'GpsLongitude' not in df.columns:
            df['GpsLongitude'] = np.nan
        if 'exif_lat' in df.columns:
            df['GpsLatitude'] = df['GpsLatitude'].fillna(df['exif_lat'])
        if 'exif_lon' in df.columns:
            df['GpsLongitude'] = df['GpsLongitude'].fillna(df['exif_lon'])

        if 'AbsoluteAltitude' not in df.columns:
            df['AbsoluteAltitude'] = np.nan
        if 'exif_gps_alt' in df.columns:
            df['AbsoluteAltitude'] = df['AbsoluteAltitude'].fillna(df['exif_gps_alt'])
        # Last resort so ned2geodetic doesn't crash -- this only affects the
        # vertical/altitude value in exports, not the horizontal error metrics
        # this dashboard is built around.
        df['AbsoluteAltitude'] = df['AbsoluteAltitude'].fillna(0.0)

        # GPS is required: the whole point of this dashboard is comparing VIO
        # estimates against real GPS during simulated dropout, so there's no
        # meaningful way to proceed without per-frame GPS coordinates. Fail
        # clearly here instead of crashing deep inside the math later.
        has_gps = 'GpsLatitude' in df.columns and 'GpsLongitude' in df.columns \
                  and df['GpsLatitude'].notna().all() and df['GpsLongitude'].notna().all()
        if not has_gps:
            st.error(
                "This dataset doesn't have per-frame GPS coordinates in its "
                "metadata (EXIF GPS tags or DJI XMP GpsLatitude/GpsLongitude). "
                "This dashboard needs real GPS on every frame as ground truth to "
                "measure VIO drift against -- without it there's nothing to "
                "compare the vision-based estimate to. Try a dataset with "
                "geotagged images (most drone footage and many phone photos "
                "qualify)."
            )
            st.stop()

        # Resolve RelativeAltitude/FlightYawDegree into normalized columns so
        # every downstream reference (metrics row, CSV export, etc.) can rely on
        # these columns always existing, instead of KeyError-ing on datasets
        # that never had DJI telemetry in the first place. Each row keeps
        # whatever real value it had; only missing rows get backfilled.
        had_dji_altitude = 'RelativeAltitude' in df.columns and df['RelativeAltitude'].notna().all()
        had_dji_yaw      = 'FlightYawDegree' in df.columns and df['FlightYawDegree'].notna().all()

        if 'RelativeAltitude' not in df.columns:
            df['RelativeAltitude'] = np.nan
        resolved_alt = []
        for _, row in df.iterrows():
            val, _src = resolve_altitude_m(row, manual_altitude_m)
            resolved_alt.append(val if val is not None else np.nan)
        df['RelativeAltitude'] = resolved_alt

        if 'FlightYawDegree' not in df.columns:
            df['FlightYawDegree'] = np.nan
        resolved_yaw = []
        for i, row in df.iterrows():
            if pd.notna(row.get('FlightYawDegree')):
                resolved_yaw.append(float(row['FlightYawDegree']))
            elif manual_yaw_deg:
                resolved_yaw.append(float(manual_yaw_deg))
            elif i + 1 < len(df) and pd.notna(df.iloc[i + 1].get('GpsLatitude')):
                resolved_yaw.append(gps_track_bearing(
                    row['GpsLatitude'], row['GpsLongitude'],
                    df.iloc[i + 1]['GpsLatitude'], df.iloc[i + 1]['GpsLongitude']
                ))
            else:
                resolved_yaw.append(0.0)
        df['FlightYawDegree'] = resolved_yaw

        # Camera intrinsics: fallback chain (manual override > DJI XMP > EXIF 35mm)
        img0 = PILImage.open(image_paths[0])
        img_w, img_h = img0.size
        fx_full, fy_full, cx_full, cy_full, dist_coeffs_full, focal_src, focal_ok = \
            resolve_camera_intrinsics(df, img_w, img_h, manual_focal_px)

        if not focal_ok:
            st.error(
                "Couldn't determine focal length from this dataset's metadata "
                "(no DJI calibration, no EXIF 35mm-equivalent focal length), and "
                "no manual override was provided. Enter a value in the sidebar "
                "under 'Camera & flight overrides' -> 'Focal length override "
                "(px, full-res)' and run the pipeline again. As a rough starting "
                f"point, the image width is {img_w}px -- for a typical smartphone "
                "or drone camera, focal length in pixels is often close to the "
                "image width, but this varies by lens."
            )
            st.stop()

        fx, fy = fx_full * SCALE, fy_full * SCALE
        cx, cy = cx_full * SCALE, cy_full * SCALE
        K           = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = dist_coeffs_full

        st.success(f"✓ Metadata extracted — {len(df)} frames, "
                   f"{df['time_s'].iloc[-1]:.0f}s flight")

        fallback_msgs = []
        if focal_src != "DJI calibrated (XMP)":
            fallback_msgs.append(f"📷 Camera focal length: **{focal_src}** ({fx_full:.1f}px full-res) "
                                  f"— not DJI-calibrated, so results will be approximate.")
        if 'timestamp_src' in df.columns and (df['timestamp_src'] == 'synthetic_sequential').any():
            fallback_msgs.append("🕐 Some frame timestamps were synthesized (1s apart) because no "
                                  "DJI filename pattern or EXIF DateTimeOriginal was found — "
                                  "speed/timing-dependent numbers are approximate.")
        if not had_dji_altitude:
            fallback_msgs.append("📏 No DJI relative-altitude telemetry found for some/all frames — "
                                  "falling back to EXIF GPS altitude or your manual override where needed.")
        if not had_dji_yaw:
            fallback_msgs.append("🧭 No DJI flight-yaw telemetry found for some/all frames — "
                                  "heading is being estimated from consecutive GPS points or your manual override instead.")
        if df['RelativeAltitude'].isna().any():
            fallback_msgs.append("❗ Some frames have no altitude from any source (no DJI telemetry, "
                                  "no EXIF GPS altitude, no manual override) — those frame pairs will "
                                  "fall back to IMU/zero motion instead of vision-based displacement.")
        if 'FlightXSpeed' not in df.columns:
            fallback_msgs.append("🚀 No IMU flight-speed telemetry found — frames where vision "
                                  "tracking fails will assume zero motion instead of an IMU estimate.")
    if fallback_msgs:
        with st.expander("⚠️ This dataset is missing some DJI-specific metadata — click for details", expanded=True):
            for msg in fallback_msgs:
                st.warning(msg)

    # Compute origin + real_north/real_east right after metadata, for BOTH
    # dataset types, so every later step can rely on these being set --
    # avoids the fragile pattern of splitting this across two separate
    # "if is_euroc" branches at different points in the script.
    n_frames = len(df)
    if is_euroc:
        # EuRoC's real_north/real_east were already set in Step 1 above
        # (from load_euroc_dataset). No real-world georeference exists for
        # an indoor MAV room, so origin is just (0,0,0) in the room's own
        # local frame.
        ORIGIN_LAT = ORIGIN_LON = ORIGIN_ALT = 0.0
    else:
        ORIGIN_LAT = float(df['GpsLatitude'].iloc[0])
        ORIGIN_LON = float(df['GpsLongitude'].iloc[0])
        ORIGIN_ALT = float(df['AbsoluteAltitude'].iloc[0])
        real_north, real_east, _ = pymap3d.geodetic2ned(
            lat=df['GpsLatitude'].values,
            lon=df['GpsLongitude'].values,
            h=df['AbsoluteAltitude'].values,
            lat0=ORIGIN_LAT, lon0=ORIGIN_LON, h0=ORIGIN_ALT
        )

    # ── Step 2: Feature tracking ────────────────────────────
    st.subheader("Step 2 / 4 — Feature tracking")
    prog2      = st.progress(0, text="Initialising ORB...")
    orb        = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8,
                                edgeThreshold=31, fastThreshold=20)
    n_pairs    = len(image_paths) - 1
    tracking   = []
    inlier_log = []

    if is_euroc:
        st.caption(
            "Stereo tracking: triangulating 3D points from cam0+cam1 at each "
            "frame (real metric scale from the known baseline), then tracking "
            "those points forward via optical flow and solving PnP for the "
            "relative pose -- this is done together with Step 3 below since "
            "the two are the same stereo-VO computation."
        )
        prog2.progress(1.0, text="Done (combined with Step 3 for the stereo pipeline)")
        prog2.empty()
    else:
        for idx in range(n_pairs):
            r = track_pair(image_paths[idx], image_paths[idx + 1], orb)
            tracking.append(r)
            inlier_log.append(r['n_inliers'] if r else 0)
            prog2.progress((idx + 1) / n_pairs,
                           text=f"Tracking pair {idx+1}/{n_pairs} — "
                                f"{inlier_log[-1]} inliers "
                                f"({'✓' if r and r['reliable'] else '⚠'})")
        prog2.empty()
        n_ok = sum(1 for r in tracking if r)
        st.success(f"✓ Tracking complete — {n_ok}/{n_pairs} pairs succeeded, "
                   f"avg {np.mean(inlier_log):.0f} inliers/pair")

    # ── Step 3: Displacements ───────────────────────────────
    st.subheader("Step 3 / 4 — Converting to real-world displacements")
    if is_euroc:
        st.caption(
            "Using stereo triangulation + PnP for metric-scale relative pose "
            "(fully independent of the IMU, unlike a monocular essential-matrix "
            "+ IMU-scale approach) -- the standard technique for a stereo "
            "camera rig, which is what EuRoC actually is."
        )
        fwd_list, right_list, sources, inlier_log = build_displacements_euroc(
            df, image_paths, image_paths_right, K, baseline, orb
        )
        n_pairs = len(fwd_list)
        n_ok = sum(1 for s in sources if s == 'stereo_vo')
        st.success(f"✓ Stereo VO complete — {n_ok}/{n_pairs} pairs succeeded, "
                   f"avg {np.mean(inlier_log):.0f} inliers/pair")

        # Rotate body-frame (forward,right) displacements into the local
        # world frame using gyro-integrated yaw (real gyro).
        frame_times = df['time_s'].values
        dN_list, dE_list, cs_yaw_est_euroc = rotate_body_disps_to_world(
            fwd_list, right_list, frame_times, imu_df['wz'].values, imu_df['time_s'].values
        )
        disp_fallback_notes = ['ok' if s == 'stereo_vo' else s for s in sources]
    else:
        displacements, sources, disp_fallback_notes = build_displacements(
            tracking, df, manual_altitude_m=manual_altitude_m, manual_yaw_deg=manual_yaw_deg
        )
    src_counts = {s: sources.count(s) for s in set(sources)}
    if not is_euroc:
        st.success(f"✓ Displacements built — {src_counts}")

    n_estimated_pairs = sum(1 for n in disp_fallback_notes if n != 'ok')
    if n_estimated_pairs:
        st.caption(
            f"ℹ️ {n_estimated_pairs}/{n_pairs if is_euroc else len(disp_fallback_notes)} frame pairs "
            f"had unreliable tracking and fell back to zero motion for that pair "
            f"{'(or used an estimated non-DJI altitude/heading value)' if not is_euroc else ''}."
        )

    # ── Step 4: VIO estimation ──────────────────────────────
    st.subheader("Step 4 / 4 — Running VIO estimator")

    if is_euroc:
        # ── Simulated vision dropout (mirrors the DJI GPS-dropout demo) ──
        vision_available = np.ones(n_frames, dtype=bool)
        vision_available[euroc_drop_start:euroc_drop_end] = False
        dN_list_dropout = list(dN_list)
        dE_list_dropout = list(dE_list)
        for i in range(euroc_drop_start, min(euroc_drop_end, len(dN_list_dropout))):
            dN_list_dropout[i] = 0.0
            dE_list_dropout[i] = 0.0

        # Pure Vision-only dead reckoning (frozen during the dropout window --
        # it has no way to "recover" a missed distance once tracking resumes)
        north_pure = np.zeros(n_frames); east_pure = np.zeros(n_frames)
        north_pure[0], east_pure[0] = real_north[0], real_east[0]
        for i in range(len(dN_list_dropout)):
            north_pure[i+1] = north_pure[i] + dN_list_dropout[i]
            east_pure[i+1]  = east_pure[i]  + dE_list_dropout[i]
    else:
        north_pure, east_pure = pure_vio(displacements, real_north, real_east)

    def errors(eN, eE):
        return np.sqrt((eN - real_north)**2 + (eE - real_east)**2)

    err_pure = errors(north_pure, east_pure)

    if is_euroc:
        # No GPS exists in this dataset at all, so the DJI path's "GPS
        # available/denied" scenarios A/B/C don't apply. Scenario D fuses
        # REAL IMU with the stereo-VO INCREMENTAL displacement, skipping
        # corrections during the simulated vision-dropout window (IMU
        # bridges the gap alone there, then vision resumes correcting).
        north_A = east_A = north_B = east_B = north_C = east_C = north_pure  # unused placeholders
        err_A = err_B = err_C = err_pure  # unused placeholders, hidden in the UI below

        north_D, east_D, ekf_bias_final = ekf_imu_fusion_euroc(
            imu_df['time_s'].values, imu_df[['ax', 'ay']].values, imu_df['wz'].values,
            df['time_s'].values, dN_list_dropout, dE_list_dropout, vision_available,
            real_north[0], real_east[0], vo_measurement_std=vo_measurement_std
        )

        # Standard rotation-only trajectory alignment (see align_trajectory
        # docstring) -- fit ONCE from the raw Pure VO trajectory, applied
        # identically to both Pure VO and the EKF-fused trajectory so the
        # comparison between them stays apples-to-apples. This is a SCORING
        # step, not fed back into either estimator.
        north_pure_raw, east_pure_raw = north_pure, east_pure
        north_pure, east_pure, align_theta_deg = align_trajectory(
            north_pure_raw, east_pure_raw, north_pure_raw, east_pure_raw, real_north, real_east
        )
        north_D, east_D, _ = align_trajectory(
            north_D, east_D, north_pure_raw, east_pure_raw, real_north, real_east
        )

        err_pure     = errors(north_pure, east_pure)
        err_D        = errors(north_D, east_D)
        ekf_bias_final = np.array(ekf_bias_final)
        north_imu_only = east_imu_only = north_pure  # no separate "IMU-only" baseline for EuRoC
        err_imu_only = err_pure
        lats_D = lons_D = None  # no real-world georeference to convert to
    else:
        mask_A   = np.ones(n_frames, dtype=bool)
        mask_B   = np.ones(n_frames, dtype=bool)
        mask_C   = np.ones(n_frames, dtype=bool)
        mask_B[dropout_b_start:dropout_b_end] = False
        mask_C[dropout_c_start:dropout_c_end] = False

        north_A, east_A       = gps_aided_vio(displacements, mask_A, real_north, real_east)
        north_B, east_B       = gps_aided_vio(displacements, mask_B, real_north, real_east)
        north_C, east_C       = gps_aided_vio(displacements, mask_C, real_north, real_east)

        err_A    = errors(north_A,    east_A)
        err_B    = errors(north_B,    east_B)
        err_C    = errors(north_C,    east_C)

        # ── Scenario D: EKF-fused IMU (uses Scenario B's dropout window) ────
        if enable_ekf:
            north_D, east_D, ekf_bias_final, north_imu_only, east_imu_only = ekf_imu_fusion_dji(
                df, real_north, real_east, mask_B,
                accel_bias=(ekf_accel_bias, -ekf_accel_bias * 0.6),
                accel_noise_std=ekf_accel_noise,
                imu_rate_hz=ekf_imu_rate,
            )
            err_D        = errors(north_D, east_D)
            err_imu_only = errors(north_imu_only, east_imu_only)
            lats_D, lons_D, _ = ned_to_gps(north_D, east_D, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

    # GPS coords for all trajectories
    if is_euroc:
        lats_real = lons_real = alts_real = None
        lats_pure = lons_pure = lats_A = lons_A = lats_B = lons_B = lats_C = lons_C = None
    else:
        lats_real = df['GpsLatitude'].values
        lons_real = df['GpsLongitude'].values
        alts_real = df['AbsoluteAltitude'].values

        lats_pure, lons_pure, _ = ned_to_gps(north_pure, east_pure, ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)
        lats_A,    lons_A,    _ = ned_to_gps(north_A,    east_A,    ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)
        lats_B,    lons_B,    _ = ned_to_gps(north_B,    east_B,    ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)
        lats_C,    lons_C,    _ = ned_to_gps(north_C,    east_C,    ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

    elapsed = time.time() - t_start
    st.success(f"✓ Pipeline complete in {elapsed:.1f}s")
    st.divider()

    # ─────────────────────────────────────────────────────────
    #  METRICS ROW
    # ─────────────────────────────────────────────────────────
    total_dist = np.sum(np.sqrt(np.diff(real_north)**2 + np.diff(real_east)**2))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frames",        f"{n_frames}")
    c2.metric("Distance",      f"{total_dist:.2f} m" if is_euroc else f"{total_dist:.0f} m")
    c3.metric("Duration",      f"{df['time_s'].iloc[-1]:.1f} s")
    if is_euroc:
        c4.metric("IMU samples",  f"{len(imu_df)}")
        c5.metric("IMU rate",     f"{len(imu_df) / max(df['time_s'].iloc[-1], 0.001):.0f} Hz")
    else:
        mean_alt   = df['RelativeAltitude'].mean()
        mean_speed = df['speed_ms'].mean()
        c4.metric("Mean altitude", f"{mean_alt:.1f} m" if pd.notna(mean_alt) else "N/A")
        c5.metric("Avg speed",     f"{mean_speed:.1f} m/s" if pd.notna(mean_speed) else "N/A")

    st.divider()

    # ─────────────────────────────────────────────────────────
    #  TAB LAYOUT
    # ─────────────────────────────────────────────────────────
    if is_euroc:
        tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Trajectory (NED)", "📉 Error Analysis",
            "🔬 Tracking Stats", "💾 Export"
        ])
        tab1 = None
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Live Map", "📈 Trajectory (NED)", "📉 Error Analysis",
            "🔬 Tracking Stats", "💾 Export"
        ])

    # ── TAB 1: Live GPS Map (DJI/geotagged only -- EuRoC has no real-world
    #    georeference, an indoor MAV room isn't on a map) ────────────────
    if tab1 is not None:
     with tab1:
        st.subheader("Flight trajectories on real map")
        fig_map = go.Figure()

        trajectories_map = [
            ("Real GPS",          lats_real, lons_real, "#e74c3c", "circle",   6),
            ("Pure VIO",          lats_pure, lons_pure, "#e67e22", "x",        5),
            ("GPS-aided — Full",  lats_A,    lons_A,    "#2ecc71", "square",   5),
            (f"GPS-aided — {dropout_b_end - dropout_b_start}fr dropout",
                                  lats_B,    lons_B,    "#3498db", "triangle-up", 5),
            (f"GPS-aided — {dropout_c_end - dropout_c_start}fr dropout",
                                  lats_C,    lons_C,    "#9b59b6", "diamond",  5),
        ]
        if enable_ekf:
            trajectories_map.append(
                ("EKF-fused IMU (Scenario D)", lats_D, lons_D, "#1abc9c", "star", 6)
            )

        for name, lats, lons, color, sym, sz in trajectories_map:
            fig_map.add_trace(go.Scattermapbox(
                lat=lats, lon=lons,
                mode="lines+markers",
                marker=dict(size=sz, color=color, symbol=sym),
                line=dict(color=color, width=2),
                name=name,
                hovertemplate=f"<b>{name}</b><br>Lat: %{{lat:.6f}}<br>Lon: %{{lon:.6f}}<extra></extra>"
            ))

        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=float(np.mean(lats_real)), lon=float(np.mean(lons_real))),
                zoom=14
            ),
            height=550,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("All five trajectories overlaid on OpenStreetMap. "
                   "Zoom in to see per-frame position differences between methods.")

    # ── TAB 2: NED Trajectory ──────────────────────────────
    with tab2:
        st.subheader("Trajectory comparison — local frame (metres)")
        fig_ned = go.Figure()

        if is_euroc:
            ned_data = [
                ("Ground truth (scoring only)", real_east, real_north, "#e74c3c", "circle", 4),
                ("Pure Stereo VO (no correction)", east_pure, north_pure, "#e67e22", "x", 3),
                ("EKF-fused VIO (real IMU + vision)", east_D, north_D, "#1abc9c", "star", 4),
            ]
        else:
            ned_data = [
                ("Real GPS",   real_east, real_north, "#e74c3c", "circle",      4),
                ("Pure VIO",   east_pure, north_pure, "#e67e22", "x",           3),
                ("Full GPS",   east_A,    north_A,    "#2ecc71", "square",      3),
                (f"{dropout_b_end - dropout_b_start}fr dropout",
                               east_B,    north_B,    "#3498db", "triangle-up", 3),
                (f"{dropout_c_end - dropout_c_start}fr dropout",
                               east_C,    north_C,    "#9b59b6", "diamond",     3),
            ]
            if enable_ekf:
                ned_data.append(("EKF-fused IMU (D)", east_D, north_D, "#1abc9c", "star", 4))

        for name, ex, no, color, sym, sz in ned_data:
            fig_ned.add_trace(go.Scatter(
                x=ex, y=no,
                mode="lines+markers",
                marker=dict(size=sz, color=color, symbol=sym),
                line=dict(color=color, width=2),
                name=name
            ))

        if not is_euroc:
            # mark denied frames
            for i in range(n_frames):
                if not mask_B[i]:
                    fig_ned.add_trace(go.Scatter(
                        x=[east_B[i]], y=[north_B[i]],
                        mode="markers",
                        marker=dict(size=10, color="#f39c12", symbol="circle-open", line=dict(width=2)),
                        showlegend=(i == dropout_b_start),
                        name="B: GPS denied frame"
                    ))
                if not mask_C[i]:
                    fig_ned.add_trace(go.Scatter(
                        x=[east_C[i]], y=[north_C[i]],
                        mode="markers",
                        marker=dict(size=10, color="#e74c3c", symbol="circle-open", line=dict(width=2)),
                        showlegend=(i == dropout_c_start),
                        name="C: GPS denied frame"
                    ))

        fig_ned.update_layout(
            xaxis_title="East (m)", yaxis_title="North (m)",
            yaxis_scaleanchor="x",
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_ned, use_container_width=True)
        if is_euroc:
            st.caption(
                "ℹ️ Axes are the auto-detected horizontal plane in the room's local "
                "frame (see Step 1 note above) -- there's no real-world lat/lon for "
                "an indoor MAV sequence, so this replaces the Live Map tab entirely."
            )

    # ── TAB 3: Error Analysis ──────────────────────────────
    with tab3:
        st.subheader("Positional error per frame")
        frame_ids = np.arange(n_frames)

        fig_err = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Error vs frame index",
                                                "Inlier count per frame pair"))

        if is_euroc:
            err_series = [
                ("Pure Stereo VO", err_pure, "#e67e22"),
                ("EKF-fused VIO", err_D, "#1abc9c"),
            ]
        else:
            err_series = [
                ("Pure VIO",   err_pure, "#e67e22"),
                ("Full GPS",   err_A,    "#2ecc71"),
                (f"{dropout_b_end - dropout_b_start}fr dropout", err_B, "#3498db"),
                (f"{dropout_c_end - dropout_c_start}fr dropout", err_C, "#9b59b6"),
            ]
            if enable_ekf:
                err_series.append(("EKF-fused IMU (D)", err_D, "#1abc9c"))

        for name, err, color in err_series:
            fig_err.add_trace(
                go.Scatter(x=frame_ids, y=err, name=name,
                           line=dict(color=color, width=2), mode="lines+markers",
                           marker=dict(size=4)),
                row=1, col=1
            )

        # shade denied windows (DJI GPS-dropout scenarios only -- EuRoC has no GPS to deny)
        if not is_euroc:
            for mask, label, color in [
                (mask_B, "B denied", "rgba(52,152,219,0.1)"),
                (mask_C, "C denied", "rgba(155,89,182,0.1)")
            ]:
                denied_frames = [i for i in range(n_frames) if not mask[i]]
                if denied_frames:
                    fig_err.add_vrect(
                        x0=denied_frames[0], x1=denied_frames[-1],
                        fillcolor=color, layer="below", line_width=0,
                        row=1, col=1
                    )

        # inlier count per pair
        fig_err.add_trace(
            go.Bar(x=list(range(n_pairs)), y=inlier_log,
                   name="Inliers",
                   marker_color="#7c3aed", opacity=0.7),
            row=2, col=1
        )
        if not is_euroc:
            fig_err.add_hline(
                y=MIN_RELIABLE_INLIERS, line_dash="dash", line_color="red",
                annotation_text="min reliable threshold",
                row=2, col=1
            )

        fig_err.update_yaxes(title_text="Error (m)", row=1, col=1)
        fig_err.update_yaxes(title_text="Inlier count", row=2, col=1)
        fig_err.update_xaxes(title_text="Frame index", row=2, col=1)
        fig_err.update_layout(height=650, showlegend=True)

        st.plotly_chart(fig_err, use_container_width=True)

        # Accuracy table
        st.subheader("Accuracy summary")
        if is_euroc:
            summary_methods = ["Pure Stereo VO (no correction)", "EKF-fused VIO (real IMU + vision)"]
            summary_errs    = [err_pure, err_D]
        else:
            summary_methods = [
                "Pure VIO (no GPS)",
                "GPS-aided — Full GPS",
                f"GPS-aided — {dropout_b_end - dropout_b_start}fr dropout",
                f"GPS-aided — {dropout_c_end - dropout_c_start}fr dropout",
            ]
            summary_errs = [err_pure, err_A, err_B, err_C]
            if enable_ekf:
                summary_methods += ["IMU-only (strapdown, synthetic)", "EKF-fused IMU (Scenario D)"]
                summary_errs   += [err_imu_only, err_D]

        summary = pd.DataFrame({
            "Method":          summary_methods,
            "Mean error (m)":  [f"{e.mean():.2f}" for e in summary_errs],
            "Max error (m)":   [f"{e.max():.2f}"  for e in summary_errs],
            "Final error (m)": [f"{e[-1]:.2f}"    for e in summary_errs],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if is_euroc:
            improvement_pct = (1 - err_D.mean() / err_pure.mean()) * 100 if err_pure.mean() > 0 else 0.0
            st.caption(
                f"ℹ️ Scenario D fuses **real** accelerometer/gyroscope data (imu0/data.csv) "
                f"with the stereo-VO incremental displacement via an Extended Kalman Filter, "
                f"skipping vision corrections during the simulated dropout window (frames "
                f"{euroc_drop_start}-{euroc_drop_end}) so real IMU alone bridges that gap. "
                f"Pure VO has no such bridge -- it assumes zero motion during the dropout and "
                f"can never recover the missed distance afterward. Ground truth was used only "
                f"to (a) score the errors above and (b) fit a single rigid rotation aligning "
                f"the local VIO frame to true North/East -- monocular/visual-inertial systems "
                f"have no absolute heading reference without a magnetometer, so this one-time "
                f"alignment (fit from Pure VO, applied identically to both trajectories) is "
                f"standard VO/SLAM evaluation practice (e.g. the `evo` toolkit's "
                f"`evo_ape --align`), not something fed into either estimator online."
            )
            if improvement_pct > 5:
                st.success(
                    f"✓ EKF fusion reduced mean error by **{improvement_pct:.0f}%** during the "
                    f"vision dropout, by bridging the gap with real IMU instead of assuming zero "
                    f"motion."
                )
            else:
                st.info(
                    f"Fusion improvement this run: {improvement_pct:.0f}%. Try widening the "
                    f"dropout window in the sidebar (a longer gap gives the IMU bridge more "
                    f"chance to matter, and makes Pure VO's zero-motion assumption look worse "
                    f"by comparison)."
                )
        elif enable_ekf:
            st.caption(
                f"ℹ️ Scenario D fuses **synthesized** high-rate IMU (since this dataset has no raw "
                f"accel/gyro — see sidebar note) with camera-frame corrections via an Extended Kalman "
                f"Filter, using Scenario B's dropout window. Estimated accelerometer bias: "
                f"{ekf_bias_final.round(3)} m/s² (true simulated bias: "
                f"({ekf_accel_bias:.2f}, {-ekf_accel_bias*0.6:.2f})). The EKF jointly estimates this "
                f"bias rather than assuming it's known — same as a real VIO/INS fusion system."
            )

    # ── TAB 4: Tracking Stats ──────────────────────────────
    with tab4:
        st.subheader("Feature tracking diagnostics")

        col_a, col_b = st.columns(2)

        with col_a:
            unique_sources = sorted(set(sources))
            src_counts_plot = {s: sources.count(s) for s in unique_sources}
            src_color_map = {
                'vision':   '#2ecc71',
                'blended':  '#f39c12',
                'imu_only': '#e74c3c',
                'stereo_vo': '#2ecc71',
                'no_features_zero_motion': '#e74c3c',
            }
            fig_src = px.pie(
                values=list(src_counts_plot.values()),
                names=list(src_counts_plot.keys()),
                title="Displacement source breakdown",
                color_discrete_map=src_color_map
            )
            st.plotly_chart(fig_src, use_container_width=True)

        with col_b:
            fig_inl = px.histogram(
                x=inlier_log,
                nbins=20,
                title="Distribution of RANSAC inlier count per frame pair",
                labels={'x': 'Inlier count', 'y': 'Frame pairs'}
            )
            fig_inl.add_vline(x=MIN_RELIABLE_INLIERS, line_dash="dash",
                              line_color="red",
                              annotation_text=f"min reliable ({MIN_RELIABLE_INLIERS})")
            st.plotly_chart(fig_inl, use_container_width=True)

        # Per-pair tracking table
        if is_euroc:
            dn_col = [f"{d:.2f}" for d in dN_list]
            de_col = [f"{d:.2f}" for d in dE_list]
        else:
            dn_col = [f"{d[0]:.2f}" for d in displacements]
            de_col = [f"{d[1]:.2f}" for d in displacements]

        tracking_df = pd.DataFrame({
            "Pair":      [f"{i}→{i+1}" for i in range(n_pairs)],
            "Inliers":   inlier_log,
            "Source":    sources,
            "dn (m)":    dn_col,
            "de (m)":    de_col,
        })
        st.dataframe(tracking_df, use_container_width=True, hide_index=True, height=300)

    # ── TAB 5: Export ──────────────────────────────────────
    with tab5:
        st.subheader("Export pipeline outputs")

        if is_euroc:
            # No real-world georeference for an indoor MAV room -- export
            # local-frame (NED-style) positions instead of lat/lon, and skip
            # KML entirely (Google Earth needs real coordinates).
            results_df = pd.DataFrame({
                'frame_id':          df['frame_id'].values,
                'time_s':            df['time_s'].values,
                'ground_truth_N':    real_north,
                'ground_truth_E':    real_east,
                'vio_pure_N':        north_pure,
                'vio_pure_E':        east_pure,
                'vio_pure_err_m':    err_pure,
                'vio_ekf_fused_N':   north_D,
                'vio_ekf_fused_E':   east_D,
                'vio_ekf_fused_err_m': err_D,
            })
            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download results CSV",
                data=csv_bytes,
                file_name="vio_euroc_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption(
                "Per-frame local-frame (North/East) positions for ground truth, "
                "Pure VIO, and the EKF-fused VIO estimate. No lat/lon or KML export "
                "here -- an indoor MAV sequence has no real-world georeference to plot "
                "on a map."
            )
        else:
            col_x, col_y = st.columns(2)

            with col_x:
                # CSV
                results_df = pd.DataFrame({
                    'frame_id':       df['frame_id'].values,
                    'time_s':         df['time_s'].values,
                    'real_lat':       lats_real,
                    'real_lon':       lons_real,
                    'real_alt':       alts_real,
                    'vio_pure_lat':   lats_pure,
                    'vio_pure_lon':   lons_pure,
                    'vio_pure_err_m': err_pure,
                    'vio_A_lat':      lats_A,
                    'vio_A_lon':      lons_A,
                    'vio_A_err_m':    err_A,
                    'vio_B_lat':      lats_B,
                    'vio_B_lon':      lons_B,
                    'vio_B_err_m':    err_B,
                    'vio_C_lat':      lats_C,
                    'vio_C_lon':      lons_C,
                    'vio_C_err_m':    err_C,
                    'altitude_m':     df['RelativeAltitude'].values,
                    'yaw_deg':        df['FlightYawDegree'].values,
                    'speed_ms':       df['speed_ms'].values,
                })
                if enable_ekf:
                    results_df['vio_D_ekf_lat']    = lats_D
                    results_df['vio_D_ekf_lon']    = lons_D
                    results_df['vio_D_ekf_err_m']  = err_D
                    results_df['vio_D_imu_only_err_m'] = err_imu_only
                csv_bytes = results_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇ Download results CSV",
                    data=csv_bytes,
                    file_name="vio_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption("Full per-frame position table for all four methods. "
                           "Compatible with QGIS, Google Maps, Excel.")

            with col_y:
                # KML
                kml_trajectories = [
                    {"name": "Real GPS (Ground Truth)", "color": "ff0000ff",
                     "lats": lats_real, "lons": lons_real, "alts": alts_real},
                    {"name": "GPS-aided VIO — Full GPS", "color": "ff00ff00",
                     "lats": lats_A,    "lons": lons_A,   "alts": alts_real},
                    {"name": f"GPS-aided — {dropout_b_end-dropout_b_start}fr dropout", "color": "ffff8800",
                     "lats": lats_B,    "lons": lons_B,   "alts": alts_real},
                    {"name": f"GPS-aided — {dropout_c_end-dropout_c_start}fr dropout", "color": "ffff00ff",
                     "lats": lats_C,    "lons": lons_C,   "alts": alts_real},
                    {"name": "Pure VIO (no GPS)", "color": "ff00ffff",
                     "lats": lats_pure, "lons": lons_pure, "alts": alts_real},
                ]
                if enable_ekf:
                    kml_trajectories.append({
                        "name": "EKF-fused IMU (Scenario D, synthetic IMU)", "color": "ff2ecc71",
                        "lats": lats_D, "lons": lons_D, "alts": alts_real
                    })
                kml_str = make_kml(kml_trajectories)
                st.download_button(
                    "⬇ Download KML (Google Earth)",
                    data=kml_str.encode(),
                    file_name="vio_trajectories.kml",
                    mime="application/vnd.google-earth.kml+xml",
                    use_container_width=True
                )
                st.caption("Open in Google Earth to see all five flight paths "
                           "overlaid on satellite imagery of the real location.")
