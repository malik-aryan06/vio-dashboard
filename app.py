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

def gps_aided_vio(displacements, gps_mask, real_north, real_east):
    n     = len(gps_mask)
    eN    = np.zeros(n)
    eE    = np.zeros(n)
    eN[0] = real_north[0]
    eE[0] = real_east[0]
    ds    = None
    ALPHA_POWER = 1.7
    for i in range(1, n):
        if gps_mask[i]:
            if ds is not None:
                dN = real_north[i] - eN[i]
                dE = real_east[i]  - eE[i]
                wl = i - ds
                for j in range(ds + 1, i + 1):
                    a     = ((j - ds) / wl) ** ALPHA_POWER
                    eN[j] += a * dN
                    eE[j] += a * dE
                ds = None
            eN[i] = real_north[i]
            eE[i] = real_east[i]
        else:
            if ds is None:
                ds = i - 1
            dn, de   = displacements[i - 1]
            eN[i]    = eN[i - 1] + dn
            eE[i]    = eE[i - 1] + de
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

    SAMPLE_ZIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "sample_dataset.zip")
    has_sample = os.path.exists(SAMPLE_ZIP_PATH)

    input_options = (["✨ Try sample dataset"] if has_sample else []) + ["Upload ZIP", "Local folder path"]
    upload_mode = st.radio(
        "Input source",
        input_options,
        help="Try the bundled sample instantly, upload your own .zip of images, or point to a local folder"
    )

    dataset_dir  = None
    image_paths  = []

    IMG_EXTS = ('jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG')

    def _find_images(root):
        found = []
        for ext in IMG_EXTS:
            found += glob.glob(f"{root}/**/*.{ext}", recursive=True)
        return sorted(set(found))

    if upload_mode == "✨ Try sample dataset":
        # Bundled with the repo -- extracted once per session and cached so
        # repeat clicks don't re-extract the zip every time.
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

    st.divider()
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

    # ── Step 2: Feature tracking ────────────────────────────
    st.subheader("Step 2 / 4 — Feature tracking")
    prog2      = st.progress(0, text="Initialising ORB...")
    orb        = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8,
                                edgeThreshold=31, fastThreshold=20)
    n_pairs    = len(image_paths) - 1
    tracking   = []
    inlier_log = []

    frame_placeholder = st.empty()

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
    displacements, sources, disp_fallback_notes = build_displacements(
        tracking, df, manual_altitude_m=manual_altitude_m, manual_yaw_deg=manual_yaw_deg
    )
    src_counts = {s: sources.count(s) for s in set(sources)}
    st.success(f"✓ Displacements built — {src_counts}")

    n_estimated_pairs = sum(1 for n in disp_fallback_notes if n != 'ok')
    if n_estimated_pairs:
        st.caption(
            f"ℹ️ {n_estimated_pairs}/{len(disp_fallback_notes)} frame pairs used an "
            f"estimated (non-DJI) altitude and/or heading value -- see the metadata "
            f"warning above for details."
        )

    # ── Step 4: VIO estimation ──────────────────────────────
    st.subheader("Step 4 / 4 — Running VIO estimator")
    ORIGIN_LAT = float(df['GpsLatitude'].iloc[0])
    ORIGIN_LON = float(df['GpsLongitude'].iloc[0])
    ORIGIN_ALT = float(df['AbsoluteAltitude'].iloc[0])

    real_north, real_east, _ = pymap3d.geodetic2ned(
        lat=df['GpsLatitude'].values,
        lon=df['GpsLongitude'].values,
        h=df['AbsoluteAltitude'].values,
        lat0=ORIGIN_LAT, lon0=ORIGIN_LON, h0=ORIGIN_ALT
    )

    n_frames = len(df)
    mask_A   = np.ones(n_frames, dtype=bool)
    mask_B   = np.ones(n_frames, dtype=bool)
    mask_C   = np.ones(n_frames, dtype=bool)
    mask_B[dropout_b_start:dropout_b_end] = False
    mask_C[dropout_c_start:dropout_c_end] = False

    north_pure, east_pure = pure_vio(displacements, real_north, real_east)
    north_A, east_A       = gps_aided_vio(displacements, mask_A, real_north, real_east)
    north_B, east_B       = gps_aided_vio(displacements, mask_B, real_north, real_east)
    north_C, east_C       = gps_aided_vio(displacements, mask_C, real_north, real_east)

    def errors(eN, eE):
        return np.sqrt((eN - real_north)**2 + (eE - real_east)**2)

    err_pure = errors(north_pure, east_pure)
    err_A    = errors(north_A,    east_A)
    err_B    = errors(north_B,    east_B)
    err_C    = errors(north_C,    east_C)

    # GPS coords for all trajectories
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
    c2.metric("Distance",      f"{total_dist:.0f} m")
    c3.metric("Duration",      f"{df['time_s'].iloc[-1]:.0f} s")
    mean_alt   = df['RelativeAltitude'].mean()
    mean_speed = df['speed_ms'].mean()
    c4.metric("Mean altitude", f"{mean_alt:.1f} m" if pd.notna(mean_alt) else "N/A")
    c5.metric("Avg speed",     f"{mean_speed:.1f} m/s" if pd.notna(mean_speed) else "N/A")

    st.divider()

    # ─────────────────────────────────────────────────────────
    #  TAB LAYOUT
    # ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Live Map", "📈 Trajectory (NED)", "📉 Error Analysis",
        "🔬 Tracking Stats", "💾 Export"
    ])

    # ── TAB 1: Live GPS Map ─────────────────────────────────
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
        st.subheader("Trajectory comparison — NED local frame (metres)")
        fig_ned = go.Figure()

        ned_data = [
            ("Real GPS",   real_east, real_north, "#e74c3c", "circle",      4),
            ("Pure VIO",   east_pure, north_pure, "#e67e22", "x",           3),
            ("Full GPS",   east_A,    north_A,    "#2ecc71", "square",      3),
            (f"{dropout_b_end - dropout_b_start}fr dropout",
                           east_B,    north_B,    "#3498db", "triangle-up", 3),
            (f"{dropout_c_end - dropout_c_start}fr dropout",
                           east_C,    north_C,    "#9b59b6", "diamond",     3),
        ]

        for name, ex, no, color, sym, sz in ned_data:
            fig_ned.add_trace(go.Scatter(
                x=ex, y=no,
                mode="lines+markers",
                marker=dict(size=sz, color=color, symbol=sym),
                line=dict(color=color, width=2),
                name=name
            ))

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

    # ── TAB 3: Error Analysis ──────────────────────────────
    with tab3:
        st.subheader("Positional error per frame")
        frame_ids = np.arange(n_frames)

        fig_err = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Error vs frame index",
                                                "Inlier count per frame pair"))

        for name, err, color in [
            ("Pure VIO",   err_pure, "#e67e22"),
            ("Full GPS",   err_A,    "#2ecc71"),
            (f"{dropout_b_end - dropout_b_start}fr dropout", err_B, "#3498db"),
            (f"{dropout_c_end - dropout_c_start}fr dropout", err_C, "#9b59b6"),
        ]:
            fig_err.add_trace(
                go.Scatter(x=frame_ids, y=err, name=name,
                           line=dict(color=color, width=2), mode="lines+markers",
                           marker=dict(size=4)),
                row=1, col=1
            )

        # shade denied windows
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
                   name="Inliers", marker_color="#7c3aed", opacity=0.7),
            row=2, col=1
        )
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
        summary = pd.DataFrame({
            "Method": [
                "Pure VIO (no GPS)",
                "GPS-aided — Full GPS",
                f"GPS-aided — {dropout_b_end - dropout_b_start}fr dropout",
                f"GPS-aided — {dropout_c_end - dropout_c_start}fr dropout"
            ],
            "Mean error (m)":  [f"{e.mean():.2f}" for e in [err_pure, err_A, err_B, err_C]],
            "Max error (m)":   [f"{e.max():.2f}"  for e in [err_pure, err_A, err_B, err_C]],
            "Final error (m)": [f"{e[-1]:.2f}"    for e in [err_pure, err_A, err_B, err_C]],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── TAB 4: Tracking Stats ──────────────────────────────
    with tab4:
        st.subheader("Feature tracking diagnostics")

        col_a, col_b = st.columns(2)

        with col_a:
            src_counts_plot = {s: sources.count(s) for s in ['vision', 'blended', 'imu_only']}
            fig_src = px.pie(
                values=list(src_counts_plot.values()),
                names=list(src_counts_plot.keys()),
                title="Displacement source breakdown",
                color_discrete_map={
                    'vision':   '#2ecc71',
                    'blended':  '#f39c12',
                    'imu_only': '#e74c3c'
                }
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
        tracking_df = pd.DataFrame({
            "Pair":      [f"{i}→{i+1}" for i in range(n_pairs)],
            "Inliers":   inlier_log,
            "Source":    sources,
            "dn (m)":    [f"{d[0]:.2f}" for d in displacements],
            "de (m)":    [f"{d[1]:.2f}" for d in displacements],
        })
        st.dataframe(tracking_df, use_container_width=True, hide_index=True, height=300)

    # ── TAB 5: Export ──────────────────────────────────────
    with tab5:
        st.subheader("Export pipeline outputs")

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
