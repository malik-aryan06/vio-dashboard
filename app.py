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
#  CAMERA INTRINSICS  (hardcoded from DJI XMP — Module 1)
# ─────────────────────────────────────────────────────────────
SCALE       = 0.25
fx          = 3725.151611 * SCALE
fy          = 3725.151611 * SCALE
cx          = 2640.0      * SCALE
cy          = 1978.0      * SCALE
K           = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
dist_coeffs = np.array([-0.112575240, 0.014874430, -0.027064110, 0.0, -0.000085720])

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
              'GpsLatitude', 'GpsLongitude']
    result = {}
    for f in fields:
        match = re.search(rf'drone-dji:{f}="([^"]+)"', xmp)
        if match:
            try:    result[f] = float(match.group(1))
            except: result[f] = match.group(1)
    return result

def extract_metadata(p):
    rec = {'filepath': p, 'filename': os.path.basename(p)}
    m   = re.search(r'DJI_(\d{8})(\d{6})_(\d+)_V', p)
    if m:
        rec['frame_id']  = int(m.group(3))
        rec['timestamp'] = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
    img      = PILImage.open(p)
    exif_raw = img._getexif()
    if exif_raw:
        exif = {TAGS.get(k, k): v for k, v in exif_raw.items()}
        gps  = exif.get('GPSInfo', {})
        if gps:
            rec['exif_lat'] = parse_gps_coord(gps[2], gps[1])
            rec['exif_lon'] = parse_gps_coord(gps[4], gps[3])
    with open(p, 'rb') as f:
        raw = f.read()
    rec.update(parse_xmp(raw))
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

def pixels_to_ned(dx_px, dy_px, altitude_m, yaw_deg):
    dX  = (dx_px / fx) * altitude_m
    dY  = (dy_px / fy) * altitude_m
    yr  = np.radians(yaw_deg)
    dN  = -(np.cos(yr) * dX + np.sin(yr) * dY)
    dE  = -(-np.sin(yr) * dX + np.cos(yr) * dY)
    return dN, dE

def build_displacements(tracking_results, df):
    displacements = []
    sources       = []
    for i, r in enumerate(tracking_results):
        row    = df.iloc[i]
        dt     = float(df.iloc[i + 1]['time_s'] - row['time_s'])
        imu_dn = float(row['FlightXSpeed']) * dt
        imu_de = float(row['FlightYSpeed']) * dt
        if r is None:
            dn, de  = imu_dn, imu_de
            src_tag = 'imu_only'
        else:
            vo_dn, vo_de = pixels_to_ned(
                r['dx'], r['dy'],
                float(row['RelativeAltitude']),
                float(row['FlightYawDegree'])
            )
            if r.get('reliable', True):
                dn, de  = vo_dn, vo_de
                src_tag = 'vision'
            else:
                dn      = 0.5 * vo_dn + 0.5 * imu_dn
                de      = 0.5 * vo_de + 0.5 * imu_de
                src_tag = 'blended'
        displacements.append((dn, de))
        sources.append(src_tag)
    return displacements, sources

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
    upload_mode = st.radio(
        "Input source",
        ["Upload ZIP", "Local folder path"],
        help="Upload a .zip of DJI JPEGs, or point to a local folder"
    )

    dataset_dir  = None
    image_paths  = []

    if upload_mode == "Upload ZIP":
        uploaded = st.file_uploader(
            "Upload dataset ZIP",
            type=["zip"],
            help="ZIP file containing DJI_*.JPG images"
        )
        if uploaded:
            tmp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(uploaded) as zf:
                zf.extractall(tmp_dir)
            image_paths = sorted(glob.glob(f"{tmp_dir}/**/*.JPG", recursive=True))
            if not image_paths:
                image_paths = sorted(glob.glob(f"{tmp_dir}/**/*.jpg", recursive=True))
            if image_paths:
                dataset_dir = os.path.dirname(image_paths[0])
                st.success(f"Found {len(image_paths)} images")
    else:
        folder = st.text_input(
            "Folder path",
            value="4thAve1",
            help="Path to folder containing DJI_*.JPG files"
        )
        if os.path.exists(folder):
            image_paths = sorted(glob.glob(f"{folder}/DJI_*.JPG"))
            if image_paths:
                dataset_dir = folder
                st.success(f"Found {len(image_paths)} images")
            else:
                st.warning("No DJI_*.JPG files found in that folder")
        else:
            st.info("Enter a valid folder path")

    st.divider()
    st.subheader("⚙️ GPS Dropout Settings")

    dropout_b_start = st.slider("Scenario B — dropout start frame", 0, 40, 20)
    dropout_b_end   = st.slider("Scenario B — dropout end frame",   1, 50, 25)
    dropout_c_start = st.slider("Scenario C — dropout start frame", 0, 40, 15)
    dropout_c_end   = st.slider("Scenario C — dropout end frame",   1, 52, 30)

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
st.caption("DJI Mavic 3 Enterprise · GPS-aided VIO with drift correction")

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
        records.append(extract_metadata(p))
        prog1.progress((idx + 1) / len(image_paths),
                       text=f"Frame {idx+1} / {len(image_paths)}")

    df          = pd.DataFrame(records).sort_values('frame_id').reset_index(drop=True)
    df['time_s']   = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    df['speed_ms'] = np.sqrt(
        df['FlightXSpeed']**2 + df['FlightYSpeed']**2 + df['FlightZSpeed']**2
    )
    prog1.empty()
    st.success(f"✓ Metadata extracted — {len(df)} frames, "
               f"{df['time_s'].iloc[-1]:.0f}s flight")

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
    displacements, sources = build_displacements(tracking, df)
    src_counts = {s: sources.count(s) for s in set(sources)}
    st.success(f"✓ Displacements built — {src_counts}")

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
    c4.metric("Mean altitude", f"{df['RelativeAltitude'].mean():.1f} m")
    c5.metric("Avg speed",     f"{df['speed_ms'].mean():.1f} m/s")

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
