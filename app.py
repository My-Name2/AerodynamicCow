"""
Wind Tunnel Simulator
=====================
2D potential flow using the Hess-Smith source panel method.

Physics:
  - Inviscid, irrotational, incompressible flow (potential flow theory)
  - Bernoulli equation for pressure: Cp = 1 - (V/V∞)²
  - Pressure force integration for lift/drag coefficients
  - Reynolds number, dynamic pressure, and stagnation point display

Reference: Hess & Smith (1967), "Calculation of Potential Flow About Arbitrary Bodies",
           Progress in Aerospace Sciences, Vol. 8.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wind Tunnel Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS for dark feel ─────────────────────────────────────────
st.markdown("""
<style>
  .metric-box { background:#1e2130; border-radius:8px; padding:10px 14px;
                margin:4px 0; border-left:3px solid #4a9eff; }
  .metric-label { color:#8899aa; font-size:0.75em; text-transform:uppercase;
                  letter-spacing:.08em; }
  .metric-value { color:#e8f0ff; font-size:1.3em; font-weight:700; }
  div[data-testid="stSidebar"] { background:#10131a; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SHAPE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

def _close(pts):
    """Ensure polygon is closed (last point == first point)."""
    pts = np.asarray(pts, dtype=float)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts

def shape_circle(cx=0.50, cy=0.50, r=0.18, n=120):
    θ = np.linspace(0, 2*np.pi, n, endpoint=False)
    return _close(np.column_stack([cx + r*np.cos(θ), cy + r*np.sin(θ)]))

def shape_naca(series="0012", chord=0.46, x0=0.15, y0=0.50,
               alpha_geom=0.0, n=160):
    """NACA 4-digit airfoil (cambered or symmetric)."""
    t  = int(series[2:]) / 100
    m  = int(series[0])  / 100
    p  = int(series[1])  / 10
    x  = np.linspace(0, 1, n // 2)
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
              + 0.2843*x**3 - 0.1015*x**4)
    if m == 0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc  = np.where(x < p,
                       m/p**2*(2*p*x - x**2),
                       m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
        dyc = np.where(x < p,
                       2*m/p**2*(p - x),
                       2*m/(1-p)**2*(p - x))
    θc = np.arctan(dyc)
    xu = x - yt*np.sin(θc);  yu = yc + yt*np.cos(θc)
    xl = x + yt*np.sin(θc);  yl = yc - yt*np.cos(θc)
    raw = np.vstack([
        np.column_stack([xu, yu]),
        np.column_stack([xl[::-1], yl[::-1]])
    ])
    # Scale, translate, rotate
    raw *= chord
    raw[:, 0] += x0
    raw[:, 1] += y0 - chord * yc[0]
    # Rotate about leading edge for geometric AoA
    if alpha_geom:
        c, s = np.cos(np.deg2rad(alpha_geom)), np.sin(np.deg2rad(alpha_geom))
        cx_, cy_ = raw[0]
        raw[:, 0] -= cx_; raw[:, 1] -= cy_
        raw = raw @ np.array([[c, s], [-s, c]])
        raw[:, 0] += cx_; raw[:, 1] += cy_
    return _close(raw)

def shape_rectangle(x0=0.35, y0=0.35, w=0.30, h=0.30):
    return _close([[x0,y0],[x0+w,y0],[x0+w,y0+h],[x0,y0+h]])

def shape_ellipse(cx=0.50, cy=0.50, a=0.28, b=0.14, n=120):
    θ = np.linspace(0, 2*np.pi, n, endpoint=False)
    return _close(np.column_stack([cx + a*np.cos(θ), cy + b*np.sin(θ)]))

def shape_cow():
    """
    Side-view cow silhouette – continuous outline tracing the outer contour
    (body + neck + head), normalized to [0,1]².
    """
    pts = [
        # ── Nose/muzzle (front, mid-height) ──
        (0.140, 0.490), (0.125, 0.510), (0.118, 0.535),
        (0.120, 0.565), (0.130, 0.590),
        # ── Forehead / poll ──
        (0.148, 0.620), (0.168, 0.638), (0.192, 0.642),
        # ── Ear bump ──
        (0.210, 0.650), (0.228, 0.660), (0.238, 0.650),
        # ── Crest of neck ──
        (0.258, 0.668), (0.290, 0.678), (0.322, 0.678),
        # ── Withers (shoulder high point) ──
        (0.355, 0.682), (0.385, 0.690),
        # ── Back (relatively flat with slight dip) ──
        (0.430, 0.685), (0.480, 0.682), (0.530, 0.684),
        # ── Loin / rump rise ──
        (0.575, 0.692), (0.610, 0.698),
        # ── Rump / tailhead ──
        (0.645, 0.695), (0.672, 0.685), (0.690, 0.668),
        # ── Pin bones / hindquarter ──
        (0.705, 0.645), (0.715, 0.615), (0.718, 0.580),
        # ── Tail (slight protrusion) ──
        (0.720, 0.545), (0.725, 0.518), (0.720, 0.492),
        # ── Rear belly / udder region ──
        (0.710, 0.450), (0.695, 0.408), (0.672, 0.382),
        # ── Belly (underline) ──
        (0.640, 0.362), (0.600, 0.348), (0.555, 0.340),
        (0.510, 0.337), (0.465, 0.338), (0.420, 0.342),
        # ── Chest / brisket ──
        (0.372, 0.350), (0.335, 0.362), (0.305, 0.382),
        # ── Throat / lower neck ──
        (0.278, 0.410), (0.260, 0.445), (0.248, 0.470),
        # ── Dewlap / chin ──
        (0.238, 0.488), (0.230, 0.505), (0.218, 0.510),
        (0.200, 0.505), (0.185, 0.498), (0.165, 0.490),
        (0.150, 0.488),
    ]
    return _close(pts)

def shape_car():
    pts = [
        # Underbody (flat)
        (0.12, 0.310), (0.82, 0.310),
        # Rear wheel arch
        (0.84, 0.330), (0.86, 0.360), (0.84, 0.385), (0.78, 0.390),
        (0.72, 0.385), (0.70, 0.360), (0.72, 0.335),
        # Rear deck / trunk
        (0.78, 0.395), (0.82, 0.430), (0.82, 0.520),
        # Rear windscreen slope
        (0.78, 0.580), (0.72, 0.620),
        # Roof
        (0.62, 0.645), (0.52, 0.650), (0.42, 0.648), (0.35, 0.640),
        # Windscreen slope
        (0.28, 0.598), (0.22, 0.535),
        # Hood
        (0.18, 0.480), (0.14, 0.440), (0.12, 0.400),
        # Front bumper
        (0.12, 0.365),
        # Front wheel arch
        (0.14, 0.335), (0.20, 0.315),
        (0.26, 0.315), (0.28, 0.335), (0.26, 0.360),
        (0.20, 0.365), (0.18, 0.350),
    ]
    return _close(pts)

def shape_truck():
    # Trailer
    trailer = _close([
        (0.06,0.29),(0.50,0.29),(0.50,0.72),(0.06,0.72)
    ])
    # Cab
    cab = _close([
        (0.51,0.29),(0.76,0.29),(0.80,0.34),(0.82,0.42),
        (0.82,0.60),(0.78,0.66),(0.71,0.68),(0.62,0.67),
        (0.55,0.63),(0.52,0.56),(0.51,0.46),
    ])
    return [trailer, cab]

def shape_cyclist():
    pts = [
        # Wheel (rear, simplified arc)
        (0.62,0.27),(0.68,0.26),(0.74,0.27),(0.78,0.31),
        (0.80,0.36),(0.78,0.42),(0.74,0.46),(0.68,0.47),
        (0.62,0.46),(0.58,0.42),(0.56,0.36),(0.58,0.31),
        # Frame / saddle bridge
        (0.60,0.50),(0.55,0.54),
        # Torso (crouched)
        (0.50,0.56),(0.44,0.62),(0.38,0.66),(0.32,0.67),
        # Head / helmet
        (0.28,0.70),(0.24,0.73),(0.22,0.76),(0.23,0.79),
        (0.27,0.81),(0.32,0.81),(0.36,0.78),(0.37,0.74),
        # Arms
        (0.34,0.69),(0.30,0.62),(0.28,0.56),(0.26,0.50),
        # Handlebar / fork
        (0.26,0.46),(0.28,0.42),
        # Front wheel
        (0.30,0.46),(0.26,0.46),(0.22,0.42),(0.20,0.36),
        (0.22,0.30),(0.28,0.26),(0.34,0.26),(0.40,0.30),
        (0.42,0.36),(0.40,0.42),(0.36,0.46),
        # Bottom bracket / crank
        (0.50,0.47),(0.54,0.44),(0.56,0.40),
    ]
    return _close(pts)

SHAPES = {
    "Cow":               (lambda: [shape_cow()],                 0.38),
    "Circle":            (lambda: [shape_circle()],              0.36),
    "NACA 0012 Airfoil": (lambda: [shape_naca("0012")],          0.46),
    "NACA 2412 Airfoil": (lambda: [shape_naca("2412")],          0.46),
    "Ellipse":           (lambda: [shape_ellipse()],             0.56),
    "Rectangle":         (lambda: [shape_rectangle()],           0.30),
    "Car (Side View)":   (lambda: [shape_car()],                 0.70),
    "Truck":             (lambda: shape_truck(),                 0.94),
    "Cyclist":           (lambda: [shape_cyclist()],             0.62),
}

# ═══════════════════════════════════════════════════════════════════════════
# SOURCE PANEL METHOD  (Hess & Smith 1967)
# ═══════════════════════════════════════════════════════════════════════════

def _panel_geometry(poly):
    """Return panel geometry arrays for a closed polygon."""
    x1, y1 = poly[:-1, 0], poly[:-1, 1]
    x2, y2 = poly[1:,  0], poly[1:,  1]
    xc = 0.5*(x1+x2);  yc = 0.5*(y1+y2)
    dx = x2-x1;         dy = y2-y1
    L   = np.hypot(dx, dy)
    β   = np.arctan2(dy, dx)
    nx  = -np.sin(β);   ny = np.cos(β)   # outward normals
    return x1, y1, xc, yc, L, β, nx, ny

def _source_influence(xp, yp, Lj):
    """
    Normal and tangential velocity at (xp, yp) in panel-j local frame
    due to a unit-strength source panel of length Lj.
    Returns (u_loc, v_loc) in local frame.
    """
    eps   = 1e-14
    r1sq  = np.maximum(xp**2 + yp**2,           eps)
    r2sq  = np.maximum((xp-Lj)**2 + yp**2,      eps)
    u_loc = 1.0/(4*np.pi) * np.log(r2sq / r1sq)
    v_loc = 1.0/(2*np.pi) * (np.arctan2(yp, xp-Lj) - np.arctan2(yp, xp))
    return u_loc, v_loc

def solve_panels(polys, U_inf=1.0, alpha_rad=0.0):
    """
    Solve for source-panel strengths on one or more body polygons.
    Returns all panel data needed for field evaluation.
    """
    # Collect geometry from all polygons
    segs = [_panel_geometry(p) for p in polys]
    x1c = np.concatenate([s[0] for s in segs])
    y1c = np.concatenate([s[1] for s in segs])
    xcc = np.concatenate([s[2] for s in segs])
    ycc = np.concatenate([s[3] for s in segs])
    Lc  = np.concatenate([s[4] for s in segs])
    βc  = np.concatenate([s[5] for s in segs])
    nxc = np.concatenate([s[6] for s in segs])
    nyc = np.concatenate([s[7] for s in segs])
    n   = len(Lc)

    # Influence matrix  A[i,j] = normal velocity at ctrl-pt i from panel j
    # Vectorised: shape (n_i, n_j)
    dxij = xcc[:, None] - x1c[None, :]      # (n, n)
    dyij = ycc[:, None] - y1c[None, :]

    cos_β = np.cos(βc[None, :]);  sin_β = np.sin(βc[None, :])
    xp = ( dxij*cos_β + dyij*sin_β)
    yp = (-dxij*sin_β + dyij*cos_β)

    u_loc, v_loc = _source_influence(xp, yp, Lc[None, :])

    # Transform induced velocity to global, then dot with panel-i normal
    u_glob = u_loc*cos_β - v_loc*sin_β
    v_glob = u_loc*sin_β + v_loc*cos_β
    A = u_glob*nxc[:, None] + v_glob*nyc[:, None]
    np.fill_diagonal(A, 0.5)

    # RHS: negative freestream normal component
    b = -(U_inf*np.cos(alpha_rad)*nxc + U_inf*np.sin(alpha_rad)*nyc)
    q = np.linalg.solve(A, b)

    return q, x1c, y1c, Lc, βc

def velocity_field(X, Y, q, x1, y1, L, β, U_inf, alpha_rad):
    """
    Evaluate (u, v) on the grid using vectorised panel superposition.
    Uses chunked evaluation to cap peak memory.
    """
    shape  = X.shape
    Xf     = X.ravel()
    Yf     = Y.ravel()
    M, N   = len(Xf), len(q)
    CHUNK  = 4096

    u_out = np.full(M, U_inf*np.cos(alpha_rad))
    v_out = np.full(M, U_inf*np.sin(alpha_rad))

    cos_β = np.cos(β);  sin_β = np.sin(β)

    for start in range(0, M, CHUNK):
        sl   = slice(start, start+CHUNK)
        dxij = Xf[sl, None] - x1[None, :]   # (chunk, N)
        dyij = Yf[sl, None] - y1[None, :]

        xp =  dxij*cos_β[None, :] + dyij*sin_β[None, :]
        yp = -dxij*sin_β[None, :] + dyij*cos_β[None, :]

        u_loc, v_loc = _source_influence(xp, yp, L[None, :])
        u_loc *= q[None, :];  v_loc *= q[None, :]

        u_out[sl] += np.sum(u_loc*cos_β[None, :] - v_loc*sin_β[None, :], axis=1)
        v_out[sl] += np.sum(u_loc*sin_β[None, :] + v_loc*cos_β[None, :], axis=1)

    return u_out.reshape(shape), v_out.reshape(shape)

def build_mask(polys, X, Y):
    """Boolean mask: True inside any body polygon."""
    pts  = np.column_stack([X.ravel(), Y.ravel()])
    mask = np.zeros(X.size, dtype=bool)
    for poly in polys:
        mask |= Path(poly).contains_points(pts)
    return mask.reshape(X.shape)

# ═══════════════════════════════════════════════════════════════════════════
# AERODYNAMIC COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════

def pressure_coeff(speed, V_inf):
    """Cp = 1 - (V/V∞)²  (Bernoulli, incompressible)."""
    return np.where(speed > 0, 1.0 - (speed / (V_inf + 1e-12))**2, 1.0)

def integrate_forces(polys, q, x1, y1, L, β, V_inf, alpha_rad, ref_length):
    """
    Integrate pressure forces around each polygon panel to get Cd and Cl.
    Returns (Cd, Cl) non-dimensionalised by ½ρV∞² · ref_length (per unit depth).
    """
    # Tangential surface speed at each panel control point
    xcc = x1 + 0.5*L*np.cos(β)
    ycc = y1 + 0.5*L*np.sin(β)
    # Evaluate field velocity at panel control points
    u_s, v_s = velocity_field(
        xcc[None, :], ycc[None, :],
        q, x1, y1, L, β, V_inf, alpha_rad
    )
    V_s  = np.hypot(u_s.ravel(), v_s.ravel())
    Cp_s = pressure_coeff(V_s, V_inf)

    # Pressure force on each panel: dF = -Cp * 0.5 * L * n̂
    nx = -np.sin(β);  ny = np.cos(β)
    Fx = np.sum(-Cp_s * nx * L)
    Fy = np.sum(-Cp_s * ny * L)

    # Rotate into drag/lift frame
    ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
    D =  Fx*ca + Fy*sa
    Lf = -Fx*sa + Fy*ca

    denom = ref_length + 1e-12
    return D/denom, Lf/denom

def stagnation_points(X, Y, u, v, mask):
    """Find approximate stagnation point(s): grid cells with min speed outside mask."""
    speed = np.hypot(u, v)
    speed_masked = np.where(mask, np.inf, speed)
    # Look only near object surface (dilate mask by a few cells)
    from scipy.ndimage import binary_dilation
    surface_zone = binary_dilation(mask, iterations=4) & ~mask
    speed_surface = np.where(surface_zone, speed_masked, np.inf)
    idx = np.unravel_index(np.argmin(speed_surface), speed_surface.shape)
    return X[idx], Y[idx], speed[idx]

# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# ✈ Wind Tunnel Simulator")
st.markdown(
    "**2D potential flow** · Hess-Smith source panel method · "
    "Bernoulli pressure · force integration"
)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Tunnel Parameters")
    shape_name = st.selectbox("Body shape", list(SHAPES.keys()))

    st.subheader("Flow Conditions")
    V_inf      = st.slider("Free-stream speed  V∞ (m/s)", 1.0, 80.0, 20.0, 1.0)
    alpha_deg  = st.slider("Angle of attack  α (°)", -20, 20, 0, 1)
    rho        = st.select_slider("Air density  ρ (kg/m³)",
                                  options=[0.413, 0.736, 1.225, 1.293, 1.654],
                                  value=1.225,
                                  format_func=lambda x: f"{x:.3f}")

    st.subheader("Solver")
    N_panels = st.select_slider("Panels per body", [40, 60, 80, 100, 140], value=80)
    grid_res = st.select_slider("Field grid resolution", [60, 80, 100, 120], value=80)

    st.subheader("Visualisation")
    vis_mode = st.selectbox("Colour field", [
        "Velocity magnitude |V|",
        "Pressure coefficient Cp",
        "Vorticity ∂v/∂x − ∂u/∂y",
        "Mach number (low-speed)",
    ])
    cmap_choice = st.selectbox("Colourmap", ["jet", "turbo", "RdBu_r", "plasma", "coolwarm"], index=0)
    show_streamlines = st.checkbox("Streamlines", True)
    show_vectors     = st.checkbox("Velocity vectors", False)
    n_streamlines    = st.slider("Streamline density", 0.8, 3.0, 1.8, 0.2)

    run = st.button("▶  Run Simulation", type="primary", use_container_width=True)

# ── Derived constants ─────────────────────────────────────────────────────
alpha_rad   = np.deg2rad(alpha_deg)
dyn_press   = 0.5 * rho * V_inf**2
# Kinematic viscosity of air at ~15 °C
nu          = 1.48e-5
shape_fn, ref_L = SHAPES[shape_name]

# ── Run / cache ───────────────────────────────────────────────────────────
cache_key = (shape_name, round(V_inf,1), alpha_deg, N_panels, grid_res)

if run or "result" not in st.session_state or st.session_state.get("cache_key") != cache_key:
    with st.spinner("Solving panel system and evaluating flow field…"):
        polys_raw = shape_fn()

        # Resample each polygon to exactly N_panels+1 points
        def resample(poly, n):
            from scipy.interpolate import interp1d
            d   = np.cumsum(np.r_[0, np.hypot(np.diff(poly[:,0]), np.diff(poly[:,1]))])
            d  /= d[-1]
            fx  = interp1d(d, poly[:,0], kind='linear')
            fy  = interp1d(d, poly[:,1], kind='linear')
            t   = np.linspace(0, 1, n+1)
            return np.column_stack([fx(t), fy(t)])

        polys = [resample(p, N_panels) for p in polys_raw]

        # Solve
        q, x1, y1, L, β = solve_panels(polys, V_inf, alpha_rad)

        # Grid
        x  = np.linspace(0, 1, grid_res)
        y  = np.linspace(0, 1, grid_res)
        X, Y = np.meshgrid(x, y)

        # Velocity field
        u, v = velocity_field(X, Y, q, x1, y1, L, β, V_inf, alpha_rad)

        # Mask
        mask = build_mask(polys, X, Y)
        u[mask] = 0;  v[mask] = 0

        speed    = np.hypot(u, v)
        Cp_field = pressure_coeff(speed, V_inf)

        # Vorticity (finite diff)
        dx_ = x[1]-x[0];  dy_ = y[1]-y[0]
        dvdx = np.gradient(v, dx_, axis=1)
        dudy = np.gradient(u, dy_, axis=0)
        vorticity = dvdx - dudy
        vorticity[mask] = 0

        # Mach (incompressible approx: M = V/343)
        mach = speed / 343.0

        # Force coefficients
        Cd, Cl = integrate_forces(polys, q, x1, y1, L, β, V_inf, alpha_rad, ref_L)

        # Stagnation point
        try:
            from scipy.ndimage import binary_dilation
            xs, ys, Vs = stagnation_points(X, Y, u, v, mask)
        except Exception:
            xs, ys, Vs = None, None, None

        st.session_state["result"]    = (X, Y, u, v, speed, Cp_field,
                                         vorticity, mach, mask, polys, Cd, Cl, xs, ys)
        # Also store solved panel data for reuse in other tabs
        st.session_state["panels"]    = (q, x1, y1, L, β)
        st.session_state["cache_key"] = cache_key

(X, Y, u, v, speed, Cp_field,
 vorticity, mach, mask, polys,
 Cd, Cl, xs, ys) = st.session_state["result"]
q_s, x1_s, y1_s, L_s, β_s = st.session_state["panels"]

Re = V_inf * ref_L / nu

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_flow, tab_pressure, tab_aero, tab_theory = st.tabs([
    "Flow Field", "Pressure Distribution", "Aerodynamic Data", "Theory & Notes"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – FLOW FIELD
# ═══════════════════════════════════════════════════════════════════════════
with tab_flow:
    colA, colB = st.columns([3, 1])

    with colA:
        fig, ax = plt.subplots(figsize=(10, 7.5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        # Choose scalar field
        if vis_mode == "Velocity magnitude |V|":
            field = np.where(mask, np.nan, speed)
            vmin, vmax = 0, V_inf * 2.2
            cbar_label = "Speed  |V|  (m/s)"
        elif vis_mode == "Pressure coefficient Cp":
            field = np.where(mask, np.nan, Cp_field)
            vmin, vmax = -3.5, 1.0
            cbar_label = "Pressure coefficient  Cp"
        elif vis_mode == "Vorticity ∂v/∂x − ∂u/∂y":
            field = np.where(mask, np.nan, vorticity)
            lim   = np.nanpercentile(np.abs(field[~mask]), 98) if np.any(~mask) else 1
            vmin, vmax = -lim, lim
            cbar_label = "Vorticity  (1/s)"
        else:  # Mach
            field = np.where(mask, np.nan, mach)
            vmin, vmax = 0, V_inf/343*2.2
            cbar_label = "Mach number  M"

        im = ax.pcolormesh(X, Y, field, cmap=cmap_choice,
                           shading="gouraud", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label, color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # Streamlines
        if show_streamlines:
            x1d = X[0, :];  y1d = Y[:, 0]
            u_sl = np.where(mask, 0, u)
            v_sl = np.where(mask, 0, v)
            try:
                ax.streamplot(x1d, y1d, u_sl, v_sl,
                              color="white", linewidth=0.55,
                              density=n_streamlines, arrowsize=0.9,
                              arrowstyle="->")
            except Exception:
                pass

        # Velocity vectors
        if show_vectors:
            step = max(1, grid_res // 18)
            Xq, Yq = X[::step, ::step], Y[::step, ::step]
            Uq, Vq = u[::step, ::step], v[::step, ::step]
            mq = mask[::step, ::step]
            Uq[mq] = 0;  Vq[mq] = 0
            ax.quiver(Xq, Yq, Uq, Vq, color="#ffdd44", alpha=0.65,
                      scale=V_inf*28, width=0.0018, headwidth=3)

        # Stagnation point marker
        if xs is not None:
            ax.plot(xs, ys, "o", color="#ff4444", ms=7, zorder=10,
                    label="Stagnation pt")
            ax.legend(facecolor="#1a1a2e", labelcolor="white",
                      fontsize=9, loc="upper right")

        # Body silhouettes
        for poly in polys:
            patch = MplPolygon(poly, closed=True,
                               facecolor="#2a2a3a", edgecolor="white",
                               linewidth=1.4, zorder=5)
            ax.add_patch(patch)

        # Wind direction arrow
        arrx = 0.04 + 0.10*np.cos(alpha_rad)
        arry = 0.06 + 0.10*np.sin(alpha_rad)
        ax.annotate("", xy=(arrx, arry), xytext=(0.04, 0.06),
                    arrowprops=dict(arrowstyle="-|>", color="#44aaff",
                                   lw=1.8, mutation_scale=14))
        ax.text(0.04, 0.03, f"V∞={V_inf} m/s  α={alpha_deg}°",
                color="#44aaff", fontsize=8, transform=ax.transData)

        ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{shape_name}  —  {vis_mode}",
                     color="white", fontsize=13, pad=10)
        ax.tick_params(colors="#888888")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
        ax.set_xlabel("x / L", color="#888888");  ax.set_ylabel("y / L", color="#888888")

        st.pyplot(fig, use_container_width=True)

    with colB:
        st.markdown("### Flow Metrics")

        speed_ext = speed[~mask]
        def _metric(label, value, unit=""):
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value} <small style="color:#667788">{unit}</small></div></div>',
                unsafe_allow_html=True
            )

        _metric("Free-stream", f"{V_inf:.1f}", "m/s")
        _metric("Max speed",   f"{np.nanmax(speed_ext):.2f}", "m/s")
        _metric("Min speed",   f"{np.nanmin(speed_ext):.2f}", "m/s")
        _metric("Dyn. pressure q∞", f"{dyn_press:.1f}", "Pa")
        _metric("Reynolds Re",
                f"{Re:.2e}".replace("e+0","×10⁰").replace("e+","×10"), "")
        _metric("Max |Cp|", f"{np.nanmax(np.abs(Cp_field[~mask])):.3f}", "")
        _metric("Cd (press.)", f"{Cd:.4f}", "")
        _metric("Cl (press.)", f"{Cl:.4f}", "")
        if xs is not None:
            _metric("Stagnation x", f"{xs:.3f}", "")
            _metric("Stagnation y", f"{ys:.3f}", "")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – PRESSURE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
with tab_pressure:
    st.markdown("#### Surface Pressure Coefficient  Cp  around body")

    # Reuse panel solution cached during main solve — no re-solve needed
    xc_sol = x1_s + 0.5*L_s*np.cos(β_s)
    yc_sol = y1_s + 0.5*L_s*np.sin(β_s)
    # Evaluate velocity at each panel control point (shape (1,N) → ravel to (N,))
    u_surf, v_surf = velocity_field(
        xc_sol[None, :], yc_sol[None, :],
        q_s, x1_s, y1_s, L_s, β_s, V_inf, alpha_rad
    )
    Vsurf   = np.hypot(u_surf.ravel(), v_surf.ravel())
    Cp_surf = pressure_coeff(Vsurf, V_inf)

    # Arc-length along surface
    s = np.cumsum(np.r_[0, L_s[:-1]])
    s /= s[-1]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")
    ax2.plot(s, Cp_surf, color="#4a9eff", lw=1.5, label="Cp")
    ax2.axhline(0,  color="#555", lw=0.8, ls="--")
    ax2.axhline(1,  color="#44aa44", lw=0.8, ls=":", label="Cp = 1 (stagnation)")
    ax2.fill_between(s, Cp_surf, 0,
                     where=(Cp_surf > 0), alpha=0.15, color="#ff6644",
                     label="Pressure (+)")
    ax2.fill_between(s, Cp_surf, 0,
                     where=(Cp_surf < 0), alpha=0.15, color="#4488ff",
                     label="Suction (−)")
    ax2.invert_yaxis()   # aerodynamic convention: Cp positive downward
    ax2.set_xlabel("Normalised arc-length  s / S", color="#aaa")
    ax2.set_ylabel("Cp  (−ve = suction)", color="#aaa")
    ax2.set_title("Surface pressure distribution", color="white")
    ax2.tick_params(colors="#888")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333")
    st.pyplot(fig2, use_container_width=True)

    with st.expander("Show surface Cp data table"):
        import pandas as pd
        df = pd.DataFrame({
            "Arc-length s/S": np.round(s,       4),
            "x":              np.round(xc_sol,   4),
            "y":              np.round(yc_sol,   4),
            "Cp":             np.round(Cp_surf,  4),
        })
        st.dataframe(df, use_container_width=True, height=300)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – AERODYNAMIC DATA
# ═══════════════════════════════════════════════════════════════════════════
with tab_aero:
    st.markdown("#### Aerodynamic Coefficients & Forces")

    # Sweep AoA for polar — compute once per shape/speed, then cache
    polar_key = (shape_name, round(V_inf, 1), N_panels)
    if st.session_state.get("polar_key") != polar_key:
        alphas = np.linspace(-15, 15, 25)
        Cds_p, Cls_p = [], []
        with st.spinner("Computing aerodynamic polar…"):
            for a in alphas:
                qa, x1a, y1a, La, βa = solve_panels(polys, V_inf, np.deg2rad(a))
                cd_, cl_ = integrate_forces(polys, qa, x1a, y1a, La, βa,
                                            V_inf, np.deg2rad(a), ref_L)
                Cds_p.append(cd_);  Cls_p.append(cl_)
        st.session_state["polar"]     = (alphas, np.array(Cds_p), np.array(Cls_p))
        st.session_state["polar_key"] = polar_key

    alphas, Cds, Cls = st.session_state["polar"]

    fig3, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig3.patch.set_facecolor("#0d1117")
    _colours = ["#4a9eff","#ff6644","#44ee88"]

    for ax3, (ydata, ylabel, colour) in zip(axes, [
        (Cls, "Lift coefficient  Cl", _colours[0]),
        (Cds, "Drag coefficient  Cd", _colours[1]),
        (Cls/(Cds+1e-6), "L/D ratio", _colours[2]),
    ]):
        ax3.set_facecolor("#0d1117")
        ax3.plot(alphas, ydata, color=colour, lw=2)
        ax3.axhline(0, color="#444", lw=0.8, ls="--")
        ax3.axvline(alpha_deg, color="#ffffff44", lw=1, ls=":")
        ax3.set_xlabel("α (°)", color="#aaa");  ax3.set_ylabel(ylabel, color="#aaa")
        ax3.tick_params(colors="#888")
        for sp in ax3.spines.values(): sp.set_edgecolor("#333")
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cd  (current α)", f"{Cd:.4f}")
    c2.metric("Cl  (current α)", f"{Cl:.4f}")
    c3.metric("L/D", f"{Cl/(Cd+1e-6):.2f}")
    c4.metric("Drag force", f"{Cd * dyn_press * ref_L:.2f} N/m")

    st.markdown(
        "> **Note:** Coefficients are from pressure integration of the *inviscid* "
        "potential flow solution.  Skin-friction drag and flow separation are not "
        "modelled; values are indicative, not definitive."
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – THEORY
# ═══════════════════════════════════════════════════════════════════════════
with tab_theory:
    st.markdown(r"""
## Physics & Numerical Method

### Governing Equations
For **steady, inviscid, irrotational, incompressible** flow the velocity
field derives from a scalar potential φ:

$$\mathbf{V} = \nabla\phi, \qquad \nabla^2\phi = 0 \quad \text{(Laplace equation)}$$

Equivalently, a stream function ψ exists such that:

$$u = \frac{\partial\psi}{\partial y}, \quad v = -\frac{\partial\psi}{\partial x}$$

### Source Panel Method  *(Hess & Smith 1967)*
The body surface is discretised into *N* straight panels.  Each panel *j*
carries a uniform source distribution of unknown strength *qⱼ* (source per
unit length).  The total velocity at any point is the superposition of the
free-stream and all panel contributions:

$$\mathbf{V}(\mathbf{r}) = V_\infty\hat{\mathbf{e}}_\infty
  + \sum_{j=1}^{N} \frac{q_j}{2\pi} \int_{\text{panel }j}
    \frac{\mathbf{r}-\mathbf{r}'}{|\mathbf{r}-\mathbf{r}'|^2}\,ds$$

The **no-penetration** boundary condition (zero normal velocity on each
panel control point) yields the *N × N* linear system:

$$\sum_{j=1}^{N} A_{ij}\,q_j = -V_\infty\,(\hat{\mathbf{e}}_\infty\cdot\hat{\mathbf{n}}_i)$$

where *A*ᵢⱼ is the analytically integrated normal-velocity influence coefficient.

### Bernoulli Pressure Coefficient
For incompressible flow the **pressure coefficient** is:

$$C_p = \frac{p - p_\infty}{\tfrac{1}{2}\rho V_\infty^2} = 1 - \left(\frac{V}{V_\infty}\right)^2$$

- *Cp = 1* : stagnation point (V = 0)
- *Cp = 0* : local speed equals free-stream
- *Cp < 0* : suction region (acceleration over curved surfaces)

### Reynolds Number
$$Re = \frac{V_\infty\,L}{\nu}$$

At *Re* ≳ 10⁵ the boundary layer is thin enough that potential-flow gives
a good approximation of the outer inviscid flow. At lower *Re*, viscous
effects (boundary layer, separation, wake) become important and are
**not captured** by this model.

### Limitations
| Effect | Modelled? |
|--------|-----------|
| Inviscid (Euler) pressure | ✅ |
| Stagnation / suction regions | ✅ |
| Streamline topology | ✅ |
| Boundary layer | ❌ |
| Flow separation & wake | ❌ |
| Turbulence | ❌ |
| Compressibility (M > 0.3) | ❌ |
| 3-D effects | ❌ |

> **Reference:**  J. L. Hess & A. M. O. Smith, "Calculation of potential flow
> about arbitrary bodies", *Progress in Aerospace Sciences* **8**, 1–138 (1967).
> Anderson, J. D., *Fundamentals of Aerodynamics*, 6th ed., McGraw-Hill, 2017.
    """)
