"""
Wind Tunnel Simulator — Professional Edition
=============================================
Live animated vapor/smoke tunnel  ·  3-D velocity surface  ·  CFD field plots
Physics: Hess-Smith 2-D source panel method (potential flow)

References
----------
Hess & Smith (1967), Progress in Aerospace Sciences 8, 1-138.
Anderson, Fundamentals of Aerodynamics, 6th ed., McGraw-Hill 2017.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Wind Tunnel Simulator", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  div[data-testid="stSidebar"] { background:#0d1117; }
  .block-container { padding-top:1.2rem; }
  .metric-box { background:#161b27; border-radius:6px; padding:9px 14px;
                margin:3px 0; border-left:3px solid #3d8bff; }
  .metric-label { color:#7a8899; font-size:.72em; text-transform:uppercase;
                  letter-spacing:.07em; }
  .metric-value { color:#dde8ff; font-size:1.25em; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SHAPE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

def _close(pts):
    pts = np.asarray(pts, dtype=float)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts

def shape_circle(cx=.50, cy=.50, r=.18, n=140):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return _close(np.column_stack([cx+r*np.cos(t), cy+r*np.sin(t)]))

def shape_naca(series="0012", chord=.46, x0=.15, y0=.50, alpha_geom=0., n=160):
    t_  = int(series[2:])/100; m = int(series[0])/100; p = int(series[1])/10
    x   = np.linspace(0, 1, n//2)
    yt  = 5*t_*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    if m == 0:
        yc = dyc = np.zeros_like(x)
    else:
        yc  = np.where(x<p, m/p**2*(2*p*x-x**2), m/(1-p)**2*((1-2*p)+2*p*x-x**2))
        dyc = np.where(x<p, 2*m/p**2*(p-x),      2*m/(1-p)**2*(p-x))
    tc  = np.arctan(dyc)
    xu  = x - yt*np.sin(tc); yu = yc + yt*np.cos(tc)
    xl  = x + yt*np.sin(tc); yl = yc - yt*np.cos(tc)
    raw = np.vstack([np.column_stack([xu,yu]), np.column_stack([xl[::-1],yl[::-1]])])
    raw *= chord; raw[:,0] += x0; raw[:,1] += y0 - chord*yc[0]
    if alpha_geom:
        c,s = np.cos(np.deg2rad(alpha_geom)), np.sin(np.deg2rad(alpha_geom))
        cx_,cy_ = raw[0]; raw[:,0]-=cx_; raw[:,1]-=cy_
        raw = raw@np.array([[c,s],[-s,c]])
        raw[:,0]+=cx_; raw[:,1]+=cy_
    return _close(raw)

def shape_ellipse(cx=.50,cy=.50,a=.28,b=.14,n=140):
    t = np.linspace(0,2*np.pi,n,endpoint=False)
    return _close(np.column_stack([cx+a*np.cos(t), cy+b*np.sin(t)]))

def shape_rectangle(x0=.35,y0=.35,w=.30,h=.30):
    return _close([[x0,y0],[x0+w,y0],[x0+w,y0+h],[x0,y0+h]])

def shape_cow():
    pts = [
        (.140,.490),(.125,.510),(.118,.535),(.120,.565),(.130,.590),
        (.148,.620),(.168,.638),(.192,.642),
        (.210,.650),(.228,.660),(.238,.650),
        (.258,.668),(.290,.678),(.322,.678),
        (.355,.682),(.385,.690),
        (.430,.685),(.480,.682),(.530,.684),
        (.575,.692),(.610,.698),
        (.645,.695),(.672,.685),(.690,.668),
        (.705,.645),(.715,.615),(.718,.580),
        (.720,.545),(.725,.518),(.720,.492),
        (.710,.450),(.695,.408),(.672,.382),
        (.640,.362),(.600,.348),(.555,.340),
        (.510,.337),(.465,.338),(.420,.342),
        (.372,.350),(.335,.362),(.305,.382),
        (.278,.410),(.260,.445),(.248,.470),
        (.238,.488),(.230,.505),(.218,.510),
        (.200,.505),(.185,.498),(.165,.490),(.150,.488),
    ]
    return _close(pts)

def shape_car():
    pts = [
        (.12,.310),(.82,.310),
        (.84,.330),(.86,.360),(.84,.385),(.78,.390),(.72,.385),(.70,.360),(.72,.335),
        (.78,.395),(.82,.430),(.82,.520),
        (.78,.580),(.72,.620),
        (.62,.645),(.52,.650),(.42,.648),(.35,.640),
        (.28,.598),(.22,.535),
        (.18,.480),(.14,.440),(.12,.400),(.12,.365),
        (.14,.335),(.20,.315),(.26,.315),(.28,.335),(.26,.360),
        (.20,.365),(.18,.350),
    ]
    return _close(pts)

def shape_truck():
    trailer = _close([(.06,.29),(.50,.29),(.50,.72),(.06,.72)])
    cab     = _close([(.51,.29),(.76,.29),(.80,.34),(.82,.42),
                      (.82,.60),(.78,.66),(.71,.68),(.62,.67),
                      (.55,.63),(.52,.56),(.51,.46)])
    return [trailer, cab]

def shape_cyclist():
    pts = [
        (.62,.27),(.68,.26),(.74,.27),(.78,.31),(.80,.36),(.78,.42),
        (.74,.46),(.68,.47),(.62,.46),(.58,.42),(.56,.36),(.58,.31),
        (.60,.50),(.55,.54),
        (.50,.56),(.44,.62),(.38,.66),(.32,.67),
        (.28,.70),(.24,.73),(.22,.76),(.23,.79),(.27,.81),(.32,.81),
        (.36,.78),(.37,.74),
        (.34,.69),(.30,.62),(.28,.56),(.26,.50),
        (.26,.46),(.28,.42),
        (.30,.46),(.26,.46),(.22,.42),(.20,.36),(.22,.30),(.28,.26),
        (.34,.26),(.40,.30),(.42,.36),(.40,.42),(.36,.46),
        (.50,.47),(.54,.44),(.56,.40),
    ]
    return _close(pts)

def shape_f1_car():
    """Simplified F1 car side-view with nose cone and rear wing."""
    # Main body
    body = _close([
        (.08,.45),(.10,.40),(.15,.36),(.25,.34),(.30,.36),
        (.32,.35),(.36,.34),(.64,.34),(.68,.35),(.72,.36),
        (.78,.38),(.82,.42),(.84,.46),(.84,.54),(.82,.57),
        (.78,.58),(.72,.60),(.68,.58),(.64,.56),
        (.62,.57),(.58,.58),(.50,.58),(.42,.57),(.38,.58),
        (.34,.57),(.30,.58),(.26,.56),(.20,.54),(.16,.52),(.12,.50),(.08,.48),
    ])
    # Front wing
    front_wing = _close([
        (.05,.36),(.22,.36),(.22,.38),(.05,.38)
    ])
    # Rear wing
    rear_wing = _close([
        (.78,.58),(.90,.58),(.90,.62),(.78,.62)
    ])
    return [body, front_wing, rear_wing]

SHAPES = {
    "Cow":               (lambda: [shape_cow()],                0.38),
    "Circle":            (lambda: [shape_circle()],             0.36),
    "NACA 0012":         (lambda: [shape_naca("0012")],         0.46),
    "NACA 2412":         (lambda: [shape_naca("2412")],         0.46),
    "NACA 4412":         (lambda: [shape_naca("4412")],         0.46),
    "Ellipse":           (lambda: [shape_ellipse()],            0.56),
    "Rectangle":         (lambda: [shape_rectangle()],          0.30),
    "Car":               (lambda: [shape_car()],                0.70),
    "Truck":             (lambda: shape_truck(),                0.94),
    "Cyclist":           (lambda: [shape_cyclist()],            0.62),
    "F1 Car":            (lambda: shape_f1_car(),               0.82),
}

# ═══════════════════════════════════════════════════════════════════════════════
# HESS-SMITH SOURCE PANEL METHOD
# ═══════════════════════════════════════════════════════════════════════════════

def _panel_geom(poly):
    x1,y1 = poly[:-1,0], poly[:-1,1]
    x2,y2 = poly[1:, 0], poly[1:, 1]
    L  = np.hypot(x2-x1, y2-y1)
    β  = np.arctan2(y2-y1, x2-x1)
    xc = .5*(x1+x2); yc = .5*(y1+y2)
    nx = -np.sin(β);  ny =  np.cos(β)
    return x1, y1, xc, yc, L, β, nx, ny

def _src(xp, yp, Lj):
    eps  = 1e-14
    r1sq = np.maximum(xp**2 + yp**2,        eps)
    r2sq = np.maximum((xp-Lj)**2 + yp**2,   eps)
    u = 1/(4*np.pi)*np.log(r2sq/r1sq)
    v = 1/(2*np.pi)*(np.arctan2(yp, xp-Lj) - np.arctan2(yp, xp))
    return u, v

def solve_panels(polys, U_inf=1., alpha_rad=0.):
    segs  = [_panel_geom(p) for p in polys]
    x1c   = np.concatenate([s[0] for s in segs])
    y1c   = np.concatenate([s[1] for s in segs])
    xcc   = np.concatenate([s[2] for s in segs])
    ycc   = np.concatenate([s[3] for s in segs])
    Lc    = np.concatenate([s[4] for s in segs])
    βc    = np.concatenate([s[5] for s in segs])
    nxc   = np.concatenate([s[6] for s in segs])
    nyc   = np.concatenate([s[7] for s in segs])

    dxij  = xcc[:,None] - x1c[None,:]
    dyij  = ycc[:,None] - y1c[None,:]
    cosβ  = np.cos(βc[None,:]); sinβ = np.sin(βc[None,:])
    xp    =  dxij*cosβ + dyij*sinβ
    yp    = -dxij*sinβ + dyij*cosβ
    ul,vl = _src(xp, yp, Lc[None,:])
    ug    = ul*cosβ - vl*sinβ
    vg    = ul*sinβ + vl*cosβ
    A     = ug*nxc[:,None] + vg*nyc[:,None]
    np.fill_diagonal(A, .5)
    b     = -(U_inf*np.cos(alpha_rad)*nxc + U_inf*np.sin(alpha_rad)*nyc)
    q     = np.linalg.solve(A, b)
    return q, x1c, y1c, Lc, βc

def velocity_field(X, Y, q, x1, y1, L, β, U_inf, alpha_rad, CHUNK=4096):
    shape = X.shape
    Xf    = X.ravel(); Yf = Y.ravel()
    M     = len(Xf)
    cosβ  = np.cos(β); sinβ = np.sin(β)
    uo    = np.full(M, U_inf*np.cos(alpha_rad))
    vo    = np.full(M, U_inf*np.sin(alpha_rad))
    for s in range(0, M, CHUNK):
        sl   = slice(s, s+CHUNK)
        dxij = Xf[sl,None] - x1[None,:]
        dyij = Yf[sl,None] - y1[None,:]
        xp   =  dxij*cosβ[None,:] + dyij*sinβ[None,:]
        yp   = -dxij*sinβ[None,:] + dyij*cosβ[None,:]
        ul,vl = _src(xp, yp, L[None,:])
        ul   *= q[None,:]; vl *= q[None,:]
        uo[sl] += np.sum(ul*cosβ[None,:] - vl*sinβ[None,:], axis=1)
        vo[sl] += np.sum(ul*sinβ[None,:] + vl*cosβ[None,:], axis=1)
    return uo.reshape(shape), vo.reshape(shape)

def build_mask(polys, X, Y):
    pts  = np.column_stack([X.ravel(), Y.ravel()])
    mask = np.zeros(X.size, dtype=bool)
    for p in polys:
        mask |= Path(p).contains_points(pts)
    return mask.reshape(X.shape)

def pressure_coeff(speed, V_inf):
    return np.where(speed > 0, 1. - (speed/(V_inf+1e-12))**2, 1.)

def integrate_forces(polys, q, x1, y1, L, β, V_inf, alpha_rad, ref_L):
    xcc  = x1 + .5*L*np.cos(β); ycc = y1 + .5*L*np.sin(β)
    us,vs = velocity_field(xcc[None,:], ycc[None,:],
                           q, x1, y1, L, β, V_inf, alpha_rad)
    Vs   = np.hypot(us.ravel(), vs.ravel())
    Cp   = pressure_coeff(Vs, V_inf)
    nx   = -np.sin(β); ny = np.cos(β)
    Fx   = np.sum(-Cp*nx*L); Fy = np.sum(-Cp*ny*L)
    ca,sa = np.cos(alpha_rad), np.sin(alpha_rad)
    D    =  Fx*ca + Fy*sa
    Lf   = -Fx*sa + Fy*ca
    return D/(ref_L+1e-12), Lf/(ref_L+1e-12)

# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE TRACING  (RK4, normalised velocities)
# ═══════════════════════════════════════════════════════════════════════════════

def trace_particles(u_grid, v_grid, x1d, y1d, mask_grid,
                    n_particles=110, n_frames=100, dt=0.008, trail=55):
    """
    RK4 particle tracing in normalised-velocity field.
    Particles that enter solid bodies are immediately reinjected at the inlet.
    Returns history (list of position arrays), speeds list, and trail length.
    """
    un = np.where(mask_grid, 0., u_grid)
    vn = np.where(mask_grid, 0., v_grid)
    V_ref = np.nanmax(np.hypot(un, vn)) + 1e-9
    un /= V_ref; vn /= V_ref

    fu = RegularGridInterpolator((y1d, x1d), un, method='linear',
                                 bounds_error=False, fill_value=0.)
    fv = RegularGridInterpolator((y1d, x1d), vn, method='linear',
                                 bounds_error=False, fill_value=0.)
    # Mask interpolator: value > 0.4 → inside a solid body
    fmask = RegularGridInterpolator(
        (y1d, x1d), mask_grid.astype(float),
        method='linear', bounds_error=False, fill_value=0.)

    def vel(pts):
        yx = np.clip(pts, [x1d[0],y1d[0]], [x1d[-1],y1d[-1]])[:,::-1]
        return fu(yx), fv(yx)

    def inside_body(pts):
        yx = np.clip(pts, [x1d[0],y1d[0]], [x1d[-1],y1d[-1]])[:,::-1]
        return fmask(yx) > 0.4

    def rk4(pts, dt):
        k1u,k1v = vel(pts);            d1 = np.c_[k1u,k1v]
        k2u,k2v = vel(pts+.5*dt*d1);  d2 = np.c_[k2u,k2v]
        k3u,k3v = vel(pts+.5*dt*d2);  d3 = np.c_[k3u,k3v]
        k4u,k4v = vel(pts+dt*d3);     d4 = np.c_[k4u,k4v]
        return pts + dt/6*(d1+2*d2+2*d3+d4)

    y_inj = np.linspace(0.02, 0.98, n_particles)
    x_inj = np.full(n_particles, 0.01)
    pts   = np.column_stack([x_inj, y_inj])

    fspd = RegularGridInterpolator(
        (y1d, x1d), np.where(mask_grid, 0., np.hypot(u_grid, v_grid)),
        method='linear', bounds_error=False, fill_value=0.)

    def raw_spd(p):
        yx = np.clip(p, [x1d[0],y1d[0]], [x1d[-1],y1d[-1]])[:,::-1]
        return fspd(yx)

    history = [pts.copy()]
    speeds  = [raw_spd(pts)]
    rng     = np.random.default_rng(0)

    for _ in range(n_frames - 1):
        new_pts = rk4(pts, dt)
        u_c, v_c = vel(new_pts); spd_n = np.hypot(u_c, v_c)
        exited  = ((new_pts[:,0] > 1.01) | (new_pts[:,0] < -.01) |
                   (new_pts[:,1] > 1.01) | (new_pts[:,1] < -.01))
        stalled = spd_n < 5e-4
        masked  = inside_body(new_pts)          # hit solid body → reinject
        ri = exited | stalled | masked
        if ri.any():
            new_pts[ri, 0] = x_inj[ri]
            new_pts[ri, 1] = y_inj[ri] + rng.uniform(-.01, .01, ri.sum())
        pts = np.clip(new_pts, 0., 1.)
        history.append(pts.copy())
        speeds.append(raw_spd(pts))

    return history, speeds, trail

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY — LIVE VAPOR TUNNEL ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def _body_traces(polys, zval=None, mode3d=False):
    """Plotly traces for body silhouettes (2-D or 3-D)."""
    out = []
    for poly in polys:
        x = np.append(poly[:,0], poly[0,0])
        y = np.append(poly[:,1], poly[0,1])
        if mode3d:
            z = np.zeros_like(x) if zval is None else np.full_like(x, zval)
            out.append(go.Scatter3d(
                x=x, y=y, z=z, mode='lines',
                line=dict(color='rgba(255,255,255,0.85)', width=3),
                showlegend=False, hoverinfo='skip'))
        else:
            out.append(go.Scatter(
                x=x, y=y, mode='lines', fill='toself',
                fillcolor='rgba(15,18,30,0.97)',
                line=dict(color='rgba(210,225,255,0.9)', width=1.8),
                showlegend=False, hoverinfo='skip'))
    return out

def _dark_axes_2d(fig, title=""):
    fig.update_layout(
        paper_bgcolor='#0b0f18', plot_bgcolor='#0b0f18',
        height=570, margin=dict(l=8,r=8,t=46,b=72),
        title=dict(text=title, font=dict(color='white',size=13),
                   x=.5, xanchor='center'),
        xaxis=dict(range=[0,1], showgrid=False, zeroline=False,
                   color='#667788', title='x / L',
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[0,1], showgrid=False, zeroline=False,
                   color='#667788', title='y / L'),
        font=dict(color='white'))

def build_vapor_figure(history, speeds, trail_len,
                       speed_field, Cp_field, X, Y, polys,
                       V_inf, alpha_deg,
                       bg_field="Speed |V|", trail_style="Vapor (white)"):
    """3-D animated vapor tunnel: particles distributed across tunnel depth."""
    n_frames = len(history)
    n_part   = history[0].shape[0]

    Z_SPREAD = 0.70                                          # tunnel depth span
    y_depth  = np.linspace(-Z_SPREAD/2, Z_SPREAD/2, n_part) # per-particle depth

    # ── Back-wall surface showing flow field ──────────────────────────────
    if bg_field == "Speed |V|":
        surf_c = np.where(np.isnan(speed_field), 0, speed_field / (V_inf + 1e-9))
        cs, zmin_bg, zmax_bg, cbar_txt = 'jet', 0, 2.5, "V / V∞"
    elif bg_field == "Pressure Cp":
        surf_c = np.clip(np.where(np.isnan(Cp_field), 0, -Cp_field), 0, 4)
        cs, zmin_bg, zmax_bg, cbar_txt = 'RdBu', 0, 4, "−Cp"
    else:
        surf_c = np.zeros_like(speed_field)
        cs, zmin_bg, zmax_bg, cbar_txt = 'greys', 0, 1, ""

    back_wall = go.Surface(
        x=X, y=np.full_like(X, -Z_SPREAD/2 - 0.04), z=Y,
        surfacecolor=surf_c, colorscale=cs, cmin=zmin_bg, cmax=zmax_bg,
        showscale=(bg_field != "Off (dark)"),
        colorbar=dict(title=cbar_txt, thickness=10, len=0.6,
                      tickfont=dict(color='#aaa')),
        lighting=dict(ambient=1.0, diffuse=0., roughness=0., specular=0.),
        opacity=0.88, showlegend=False, hoverinfo='skip', name='field')

    # ── Extruded body: rings at multiple depth slices + spanwise ribs ─────
    bodies_3d = []
    ring_depths = np.linspace(-Z_SPREAD/2, Z_SPREAD/2, 9)
    for poly in polys:
        xb = np.append(poly[:,0], poly[0,0])
        zb = np.append(poly[:,1], poly[0,1])   # flow-y → Plotly-z (height)
        for yp in ring_depths:
            bodies_3d.append(go.Scatter3d(
                x=xb, y=np.full(len(xb), float(yp)), z=zb,
                mode='lines',
                line=dict(color='rgba(200,220,255,0.50)', width=2),
                showlegend=False, hoverinfo='skip'))
        # Spanwise ribs connecting front/back faces
        n_ribs = 14
        step = max(1, len(poly) // n_ribs)
        for i in range(0, len(poly), step):
            bodies_3d.append(go.Scatter3d(
                x=[poly[i,0], poly[i,0]],
                y=[-Z_SPREAD/2, Z_SPREAD/2],
                z=[poly[i,1], poly[i,1]],
                mode='lines',
                line=dict(color='rgba(200,220,255,0.30)', width=1),
                showlegend=False, hoverinfo='skip'))

    n_static  = 1 + len(bodies_3d)
    trail_idx = n_static

    # ── 3-D trail builder (x=flow-x, y=depth, z=flow-y/height) ──────────
    def make_trail(k):
        t0   = max(0, k - trail_len + 1)
        ages = list(range(t0, k + 1))
        xs, ys, zs, cs_arr = [], [], [], []

        for p in range(n_part):
            yp     = float(y_depth[p])
            prev_x = prev_z = None
            for t in ages:
                cx = float(history[t][p, 0])
                cz = float(history[t][p, 1])
                # Insert gap when particle teleports back to inlet
                if prev_x is not None and abs(cx - prev_x) + abs(cz - prev_z) > 0.15:
                    xs.append(None); ys.append(None)
                    zs.append(None); cs_arr.append(None)
                xs.append(cx); ys.append(yp); zs.append(cz)
                cs_arr.append(float(speeds[t][p]))
                prev_x, prev_z = cx, cz
            xs.append(None); ys.append(None); zs.append(None); cs_arr.append(None)

        if trail_style == "Vapor (white)":
            return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                line=dict(color='rgba(195,225,255,0.55)', width=2),
                showlegend=False, hoverinfo='skip', name='trail')
        elif trail_style == "Neon":
            return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                line=dict(color='rgba(60,255,180,0.60)', width=2),
                showlegend=False, hoverinfo='skip', name='trail')
        else:  # Speed-colored dots
            return go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                marker=dict(color=cs_arr, colorscale='hot',
                            cmin=0, cmax=float(V_inf * 2), size=2,
                            showscale=False),
                showlegend=False, hoverinfo='skip', name='trail')

    init_trail = make_trail(0)
    all_data   = [back_wall] + bodies_3d + [init_trail]

    frames = [
        go.Frame(data=[make_trail(k)], traces=[trail_idx], name=str(k))
        for k in range(n_frames)
    ]

    fig = go.Figure(data=all_data, frames=frames)
    fig.update_layout(
        paper_bgcolor='#0b0f18', height=640,
        margin=dict(l=0, r=0, t=50, b=72),
        title=dict(
            text=(f"Live 3D Vapor Tunnel  ·  {n_part} streams  ·  "
                  f"V∞={V_inf:.0f} m/s  ·  α={alpha_deg}°"),
            font=dict(color='white', size=13), x=.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title=dict(text='x / L', font=dict(color='#778899')),
                       gridcolor='#1e2535', zerolinecolor='#1e2535',
                       tickfont=dict(color='#778899'),
                       showbackground=True, backgroundcolor='#0b0f18'),
            yaxis=dict(title=dict(text='depth', font=dict(color='#778899')),
                       gridcolor='#1e2535', zerolinecolor='#1e2535',
                       tickfont=dict(color='#778899'),
                       showbackground=True, backgroundcolor='#0b0f18'),
            zaxis=dict(title=dict(text='y / L', font=dict(color='#778899')),
                       gridcolor='#1e2535', zerolinecolor='#1e2535',
                       tickfont=dict(color='#778899'),
                       showbackground=True, backgroundcolor='#0b0f18'),
            camera=dict(eye=dict(x=-1.5, y=-2.0, z=1.0),
                        up=dict(x=0, y=0, z=1)),
            aspectmode='auto'),
        font=dict(color='white'))

    # ── Animation controls ────────────────────────────────────────────────
    btn_style = dict(bgcolor='#1c2235', bordercolor='#3d8bff',
                     font=dict(color='white'))
    fig.update_layout(
        updatemenus=[dict(
            type='buttons', showactive=False, **btn_style,
            y=-.08, x=.5, xanchor='center', yanchor='top',
            pad=dict(t=0),
            buttons=[
                dict(label='▶  Play', method='animate',
                     args=[None, dict(frame=dict(duration=60, redraw=True),
                                     fromcurrent=True, loop=True,
                                     transition=dict(duration=0))]),
                dict(label='⏸  Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=True),
                                        mode='immediate',
                                        transition=dict(duration=0))])
            ])],
        sliders=[dict(
            active=0, bgcolor='#1c2235', bordercolor='#3d8bff',
            tickcolor='#667788', font=dict(color='#667788', size=9),
            yanchor='top', xanchor='left',
            currentvalue=dict(visible=False),
            pad=dict(b=4, t=52), len=1., x=0., y=0.,
            steps=[dict(method='animate',
                        args=[[str(k)], dict(mode='immediate',
                              frame=dict(duration=60, redraw=True),
                              transition=dict(duration=0))],
                        label='') for k in range(n_frames)])])
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY — 3-D VELOCITY SURFACE
# ═══════════════════════════════════════════════════════════════════════════════

def build_3d_figure(X, Y, speed_field, Cp_field, polys, V_inf, field_3d="Speed |V|"):
    x1d = X[0,:]; y1d = Y[:,0]

    if field_3d == "Speed |V|":
        Z   = np.where(np.isnan(speed_field), 0, speed_field/V_inf)
        cs  = 'jet'; zlab = 'V / V∞'; zlim = [0, 2.5]
    else:
        Z   = np.clip(np.where(np.isnan(Cp_field), 0, -Cp_field), 0, 4)
        cs  = 'RdBu'; zlab = '−Cp'; zlim = [0, 4]

    surf = go.Surface(
        x=X, y=Y, z=Z,
        colorscale=cs, cmin=zlim[0], cmax=zlim[1],
        showscale=True,
        colorbar=dict(title=zlab, tickfont=dict(color='#aaa'),
                      thickness=12, len=.65),
        lighting=dict(ambient=.45, diffuse=.85, roughness=.4,
                      specular=.4, fresnel=.3),
        opacity=.96, name='field')

    # Project body onto floor (z = 0) and lift wall at z=0 plane
    bodies_3d = _body_traces(polys, zval=0., mode3d=True)

    # Also draw body "walls" as thin vertical ribbons for realism
    wall_traces = []
    for poly in polys:
        for i in range(len(poly)-1):
            x0,y0 = poly[i]; x1,y1 = poly[i+1]
            # Get Z values at these two points (bilinear lookup)
            ix0 = int(np.clip(x0*(len(x1d)-1), 0, len(x1d)-2))
            iy0 = int(np.clip(y0*(len(y1d)-1), 0, len(y1d)-2))
            z_h = float(Z[iy0, ix0]) if 0<=iy0<Z.shape[0] and 0<=ix0<Z.shape[1] else 0.
            wall_traces.append(go.Scatter3d(
                x=[x0,x0], y=[y0,y0], z=[0., z_h],
                mode='lines',
                line=dict(color='rgba(200,215,255,0.25)', width=1),
                showlegend=False, hoverinfo='skip'))

    fig3 = go.Figure(data=[surf] + bodies_3d + wall_traces)
    fig3.update_layout(
        paper_bgcolor='#0b0f18', height=620,
        margin=dict(l=0,r=0,t=50,b=0),
        title=dict(text=f'3D {zlab} Surface  ·  drag to rotate  ·  scroll to zoom',
                   font=dict(color='white', size=13), x=.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title=dict(text='x / L', font=dict(color='#778899')),
                       gridcolor='#202535', zerolinecolor='#202535',
                       tickfont=dict(color='#778899'),
                       showbackground=True, backgroundcolor='#0d1420'),
            yaxis=dict(title=dict(text='y / L', font=dict(color='#778899')),
                       gridcolor='#202535', zerolinecolor='#202535',
                       tickfont=dict(color='#778899'),
                       showbackground=True, backgroundcolor='#0d1420'),
            zaxis=dict(title=dict(text=zlab, font=dict(color='#778899')),
                       range=zlim, gridcolor='#202535', zerolinecolor='#202535',
                       tickfont=dict(color='#778899'),
                       showbackground=True, backgroundcolor='#0d1420'),
            camera=dict(eye=dict(x=-1.35, y=-1.75, z=1.45),
                        up=dict(x=0, y=0, z=1)),
            aspectmode='cube'),
        font=dict(color='white'))
    return fig3

# ═══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB STATIC CFD PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def build_cfd_figure(X, Y, u, v, speed, Cp_field, vorticity, mach, mask,
                     polys, V_inf, alpha_deg, xs, ys,
                     vis_mode, cmap_choice, show_sl, show_vec, n_sl, grid_res):
    fig, ax = plt.subplots(figsize=(10,7.5))
    fig.patch.set_facecolor("#0b0f18"); ax.set_facecolor("#0b0f18")

    if vis_mode == "Speed |V|":
        field = np.where(mask, np.nan, speed)
        vmin,vmax,clab = 0, V_inf*2.2, "Speed  (m/s)"
    elif vis_mode == "Pressure Cp":
        field = np.where(mask, np.nan, Cp_field)
        vmin,vmax,clab = -3.5, 1., "Pressure coefficient  Cp"
    elif vis_mode == "Vorticity":
        field = np.where(mask, np.nan, vorticity)
        lim   = np.nanpercentile(np.abs(field[~mask]),97) if np.any(~mask) else 1
        vmin,vmax,clab = -lim, lim, "Vorticity  (1/s)"
    else:
        field = np.where(mask, np.nan, mach)
        vmin,vmax,clab = 0, V_inf/343*2.2, "Mach  M"

    im   = ax.pcolormesh(X, Y, field, cmap=cmap_choice, shading='gouraud',
                         vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cbar.set_label(clab, color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    if show_sl:
        u_sl = np.where(mask,0,u); v_sl = np.where(mask,0,v)
        try:
            ax.streamplot(X[0,:], Y[:,0], u_sl, v_sl,
                          color='white', linewidth=.55,
                          density=n_sl, arrowsize=.9, arrowstyle='->')
        except Exception:
            pass

    if show_vec:
        st = max(1, grid_res//18)
        Xq,Yq = X[::st,::st], Y[::st,::st]
        Uq,Vq = u[::st,::st].copy(), v[::st,::st].copy()
        mq    = mask[::st,::st]; Uq[mq]=0; Vq[mq]=0
        ax.quiver(Xq,Yq,Uq,Vq,color='#ffdd44',alpha=.65,
                  scale=V_inf*28, width=.0018, headwidth=3)

    if xs is not None:
        ax.plot(xs, ys, 'o', color='#ff4444', ms=7, zorder=10,
                label='Stagnation pt')
        ax.legend(facecolor='#1a1a2e', labelcolor='white',
                  fontsize=9, loc='upper right')

    for poly in polys:
        ax.add_patch(MplPolygon(poly, closed=True,
                                facecolor='#1e2235', edgecolor='white',
                                linewidth=1.5, zorder=5))

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    ax.set_title(f"{vis_mode}  ·  V∞={V_inf:.0f} m/s  ·  α={alpha_deg}°",
                 color='white', fontsize=12, pad=8)
    ax.tick_params(colors='#668899')
    for sp in ax.spines.values(): sp.set_edgecolor('#2a3040')
    ax.set_xlabel('x / L', color='#668899'); ax.set_ylabel('y / L', color='#668899')
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("## ✈  Wind Tunnel Simulator")
st.markdown(
    "**2D potential flow** · Hess-Smith source panel method · "
    "live animated smoke tunnel · interactive 3D surface")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🌬 Tunnel Controls")
    shape_name = st.selectbox("Body shape", list(SHAPES.keys()))

    st.subheader("Flow Conditions")
    V_inf     = st.slider("Free-stream  V∞  (m/s)", 1., 80., 20., 1.)
    alpha_deg = st.slider("Angle of attack  α  (°)", -20, 20, 0, 1)
    rho       = st.select_slider("Air density  ρ  (kg/m³)",
                                 [0.413,0.736,1.225,1.293,1.654], value=1.225,
                                 format_func=lambda x: f"{x:.3f}")

    st.subheader("Solver")
    N_panels = st.select_slider("Panels per body", [40,60,80,100,140], value=80)
    grid_res = st.select_slider("Grid resolution", [60,80,100,120], value=80)

    st.subheader("Smoke Tunnel")
    n_particles  = st.slider("Particle streams", 50, 180, 110, 10)
    n_frames     = st.slider("Animation frames", 60, 160, 100, 10)
    trail_len    = st.slider("Trail length", 10, 100, 55, 5)
    trail_style  = st.selectbox("Trail style",
                                ["Vapor (white)", "Neon", "Speed-colored dots"])
    bg_field_ani = st.selectbox("Background (animation)",
                                ["Speed |V|", "Pressure Cp", "Off (dark)"])

    st.subheader("CFD Field Plot")
    vis_mode     = st.selectbox("Colour field",
                                ["Speed |V|","Pressure Cp","Vorticity","Mach"])
    cmap_choice  = st.selectbox("Colourmap",
                                ["jet","turbo","RdBu_r","plasma","coolwarm"])
    show_sl      = st.checkbox("Streamlines", True)
    show_vec     = st.checkbox("Velocity vectors", False)
    n_sl         = st.slider("Streamline density", 0.8, 3.0, 1.8, 0.2)

    st.subheader("3D View")
    field_3d = st.selectbox("3D field", ["Speed |V|", "Pressure Cp"])

    run = st.button("▶  Run Simulation", type="primary", use_container_width=True)

# ── Derived ───────────────────────────────────────────────────────────────────
alpha_rad = np.deg2rad(alpha_deg)
dyn_press = .5 * rho * V_inf**2
nu        = 1.48e-5
shape_fn, ref_L = SHAPES[shape_name]

# ── Resample helper ───────────────────────────────────────────────────────────
def resample(poly, n):
    from scipy.interpolate import interp1d
    d  = np.cumsum(np.r_[0, np.hypot(np.diff(poly[:,0]), np.diff(poly[:,1]))])
    d /= d[-1]
    t  = np.linspace(0, 1, n+1)
    return np.column_stack([interp1d(d,poly[:,0])(t), interp1d(d,poly[:,1])(t)])

# ── Solve / cache ──────────────────────────────────────────────────────────────
solve_key   = (shape_name, round(V_inf,1), alpha_deg, N_panels, grid_res)
particle_key = (*solve_key, n_particles, n_frames, trail_len)

if run or "solve_key" not in st.session_state or st.session_state["solve_key"] != solve_key:
    with st.spinner("Solving panel system and computing flow field…"):
        polys_raw = shape_fn()
        polys     = [resample(p, N_panels) for p in polys_raw]
        q, x1,y1,L,β = solve_panels(polys, V_inf, alpha_rad)

        x   = np.linspace(0,1,grid_res); y = np.linspace(0,1,grid_res)
        X,Y = np.meshgrid(x, y)
        u,v = velocity_field(X, Y, q, x1,y1,L,β, V_inf, alpha_rad)
        mask = build_mask(polys, X, Y)
        u[mask]=0; v[mask]=0

        speed    = np.hypot(u, v)
        Cp_field = pressure_coeff(speed, V_inf)
        Cp_field[mask] = np.nan; speed_disp = speed.copy(); speed_disp[mask] = np.nan

        dx_=x[1]-x[0]; dy_=y[1]-y[0]
        vorticity = np.gradient(v,dx_,axis=1) - np.gradient(u,dy_,axis=0)
        vorticity[mask]=0
        mach = speed/343.

        Cd,Cl = integrate_forces(polys, q, x1,y1,L,β, V_inf, alpha_rad, ref_L)

        try:
            sz  = binary_dilation(mask, iterations=4) & ~mask
            spd_sz = np.where(sz, np.hypot(u,v), np.inf)
            idx = np.unravel_index(np.argmin(spd_sz), spd_sz.shape)
            xs,ys = float(X[idx]), float(Y[idx])
        except Exception:
            xs=ys=None

        st.session_state.update({
            "solve_key": solve_key,
            "panels":    (q,x1,y1,L,β),
            "flow":      (X,Y,u,v,speed_disp,Cp_field,vorticity,mach,mask,polys),
            "forces":    (Cd,Cl,xs,ys),
        })

if run or st.session_state.get("particle_key") != particle_key:
    with st.spinner("Tracing particle streams…"):
        _X,_Y,_u,_v,*_ = st.session_state["flow"]
        mask_p = st.session_state["flow"][8]
        history, spd_hist, _tl = trace_particles(
            _u, _v, _X[0,:], _Y[:,0], mask_p,
            n_particles=n_particles, n_frames=n_frames, trail=trail_len)
        st.session_state["particles"]   = (history, spd_hist, trail_len)
        st.session_state["particle_key"] = particle_key

# ── Unpack ────────────────────────────────────────────────────────────────────
X,Y,u,v,speed_disp,Cp_field,vorticity,mach,mask,polys = st.session_state["flow"]
Cd,Cl,xs,ys = st.session_state["forces"]
history, spd_hist, _tl = st.session_state["particles"]
q,x1,y1,L,β = st.session_state["panels"]
Re = V_inf * ref_L / nu

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
t_vapor, t_3d, t_cfd, t_pres, t_aero, t_theory = st.tabs([
    "🌫 Live Vapor Tunnel", "🏔 3D Surface", "🎨 CFD Field",
    "📈 Pressure Dist.", "📊 Aero Data", "📐 Theory"
])

# ── Tab 1: Live Vapor Tunnel ─────────────────────────────────────────────────
with t_vapor:
    st.markdown(
        "Press **▶ Play** below the chart to start the animation.  "
        "The slider lets you scrub through frames manually.")
    vapor_fig = build_vapor_figure(
        history, spd_hist, trail_len,
        speed_disp, Cp_field, X, Y, polys,
        V_inf, alpha_deg, bg_field_ani, trail_style)
    st.plotly_chart(vapor_fig, use_container_width=True,
                    config=dict(displayModeBar=False))

    ca,cb,cc,cd_ = st.columns(4)
    def _m(col,label,val,unit=""):
        col.markdown(
            f'<div class="metric-box"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}'
            f'<small style="color:#556677"> {unit}</small></div></div>',
            unsafe_allow_html=True)
    _m(ca,"V∞",f"{V_inf:.0f}","m/s")
    _m(cb,"Re",f"{Re:.2e}".replace("e+","×10^"),"")
    _m(cc,"Cd (press.)",f"{Cd:.4f}","")
    _m(cd_,"Cl (press.)",f"{Cl:.4f}","")

# ── Tab 2: 3D Surface ─────────────────────────────────────────────────────────
with t_3d:
    st.markdown(
        "**Drag** to rotate · **Scroll** to zoom · **Double-click** to reset view.  "
        "Z-axis shows the selected scalar lifted into 3D space.")
    fig3d = build_3d_figure(X, Y, speed_disp, Cp_field, polys, V_inf, field_3d)
    st.plotly_chart(fig3d, use_container_width=True,
                    config=dict(displayModeBar=True,
                                modeBarButtonsToRemove=['toImage']))

# ── Tab 3: CFD Field ──────────────────────────────────────────────────────────
with t_cfd:
    colA, colB = st.columns([3,1])
    with colA:
        cfd_fig = build_cfd_figure(
            X,Y,u,v,speed_disp,Cp_field,vorticity,mach,mask,polys,
            V_inf,alpha_deg,xs,ys,
            vis_mode,cmap_choice,show_sl,show_vec,n_sl,grid_res)
        st.pyplot(cfd_fig, use_container_width=True)
    with colB:
        st.markdown("### Metrics")
        ext = speed_disp[~mask]
        _m(colB,"V∞",f"{V_inf:.1f}","m/s")
        _m(colB,"Max speed",f"{np.nanmax(ext):.2f}","m/s")
        _m(colB,"Min speed",f"{np.nanmin(ext):.4f}","m/s")
        _m(colB,"Dyn. press. q∞",f"{dyn_press:.1f}","Pa")
        _m(colB,"Reynolds Re",f"{Re:.2e}".replace("e+","×10^"),"")
        _m(colB,"Max |Cp|",f"{np.nanmax(np.abs(Cp_field[~mask])):.3f}","")
        _m(colB,"Cd",f"{Cd:.4f}","")
        _m(colB,"Cl",f"{Cl:.4f}","")
        if xs:
            _m(colB,"Stag. x",f"{xs:.3f}","")
            _m(colB,"Stag. y",f"{ys:.3f}","")

# ── Tab 4: Pressure Distribution ─────────────────────────────────────────────
with t_pres:
    st.markdown("#### Surface  Cp  distribution")
    xc_s = x1 + .5*L*np.cos(β); yc_s = y1 + .5*L*np.sin(β)
    us_,vs_ = velocity_field(xc_s[None,:], yc_s[None,:],
                             q, x1,y1,L,β, V_inf, alpha_rad)
    Cp_surf = pressure_coeff(np.hypot(us_.ravel(), vs_.ravel()), V_inf)
    s = np.cumsum(np.r_[0, L[:-1]]); s /= s[-1]

    fig2, ax2 = plt.subplots(figsize=(10,4))
    fig2.patch.set_facecolor("#0b0f18"); ax2.set_facecolor("#0b0f18")
    ax2.plot(s, Cp_surf, color='#4a9eff', lw=1.6, label='Cp')
    ax2.axhline(0, color='#445',lw=.8,ls='--')
    ax2.axhline(1, color='#44aa55',lw=.8,ls=':',label='Cp=1 (stagnation)')
    ax2.fill_between(s, Cp_surf, 0, where=Cp_surf>0,
                     alpha=.15, color='#ff6644', label='Pressure (+)')
    ax2.fill_between(s, Cp_surf, 0, where=Cp_surf<0,
                     alpha=.15, color='#4488ff', label='Suction (−)')
    ax2.invert_yaxis()
    ax2.set_xlabel('Normalised arc-length  s/S', color='#aaa')
    ax2.set_ylabel('Cp  (−ve = suction, aero convention)', color='#aaa')
    ax2.set_title('Surface pressure coefficient', color='white')
    ax2.tick_params(colors='#888')
    ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor('#2a3040')
    st.pyplot(fig2, use_container_width=True)

    with st.expander("Show raw Cp data table"):
        df = pd.DataFrame({
            's/S': np.round(s,4), 'x':np.round(xc_s,4),
            'y':np.round(yc_s,4), 'Cp':np.round(Cp_surf,4)})
        st.dataframe(df, use_container_width=True, height=280)

# ── Tab 5: Aero Data ──────────────────────────────────────────────────────────
with t_aero:
    st.markdown("#### Aerodynamic polar  (Cl, Cd, L/D  vs  α)")
    polar_key = (shape_name, round(V_inf,1), N_panels)
    if st.session_state.get("polar_key") != polar_key:
        alphas_p = np.linspace(-15,15,25)
        Cds_p,Cls_p = [],[]
        with st.spinner("Computing polar sweep…"):
            for a in alphas_p:
                qa,x1a,y1a,La,βa = solve_panels(polys, V_inf, np.deg2rad(a))
                cd_,cl_ = integrate_forces(polys,qa,x1a,y1a,La,βa,
                                           V_inf,np.deg2rad(a),ref_L)
                Cds_p.append(cd_); Cls_p.append(cl_)
        st.session_state["polar"]     = (alphas_p,np.array(Cds_p),np.array(Cls_p))
        st.session_state["polar_key"] = polar_key

    alphas_p,Cds_p,Cls_p = st.session_state["polar"]
    LD = Cls_p/(Cds_p+1e-9)

    fig_p = go.Figure()
    for ydata,name,col in [
        (Cls_p,"Cl","#4a9eff"),(Cds_p,"Cd","#ff6644"),(LD,"L/D","#44ee88")]:
        fig_p.add_trace(go.Scatter(x=alphas_p, y=ydata, name=name,
                                   line=dict(color=col,width=2.5)))
    fig_p.add_vline(x=alpha_deg, line=dict(color='white',width=1,dash='dot'),
                    annotation_text=f"α={alpha_deg}°",
                    annotation_font_color='white')
    fig_p.update_layout(
        paper_bgcolor='#0b0f18', plot_bgcolor='#0b0f18',
        height=420, margin=dict(l=10,r=10,t=40,b=40),
        xaxis=dict(title='α (°)',color='#aaa',gridcolor='#1e2535',zeroline=True,
                   zerolinecolor='#445'),
        yaxis=dict(color='#aaa',gridcolor='#1e2535'),
        legend=dict(bgcolor='#161c28',bordercolor='#2a3040',font=dict(color='white')),
        font=dict(color='white'),
        title=dict(text='Aerodynamic polar', font=dict(color='white'),
                   x=.5, xanchor='center'))
    st.plotly_chart(fig_p, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Cd  (current α)", f"{Cd:.4f}")
    c2.metric("Cl  (current α)", f"{Cl:.4f}")
    c3.metric("L/D", f"{Cl/(Cd+1e-9):.2f}")
    c4.metric("Drag force  (N/m)", f"{Cd*dyn_press*ref_L:.2f}")
    st.caption(
        "Coefficients from pressure integration of inviscid potential flow.  "
        "Skin-friction drag and flow separation are not modelled.")

# ── Tab 6: Theory ─────────────────────────────────────────────────────────────
with t_theory:
    st.markdown(r"""
## Physics & Numerical Method

### Governing Equations
For **steady, inviscid, irrotational, incompressible** flow the velocity
derives from a scalar potential φ:

$$\mathbf{V} = \nabla\phi, \qquad \nabla^2\phi = 0$$

### Hess-Smith Source Panel Method (1967)
The body is discretised into *N* straight panels each carrying a uniform
source distribution of unknown strength *qⱼ*:

$$\mathbf{V}(\mathbf{r}) = V_\infty\hat{\mathbf{e}}_\infty
  + \sum_j \frac{q_j}{2\pi}
    \int_{\text{panel}\,j}\!\frac{\mathbf{r}-\mathbf{r}'}{|\mathbf{r}-\mathbf{r}'|^2}\,ds$$

Zero normal velocity on each control point gives the **N × N** linear system:

$$\sum_j A_{ij}\,q_j = -V_\infty\,(\hat{\mathbf{e}}_\infty\cdot\hat{\mathbf{n}}_i)$$

Influence coefficients *Aᵢⱼ* are computed analytically.

### Bernoulli Pressure
$$C_p = 1 - \left(\frac{V}{V_\infty}\right)^2$$

| Cp value | Physical meaning |
|----------|-----------------|
| Cp = 1 | Stagnation point — full kinetic → static pressure |
| Cp = 0 | Local speed equals free-stream |
| Cp < 0 | Suction region — flow accelerated over curved surface |
| Cp ≈ −3 | Highly accelerated tip/corner region |

### Particle Tracing
Particles are advanced by **4th-order Runge-Kutta** through the velocity field
(bilinear interpolation via `scipy.interpolate.RegularGridInterpolator`).
Velocities are normalised by *V*max before tracing so animation speed is
view-independent of wind speed.

### Reynolds Number
$$Re = \frac{V_\infty\,L}{\nu}, \qquad \nu_\text{air, 15°C} = 1.48\times10^{-5}\ \text{m}^2/\text{s}$$

### Model Limitations
| Effect | Modelled? |
|--------|-----------|
| Inviscid pressure distribution | ✅ |
| Stagnation / suction regions | ✅ |
| Streamline topology | ✅ |
| Boundary layer / friction drag | ❌ |
| Flow separation & turbulent wake | ❌ |
| Lift from circulation (Kutta condition) | ❌ |
| Compressibility (*M* > 0.3) | ❌ |
| 3-D effects | ❌ |

> **References:**
> J. L. Hess & A. M. O. Smith, *Progress in Aerospace Sciences* **8** (1967).
> J. D. Anderson, *Fundamentals of Aerodynamics*, 6th ed., McGraw-Hill (2017).
""")
