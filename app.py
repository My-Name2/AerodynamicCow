import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

st.set_page_config(page_title="Wind Tunnel Simulator", layout="wide")

st.title("Wind Tunnel Simulator")
st.markdown("Simulate airflow around objects using a 2D potential flow model with CFD-style visualization.")

# --- Object shape definitions ---
def get_cow_polygon():
    """Returns a rough 2D cow silhouette as a polygon (normalized ~0-1 range)."""
    # Body
    body = np.array([
        [0.20, 0.35], [0.22, 0.30], [0.25, 0.28], [0.30, 0.26],
        [0.35, 0.25], [0.60, 0.25], [0.65, 0.27], [0.68, 0.30],
        [0.70, 0.35], [0.72, 0.42], [0.70, 0.55], [0.65, 0.62],
        [0.60, 0.65], [0.55, 0.66], # back / rump
        [0.50, 0.67], [0.45, 0.67],
        [0.40, 0.66], [0.35, 0.65], # mid back
        [0.30, 0.63], [0.25, 0.60],
        [0.20, 0.55], [0.18, 0.48], [0.18, 0.42], [0.20, 0.35],
    ])
    # Head (appended as a bump on the front)
    head = np.array([
        [0.10, 0.45], [0.08, 0.50], [0.07, 0.56], [0.09, 0.62],
        [0.13, 0.65], [0.17, 0.66], [0.21, 0.64], [0.23, 0.60],
        [0.22, 0.55], [0.20, 0.50], [0.18, 0.48],
    ])
    return body, head

def get_shape_polygons(shape_name):
    """Return list of (polygon_array, is_hole) for a given shape name."""
    if shape_name == "Cow":
        body, head = get_cow_polygon()
        # Combine: head attached to body
        combined = np.vstack([
            body[:10],
            head,
            body[10:],
        ])
        return [combined]
    elif shape_name == "Circle":
        theta = np.linspace(0, 2*np.pi, 100)
        r = 0.18
        cx, cy = 0.5, 0.5
        pts = np.column_stack([cx + r*np.cos(theta), cy + r*np.sin(theta)])
        return [pts]
    elif shape_name == "Airfoil (NACA 0012)":
        return [naca0012_polygon()]
    elif shape_name == "Rectangle":
        pts = np.array([[0.35,0.35],[0.65,0.35],[0.65,0.65],[0.35,0.65]])
        return [pts]
    elif shape_name == "Car (Side View)":
        pts = np.array([
            [0.15,0.35],[0.18,0.30],[0.25,0.28],[0.70,0.28],[0.78,0.30],
            [0.80,0.38],[0.80,0.48],[0.70,0.50],[0.62,0.60],[0.40,0.62],
            [0.28,0.58],[0.18,0.50],[0.15,0.42],[0.15,0.35],
        ])
        return [pts]
    elif shape_name == "Truck":
        cab = np.array([
            [0.55,0.28],[0.80,0.28],[0.82,0.32],[0.82,0.62],[0.78,0.65],
            [0.55,0.65],[0.53,0.62],[0.53,0.32],[0.55,0.28],
        ])
        trailer = np.array([
            [0.15,0.28],[0.52,0.28],[0.52,0.70],[0.15,0.70],[0.15,0.28],
        ])
        return [cab, trailer]
    elif shape_name == "Bicycle Rider":
        # Simplified silhouette
        rider = np.array([
            [0.35,0.28],[0.65,0.28],[0.67,0.32],[0.65,0.40],
            [0.60,0.50],[0.62,0.60],[0.58,0.70],[0.52,0.72],
            [0.46,0.70],[0.42,0.60],[0.44,0.50],[0.38,0.40],
            [0.33,0.32],[0.35,0.28],
        ])
        return [rider]
    else:
        # Default: circle
        theta = np.linspace(0, 2*np.pi, 80)
        r = 0.18
        cx, cy = 0.5, 0.5
        pts = np.column_stack([cx + r*np.cos(theta), cy + r*np.sin(theta)])
        return [pts]

def naca0012_polygon():
    """Generate NACA 0012 airfoil polygon."""
    t = 0.12
    c = 0.45
    x_start, y_center = 0.15, 0.50
    x = np.linspace(0, 1, 80)
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    xu = x * c + x_start
    xl = xu.copy()
    yu = y_center + yt * c
    yl = y_center - yt * c
    upper = np.column_stack([xu, yu])
    lower = np.column_stack([xl[::-1], yl[::-1]])
    return np.vstack([upper, lower])

# --- Panel / Potential Flow Simulation ---
def simulate_flow(shape_polygons, grid_size=80, wind_speed=1.0, angle_deg=0.0):
    """
    Simple 2D potential flow using source panel method approximation.
    For complex shapes we use a mask-based approach with Laplace relaxation.
    """
    N = grid_size
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    angle_rad = np.deg2rad(angle_deg)
    U_inf = wind_speed * np.cos(angle_rad)
    V_inf = wind_speed * np.sin(angle_rad)

    # Build object mask
    from matplotlib.path import Path
    mask = np.zeros((N, N), dtype=bool)
    for poly in shape_polygons:
        path = Path(poly)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        inside = path.contains_points(pts).reshape(N, N)
        mask |= inside

    # Stream function approach via Laplace relaxation
    psi = U_inf * Y - V_inf * X  # free-stream stream function

    # Set boundary stream function on object boundary
    # We enforce psi = const on the surface (streamline condition)
    # Find border of mask (cells inside mask)
    interior_psi_val = 0.5 * (U_inf - V_inf)  # roughly center value

    for _ in range(300):
        psi_old = psi.copy()
        # Laplace update
        psi[1:-1, 1:-1] = 0.25 * (
            psi[2:, 1:-1] + psi[:-2, 1:-1] +
            psi[1:-1, 2:] + psi[1:-1, :-2]
        )
        # Enforce boundary conditions
        # Inlet (left): psi = U_inf*y - V_inf*x
        psi[:, 0] = U_inf * y - V_inf * x[0]
        # Outlet (right): zero gradient
        psi[:, -1] = psi[:, -2]
        # Top/bottom: freestream
        psi[0, :] = U_inf * y[0] - V_inf * x
        psi[-1, :] = U_inf * y[-1] - V_inf * x
        # Inside object: fixed stream value
        psi[mask] = interior_psi_val

        if np.max(np.abs(psi - psi_old)) < 1e-4:
            break

    # Compute velocities from stream function: u = dpsi/dy, v = -dpsi/dx
    dy = y[1] - y[0]
    dx = x[1] - x[0]
    u = np.gradient(psi, dy, axis=0)
    v = -np.gradient(psi, dx, axis=1)

    # Zero velocity inside object
    u[mask] = 0
    v[mask] = 0

    speed = np.sqrt(u**2 + v**2)

    return X, Y, u, v, speed, mask, psi

def compute_pressure(speed, wind_speed):
    """Bernoulli pressure coefficient: Cp = 1 - (V/V_inf)^2"""
    V_inf = wind_speed if wind_speed > 0 else 1.0
    cp = 1 - (speed / V_inf)**2
    return np.clip(cp, -4, 1)

# --- Streamlit UI ---
col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.header("Settings")
    shape = st.selectbox("Object", [
        "Cow", "Circle", "Airfoil (NACA 0012)", "Rectangle", "Car (Side View)", "Truck", "Bicycle Rider"
    ])
    wind_speed = st.slider("Wind Speed", 0.5, 5.0, 1.0, 0.1)
    angle = st.slider("Wind Angle (deg)", -30, 30, 0, 1)
    grid_size = st.select_slider("Grid Resolution", options=[40, 60, 80, 100], value=60)
    vis_mode = st.selectbox("Visualization", [
        "Speed (CFD Colors)", "Pressure Coefficient", "Streamlines + Speed", "Velocity Vectors"
    ])
    show_streamlines = st.checkbox("Overlay Streamlines", value=True)
    show_vectors = st.checkbox("Overlay Velocity Vectors", value=False)
    st.markdown("---")
    run = st.button("Run Simulation", type="primary")

if run or "sim_result" not in st.session_state:
    with st.spinner("Running simulation..."):
        polys = get_shape_polygons(shape)
        X, Y, u, v, speed, mask, psi = simulate_flow(
            polys, grid_size=grid_size, wind_speed=wind_speed, angle_deg=angle
        )
        pressure = compute_pressure(speed, wind_speed)
        st.session_state["sim_result"] = (X, Y, u, v, speed, mask, psi, pressure, polys)

X, Y, u, v, speed, mask, psi, pressure, polys = st.session_state["sim_result"]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#111111")
ax.set_facecolor("#111111")

cmap_speed = "jet"
cmap_pressure = "RdBu_r"

if vis_mode == "Speed (CFD Colors)":
    data = speed.copy()
    data[mask] = np.nan
    im = ax.pcolormesh(X, Y, data, cmap=cmap_speed, shading="auto",
                       vmin=0, vmax=wind_speed * 2.5)
    plt.colorbar(im, ax=ax, label="Speed (m/s)")
    title = "Flow Speed"

elif vis_mode == "Pressure Coefficient":
    data = pressure.copy()
    data[mask] = np.nan
    im = ax.pcolormesh(X, Y, data, cmap=cmap_pressure, shading="auto",
                       vmin=-4, vmax=1)
    plt.colorbar(im, ax=ax, label="Pressure Coefficient Cp")
    title = "Pressure Distribution"

elif vis_mode == "Streamlines + Speed":
    data = speed.copy()
    data[mask] = np.nan
    im = ax.pcolormesh(X, Y, data, cmap=cmap_speed, shading="auto",
                       vmin=0, vmax=wind_speed * 2.5)
    plt.colorbar(im, ax=ax, label="Speed (m/s)")
    # Draw streamlines
    nx, ny = X.shape[1], X.shape[0]
    x1d = X[0, :]
    y1d = Y[:, 0]
    u_sl = u.copy(); u_sl[mask] = np.nan
    v_sl = v.copy(); v_sl[mask] = np.nan
    u_sl = np.nan_to_num(u_sl)
    v_sl = np.nan_to_num(v_sl)
    ax.streamplot(x1d, y1d, u_sl, v_sl, color="white", linewidth=0.6,
                  density=2.0, arrowsize=0.8, arrowstyle="->")
    title = "Streamlines + Speed"
    show_streamlines = False  # already done

elif vis_mode == "Velocity Vectors":
    data = speed.copy()
    data[mask] = np.nan
    im = ax.pcolormesh(X, Y, data, cmap=cmap_speed, shading="auto",
                       vmin=0, vmax=wind_speed * 2.5)
    plt.colorbar(im, ax=ax, label="Speed (m/s)")
    step = max(1, grid_size // 20)
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    Uq = u[::step, ::step]
    Vq = v[::step, ::step]
    mq = mask[::step, ::step]
    Uq[mq] = 0; Vq[mq] = 0
    ax.quiver(Xq, Yq, Uq, Vq, color="white", alpha=0.7, scale=wind_speed * 25,
              width=0.002, headwidth=3)
    title = "Velocity Vectors"
    show_vectors = False

else:
    title = ""

# Overlay streamlines
if show_streamlines and vis_mode not in ("Streamlines + Speed",):
    x1d = X[0, :]
    y1d = Y[:, 0]
    u_sl = np.nan_to_num(u.copy())
    v_sl = np.nan_to_num(v.copy())
    u_sl[mask] = 0; v_sl[mask] = 0
    ax.streamplot(x1d, y1d, u_sl, v_sl, color="white", linewidth=0.5,
                  density=1.8, arrowsize=0.7, arrowstyle="->")

# Overlay vectors
if show_vectors and vis_mode not in ("Velocity Vectors",):
    step = max(1, grid_size // 18)
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    Uq = u[::step, ::step].copy()
    Vq = v[::step, ::step].copy()
    mq = mask[::step, ::step]
    Uq[mq] = 0; Vq[mq] = 0
    ax.quiver(Xq, Yq, Uq, Vq, color="yellow", alpha=0.6, scale=wind_speed * 30,
              width=0.0015, headwidth=3)

# Draw object silhouette(s)
for poly in polys:
    patch = plt.Polygon(poly, closed=True, facecolor="#333333",
                        edgecolor="white", linewidth=1.5, zorder=5)
    ax.add_patch(patch)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_title(f"{shape} — {title}  |  Wind: {wind_speed} m/s @ {angle}°",
             color="white", fontsize=13)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#555555")
ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")

with col_main:
    st.pyplot(fig)

    # Stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    max_speed = float(np.nanmax(speed[~mask]))
    min_speed = float(np.nanmin(speed[~mask]))
    avg_speed = float(np.nanmean(speed[~mask]))
    max_cp = float(np.nanmax(pressure[~mask]))
    min_cp = float(np.nanmin(pressure[~mask]))
    col1.metric("Max Speed", f"{max_speed:.3f} m/s")
    col2.metric("Min Speed", f"{min_speed:.3f} m/s")
    col3.metric("Max Cp", f"{max_cp:.3f}")
    col4.metric("Min Cp (suction)", f"{min_cp:.3f}")

    with st.expander("About this simulation"):
        st.markdown("""
**Method:** 2D potential flow via Laplace stream-function relaxation (finite differences).

**What's shown:**
- **Speed (CFD Colors):** Velocity magnitude — red/yellow = fast, blue = slow/wake
- **Pressure Coefficient:** Cp = 1 − (V/V∞)² via Bernoulli — red = high pressure (stagnation), blue = low pressure (suction)
- **Streamlines:** Lines tangent to velocity field — show flow path around the object
- **Velocity Vectors:** Arrows showing direction and magnitude at grid points

**Limitations:** This is an irrotational, inviscid 2D model (potential flow). It does not capture turbulence, boundary layers, or 3D effects, but it produces qualitatively correct results for educational visualization.
        """)
