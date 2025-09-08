#PSO code 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import animation
from matplotlib.animation import PillowWriter  # 
from matplotlib.gridspec import GridSpec

# -------------------- Objective --------------------
def costfunction(x: np.ndarray) -> float:
    return float(x[0]**2 + x[1]**2)

# -------------------- clip--------------------
def bound(x, lb, ub):
    return np.clip(x, lb, ub)


def make_grid(lb=-3.0, ub=3.0, res=181):
    xs = np.linspace(lb, ub, res)
    ys = np.linspace(lb, ub, res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = X**2 + Y**2
    return X, Y, Z

def draw_wireframe_bowl(ax, X, Y, Z, set_limits=True):
    ax.plot_wireframe(X, Y, Z, rstride=8, cstride=8,
                      linewidth=0.6, color="#7aa6ff", alpha=0.6)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("f(X,Y)")
    if set_limits:
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(0, 18)

# --------------------  console log --------------------
def print_iteration_log(t, P, V, pbest_X, pbest_F, gbest_x, gbest_f,
                        w=None, c1=None, c2=None, r1_all=None, r2_all=None):
    print(f"\n----- Iteration {t} -----")
    if w is not None and c1 is not None and c2 is not None:
        print(f"Params: w={w:.4f}, c1={c1:.4f}, c2={c2:.4f}")
    for i in range(P.shape[0]):
        pos = P[i]; vel = V[i]; f = costfunction(P[i])
        line = (f"Particle {i+1:>2}: "
                f"pos=({pos[0]:+7.4f}, {pos[1]:+7.4f})  "
                f"vel=({vel[0]:+7.4f}, {vel[1]:+7.4f})  "
                f"fit={f:8.5f}  "
                f"pbest=({pbest_X[i,0]:+7.4f}, {pbest_X[i,1]:+7.4f}) "
                f"pbest_fit={pbest_F[i]:8.5f}")
        if r1_all is not None and r2_all is not None:
            r1 = r1_all[i]; r2 = r2_all[i]
            line += (f"\n           r1=({r1[0]:.4f}, {r1[1]:.4f})  "
                     f"r2=({r2[0]:.4f}, {r2[1]:.4f})")
        print(line)
    print(f"GBEST: pos=({gbest_x[0]:+7.4f}, {gbest_x[1]:+7.4f})  fit={gbest_f:8.5f}")

# -------------------- PSO w --------------------
def pso_with_history(Positions, lowerbound, upperbound, iters=100, seed=7, log_first_n=3):

    rng = np.random.default_rng(seed)
    n, dim = Positions.shape
    assert n == 4 and dim == 2, " 4 particles in 2D"

    lb = np.array([lowerbound, lowerbound], dtype=float) if np.isscalar(lowerbound) else np.asarray(lowerbound, float)
    ub = np.array([upperbound, upperbound], dtype=float) if np.isscalar(upperbound) else np.asarray(upperbound, float)

    w_max, w_min = 0.9, 0.4
    c1 = 1.5; c2 = 1.5


    P = Positions.copy()
    V = np.zeros_like(P)
    pbest_X = P.copy()
    pbest_F = np.array([costfunction(P[i]) for i in range(n)], float)
    g_idx = int(np.argmin(pbest_F))
    gbest_x = pbest_X[g_idx].copy()
    gbest_f = float(pbest_F[g_idx])

    P_hist      = [P.copy()]
    V_hist      = [V.copy()]
    pbestX_hist = [pbest_X.copy()]
    pbestF_hist = [pbest_F.copy()]
    g_hist      = [gbest_x.copy()]
    gbestF_hist = [gbest_f]
    w_hist      = [w_max]  

    # 
    for t in range(1, iters + 1):
        w = w_max - t * ((w_max - w_min) / iters)
        r1_all = rng.random((n, dim))
        r2_all = rng.random((n, dim))

        for i in range(n):
            r1 = r1_all[i]; r2 = r2_all[i]

            V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - P[i]) + c2 * r2 * (gbest_x - P[i])
            P[i] = bound(P[i] + V[i], lb, ub)


            f = costfunction(P[i])
            if f < pbest_F[i]:
                pbest_F[i] = f
                pbest_X[i] = P[i].copy()
            if f < gbest_f:
                gbest_f = f
                gbest_x = P[i].copy()

        if t <= log_first_n:
            print_iteration_log(t, P, V, pbest_X, pbest_F, gbest_x, gbest_f,
                                w=w, c1=c1, c2=c2, r1_all=r1_all, r2_all=r2_all)
        P_hist.append(P.copy())
        V_hist.append(V.copy())
        pbestX_hist.append(pbest_X.copy())
        pbestF_hist.append(pbest_F.copy())
        g_hist.append(gbest_x.copy())
        gbestF_hist.append(gbest_f)
        w_hist.append(w)

    return P_hist, V_hist, pbestX_hist, pbestF_hist, g_hist, gbestF_hist, w_hist

#  Main 
if __name__ == "__main__":
    lowerbound, upperbound = -3.0, 3.0
    FOUR_POS = np.array([
        [-2.2,  2.5],
        [ 1.6, -1.8],
        [-0.7,  0.4],
        [ 2.4,  0.9],
    ], dtype=float)
    FOUR_POS = bound(FOUR_POS, lowerbound, upperbound)

    # Run PSO
    ITERATIONS = 100
    LOG_FIRST_N = 3
    SEED = 7
    (P_hist, V_hist, pbestX_hist, pbestF_hist,
     g_hist, gbestF_hist, w_hist) = pso_with_history(
        FOUR_POS.copy(), lowerbound, upperbound,
        iters=ITERATIONS, seed=SEED, log_first_n=LOG_FIRST_N
    )
    Xg, Yg, Zg = make_grid(lowerbound, upperbound, res=181)

    fig = plt.figure(figsize=(13.0, 8.0))
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[4.0, 1.6], hspace=0.35, wspace=0.35)

    # 3D 
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    draw_wireframe_bowl(ax3d, Xg, Yg, Zg)
    ax3d.set_title("3D wireframe: PSO over all iterations")
    ax3d.view_init(elev=32, azim=-60)

    # 2D 
    ax2d = fig.add_subplot(gs[0, 1])
    ax2d.contour(Xg, Yg, Zg, levels=22, colors="#8fa6bf", linewidths=0.7)
    ax2d.set_xlim(-3, 3); ax2d.set_ylim(-3, 3)
    ax2d.set_xlabel("X"); ax2d.set_ylabel("Y")
    ax2d.set_title("2D plane: PSO over all iterations")

    #curve 
    axw = fig.add_subplot(gs[0, 2])
    axw.plot(range(len(w_hist)), w_hist, linewidth=1.8)
    axw.set_xlim(0, ITERATIONS)
    axw.set_ylim(min(w_hist)-0.02, max(w_hist)+0.02)
    axw.set_xlabel("Iteration")
    axw.set_ylabel("w")
    axw.set_title("Inertia weight (w) over iterations")
    current_w_marker, = axw.plot([], [], 'o', ms=8)

    axtext = fig.add_subplot(gs[1, :])
    axtext.axis("off")
    status_text = axtext.text(
        0.01, 0.98, "", va="top", ha="left",
        family="monospace", fontsize=9, transform=axtext.transAxes
    )

    scat3d = ax3d.scatter([], [], [], s=58, c="red", alpha=0.80, edgecolors="none")
    gbest3d = ax3d.scatter([], [], [], s=100, c="blue")
    scat2d = ax2d.scatter([], [], s=58, c="red", alpha=0.80)
    gbest2d = ax2d.scatter([], [], s=100, c="blue")

    arrows3d = []
    arrows2d = []

    def clear_arrows():
        for a in arrows3d:
            try: a.remove()
            except Exception: pass
        for a in arrows2d:
            try: a.remove()
            except Exception: pass
        arrows3d.clear()
        arrows2d.clear()

    def build_status_string(frame: int) -> str:
        pop    = P_hist[frame]
        vel    = V_hist[frame]
        pbestX = pbestX_hist[frame]
        pbestF = pbestF_hist[frame]
        gb     = g_hist[frame]
        gbF    = gbestF_hist[frame]
        w_now  = w_hist[frame]

        lines = []
        lines.append(f"Iteration: {frame:3d} / {ITERATIONS:3d}   |   w = {w_now:0.4f}")
        lines.append("Particles (pos, vel, fit, pbest, pbest_fit):")
        for i in range(pop.shape[0]):
            x, y = pop[i]
            vx, vy = vel[i]
            fit = x*x + y*y
            px, py = pbestX[i]
            pfit = pbestF[i]
            lines.append(
                f"  #{i+1}: pos=({x:+7.4f},{y:+7.4f})  "
                f"vel=({vx:+7.4f},{vy:+7.4f})  "
                f"fit={fit:8.5f}  "
                f"pbest=({px:+7.4f},{py:+7.4f})  pbest_fit={pfit:8.5f}"
            )
        lines.append(f"Global Best: gbest=({gb[0]:+7.4f},{gb[1]:+7.4f})   gbest_fit={gbF:8.5f}")
        lines.append("True Optimum: x*=(+0.0000,+0.0000)   f(x*)=0.00000")

        if frame == ITERATIONS:
            lines.append("")
            lines.append("=result")
            lines.append(f"Optimal: gbest=({gb[0]:+7.6f},{gb[1]:+7.6f})   f={gbF:0.8f}")

        return "\n".join(lines)

    def init_anim():
        clear_arrows()
        scat3d._offsets3d = ([], [], [])
        gbest3d._offsets3d = ([], [], [])
        scat2d.set_offsets(np.empty((0, 2)))
        gbest2d.set_offsets(np.empty((0, 2)))
        status_text.set_text(build_status_string(0))
        current_w_marker.set_data([0], [w_hist[0]])
        return scat3d, gbest3d, scat2d, gbest2d, status_text, current_w_marker

    def update_anim(frame):
        clear_arrows()

        pop = P_hist[frame]
        gb  = g_hist[frame]
        zpop = pop[:, 0]**2 + pop[:, 1]**2
        zg   = gb[0]**2 + gb[1]**2

        scat3d._offsets3d = (pop[:, 0], pop[:, 1], zpop)
        gbest3d._offsets3d = ([gb[0]], [gb[1]], [zg])
        scat2d.set_offsets(pop[:, :2])
        gbest2d.set_offsets(np.array([[gb[0], gb[1]]]))

        if frame >= 1:
            prev = P_hist[frame - 1]
            zprev = prev[:, 0]**2 + prev[:, 1]**2
            dx = pop[:, 0] - prev[:, 0]
            dy = pop[:, 1] - prev[:, 1]
            dz = zpop - zprev

            for k in range(pop.shape[0]):
                q = ax3d.quiver(prev[k, 0], prev[k, 1], zprev[k],
                                dx[k], dy[k], dz[k],
                                color='k', alpha=0.55, linewidths=1.0, normalize=False)
                arrows3d.append(q)

            for k in range(pop.shape[0]):
                a = ax2d.arrow(prev[k, 0], prev[k, 1],
                               dx[k], dy[k],
                               head_width=0.08, head_length=0.12,
                               length_includes_head=True,
                               fc='k', ec='k', alpha=0.65)
                arrows2d.append(a)

        status_text.set_text(build_status_string(frame))
        current_w_marker.set_data([frame], [w_hist[frame]])

        return (*arrows3d, *arrows2d, scat3d, gbest3d, scat2d, gbest2d, status_text, current_w_marker)

    anim = animation.FuncAnimation(
        fig, update_anim, init_func=init_anim,
        frames=len(P_hist), interval=120, blit=False
    )

    # Save GIF
    writer = PillowWriter(fps=12, metadata={"artist": "PSO"})
    def _progress(i, n):
        if i % max(1, n // 10) == 0 or i == n-1:
            print(f"Saving GIF: frame {i+1}/{n}")
    anim.save("pso.gif", writer=writer, dpi=120, progress_callback=_progress)

    # Keep reference
    globals().update(dict(_anim=anim))

    plt.tight_layout()
    plt.show()
