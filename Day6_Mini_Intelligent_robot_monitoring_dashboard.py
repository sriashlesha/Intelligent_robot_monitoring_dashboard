import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.style.use("dark_background")
np.random.seed(42)

t = np.linspace(0, 79.94, 800)
dt = t[1] - t[0]
smooth = lambda a, n=5: np.convolve(a, np.ones(n) / n, mode="same")   # makes odom follow cmd

# ---- X: step commands, odom = smoothed command ----
cmd_x = np.zeros_like(t)
cmd_x[t >= 5] = -0.25
cmd_x[t >= 32] = -0.40
cmd_x[t >= 52] = -0.10
cmd_x[t >= 69] = 0
odom_x = smooth(cmd_x)

# ---- Y: damped wobble that settles to 0 ----
odom_y = -0.05 * np.exp(-t / 5) * np.sin(2 * np.pi * t / 2.6)
cmd_y = smooth(odom_y, 9)

# ---- Yaw: one pulse ----
cmd_yaw = np.where((t > 4) & (t < 6.5), 0.075, 0)
odom_yaw = smooth(cmd_yaw)

# ---- Forward obstacle: distance falls as the robot moves ----
travel = np.cumsum(abs(odom_x)) * dt
fwd = np.where(t < 5, 1.5, travel.max() - travel)

# ---- Save CSV + report ----
pd.DataFrame(dict(t=t, cmd_x=cmd_x, odom_x=odom_x, odom_y=odom_y,
                  odom_yaw=odom_yaw, forward_obstacle=fwd)).to_csv("sensor_data.csv", index=False)

report = (f"FINAL PERFORMANCE REPORT\n{'-'*30}\n"
          f"Duration          : {t[-1]:.2f} s\n"
          f"Distance travelled: {travel[-1]:.2f} m\n"
          f"Max reverse speed : {abs(odom_x).max():.2f} m/s\n"
          f"Max Y deviation   : {abs(odom_y).max():.3f} m\n"
          f"Max yaw           : {odom_yaw.max():.3f} rad\n"
          f"Final obstacle    : {fwd[-1]:.2f} m")
open("performance_report.txt", "w").write(report)

# ---- 3D LiDAR cloud ----
a = np.random.uniform(0, 2 * np.pi, 400)
r = np.random.uniform(1, 12, 400)
cloud = np.c_[r * np.cos(a), r * np.sin(a), np.random.rand(400) * 2]

# ---- Dashboard ----
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(4, 5, left=0.05, right=0.98, top=0.93, bottom=0.06, hspace=0.55, wspace=0.5)
axs = [fig.add_subplot(gs[i, 0:2]) for i in range(4)]
a3d = fig.add_subplot(gs[0:2, 2:], projection="3d")
arep = fig.add_subplot(gs[2:, 2:]); arep.axis("off")

def plot(ax, title, lines, ylim):
    for y, color, lw in lines: ax.plot(t, y, color, lw=lw)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlim(0, 79.94); ax.set_ylim(*ylim)
    ax.grid(color="#444"); ax.axvline(9, color="white", lw=0.8)
    ax.tick_params(labelsize=7)

plot(axs[0], "Cmd Vel vs. Odom X",   [(cmd_x, "#22d3ee", 2.5), (odom_x, "#d946ef", 1.2)],   (-0.45, 0.05))
plot(axs[1], "Cmd Vel vs. Odom Y",   [(cmd_y, "#facc15", 2.5), (odom_y, "#fb923c", 1.2)],   (-0.065, 0.022))
plot(axs[2], "Cmd Vel vs. Odom Yaw", [(cmd_yaw, "#2dd4bf", 2.5), (odom_yaw, "#d946ef", 1.2)], (-0.02, 0.085))
plot(axs[3], "Forward Obstacle Distance", [(fwd, "#d946ef", 2.5)], (0, 18.5))

a3d.scatter(*cloud.T, c=cloud[:, 2], cmap="cool", s=6)
a3d.set_title("3D LiDAR Point Cloud", fontsize=10)
arep.text(0, 1, report, va="top", family="monospace", fontsize=11, color="#86efac")

fig.suptitle("Physical AI Robot Monitoring Dashboard", fontsize=14)
plt.show()