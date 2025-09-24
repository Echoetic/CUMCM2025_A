import numpy as np
import math
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 常量与输入 ----------
g = 9.8
R0 = 10.0
sink_v = 3.0
v_missile = 300.0
FY0 = np.array([17800.0, 0.0, 1800.0])
M0 = np.array([20000.0, 0.0, 2000.0])
T = np.array([0.0, 200.0, 5.0])
active_window = 20.0
base_angle = math.atan2(-FY0[1], -FY0[0])
uM = -(M0 / np.linalg.norm(M0))

def missile_pos(t):
    return M0 + v_missile * t * uM

def dist_point_to_segment(P, A, B):
    AB = B - A
    AP = P - A
    ab2 = np.dot(AB, AB)
    if ab2 == 0.0:
        return np.linalg.norm(P - A), 0.0, A
    s = np.dot(AP, AB) / ab2
    s_clamped = max(0.0, min(1.0, s))
    closest = A + s_clamped * AB
    return np.linalg.norm(P - closest), s, closest

def compute_cover_time(theta_offset, v_uav, t_release, t_fuze, dt=0.01):
    heading = base_angle + theta_offset
    uav_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
    FY_release = FY0 + v_uav * t_release * uav_dir
    v0 = v_uav * uav_dir
    C0 = FY_release + v0 * t_fuze + np.array([0,0,-0.5*g*t_fuze*t_fuze])
    def cloud_center(t):
        if t < t_release + t_fuze:
            return C0
        return C0 + np.array([0,0,-sink_v*(t - (t_release + t_fuze))])
    t_start = t_release + t_fuze
    t_end = t_start + active_window
    t_vals = np.arange(t_start, t_end+1e-12, dt)
    covered = []
    for t in t_vals:
        d, s, _ = dist_point_to_segment(cloud_center(t), missile_pos(t), T)
        covered.append((d <= R0) and (0 < s < 1))
    total = 0.0
    in_cov = False
    t0 = 0.0
    for i, flag in enumerate(covered):
        if flag and not in_cov:
            t0 = t_vals[i]; in_cov = True
        if in_cov and (not flag):
            total += t_vals[i-1] - t0
            in_cov = False
    if in_cov:
        total += t_vals[-1] - t0
    return total

# 优化目标
def objective(x):
    theta, v, trel, tfuz = x
    # 限制搜索在合理范围
    if v < 70 or v > 140 or trel < 0 or tfuz < 0:
        return 1e6
    return -compute_cover_time(theta, v, trel, tfuz, dt=0.02)

# 注意：收紧搜索范围，避免搜索无效解
bounds = [
    (-math.pi/2, math.pi/2),  # 航向偏移：只搜 ±90°，防止乱飞
    (70, 140),                # 速度
    (0, 3),                   # 投放延时
    (0.5, 5)                  # 引信延时
]

# 记录优化历史
history = {'x': [], 'fval': []}
def callback(xk, convergence):
    history['x'].append(xk.copy())
    history['fval'].append(-objective(xk))
    return False

# 运行优化
result = differential_evolution(objective, bounds, maxiter=50, popsize=25, 
                               polish=True, tol=1e-6, seed=42, callback=callback)
theta_opt, v_opt, trel_opt, tfuz_opt = result.x
final_cover = compute_cover_time(theta_opt, v_opt, trel_opt, tfuz_opt, dt=0.001)

print("=== 优化结果 ===")
print(f"航向偏移 theta_offset = {theta_opt:.6f} rad  ({math.degrees(theta_opt):.3f}°)")
print(f"无人机速度 v = {v_opt:.3f} m/s")
print(f"投放延时 t_release = {trel_opt:.3f} s")
print(f"引信延时 t_fuze = {tfuz_opt:.3f} s")
print(f"最大遮蔽时长 = {final_cover:.3f} s")

# 可视化
fig = plt.figure(figsize=(20, 15))
fig.suptitle('烟幕遮蔽优化分析', fontsize=16)

# 1. 优化过程收敛图
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(range(1, len(history['fval'])+1), history['fval'], 'b-', linewidth=2)
ax1.set_xlabel('迭代次数')
ax1.set_ylabel('遮蔽时间 (s)')
ax1.set_title('优化过程收敛图')
ax1.grid(True)
ax1.annotate(f'最优值: {final_cover:.3f}s', 
            xy=(len(history['fval']), final_cover),
            xytext=(len(history['fval'])*0.7, final_cover*0.8),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=12)

# 2. 参数演化过程
ax2 = fig.add_subplot(3, 3, 2)
iterations = range(1, len(history['x'])+1)
thetas = [math.degrees(x[0]) for x in history['x']]
vs = [x[1] for x in history['x']]
trels = [x[2] for x in history['x']]
tfuzs = [x[3] for x in history['x']]

ax2.plot(iterations, thetas, label='航向偏移 (°)')
ax2.plot(iterations, vs, label='速度 (m/s)')
ax2.plot(iterations, trels, label='投放延时 (s)')
ax2.plot(iterations, tfuzs, label='引信延时 (s)')
ax2.set_xlabel('迭代次数')
ax2.set_title('参数演化过程')
ax2.legend()
ax2.grid(True)

# 3. 参数相关性热图
ax3 = fig.add_subplot(3, 3, 3)
params = np.array(history['x'])
param_names = ['航向偏移', '速度', '投放延时', '引信延时']
correlation_matrix = np.corrcoef(params.T)

im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax3.set_xticks(range(len(param_names)))
ax3.set_yticks(range(len(param_names)))
ax3.set_xticklabels(param_names)
ax3.set_yticklabels(param_names)
ax3.set_title('参数相关性热图')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('相关系数')

# 4. 三维轨迹可视化
ax4 = fig.add_subplot(3, 3, 4, projection='3d')

# 计算导弹轨迹
missile_t = np.linspace(0, np.linalg.norm(T - M0)/v_missile, 100)
missile_points = np.array([missile_pos(t) for t in missile_t])

# 计算无人机轨迹
heading_opt = base_angle + theta_opt
uav_dir_opt = np.array([math.cos(heading_opt), math.sin(heading_opt), 0.0])
uav_t = np.linspace(0, trel_opt, 100)
uav_points = np.array([FY0 + v_opt * t * uav_dir_opt for t in uav_t])

# 绘制导弹轨迹
ax4.plot(missile_points[:, 0], missile_points[:, 1], missile_points[:, 2], 
         'r-', linewidth=2, label='导弹轨迹')
ax4.scatter(M0[0], M0[1], M0[2], c='red', s=100, marker='o', label='导弹起始点')
ax4.scatter(T[0], T[1], T[2], c='red', s=100, marker='*', label='目标点')

# 绘制无人机轨迹
ax4.plot(uav_points[:, 0], uav_points[:, 1], uav_points[:, 2], 
         'b-', linewidth=2, label='无人机轨迹')
ax4.scatter(FY0[0], FY0[1], FY0[2], c='blue', s=100, marker='o', label='无人机起始点')

# 标记烟幕释放点
release_point = FY0 + v_opt * trel_opt * uav_dir_opt
ax4.scatter(release_point[0], release_point[1], release_point[2], 
           c='green', s=150, marker='X', label='烟幕释放点')

# 绘制烟幕扩散过程
t_start = trel_opt + tfuz_opt
t_end = t_start + active_window
t_samples = np.linspace(t_start, t_end, 5)
colors = plt.cm.viridis(np.linspace(0, 1, len(t_samples)))

for i, t in enumerate(t_samples):
    if t < t_start + tfuz_opt:
        cloud_center = release_point + v_opt * uav_dir_opt * tfuz_opt + np.array([0, 0, -0.5*g*tfuz_opt**2])
    else:
        cloud_center = release_point + v_opt * uav_dir_opt * tfuz_opt + np.array([0, 0, -0.5*g*tfuz_opt**2 - sink_v*(t - t_start)])
    
    # 绘制烟幕球体
    u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:6j]
    x = R0 * np.cos(u) * np.sin(v) + cloud_center[0]
    y = R0 * np.sin(u) * np.sin(v) + cloud_center[1]
    z = R0 * np.cos(v) + cloud_center[2]
    ax4.plot_wireframe(x, y, z, color=colors[i], alpha=0.3, label=f't={t:.1f}s')

ax4.set_xlabel('X坐标 (m)')
ax4.set_ylabel('Y坐标 (m)')
ax4.set_zlabel('Z坐标 (m)')
ax4.set_title('三维轨迹与烟幕扩散')
ax4.legend(loc='upper left', bbox_to_anchor=(0, 1))

# 5. 烟幕遮蔽时间线
ax5 = fig.add_subplot(3, 3, 5)
dt = 0.01
t_vals = np.arange(t_start, t_end + dt, dt)
covered = []
distances = []

for t in t_vals:
    if t < t_start + tfuz_opt:
        cloud_center = release_point + v_opt * uav_dir_opt * tfuz_opt + np.array([0, 0, -0.5*g*tfuz_opt**2])
    else:
        cloud_center = release_point + v_opt * uav_dir_opt * tfuz_opt + np.array([0, 0, -0.5*g*tfuz_opt**2 - sink_v*(t - t_start)])
    
    d, s, _ = dist_point_to_segment(cloud_center, missile_pos(t), T)
    covered.append((d <= R0) and (0 < s < 1))
    distances.append(d)

ax5.plot(t_vals, covered, 'b-', drawstyle='steps-post', label='遮蔽状态')
ax5.set_xlabel('时间 (s)')
ax5.set_ylabel('是否遮蔽')
ax5.set_title('烟幕遮蔽时间线')
ax5.set_yticks([0, 1])
ax5.set_yticklabels(['否', '是'])
ax5.grid(True)
ax5.legend(loc='upper right')

# 6. 烟幕与导弹距离变化
ax6 = fig.add_subplot(3, 3, 6)
ax6.plot(t_vals, distances, 'b-', label='烟幕中心与导弹距离')
ax6.axhline(y=R0, color='r', linestyle='--', label='有效遮蔽半径')
ax6.fill_between(t_vals, 0, R0, where=np.array(distances) <= R0, 
                 color='green', alpha=0.3, label='有效遮蔽区域')
ax6.set_xlabel('时间 (s)')
ax6.set_ylabel('距离 (m)')
ax6.set_title('烟幕中心与导弹距离变化')
ax6.legend()
ax6.grid(True)

# 7. 参数敏感度分析
ax7 = fig.add_subplot(3, 3, 7)
param_names = ['航向偏移', '速度', '投放延时', '引信延时']
param_values = [theta_opt, v_opt, trel_opt, tfuz_opt]
sensitivities = []

for i in range(4):
    perturbed_params = param_values.copy()
    perturbation = 0.05 * (bounds[i][1] - bounds[i][0])  # 5%的参数范围扰动
    perturbed_params[i] += perturbation
    perturbed_cover = compute_cover_time(perturbed_params[0], perturbed_params[1], 
                                        perturbed_params[2], perturbed_params[3])
    sensitivity = abs(perturbed_cover - final_cover) / perturbation
    sensitivities.append(sensitivity)

ax7.bar(param_names, sensitivities)
ax7.set_xlabel('参数')
ax7.set_ylabel('敏感度')
ax7.set_title('参数对遮蔽时间的敏感度分析')
ax7.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# 8. 参数空间采样可视化
ax8 = fig.add_subplot(3, 3, 8)
# 随机采样参数空间，评估遮蔽时间
n_samples = 100
theta_samples = np.random.uniform(bounds[0][0], bounds[0][1], n_samples)
v_samples = np.random.uniform(bounds[1][0], bounds[1][1], n_samples)
cover_times = []

for theta, v in zip(theta_samples, v_samples):
    # 使用最优的trel和tfuz，只变化theta和v
    cover_time = compute_cover_time(theta, v, trel_opt, tfuz_opt, dt=0.05)
    cover_times.append(cover_time)

sc = ax8.scatter(np.degrees(theta_samples), v_samples, c=cover_times, 
                cmap='viridis', alpha=0.7)
ax8.set_xlabel('航向偏移 (°)')
ax8.set_ylabel('速度 (m/s)')
ax8.set_title('参数空间采样 (颜色表示遮蔽时间)')
plt.colorbar(sc, ax=ax8, label='遮蔽时间 (s)')

# 标记最优解
ax8.scatter(math.degrees(theta_opt), v_opt, c='red', s=100, marker='*', label='最优解')
ax8.legend()

# 9. 结果摘要
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
textstr = '\n'.join((
    f'最优航向偏移: {math.degrees(theta_opt):.2f}°',
    f'最优速度: {v_opt:.2f} m/s',
    f'最优投放延时: {trel_opt:.2f} s',
    f'最优引信延时: {tfuz_opt:.2f} s',
    f'最大遮蔽时间: {final_cover:.2f} s',
    f'总优化迭代: {len(history["x"])} 次',
    f'优化状态: {"成功" if result.success else "失败"}'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax9.text(0.5, 0.5, textstr, transform=ax9.transAxes, fontsize=12,
        verticalalignment='center', horizontalalignment='center', bbox=props)

plt.tight_layout()
plt.show()

# 创建动画展示烟幕扩散过程
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

# 绘制固定元素
ax_anim.plot(missile_points[:, 0], missile_points[:, 1], missile_points[:, 2], 
            'r-', linewidth=2, label='导弹轨迹')
ax_anim.scatter(M0[0], M0[1], M0[2], c='red', s=100, marker='o', label='导弹起始点')
ax_anim.scatter(T[0], T[1], T[2], c='red', s=100, marker='*', label='目标点')
ax_anim.plot(uav_points[:, 0], uav_points[:, 1], uav_points[:, 2], 
            'b-', linewidth=2, label='无人机轨迹')
ax_anim.scatter(FY0[0], FY0[1], FY0[2], c='blue', s=100, marker='o', label='无人机起始点')
ax_anim.scatter(release_point[0], release_point[1], release_point[2], 
               c='green', s=150, marker='X', label='烟幕释放点')

# 初始化烟幕球体
u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:6j]
cloud_plot = ax_anim.plot_wireframe(np.zeros_like(u), np.zeros_like(u), np.zeros_like(u), 
                                   color='orange', alpha=0.5, label='烟幕')

ax_anim.set_xlabel('X坐标 (m)')
ax_anim.set_ylabel('Y坐标 (m)')
ax_anim.set_zlabel('Z坐标 (m)')
ax_anim.set_title('烟幕扩散过程动画')
ax_anim.legend(loc='upper left')

# 动画更新函数
def update(frame):
    t = t_start + frame * 0.2  # 每帧0.2秒
    if t > t_end:
        t = t_end
    
    if t < t_start + tfuz_opt:
        cloud_center = release_point + v_opt * uav_dir_opt * tfuz_opt + np.array([0, 0, -0.5*g*tfuz_opt**2])
    else:
        cloud_center = release_point + v_opt * uav_dir_opt * tfuz_opt + np.array([0, 0, -0.5*g*tfuz_opt**2 - sink_v*(t - t_start)])
    
    # 更新烟幕位置
    x = R0 * np.cos(u) * np.sin(v) + cloud_center[0]
    y = R0 * np.sin(u) * np.sin(v) + cloud_center[1]
    z = R0 * np.cos(v) + cloud_center[2]
    
    cloud_plot._verts3d = (x.ravel(), y.ravel(), z.ravel())
    
    # 计算当前导弹位置
    missile_pos_now = missile_pos(t)
    ax_anim.title.set_text(f'烟幕扩散过程动画 (时间: {t:.1f}s, 导弹高度: {missile_pos_now[2]:.1f}m)')
    
    return cloud_plot,

# 创建动画
ani = FuncAnimation(fig_anim, update, frames=int((t_end-t_start)/0.2), 
                    interval=50, blit=False)

plt.tight_layout()
plt.show()

# 保存动画（可选）
ani.save('smoke_diffusion.gif', writer='pillow', fps=20)