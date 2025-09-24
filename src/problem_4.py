import numpy as np
import math
import sys
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def uprint(*args, sep=' ', end='\n'):
    """确保UTF-8编码的中文能正确打印。"""
    s = sep.join(map(str, args)) + end
    sys.stdout.buffer.write(s.encode('utf-8'))

# ---------- 常量与输入 ----------
G = 9.8
R_SMOKE = 10.0
V_SINK = 3.0
V_MISSILE = 300.0
T_SMOKE_EFFECTIVE = 20.0

# 初始位置
P_UAVS_INITIAL = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0])
}
P_M1 = np.array([20000.0, 0.0, 2000.0])
P_TARGET_REAL = np.array([0.0, 200.0, 5.0])
P_TARGET_FAKE = np.array([0.0, 0.0, 0.0])

# 预计算的常量向量
MISSILE_DIR = (P_TARGET_FAKE - P_M1) / np.linalg.norm(P_TARGET_FAKE - P_M1)
BASE_ANGLES = {
    name: math.atan2(P_TARGET_FAKE[1] - pos[1], P_TARGET_FAKE[0] - pos[0])
    for name, pos in P_UAVS_INITIAL.items()
}
UAV_NAMES = list(P_UAVS_INITIAL.keys())

# ---------- 工具函数 ----------
def missile_pos(t):
    return P_M1 + V_MISSILE * t * MISSILE_DIR

def dist_point_to_segment(P, A, B):
    AB = B - A
    AP = P - A
    ab2 = np.dot(AB, AB)
    if ab2 == 0.0: return np.linalg.norm(P - A), 0.0
    s = np.dot(AP, AB) / ab2
    return np.linalg.norm(P - (A + s * AB)), s

# ---------- 核心计算函数 ----------
def compute_cover_time_multi_uavs(x, dt=0.02):
    """
    计算多无人机、各1枚弹的总遮蔽时长（并集）。
    x: [theta1, v1, tr1, tf1, theta2, v2, tr2, tf2, ...] (12个变量)
    """
    blast_params = []
    num_uavs = len(x) // 4
    for i in range(num_uavs):
        uav_name = UAV_NAMES[i]
        params = x[i*4 : (i+1)*4]
        theta_offset, v_uav, t_rel, t_fuz = params

        heading = BASE_ANGLES[uav_name] + theta_offset
        uav_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
        v_uav_vec = v_uav * uav_dir
        
        p_release = P_UAVS_INITIAL[uav_name] + v_uav_vec * t_rel
        p_blast = p_release + v_uav_vec * t_fuz + np.array([0, 0, -0.5 * G * t_fuz**2])
        t_blast = t_rel + t_fuz
        blast_params.append({'p_blast': p_blast, 't_blast': t_blast, 'uav_name': uav_name})

    # --- 模拟与扫描 ---
    if not blast_params: return 0.0, blast_params
    t_start_scan = min(p['t_blast'] for p in blast_params)
    t_end_scan = max(p['t_blast'] for p in blast_params) + T_SMOKE_EFFECTIVE
    t_vals = np.arange(t_start_scan, t_end_scan, dt)
    
    total_time = 0
    cover_status = []
    for t in t_vals:
        p_missile_t = missile_pos(t)
        is_covered_at_t = False
        for params in blast_params:
            t_after_blast = t - params['t_blast']
            if 0 <= t_after_blast < T_SMOKE_EFFECTIVE:
                p_cloud_center = params['p_blast'] + np.array([0, 0, -V_SINK * t_after_blast])
                d, s = dist_point_to_segment(p_cloud_center, p_missile_t, P_TARGET_REAL)
                if d < R_SMOKE and 0 < s < 1:
                    is_covered_at_t = True
                    break
        if is_covered_at_t:
            total_time += dt
        cover_status.append((t, is_covered_at_t, p_missile_t))
            
    return total_time, blast_params, cover_status

def compute_single_cover_time(uav_name, single_x, dt=0.01):
    """
    计算单架无人机单枚弹的独立遮蔽时长。
    uav_name: 'FY1', 'FY2', or 'FY3'
    single_x: [theta_offset, v_uav, t_release, t_fuze]
    """
    theta_offset, v_uav, t_release, t_fuze = single_x

    heading = BASE_ANGLES[uav_name] + theta_offset
    uav_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
    v_uav_vec = v_uav * uav_dir
    
    p_release = P_UAVS_INITIAL[uav_name] + v_uav_vec * t_release
    p_blast = p_release + v_uav_vec * t_fuze + np.array([0, 0, -0.5 * G * t_fuze**2])
    t_blast = t_release + t_fuze

    t_start_scan = t_blast
    t_end_scan = t_blast + T_SMOKE_EFFECTIVE
    t_vals = np.arange(t_start_scan, t_end_scan, dt)
    
    cover_time = 0
    for t in t_vals:
        p_missile_t = missile_pos(t)
        t_after_blast = t - t_blast
        p_cloud_center = p_blast + np.array([0, 0, -V_SINK * t_after_blast])
        d, s = dist_point_to_segment(p_cloud_center, p_missile_t, P_TARGET_REAL)
        if d < R_SMOKE and 0 < s < 1:
            cover_time += dt
            
    return cover_time

# ---------- 可视化函数 ----------
def visualize_solution(x_opt):
    """可视化最优解的三维轨迹和遮蔽效果"""
    # 计算遮蔽时间和获取详细参数
    cover_time, blast_params, cover_status = compute_cover_time_multi_uavs(x_opt, dt=0.05)
    
    # 创建3D图形
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色设置
    colors = {'FY1': 'red', 'FY2': 'blue', 'FY3': 'green'}
    
    # 绘制初始位置和目标点
    ax.scatter(P_M1[0], P_M1[1], P_M1[2], c='orange', marker='*', s=200, label='导弹起始点')
    ax.scatter(P_TARGET_REAL[0], P_TARGET_REAL[1], P_TARGET_REAL[2], c='purple', marker='X', s=200, label='真实目标')
    ax.scatter(P_TARGET_FAKE[0], P_TARGET_FAKE[1], P_TARGET_FAKE[2], c='gray', marker='X', s=200, label='假目标')
    
    # 绘制导弹轨迹
    missile_flight_time = np.linalg.norm(P_TARGET_FAKE - P_M1) / V_MISSILE
    t_missile = np.linspace(0, missile_flight_time, 100)
    missile_path = np.array([missile_pos(t) for t in t_missile])
    ax.plot(missile_path[:, 0], missile_path[:, 1], missile_path[:, 2], 'y-', linewidth=2, label='导弹轨迹')
    
    # 绘制无人机轨迹和烟幕
    for i, uav_name in enumerate(UAV_NAMES):
        params = x_opt[i*4 : (i+1)*4]
        theta_offset, v_uav, t_rel, t_fuz = params
        
        # 计算无人机轨迹
        heading = BASE_ANGLES[uav_name] + theta_offset
        uav_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
        v_uav_vec = v_uav * uav_dir
        
        # 无人机从初始位置到投放点的轨迹
        t_uav = np.linspace(0, t_rel, 50)
        uav_path = np.array([P_UAVS_INITIAL[uav_name] + v_uav_vec * t for t in t_uav])
        ax.plot(uav_path[:, 0], uav_path[:, 1], uav_path[:, 2], 
                color=colors[uav_name], linestyle='-', linewidth=2, label=f'{uav_name}轨迹')
        
        # 投放点
        p_release = P_UAVS_INITIAL[uav_name] + v_uav_vec * t_rel
        ax.scatter(p_release[0], p_release[1], p_release[2], 
                  color=colors[uav_name], marker='o', s=100, label=f'{uav_name}投放点')
        
        # 烟幕弹轨迹 (从投放到爆炸)
        t_smoke = np.linspace(0, t_fuz, 20)
        smoke_path = np.array([p_release + v_uav_vec * t + np.array([0, 0, -0.5 * G * t**2]) for t in t_smoke])
        ax.plot(smoke_path[:, 0], smoke_path[:, 1], smoke_path[:, 2], 
                color=colors[uav_name], linestyle='--', linewidth=1, label=f'{uav_name}烟幕弹轨迹')
        
        # 爆炸点
        p_blast = p_release + v_uav_vec * t_fuz + np.array([0, 0, -0.5 * G * t_fuz**2])
        ax.scatter(p_blast[0], p_blast[1], p_blast[2], 
                  color=colors[uav_name], marker='*', s=150, label=f'{uav_name}爆炸点')
        
        # 烟幕下沉轨迹 (爆炸后)
        for param in blast_params:
            if param['uav_name'] == uav_name:
                t_after_blast = np.linspace(0, T_SMOKE_EFFECTIVE, 10)
                smoke_sink = np.array([param['p_blast'] + np.array([0, 0, -V_SINK * t]) for t in t_after_blast])
                ax.plot(smoke_sink[:, 0], smoke_sink[:, 1], smoke_sink[:, 2], 
                        color=colors[uav_name], linestyle=':', linewidth=1, alpha=0.7, label=f'{uav_name}烟幕下沉')
                
                # 在几个关键点绘制烟幕范围
                for t_idx in [0, 5, 10, 15]:
                    if t_idx < len(t_after_blast):
                        center = smoke_sink[t_idx]
                        # 绘制烟幕圆柱体 (简化表示)
                        theta = np.linspace(0, 2*np.pi, 20)
                        x = center[0] + R_SMOKE * np.cos(theta)
                        y = center[1] + R_SMOKE * np.sin(theta)
                        z = np.full_like(theta, center[2])
                        ax.plot(x, y, z, color=colors[uav_name], alpha=0.3)
    
    # 标记被遮蔽的导弹段
    covered_segments = []
    current_segment = []
    for t, covered, pos in cover_status:
        if covered:
            current_segment.append(pos)
        elif current_segment:
            covered_segments.append(np.array(current_segment))
            current_segment = []
    
    if current_segment:
        covered_segments.append(np.array(current_segment))
    
    for i, segment in enumerate(covered_segments):
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], 'r-', linewidth=3, alpha=0.8, 
                label='被遮蔽段' if i == 0 else "")
    
    # 设置图形属性
    ax.set_xlabel('X坐标 (m)')
    ax.set_ylabel('Y坐标 (m)')
    ax.set_zlabel('Z坐标 (m)')
    ax.set_title(f'无人机烟幕遮蔽优化方案 (总遮蔽时间: {cover_time:.2f}s)')
    
    # 添加图例 (避免重复)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(0, 1))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('uav_smoke_optimization_3d.png', dpi=300)
    plt.show()
    
    # 绘制时间-遮蔽状态图
    plt.figure(figsize=(12, 6))
    times = [status[0] for status in cover_status]
    covered = [status[1] for status in cover_status]
    missile_positions = [status[2] for status in cover_status]
    
    # 计算导弹到目标的距离
    distances = [np.linalg.norm(pos - P_TARGET_REAL) for pos in missile_positions]
    
    plt.plot(times, covered, 'b-', linewidth=2, label='遮蔽状态 (1=被遮蔽)')
    plt.fill_between(times, 0, covered, alpha=0.3, color='blue')
    plt.xlabel('时间 (s)')
    plt.ylabel('遮蔽状态')
    plt.title('导弹飞行过程中的遮蔽状态')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加第二个y轴显示导弹到目标的距离
    ax2 = plt.gca().twinx()
    ax2.plot(times, distances, 'r-', linewidth=2, alpha=0.7, label='导弹到目标距离')
    ax2.set_ylabel('距离 (m)')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('cover_status_vs_time.png', dpi=300)
    plt.show()

# ---------- 优化目标函数 ----------
def objective(x):
    cover_time, _, _ = compute_cover_time_multi_uavs(x, dt=0.1)
    return -cover_time

# ---------- 主程序：差分进化优化 ----------
def solve_problem4():
    uprint("--- 问题4：三无人机各一弹最优策略求解 ---")
    uprint("正在使用差分进化算法进行全局优化，变量维度较高，预计需要很长时间...")

    # 决策变量边界: 3 * 4 = 12个变量
    # [theta1, v1, tr1, tf1, theta2, v2, tr2, tf2, theta3, v3, tr3, tf3]
    bounds = []
    for name in UAV_NAMES:
        # 航向偏移, 速度, 投放延时, 引信延时
        uav_bounds = [
            (-math.pi / 3, math.pi / 3), # 航向偏移: ±60°
            (100, 140),                  # 速度 (m/s)
            (1.0, 25.0),                 # 投放延时 (s)
            (1.0, 20.0),                 # 引信延时 (s)
        ]
        bounds.extend(uav_bounds)

    # 调用差分进化求解器
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=500,      # 针对高维度问题，增加迭代次数
        popsize=20,       # 种群大小
        polish=True, 
        tol=1e-5,
        updating='deferred',
        workers=-1,
        seed=42
    )

    uprint("\n优化完成！")
    
    # --- 使用高精度dt计算最终结果 ---
    final_cover_time, blast_params, _ = compute_cover_time_multi_uavs(result.x, dt=0.005)
    
    # --- 结果输出 ---
    uprint("\n" + "="*20 + " 最优策略 " + "="*20)
    
    total_individual_time = 0
    for i in range(len(UAV_NAMES)):
        uav_name = UAV_NAMES[i]
        params = result.x[i*4 : (i+1)*4]
        theta_opt, v_opt, tr_opt, tf_opt = params
        
        # 计算单弹独立贡献
        single_cover = compute_single_cover_time(uav_name, params, dt=0.005)
        total_individual_time += single_cover

        uprint(f"\n--- 无人机 {uav_name} 策略 ---")
        uprint(f"  飞行速度: {v_opt:.4f} m/s")
        uprint(f"  飞行航向: {np.rad2deg(BASE_ANGLES[uav_name] + theta_opt):.4f} 度 (相对基准偏移 {np.rad2deg(theta_opt):.4f} 度)")
        uprint(f"  投放时间: {tr_opt:.4f} s")
        uprint(f"  引信时间: {tf_opt:.4f} s")
        uprint(f"  起爆时刻: {tr_opt + tf_opt:.4f} s")
        uprint(f"  [独立遮蔽贡献: {single_cover:.4f} s]")

    uprint(f"\n注意: 各弹独立遮蔽时长之和 ({total_individual_time:.4f} s) 可能因时间重叠而大于总有效遮蔽时长。")
    uprint("\n" + "-"*50)
    uprint(f"找到的总有效遮蔽时长（并集）为: {final_cover_time:.4f} 秒")
    uprint("-"*50)
    
    # 可视化最优解
    uprint("\n正在生成可视化图表...")
    visualize_solution(result.x)

if __name__ == '__main__':
    solve_problem4()