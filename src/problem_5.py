import numpy as np
import math
import sys
from scipy.optimize import differential_evolution

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
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0])
}
P_MISSILES_INITIAL = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0])
}
P_TARGET_REAL = np.array([0.0, 200.0, 5.0])
P_TARGET_FAKE = np.array([0.0, 0.0, 0.0])

# 预计算的常量向量
MISSILE_DIRS = {name: (P_TARGET_FAKE - pos) / np.linalg.norm(P_TARGET_FAKE - pos) for name, pos in P_MISSILES_INITIAL.items()}
BASE_ANGLES = {name: math.atan2(P_TARGET_FAKE[1] - pos[1], P_TARGET_FAKE[0] - pos[0]) for name, pos in P_UAVS_INITIAL.items()}
UAV_NAMES = list(P_UAVS_INITIAL.keys())
MISSILE_NAMES = list(P_MISSILES_INITIAL.keys())

# ---------- 工具函数 ----------
def missile_pos(missile_name, t):
    return P_MISSILES_INITIAL[missile_name] + V_MISSILE * t * MISSILE_DIRS[missile_name]

def dist_point_to_segment(P, A, B):
    AB = B - A
    AP = P - A
    ab2 = np.dot(AB, AB)
    if ab2 == 0.0: return np.linalg.norm(P - A), 0.0
    s = np.dot(AP, AB) / ab2
    return np.linalg.norm(P - (A + s * AB)), s

# ---------- 核心计算函数 ----------
def compute_total_cover_time(x, dt=0.1):
    """
    计算所有烟幕弹对所有导弹的总遮蔽时长（并集）。
    x: 40维决策变量向量
    """
    blast_params = []
    for i in range(len(UAV_NAMES)):
        uav_name = UAV_NAMES[i]
        # 每个无人机8个参数: [theta, v, tr1, tf1, dtr2, tf2, dtr3, tf3]
        uav_x = x[i*8 : (i+1)*8]
        theta_offset, v_uav, t_rel1, t_fuz1, dt_rel2, t_fuz2, dt_rel3, t_fuz3 = uav_x

        heading = BASE_ANGLES[uav_name] + theta_offset
        uav_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
        v_uav_vec = v_uav * uav_dir

        release_times = [t_rel1, t_rel1 + dt_rel2, t_rel1 + dt_rel2 + dt_rel3]
        fuze_times = [t_fuz1, t_fuz2, t_fuz3]

        for t_rel, t_fuz in zip(release_times, fuze_times):
            p_release = P_UAVS_INITIAL[uav_name] + v_uav_vec * t_rel
            p_blast = p_release + v_uav_vec * t_fuz + np.array([0, 0, -0.5 * G * t_fuz**2])
            t_blast = t_rel + t_fuz
            blast_params.append({'p_blast': p_blast, 't_blast': t_blast})

    # --- 模拟与扫描 ---
    if not blast_params: return 0.0
    t_start_scan = min(p['t_blast'] for p in blast_params)
    t_end_scan = max(p['t_blast'] for p in blast_params) + T_SMOKE_EFFECTIVE
    t_vals = np.arange(t_start_scan, t_end_scan, dt)
    
    total_time = 0
    for t in t_vals:
        missile_positions_t = [missile_pos(name, t) for name in MISSILE_NAMES]
        is_covered_at_t = False
        
        for p_missile in missile_positions_t:
            if p_missile[0] < P_TARGET_REAL[0]: continue # 导弹已飞过目标
            
            is_this_missile_blocked = False
            for smoke in blast_params:
                t_after_blast = t - smoke['t_blast']
                if 0 <= t_after_blast < T_SMOKE_EFFECTIVE:
                    p_cloud = smoke['p_blast'] + np.array([0, 0, -V_SINK * t_after_blast])
                    d, s = dist_point_to_segment(p_cloud, p_missile, P_TARGET_REAL)
                    if d < R_SMOKE and 0 < s < 1:
                        is_this_missile_blocked = True
                        break
            
            if is_this_missile_blocked:
                is_covered_at_t = True
                break
        
        if is_covered_at_t:
            total_time += dt
            
    return total_time

def compute_single_grenade_cover_details(uav_name, uav_params, grenade_params, dt=0.01):
    """
    计算单枚弹对所有导弹的独立遮蔽详情。
    返回: (遮蔽时长, {被干扰的导弹集合}, 投放点, 起爆点)
    """
    theta_offset, v_uav = uav_params
    t_rel, t_fuz = grenade_params

    heading = BASE_ANGLES[uav_name] + theta_offset
    uav_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
    v_uav_vec = v_uav * uav_dir
    
    p_release = P_UAVS_INITIAL[uav_name] + v_uav_vec * t_rel
    p_blast = p_release + v_uav_vec * t_fuz + np.array([0, 0, -0.5 * G * t_fuz**2])
    t_blast = t_rel + t_fuz

    t_start_scan = t_blast
    t_end_scan = t_blast + T_SMOKE_EFFECTIVE
    t_vals = np.arange(t_start_scan, t_end_scan, dt)
    
    covered_time_points = set()
    interfered_missiles = set()

    for t in t_vals:
        is_covered_at_t = False
        for m_name in MISSILE_NAMES:
            p_missile_t = missile_pos(m_name, t)
            if p_missile_t[0] < P_TARGET_REAL[0]: continue

            t_after_blast = t - t_blast
            p_cloud_center = p_blast + np.array([0, 0, -V_SINK * (t - t_blast)])
            d, s = dist_point_to_segment(p_cloud_center, p_missile_t, P_TARGET_REAL)
            if d < R_SMOKE and 0 < s < 1:
                is_covered_at_t = True
                interfered_missiles.add(m_name)
        
        if is_covered_at_t:
            # 使用 round 避免浮点数精度问题导致集合元素过多
            covered_time_points.add(round(t, 5))
            
    cover_duration = len(covered_time_points) * dt
    return cover_duration, interfered_missiles, p_release, p_blast

# ---------- 优化目标函数 ----------
def objective(x):
    return -compute_total_cover_time(x, dt=0.2) # 使用非常粗糙的步长加速优化

# ---------- 主程序：差分进化优化 ----------
def solve_problem5():
    uprint("--- 问题5：五无人机、三导弹、每机最多三弹最优策略求解 ---")
    uprint("警告：这是一个40维的超高维优化问题，计算将极其耗时（可能需要数小时）。")
    uprint("正在使用差分进化算法进行全局优化...")

    # 决策变量边界: 5 * 8 = 40个变量
    bounds = []
    for name in UAV_NAMES:
        # [theta, v, tr1, tf1, dtr2, tf2, dtr3, tf3]
        uav_bounds = [
            (-math.pi / 2, math.pi / 2), # 航向偏移: ±90°
            (100, 140),                  # 速度 (m/s)
            (1.0, 40.0),                 # 第1枚弹投放延时 (s)
            (1.0, 25.0),                 # 第1枚弹引信延时 (s)
            (1.0, 30.0),                 # 第2枚弹投放间隔 (s)
            (1.0, 25.0),                 # 第2枚弹引信延时 (s)
            (1.0, 30.0),                 # 第3枚弹投放间隔 (s)
            (1.0, 25.0),                 # 第3枚弹引信延时 (s)
        ]
        bounds.extend(uav_bounds)

    # 调用差分进化求解器
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=1000,     # 针对超高维度问题，必须增加迭代次数
        popsize=25,       # 40维问题，种群大小至少20-30
        polish=False,     # 在高维问题中，polish可能非常耗时且效果不佳
        tol=1e-4,
        updating='deferred',
        workers=-1,       # 使用所有CPU核心
        seed=42
    )

    uprint("\n优化完成！")
    
    # --- 使用高精度dt计算最终结果 ---
    final_cover_time = compute_total_cover_time(result.x, dt=0.01)
    
    # --- 详细结果输出 ---
    uprint("\n" + "="*25 + " 详细投放策略与效果评估 " + "="*25)
    header = (
        f"{'UAV':<5} {'弹号':<4} {'航向(°)':<8} {'速度(m/s)':<10} "
        f"{'投放点(x,y,z)':<32} {'起爆点(x,y,z)':<32} "
        f"{'独立时长(s)':<12} {'干扰目标':<10}"
    )
    uprint(header)
    uprint("-" * len(header))

    for i in range(len(UAV_NAMES)):
        uav_name = UAV_NAMES[i]
        uav_x = result.x[i*8 : (i+1)*8]
        theta_opt, v_opt, tr1_opt, tf1_opt, dtr2_opt, tf2_opt, dtr3_opt, tf3_opt = uav_x
        
        uav_params = (theta_opt, v_opt)
        
        t_rels = [tr1_opt, tr1_opt + dtr2_opt, tr1_opt + dtr2_opt + dtr3_opt]
        t_fuzs = [tf1_opt, tf2_opt, tf3_opt]

        for j in range(3):
            grenade_params = (t_rels[j], t_fuzs[j])
            
            # 计算这枚弹的详细信息
            cover_duration, interfered_set, p_rel, p_blast = compute_single_grenade_cover_details(
                uav_name, uav_params, grenade_params, dt=0.01
            )
            
            # 格式化输出
            heading_deg = np.rad2deg(BASE_ANGLES[uav_name] + theta_opt)
            p_rel_str = f"({p_rel[0]:.1f}, {p_rel[1]:.1f}, {p_rel[2]:.1f})"
            p_blast_str = f"({p_blast[0]:.1f}, {p_blast[1]:.1f}, {p_blast[2]:.1f})"
            interfered_str = ','.join(sorted(list(interfered_set))) if interfered_set else "无"

            line = (
                f"{uav_name:<5} {j+1:<4} {heading_deg:<8.2f} {v_opt:<10.2f} "
                f"{p_rel_str:<32} {p_blast_str:<32} "
                f"{cover_duration:<12.3f} {interfered_str:<10}"
            )
            uprint(line)

    uprint("\n" + "-"*60)
    uprint(f"找到的总有效遮蔽时长（并集）为: {final_cover_time:.4f} 秒")
    uprint("-"*60)

if __name__ == '__main__':
    solve_problem5()