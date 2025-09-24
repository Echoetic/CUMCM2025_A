import numpy as np
import math
import sys
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

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
P_UAV1_INITIAL = np.array([17800.0, 0.0, 1800.0])
P_M1 = np.array([20000.0, 0.0, 2000.0])
P_TARGET_REAL = np.array([0.0, 200.0, 5.0])
P_TARGET_FAKE = np.array([0.0, 0.0, 0.0])

# 预计算的常量向量
MISSILE_DIR = (P_TARGET_FAKE - P_M1) / np.linalg.norm(P_TARGET_FAKE - P_M1)
BASE_ANGLE = math.atan2(P_TARGET_FAKE[1] - P_UAV1_INITIAL[1], P_TARGET_FAKE[0] - P_UAV1_INITIAL[0])

# ---------- 工具函数 ----------
def missile_pos(t):
    return P_M1 + V_MISSILE * t * MISSILE_DIR

def dist_point_to_segment(P, A, B):
    AB = B - A
    AP = P - A
    ab2 = np.dot(AB, AB)
    if ab2 == 0.0: return np.linalg.norm(P - A), 0.0
    s = np.dot(AP, AB) / ab2
    # s_clamped = max(0.0, min(1.0, s)) # 在判断逻辑中处理s的范围
    closest = A + s * AB
    return np.linalg.norm(P - closest), s

# ---------- 核心计算函数 (3枚弹) ----------
def compute_cover_time_3_grenades(x, dt=0.02):
    """
    计算3枚弹的总遮蔽时长（并集）。
    x: [theta_offset, v_uav, t_rel1, t_fuz1, dt_rel2, t_fuz2, dt_rel3, t_fuz3]
    """
    theta_offset, v_uav, t_rel1, t_fuz1, dt_rel2, t_fuz2, dt_rel3, t_fuz3 = x

    # --- 计算弹道与起爆点 ---
    heading = BASE_ANGLE + theta_offset
    uav_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
    v_uav_vec = v_uav * uav_dir

    # 计算真实的投放时间
    t_rel2 = t_rel1 + dt_rel2
    t_rel3 = t_rel2 + dt_rel3
    
    release_times = [t_rel1, t_rel2, t_rel3]
    fuze_times = [t_fuz1, t_fuz2, t_fuz3]
    
    blast_params = []
    for t_rel, t_fuz in zip(release_times, fuze_times):
        p_release = P_UAV1_INITIAL + v_uav_vec * t_rel
        p_blast = p_release + v_uav_vec * t_fuz + np.array([0, 0, -0.5 * G * t_fuz**2])
        t_blast = t_rel + t_fuz
        blast_params.append({'p_blast': p_blast, 't_blast': t_blast})

    # --- 模拟与扫描 ---
    # 确定仿真时间范围
    if not blast_params: return 0.0
    t_start_scan = min(p['t_blast'] for p in blast_params)
    t_end_scan = max(p['t_blast'] for p in blast_params) + T_SMOKE_EFFECTIVE
    t_vals = np.arange(t_start_scan, t_end_scan, dt)
    
    total_time = 0
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
                    break # 当前时间点已被覆盖，无需检查其他弹
        
        if is_covered_at_t:
            total_time += dt
            
    return total_time

def compute_single_grenade_cover_time(x_single, dt=0.01):
    """
    计算单枚弹的独立遮蔽时长。
    x_single: [theta_offset, v_uav, t_release, t_fuze]
    """
    theta_offset, v_uav, t_release, t_fuze = x_single

    # --- 计算弹道与起爆点 ---
    heading = BASE_ANGLE + theta_offset
    uav_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
    v_uav_vec = v_uav * uav_dir
    
    p_release = P_UAV1_INITIAL + v_uav_vec * t_release
    p_blast = p_release + v_uav_vec * t_fuze + np.array([0, 0, -0.5 * G * t_fuze**2])
    t_blast = t_release + t_fuze

    # --- 模拟与扫描 ---
    t_start_scan = t_blast
    t_end_scan = t_blast + T_SMOKE_EFFECTIVE
    t_vals = np.arange(t_start_scan, t_end_scan, dt)
    
    single_cover_time = 0
    for t in t_vals:
        p_missile_t = missile_pos(t)
        p_cloud_center = p_blast + np.array([0, 0, -V_SINK * (t - t_blast)])
        d, s = dist_point_to_segment(p_cloud_center, p_missile_t, P_TARGET_REAL)
        if d < R_SMOKE and 0 < s < 1:
            single_cover_time += dt
            
    return single_cover_time

# ---------- 优化目标函数 ----------
def objective(x):
    """优化器调用的目标函数，返回负时长。"""
    # 使用粗糙的dt进行快速评估
    return -compute_cover_time_3_grenades(x, dt=0.1)

# ---------- 可视化辅助函数 ----------
def compute_cover_time_3_grenades_with_details(x, dt=0.02):
    """
    计算3枚弹的总遮蔽时长（并集），并返回详细信息用于可视化。
    这个函数与compute_cover_time_3_grenades逻辑完全相同，只是返回更多信息。
    """
    theta_offset, v_uav, t_rel1, t_fuz1, dt_rel2, t_fuz2, dt_rel3, t_fuz3 = x

    # --- 计算弹道与起爆点 ---
    heading = BASE_ANGLE + theta_offset
    uav_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
    v_uav_vec = v_uav * uav_dir

    # 计算真实的投放时间
    t_rel2 = t_rel1 + dt_rel2
    t_rel3 = t_rel2 + dt_rel3
    
    release_times = [t_rel1, t_rel2, t_rel3]
    fuze_times = [t_fuz1, t_fuz2, t_fuz3]
    
    blast_params = []
    for t_rel, t_fuz in zip(release_times, fuze_times):
        p_release = P_UAV1_INITIAL + v_uav_vec * t_rel
        p_blast = p_release + v_uav_vec * t_fuz + np.array([0, 0, -0.5 * G * t_fuz**2])
        t_blast = t_rel + t_fuz
        blast_params.append({
            'p_blast': p_blast, 
            't_blast': t_blast,
            't_rel': t_rel,
            't_fuz': t_fuz
        })

    # --- 模拟与扫描 ---
    if not blast_params: 
        return 0.0, [], [], []
        
    t_start_scan = min(p['t_blast'] for p in blast_params)
    t_end_scan = max(p['t_blast'] for p in blast_params) + T_SMOKE_EFFECTIVE
    t_vals = np.arange(t_start_scan, t_end_scan, dt)
    
    total_time = 0
    covered = []  # 记录每个时间点是否被遮蔽
    cover_states = [[] for _ in range(3)]  # 记录每个烟幕弹的遮蔽状态
    
    for t in t_vals:
        p_missile_t = missile_pos(t)
        is_covered_at_t = False
        
        for i, params in enumerate(blast_params):
            t_after_blast = t - params['t_blast']
            if 0 <= t_after_blast < T_SMOKE_EFFECTIVE:
                p_cloud_center = params['p_blast'] + np.array([0, 0, -V_SINK * t_after_blast])
                d, s = dist_point_to_segment(p_cloud_center, p_missile_t, P_TARGET_REAL)
                is_grenade_covered = (d < R_SMOKE and 0 < s < 1)
                cover_states[i].append(is_grenade_covered)
                
                if is_grenade_covered:
                    is_covered_at_t = True
            else:
                cover_states[i].append(False)
                
        covered.append(is_covered_at_t)
        if is_covered_at_t:
            total_time += dt
            
    return total_time, blast_params, covered, cover_states

def visualize_results(x_opt, cover_time, bounds):
    """可视化优化结果"""
    # 计算详细信息用于可视化
    detailed_cover_time, blast_params, covered, cover_states = compute_cover_time_3_grenades_with_details(x_opt, dt=0.05)
    
    theta_opt, v_opt, tr1_opt, tf1_opt, dtr2_opt, tf2_opt, dtr3_opt, tf3_opt = x_opt
    tr2_opt = tr1_opt + dtr2_opt
    tr3_opt = tr2_opt + dtr3_opt
    
    # 计算每枚弹的独立贡献时长
    high_precision_dt = 0.005
    cover1 = compute_single_grenade_cover_time([theta_opt, v_opt, tr1_opt, tf1_opt], dt=high_precision_dt)
    cover2 = compute_single_grenade_cover_time([theta_opt, v_opt, tr2_opt, tf2_opt], dt=high_precision_dt)
    cover3 = compute_single_grenade_cover_time([theta_opt, v_opt, tr3_opt, tf3_opt], dt=high_precision_dt)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建综合可视化
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('单无人机三弹最优策略可视化分析', fontsize=16)
    
    # 使用GridSpec创建复杂的布局
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # 1. 三维轨迹可视化
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # 计算导弹轨迹
    missile_t = np.linspace(0, np.linalg.norm(P_TARGET_FAKE - P_M1)/V_MISSILE, 100)
    missile_points = np.array([missile_pos(t) for t in missile_t])
    
    # 计算无人机轨迹
    heading_opt = BASE_ANGLE + theta_opt
    uav_dir_opt = np.array([math.cos(heading_opt), math.sin(heading_opt), 0.0])
    max_release_time = max([tr1_opt, tr2_opt, tr3_opt])
    uav_t = np.linspace(0, max_release_time, 100)
    uav_points = np.array([P_UAV1_INITIAL + v_opt * t * uav_dir_opt for t in uav_t])
    
    # 绘制导弹轨迹
    ax1.plot(missile_points[:, 0], missile_points[:, 1], missile_points[:, 2], 
             'r-', linewidth=2, label='导弹轨迹')
    ax1.scatter(P_M1[0], P_M1[1], P_M1[2], c='red', s=100, marker='o', label='导弹起始点')
    ax1.scatter(P_TARGET_REAL[0], P_TARGET_REAL[1], P_TARGET_REAL[2], c='red', s=100, marker='*', label='目标点')
    
    # 绘制无人机轨迹
    ax1.plot(uav_points[:, 0], uav_points[:, 1], uav_points[:, 2], 
             'b-', linewidth=2, label='无人机轨迹')
    ax1.scatter(P_UAV1_INITIAL[0], P_UAV1_INITIAL[1], P_UAV1_INITIAL[2], c='blue', s=100, marker='o', label='无人机起始点')
    
    # 标记烟幕释放点和爆炸点
    colors = ['green', 'orange', 'purple']
    labels = ['弹1', '弹2', '弹3']
    
    for i, params in enumerate(blast_params):
        # 释放点
        release_point = P_UAV1_INITIAL + v_opt * params['t_rel'] * uav_dir_opt
        ax1.scatter(release_point[0], release_point[1], release_point[2], 
                   c=colors[i], s=100, marker='X', label=f'{labels[i]}释放点')
        
        # 爆炸点
        ax1.scatter(params['p_blast'][0], params['p_blast'][1], params['p_blast'][2], 
                   c=colors[i], s=150, marker='*', label=f'{labels[i]}爆炸点')
    
    ax1.set_xlabel('X坐标 (m)')
    ax1.set_ylabel('Y坐标 (m)')
    ax1.set_zlabel('Z坐标 (m)')
    ax1.set_title('三维轨迹与烟幕弹位置')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)
    
    # 2. 烟幕遮蔽时间线
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 计算时间值
    t_start_scan = min(p['t_blast'] for p in blast_params)
    t_end_scan = max(p['t_blast'] for p in blast_params) + T_SMOKE_EFFECTIVE
    t_vals = np.linspace(t_start_scan, t_end_scan, len(covered))
    
    # 绘制遮蔽时间线
    for i in range(3):
        ax2.plot(t_vals, [1.2*i + 0.1 + 0.8*state for state in cover_states[i]], 
                label=f'{labels[i]}遮蔽', color=colors[i], linewidth=2)
    
    ax2.plot(t_vals, [3.5 + 0.8*state for state in covered], 
            label='合并遮蔽', color='red', linewidth=3)
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('遮蔽状态')
    ax2.set_yticks([0.5, 1.3, 2.1, 3.9])
    ax2.set_yticklabels(['弹1', '弹2', '弹3', '合并'])
    ax2.set_title('烟幕遮蔽时间线')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 独立与合并遮蔽时间对比
    ax3 = fig.add_subplot(gs[0, 2])
    labels = ['弹1', '弹2', '弹3', '合并遮蔽']
    times = [cover1, cover2, cover3, cover_time]
    colors = ['green', 'orange', 'purple', 'red']
    
    bars = ax3.bar(labels, times, color=colors)
    ax3.set_ylabel('遮蔽时间 (s)')
    ax3.set_title('独立与合并遮蔽时间对比')
    
    # 在柱状图上添加数值标签
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time:.2f}s', ha='center', va='bottom')
    
    # 4. 烟幕弹时间安排
    ax4 = fig.add_subplot(gs[1, 0])
    
    # 创建时间线图
    for i, (t_rel, t_fuz) in enumerate(zip([tr1_opt, tr2_opt, tr3_opt], [tf1_opt, tf2_opt, tf3_opt])):
        t_blast = t_rel + t_fuz
        ax4.barh(i, T_SMOKE_EFFECTIVE, left=t_blast, height=0.6, 
                color=colors[i], alpha=0.5, label=f'{labels[i]}有效时间')
        ax4.arrow(t_rel, i, t_fuz, 0, head_width=0.2, head_length=0.5, 
                 fc=colors[i], ec=colors[i], label=f'{labels[i]}引信时间')
        ax4.scatter(t_rel, i, color=colors[i], s=100, marker='o', label=f'{labels[i]}投放时间')
    
    ax4.set_xlabel('时间 (s)')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['弹1', '弹2', '弹3'])
    ax4.set_title('烟幕弹时间安排')
    ax4.grid(True)
    ax4.legend(loc='upper right')
    
    # 5. 参数敏感度分析
    ax5 = fig.add_subplot(gs[1, 1])
    param_names = ['航向偏移', '速度', '投放延时1', '引信延时1', '投放间隔2', '引信延时2', '投放间隔3', '引信延时3']
    sensitivities = []
    
    for i in range(8):
        perturbed_x = x_opt.copy()
        perturbation = 0.05 * (bounds[i][1] - bounds[i][0])
        perturbed_x[i] += perturbation
        perturbed_cover = compute_cover_time_3_grenades(perturbed_x, dt=0.05)
        sensitivity = abs(perturbed_cover - cover_time) / perturbation
        sensitivities.append(sensitivity)
    
    ax5.bar(range(8), sensitivities)
    ax5.set_xticks(range(8))
    ax5.set_xticklabels(param_names, rotation=45, ha='right')
    ax5.set_ylabel('敏感度')
    ax5.set_title('参数对遮蔽时间的敏感度分析')
    ax5.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 6. 结果摘要
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    textstr = '\n'.join((
        f'最优航向偏移: {np.degrees(theta_opt):.2f}°',
        f'最优速度: {v_opt:.2f} m/s',
        '',
        '弹1:',
        f'  投放时间: {tr1_opt:.2f} s',
        f'  引信时间: {tf1_opt:.2f} s',
        f'  独立遮蔽: {cover1:.2f} s',
        '',
        '弹2:',
        f'  投放时间: {tr2_opt:.2f} s',
        f'  引信时间: {tf2_opt:.2f} s',
        f'  独立遮蔽: {cover2:.2f} s',
        '',
        '弹3:',
        f'  投放时间: {tr3_opt:.2f} s',
        f'  引信时间: {tf3_opt:.2f} s',
        f'  独立遮蔽: {cover3:.2f} s',
        '',
        f'总遮蔽时间: {cover_time:.2f} s'
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax6.text(0.5, 0.5, textstr, transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.show()

# ---------- 主程序：差分进化优化 ----------
def solve_problem3():
    uprint("--- 问题3：单无人机三弹最优策略求解 ---")
    uprint("正在使用差分进化算法进行全局优化，这可能需要较长时间...")

    # 决策变量边界: 8个变量
    # 借鉴A3.py的思路，收紧边界以提高效率
    # [theta_offset, v_uav, t_rel1, t_fuz1, dt_rel2, t_fuz2, dt_rel3, t_fuz3]
    bounds = [
        (-math.pi / 4, math.pi / 4), # 航向偏移: 进一步收紧至 ±45°
        (100, 140),                  # 速度 (m/s): 高速通常更有利于快速部署
        (0.5, 5.0),                  # 第1枚弹投放延时 (s)
        (1.0, 8.0),                  # 第1枚弹引信延时 (s)
        (1.0, 15.0),                 # 第2枚弹投放间隔 (s), 下限为1s
        (1.0, 8.0),                  # 第2枚弹引信延时 (s)
        (1.0, 15.0),                 # 第3枚弹投放间隔 (s), 下限为1s
        (1.0, 8.0),                  # 第3枚弹引信延时 (s)
    ]

    # 调用差分进化求解器
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=300,      # 增加迭代次数以进行更充分的搜索
        popsize=25,       # 增加种群大小
        polish=True, 
        tol=1e-6,
        updating='deferred', # 并行计算
        workers=-1,          # 使用所有CPU核心
        seed=42
    )

    uprint("\n优化完成！")
    
    # --- 使用高精度dt计算最终结果 ---
    final_cover_time = compute_cover_time_3_grenades(result.x, dt=0.005)
    
    # --- 结果输出 ---
    uprint("\n" + "="*20 + " 最优策略 " + "="*20)
    
    theta_opt, v_opt, tr1_opt, tf1_opt, dtr2_opt, tf2_opt, dtr3_opt, tf3_opt = result.x
    tr2_opt = tr1_opt + dtr2_opt
    tr3_opt = tr2_opt + dtr3_opt

    # 计算每枚弹的独立贡献时长
    high_precision_dt = 0.005
    cover1 = compute_single_grenade_cover_time([theta_opt, v_opt, tr1_opt, tf1_opt], dt=high_precision_dt)
    cover2 = compute_single_grenade_cover_time([theta_opt, v_opt, tr2_opt, tf2_opt], dt=high_precision_dt)
    cover3 = compute_single_grenade_cover_time([theta_opt, v_opt, tr3_opt, tf3_opt], dt=high_precision_dt)

    uprint(f"无人机飞行速度: {v_opt:.4f} m/s")
    uprint(f"无人机飞行航向: {np.rad2deg(BASE_ANGLE + theta_opt):.4f} 度 (相对基准偏移 {np.rad2deg(theta_opt):.4f} 度)")
    uprint("\n--- 干扰弹投放详情 ---")
    uprint(f"弹1: 投放时间 = {tr1_opt:.4f} s, 引信时间 = {tf1_opt:.4f} s, 起爆时刻 = {tr1_opt + tf1_opt:.4f} s, [独立遮蔽: {cover1:.4f} s]")
    uprint(f"弹2: 投放时间 = {tr2_opt:.4f} s, 引信时间 = {tf2_opt:.4f} s, 起爆时刻 = {tr2_opt + tf2_opt:.4f} s, [独立遮蔽: {cover2:.4f} s]")
    uprint(f"弹3: 投放时间 = {tr3_opt:.4f} s, 引信时间 = {tf3_opt:.4f} s, 起爆时刻 = {tr3_opt + tf3_opt:.4f} s, [独立遮蔽: {cover3:.4f} s]")
    
    uprint(f"\n注意: 各弹独立遮蔽时长之和 ({cover1+cover2+cover3:.4f} s) 可能因时间重叠而大于总有效遮蔽时长。")
    uprint("\n" + "-"*50)
    uprint(f"找到的总有效遮蔽时长为: {final_cover_time:.4f} 秒")
    uprint("-"*50)
    
    # 可视化结果
    visualize_results(result.x, final_cover_time, bounds)

if __name__ == '__main__':
    solve_problem3()