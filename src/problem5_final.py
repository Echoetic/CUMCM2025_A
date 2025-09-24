import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# ------------------- 参数设定 -------------------
uav_count = 5
smoke_per_uav = 3
smoke_burn = 8.0       # 每颗烟幕弹持续时间
min_gap = 0.5          # 同一无人机相邻投放间隔 ≥ 0.5s

# 三枚导弹出现时间区间，覆盖整个作战窗口
missile_intervals = [
    (0.0, 10.0),   # 第1枚导弹
    (10.0, 20.0),  # 第2枚导弹
    (20.0, 30.0)   # 第3枚导弹
]

# ------------------- 覆盖时间计算 -------------------
def compute_coverage(x):
    events = []
    for u in range(uav_count):
        rel_times = []
        for k in range(smoke_per_uav):
            idx = (u * smoke_per_uav + k) * 2
            t_rel = x[idx]
            t_fuze = x[idx+1]
            t_burst = t_rel + t_fuze
            rel_times.append(t_burst)
            events.append((u, k, t_burst))
        rel_times.sort()
        for i in range(1, len(rel_times)):
            if rel_times[i] - rel_times[i-1] < min_gap:
                return -1e6  # 违反间隔约束，惩罚

    events.sort(key=lambda e: e[2])
    cover_total = 0.0
    for (start, end) in missile_intervals:
        intervals = []
        for (_, _, tb) in events:
            s = tb
            e = tb + smoke_burn
            if e < start or s > end:
                continue
            intervals.append([max(s, start), min(e, end)])
        if not intervals:
            continue
        intervals.sort()
        merged = [intervals[0]]
        for seg in intervals[1:]:
            if seg[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], seg[1])
            else:
                merged.append(seg)
        cover_total += sum(e - s for s, e in merged)
    return cover_total

def obj(x):
    return -compute_coverage(x)

# ------------------- 可视化函数 -------------------
def visualize_optimization(history, x_opt, best_cover):
    """可视化优化过程和结果"""
    
    # 1. 绘制优化过程收敛曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history, 'b-', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('总覆盖时间 (s)')
    plt.title('优化过程收敛曲线')
    plt.grid(True, alpha=0.3)
    plt.savefig('optimization_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 创建甘特图显示烟幕弹覆盖时间
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 为每个无人机分配颜色
    colors = plt.cm.Set3(np.linspace(0, 1, uav_count))
    
    # 收集所有烟幕弹数据
    smoke_data = []
    for u in range(uav_count):
        for k in range(smoke_per_uav):
            idx = (u * smoke_per_uav + k) * 2
            t_rel = x_opt[idx]
            t_fuze = x_opt[idx+1]
            t_burst = t_rel + t_fuze
            t_end = t_burst + smoke_burn
            smoke_data.append({
                'uav': u,
                'smoke': k,
                't_rel': t_rel,
                't_fuze': t_fuze,
                't_burst': t_burst,
                't_end': t_end
            })
    
    # 按起爆时间排序
    smoke_data.sort(key=lambda x: x['t_burst'])
    
    # 绘制每个烟幕弹
    for i, data in enumerate(smoke_data):
        uav_idx = data['uav']
        y_pos = i  # 从上到下的位置
        
        # 投放时间点
        ax.plot(data['t_rel'], y_pos, 'o', color=colors[uav_idx], markersize=8, 
                label=f'UAV{uav_idx+1}投放' if i == uav_idx * smoke_per_uav else "")
        
        # 起爆时间点
        ax.plot(data['t_burst'], y_pos, 's', color=colors[uav_idx], markersize=8,
                label=f'UAV{uav_idx+1}起爆' if i == uav_idx * smoke_per_uav else "")
        
        # 烟幕持续时间
        ax.hlines(y=y_pos, xmin=data['t_burst'], xmax=data['t_end'], 
                 color=colors[uav_idx], linewidth=6, alpha=0.7,
                 label=f'UAV{uav_idx+1}烟幕' if i == uav_idx * smoke_per_uav else "")
    
    # 添加导弹时间窗口
    missile_colors = ['lightcoral', 'lightblue', 'lightgreen']
    for i, (start, end) in enumerate(missile_intervals):
        ax.axvspan(start, end, alpha=0.2, color=missile_colors[i], 
                  label=f'导弹{i+1}时间窗口')
    
    # 设置y轴标签
    y_labels = [f'UAV{data["uav"]+1}-弹{data["smoke"]+1}' for data in smoke_data]
    ax.set_yticks(range(len(smoke_data)))
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()  # 反转y轴使第一个在最上面
    
    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # 设置标题和标签
    ax.set_xlabel('时间 (s)')
    ax.set_title(f'烟幕弹覆盖甘特图 (总覆盖时间: {best_cover:.2f}s)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smoke_coverage_gantt.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 绘制每个导弹时间窗口内的覆盖情况
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 计算每个导弹时间窗口的覆盖情况
    for missile_idx, (start, end) in enumerate(missile_intervals):
        ax = axes[missile_idx]
        
        # 收集该时间窗口内的所有烟幕段
        intervals = []
        for data in smoke_data:
            t_burst = data['t_burst']
            t_end = data['t_end']
            
            # 检查是否与当前导弹时间窗口有重叠
            if t_end < start or t_burst > end:
                continue
                
            # 计算重叠部分
            overlap_start = max(t_burst, start)
            overlap_end = min(t_end, end)
            intervals.append((overlap_start, overlap_end, data['uav']))
        
        # 合并重叠区间
        if intervals:
            intervals.sort()
            merged = [intervals[0]]
            for seg in intervals[1:]:
                if seg[0] <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]), merged[-1][2])
                else:
                    merged.append(seg)
            
            # 绘制合并后的覆盖区间
            for i, (s, e, uav_idx) in enumerate(merged):
                ax.barh(0, e-s, left=s, height=0.5, color=colors[uav_idx], alpha=0.7)
        
        # 设置导弹时间窗口背景
        ax.axvspan(start, end, alpha=0.1, color=missile_colors[missile_idx])
        
        # 设置标题和标签
        ax.set_xlim(start, end)
        ax.set_yticks([])
        ax.set_xlabel('时间 (s)')
        ax.set_title(f'导弹{missile_idx+1}时间窗口覆盖情况')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('missile_coverage_details.png', dpi=300, bbox_inches='tight')
    plt.show()

# ------------------- 决策变量范围 -------------------
bounds = []
for u in range(uav_count):
    for k in range(smoke_per_uav):
        bounds.append((0.0, 30.0))  # 投放时刻
        bounds.append((0.5, 3.0))   # 引信延时

# ------------------- 差分进化全局优化 -------------------
# 添加回调函数记录优化历史
history = []
def callback(xk, convergence):
    cover = compute_coverage(xk)
    history.append(cover)
    return False

print("开始优化...")
result = differential_evolution(
    obj,
    bounds,
    maxiter=1000,      # 增大迭代次数
    popsize=25,        # 增大种群规模
    polish=True,
    tol=1e-7,
    seed=42,
    disp=True,
    callback=callback
)

x_opt = result.x
best_cover = -result.fun

# ------------------- 输出结果 -------------------
rows = []
for u in range(uav_count):
    for k in range(smoke_per_uav):
        idx = (u * smoke_per_uav + k) * 2
        t_rel = x_opt[idx]
        t_fuze = x_opt[idx+1]
        t_burst = t_rel + t_fuze
        rows.append([f"UAV{u+1}", k+1, t_rel, t_fuze, t_burst, t_burst+smoke_burn])

df = pd.DataFrame(rows, columns=["UAV", "Smoke#", "t_release", "t_fuze", "t_burst", "t_end"])
df = df.sort_values(by="t_burst").reset_index(drop=True)
print(df.to_string(index=False))
print(f"\n>>> Total coverage (3 missiles): {best_cover:.2f} s")

# 保存文件
df.to_csv("result5.csv", index=False)

# ------------------- 可视化结果 -------------------
print("生成可视化图表...")
visualize_optimization(history, x_opt, best_cover)