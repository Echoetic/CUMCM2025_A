```mermaid
graph TD
    A[开始] --> B[1. 初始化参数<br/>- 初始温度 T_start<br/>- 终止温度 T_end<br/>- 冷却系数 alpha (a < 1)<br/>- 每个温度下的迭代次数 L];
    B --> C[2. 生成初始解<br/>- 随机生成一个解 S_current<br/>- 令最优解 S_best = S_current];
    
    C --> D{Outer Loop: while T > T_end};
    D -- 是 --> E[Inner Loop: for i = 1 to L];
    
    E --> F[3. 生成新解<br/>在 S_current 的邻域内随机产生一个新解 S_new];
    F --> G[4. 计算能量差<br/>ΔE = Cost(S_new) - Cost(S_current)];
    
    G --> H{ΔE < 0 ?<br/>(新解是否更优?)};
    H -- 是 --> I[5a. 接受新解<br/>S_current = S_new<br/>S_best = S_new];
    H -- 否 --> J{exp(-ΔE / T) > random(0, 1) ?<br/>(是否按概率接受劣解?)};
    
    J -- 是 --> K[5b. 概率接受劣解<br/>S_current = S_new];
    J -- 否 --> L[5c. 拒绝新解<br/>(S_current 保持不变)];
    
    I --> M{i < L ?};
    K --> M;
    L --> M;
    
    M -- 是, 继续内循环 --> E;
    M -- 否, 内循环结束 --> N[6. 降温<br/>T = T * alpha];
    
    N --> D;
    
    D -- 否, 循环结束 --> Z[7. 输出结果<br/>返回找到的最优解 S_best];
    Z --> Z_END[结束];
```