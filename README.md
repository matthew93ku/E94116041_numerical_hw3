第一題程式碼
    import math

    def lagrange_interpolation(xs, ys):
    """
    使用 Lagrange 插值法生成插值多項式 P(x)。
    
    參數:
        xs: 已知 x 值的列表
        ys: 對應的 y 值列表
        
    回傳:
        P(x): 一個函數，輸入 x 可以計算對應的插值結果
    """
    if len(xs) != len(ys):
        raise ValueError("xs 跟 ys 長度不一致")

    def P(x):
        total = 0.0
        n = len(xs)
        for j in range(n):
            Lj = 1.0
            for m in range(n):
                if m != j:
                    Lj *= (x - xs[m]) / (xs[j] - xs[m])
            total += ys[j] * Lj
        return total

    return P 

    if __name__ == "__main__":
    # 已知資料點
    x_data = [0.698, 0.733, 0.768, 0.803]
    y_data = [0.7661, 0.7432, 0.7193, 0.6946]

    # 要近似的點
    x_target = 0.750
    actual_value = 0.7317  # 題目提供的真值

    # 一次多項式 (使用兩個點)
    xs_deg1 = [0.733, 0.768]
    ys_deg1 = [0.7432, 0.7193]
    P1 = lagrange_interpolation(xs_deg1, ys_deg1)
    approx_deg1 = P1(x_target)
    error_deg1 = actual_value - approx_deg1

    # 二次多項式 (使用三個點)
    xs_deg2 = [0.698, 0.733, 0.768]
    ys_deg2 = [0.7661, 0.7432, 0.7193]
    P2 = lagrange_interpolation(xs_deg2, ys_deg2)
    approx_deg2 = P2(x_target)
    error_deg2 = approx_deg2 - actual_value

    # 三次多項式 (使用四個點)
    xs_deg3 = x_data
    ys_deg3 = y_data
    P3 = lagrange_interpolation(xs_deg3, ys_deg3)
    approx_deg3 = P3(x_target)
    error_deg3 = approx_deg3 - actual_value

    # 四次多項式 (使用五個點，包括 x_target)
    xs_deg4 = [0.698, 0.733, 0.750, 0.768, 0.803]
    ys_deg4 = [0.7661, 0.7432, 0.7317, 0.7193, 0.6946]
    P4 = lagrange_interpolation(xs_deg4, ys_deg4)
    approx_deg4 = P4(x_target)
    error_deg4 = approx_deg4 - actual_value

    # 輸出結果
    print("==== Lagrange Interpolation for f(0.750) ====")
    print(f"Degree 1 approximation: {approx_deg1:.6f}, error = {error_deg1:.6f}")
    print(f"Degree 2 approximation: {approx_deg2:.6f}, error = {error_deg2:.6f}")
    print(f"Degree 3 approximation: {approx_deg3:.6f}, error = {error_deg3:.6f}")
    print(f"Degree 4 approximation: {approx_deg4:.6f}, error = {error_deg4:.6f}")
第二題程式碼

    import math
    def f(x):
    """ 定義函數 f(x) = x - e^(-x) """
    return x - math.exp(-x)

    def inverse_quadratic_interp_3points(x0, x1, x2):
    """
    使用三點逆二次插值法 (Inverse Quadratic Interpolation) 找 f(x) = 0 的近似解。

    公式：
      x3 = x0 * (f1 * f2) / ((f0 - f1) * (f0 - f2))
         + x1 * (f0 * f2) / ((f1 - f0) * (f1 - f2))
         + x2 * (f0 * f1) / ((f2 - f0) * (f2 - f1))
         
    其中：
      f0 = f(x0), f1 = f(x1), f2 = f(x2)

    參數:
        x0, x1, x2: 目前的三個近似根

    回傳:
        x3: 新的近似根
    """
    f0, f1, f2 = f(x0), f(x1), f(x2)

    x3 = (x0 * (f1 * f2) / ((f0 - f1) * (f0 - f2)) +
          x1 * (f0 * f2) / ((f1 - f0) * (f1 - f2)) +
          x2 * (f0 * f1) / ((f2 - f0) * (f2 - f1)))

    return x3

    if __name__ == "__main__":
    # 初始值
    x0, x1, x2 = 0.4, 0.5, 0.6  
    tol = 1e-8  # 允許誤差
    max_iter = 200  # 最大迭代次數

    print("\n==== Inverse Quadratic Interpolation for f(x) = x - e^(-x) ====")
    print("Iter |    x0     |    x1     |    x2     |    x_new    |   f(x_new)   ")
    print("-----+-----------+-----------+-----------+-------------+-------------")

    for i in range(max_iter):
        x3 = inverse_quadratic_interp_3points(x0, x1, x2)
        fx3 = f(x3)

        print(f"{i+1:3d}  | {x0:9.6f} | {x1:9.6f} | {x2:9.6f} | {x3:11.8f} | {fx3:12.6e}")

        # 收斂條件：|f(x3)| < tol
        if abs(fx3) < tol:
            print(f"\n收斂於 x = {x3:.8f}")
            break

        # 更新 x0, x1, x2：捨棄最舊的點，保留最新的 x3
        x0, x1, x2 = x1, x2, x3

    print(f"\n最終近似解 x = {x3:.8f}, f(x) = {fx3:.8f}\n")
第三題程式碼
    # === 1. 設定已知數據 ===
    T_data = [0, 3, 8, 13]      # 時間 (秒)
    D_data = [0, 200, 620, 990] # 位置 (英尺)
    V_data = [75, 77, 74, 72]   # 速度 (英尺/秒)

    # 55 mph 轉換為 ft/s
    speed_limit = 55 * 5280 / 3600  # ≈ 80.67 ft/s
    
    # === 2. 定義 Hermite 插值基底函數 ===
    def h00(tau): return 2*tau**3 - 3*tau**2 + 1
    def h10(tau): return tau**3 - 2*tau**2 + tau
    def h01(tau): return -2*tau**3 + 3*tau**2
    def h11(tau): return tau**3 - tau**2

    # 對 τ 微分
    def dh00_dtau(tau): return 6*tau**2 - 6*tau
    def dh10_dtau(tau): return 3*tau**2 - 4*tau + 1
    def dh01_dtau(tau): return -6*tau**2 + 6*tau
    def dh11_dtau(tau): return 3*tau**2 - 2*tau
    
    # === 3. Hermite 插值函數 ===
    def hermite_segment(t, t0, t1, d0, d1, v0, v1):
    """
    在單一區間 [t0, t1] 進行 Hermite 插值，計算 t 時的位置與速度。
    """
    L = t1 - t0  # 區間長度
    tau = (t - t0) / L  # 無因次變數

    # 計算位置 H(t)
    pos = h00(tau) * d0 + h10(tau) * L * v0 + h01(tau) * d1 + h11(tau) * L * v1

    # 計算速度 dH(t)/dt
    dpos_dtau = (dh00_dtau(tau) * d0 + dh10_dtau(tau) * L * v0 +
                 dh01_dtau(tau) * d1 + dh11_dtau(tau) * L * v1)
    vel = dpos_dtau / L  # dH/dt = dH/dτ * dτ/dt

    return pos, vel
    
    # === 4. 找到 t 所屬的區間，並計算 Hermite 插值 ===
    def car_position_speed(t, T, D, V):
    """
    在時間 t 查詢車輛的 (位置, 速度)，使用 Hermite 插值法。
    """
    n = len(T) - 1
    for i in range(n):
        if T[i] <= t <= T[i+1]:
            return hermite_segment(t, T[i], T[i+1], D[i], D[i+1], V[i], V[i+1])

    # 若 t 超出範圍，返回邊界值
    return (D[0], V[0]) if t < T[0] else (D[-1], V[-1])

    # === 5. (a) 計算 t=10 秒時的 位置與速度 ===
    t_query = 10
    pos_10, vel_10 = car_position_speed(t_query, T_data, D_data, V_data)
    print(f"(a) t={t_query} s: position = {pos_10:.2f} ft, speed = {vel_10:.2f} ft/s")

    # === 6. (b) 找到第一次超過 55 mph (≈80.67 ft/s) 的時間 ===
    def first_time_exceed_speed(T, D, V, limit_speed):
    """
    找到車輛第一次超過 speed_limit 的時間，若無則回傳 None。
    """
    n = len(T) - 1
    for i in range(n):
        t0, t1 = T[i], T[i+1]

        # 定義速度函數 speed_fun(t)
        def speed_fun(t):
            _, v = hermite_segment(t, t0, t1, D[i], D[i+1], V[i], V[i+1])
            return v

        # 先檢查區間端點
        if speed_fun(t0) > limit_speed:
            return t0
        if speed_fun(t1) > limit_speed:
            return t1

        # 在 (t0, t1) 內尋找速度是否超過 limit_speed
        steps = 20
        for k in range(steps+1):
            tk = t0 + (t1 - t0) * k / steps
            if speed_fun(tk) > limit_speed:
                return tk
    return None

    exceed_time = first_time_exceed_speed(T_data, D_data, V_data, speed_limit)
    if exceed_time is None:
        print("(b) The car never exceeds 55 mph.")
    else:
        print(f"(b) The car first exceeds 55 mph at t ≈ {exceed_time:.4f} s.")

    # === 7. (c) 找到全程最大速度及對應時間 ===
    def find_max_speed(T, D, V):
    """
    找出整個區間 [T[0], T[-1]] 內的最大速度及其對應時間。
    """
    n = len(T) - 1
    max_speed = -float('inf')
    time_at_max = None

    for i in range(n):
        t0, t1 = T[i], T[i+1]

        # 定義速度函數 speed_fun(t)
        def speed_fun(t):
            _, v = hermite_segment(t, t0, t1, D[i], D[i+1], V[i], V[i+1])
            return v

        # 檢查端點速度
        candidates = [(t0, speed_fun(t0)), (t1, speed_fun(t1))]

        # 以細分區間方式找出速度最大值
        steps = 50
        for k in range(steps+1):
            tk = t0 + (t1 - t0) * k / steps
            candidates.append((tk, speed_fun(tk)))

        # 找到該區間最大速度
        for (t, v) in candidates:
            if v > max_speed:
                max_speed = v
                time_at_max = t

    return max_speed, time_at_max
    
    max_speed, time_at_max = find_max_speed(T_data, D_data, V_data)
    mph_at_max = max_speed * 3600 / 5280  # 轉換為 mph
    
    print(f"(c) The car's maximum speed is {max_speed:.2f} ft/s at t={time_at_max:.2f} s.")
    print(f"    (which is about {mph_at_max:.2f} mph).")
