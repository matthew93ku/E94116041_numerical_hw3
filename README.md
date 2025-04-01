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

    import numpy as np
    from scipy.optimize import fsolve
    
    # 給定的數據點
    t_points = [0, 3, 5, 8, 13]  # 時間 T (秒)
    d_points = [0, 200, 375, 620, 990]  # 距離 D (英尺)
    v_points = [75, 77, 80, 74, 72]  # 速度 V (英尺/秒)
    
    # Hermite 插值的基礎：計算除法差分表
    def divided_differences(t_points, d_points, v_points):
        n = len(t_points)
        z = np.zeros(2 * n)  
        Q = np.zeros((2 * n, 2 * n))  
        
        for i in range(n):
            z[2 * i] = t_points[i]
            z[2 * i + 1] = t_points[i]
        
        for i in range(n):
            Q[2 * i, 0] = d_points[i]  
            Q[2 * i + 1, 0] = d_points[i]
            Q[2 * i + 1, 1] = v_points[i]  
        
        for i in range(2, 2 * n):
            for j in range(2, i + 1):
                if j == 2 and i % 2 == 1:
                    continue  
                Q[i, j] = (Q[i, j-1] - Q[i-1, j-1]) / (z[i] - z[i-j])
        
        coeffs = [Q[i, i] for i in range(2 * n)]
        return z, coeffs
    
    # 計算 Hermite 插值多項式的值
    def hermite_interpolation(t, t_points, d_points, v_points):
        z, coeffs = divided_differences(t_points, d_points, v_points)
        n = len(t_points)
        result = coeffs[0]  
        product = 1.0
        for i in range(1, 2 * n):
            product *= (t - z[i-1])
            result += coeffs[i] * product
        return result
    
    # 計算 Hermite 插值多項式的導數
    def hermite_derivative(t, t_points, d_points, v_points):
        z, coeffs = divided_differences(t_points, d_points, v_points)
        n = len(t_points)
        result = 0.0
        for i in range(1, 2 * n):
            term = 0.0
            for j in range(i):
                prod = 1.0
                for k in range(i):
                    if k != j:
                        prod *= (t - z[k])
                term += prod
            result += coeffs[i] * term
        return result
    
    # 計算 Hermite 插值多項式的二階導數
    def hermite_second_derivative(t, t_points, d_points, v_points, h=1e-5):
        return (hermite_derivative(t + h, t_points, d_points, v_points) - 
                hermite_derivative(t - h, t_points, d_points, v_points)) / (2 * h)
    
    # a. 預測 t = 10 時的位置和速度
    t_target = 10
    position = hermite_interpolation(t_target, t_points, d_points, v_points)
    speed = hermite_derivative(t_target, t_points, d_points, v_points)
    print(f"\na. At t = {t_target} seconds:")
    print(f"Position: {position:.2f} feet")
    print(f"Speed: {speed:.2f} feet/second")
    
    # b. 何時首次超過 55 mi/h
    speed_limit_mph = 55
    speed_limit_fps = speed_limit_mph * 5280 / 3600  
    print(f"\nb. Speed limit: {speed_limit_mph} mi/h = {speed_limit_fps:.2f} feet/second")
    
    # 定義方程：H'(t) - speed_limit = 0
    def speed_equation(t):
        return hermite_derivative(t, t_points, d_points, v_points) - speed_limit_fps
    
    t_first_exceed = None
    for i in range(len(t_points) - 1):
        t_start = t_points[i]
        t_end = t_points[i + 1]
        
        speed_start = hermite_derivative(t_start, t_points, d_points, v_points)
        speed_end = hermite_derivative(t_end, t_points, d_points, v_points)
        
        if (speed_start - speed_limit_fps) * (speed_end - speed_limit_fps) < 0:
            t_exceed = fsolve(speed_equation, [(t_start + t_end) / 2])[0]
            if t_start <= t_exceed <= t_end:  
                t_first_exceed = t_exceed
                break  
    
    if t_first_exceed is not None:
        print(f"First time exceeding speed limit: t = {t_first_exceed:.2f} seconds")
    else:
        print("The car never exceeds the speed limit of 55 mi/h.")
    
    # c. 預測最大速度
    def second_derivative_equation(t):
        return hermite_second_derivative(t, t_points, d_points, v_points)
    
    critical_points = []
    for i in range(len(t_points) - 1):
        t_start = t_points[i]
        t_end = t_points[i + 1]
        
        t_critical = fsolve(second_derivative_equation, [(t_start + t_end) / 2])[0]
        if t_start <= t_critical <= t_end:
            critical_points.append(t_critical)
    
    critical_points.extend([t_points[0], t_points[-1]])
    
    speeds = [hermite_derivative(t, t_points, d_points, v_points) for t in critical_points]
    max_speed = max(speeds)
    max_speed_mph = max_speed * 3600 / 5280  
    
    print(f"\nc. Predicted maximum speed: {max_speed:.2f} feet/second = {max_speed_mph:.2f} mi/h")
    print(f"Critical points for speed: {[f'{t:.2f}' for t in critical_points]}")

