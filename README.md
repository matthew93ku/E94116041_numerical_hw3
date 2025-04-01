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
