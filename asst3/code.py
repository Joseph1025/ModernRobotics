import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'Python'))

import modern_robotics as mr
import matplotlib.pyplot as plt


def _compute_error_twist(T_actual, T_desired):
    """Compute the error twist Vb = log(T_actual^-1 * T_desired)."""
    return mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(T_actual), T_desired)))


def _compute_error_magnitudes(Vb):
    """Extract angular and linear error magnitudes from error twist."""
    angular_error = np.linalg.norm(Vb[:3])
    linear_error = np.linalg.norm(Vb[3:])
    return angular_error, linear_error


def _print_iteration(iteration, thetalist, T_act, Vb, angular_error, linear_error):
    """Print iteration information in formatted output."""
    print(f" Iteration {iteration}:\n\n"
          f"        Joint Vector:\n"
          f"        {thetalist[0]:6.4f}, {thetalist[1]:6.4f}, {thetalist[2]:6.4f}, "
          f"{thetalist[3]:6.4f}, {thetalist[4]:6.4f}, {thetalist[5]:6.4f}\n\n"
          f"        SE(3) End-Effector Configuration:\n"
          f"        {T_act[0,0]:6.4f} {T_act[0,1]:6.4f} {T_act[0,2]:6.4f} {T_act[0,3]:6.4f}\n"
          f"        {T_act[1,0]:6.4f} {T_act[1,1]:6.4f} {T_act[1,2]:6.4f} {T_act[1,3]:6.4f}\n"
          f"        {T_act[2,0]:6.4f} {T_act[2,1]:6.4f} {T_act[2,2]:6.4f} {T_act[2,3]:6.4f}\n"
          f"        {T_act[3,0]:6.4f} {T_act[3,1]:6.4f} {T_act[3,2]:6.4f} {T_act[3,3]:6.4f}\n\n"
          f"        Error Twist Vb:\n"
          f"        {Vb[0]:6.4f}, {Vb[1]:6.4f}, {Vb[2]:6.4f}, "
          f"{Vb[3]:6.4f}, {Vb[4]:6.4f}, {Vb[5]:6.4f}\n\n"
          f"        Angular Error Magnitude || omega_b ||: {angular_error:6.5f}\n"
          f"        Linear Error Magnitude || v_b ||: {linear_error:6.5f}\n")


def _save_iterates_to_csv(iter_thetas, filename="long_iterates.csv"):
    """Save joint angle iterations to CSV file."""
    with open(filename, "w") as f:
        for theta_row in iter_thetas:
            f.write(f"{theta_row[0]:7.6f},{theta_row[1]:7.6f},{theta_row[2]:7.6f},"
                    f"{theta_row[3]:7.6f},{theta_row[4]:7.6f},{theta_row[5]:7.6f}\n")


def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev, csv_filename="iterates.csv"):
    """
    Iterative inverse kinematics solver using body Jacobian.
    
    Args:
        Blist: Screw axes in body frame (6x6 numpy array)
        M: Home configuration transformation matrix (4x4 SE(3))
        T: Desired end-effector configuration (4x4 SE(3))
        thetalist0: Initial guess for joint angles (6x1)
        eomg: Angular error tolerance
        ev: Linear error tolerance
        csv_filename: Filename for saving joint angle iterations (default: "iterates.csv")
    
    Returns:
        iter_thetas: All joint angle iterations (Nx6 numpy array)
        success: Boolean indicating convergence
        positions: List of end-effector positions at each iteration
        linear_errors: List of linear error magnitudes
        angular_errors: List of angular error magnitudes
    """
    # Initialize tracking variables
    positions = []
    linear_errors = []
    angular_errors = []
    
    # Initialize joint angles
    thetalist = np.array(thetalist0).copy()
    iter_thetas = np.array([thetalist.copy()])
    
    # Compute initial forward kinematics and error
    T_act = mr.FKinBody(M, Blist, thetalist)
    Vb = _compute_error_twist(T_act, T)
    angular_error, linear_error = _compute_error_magnitudes(Vb)
    
    # Check convergence
    converged = angular_error <= eomg and linear_error <= ev
    
    # Print initial iteration
    _print_iteration(0, thetalist, T_act, Vb, angular_error, linear_error)
    
    # Iterative Newton-Raphson loop
    iteration = 0
    while not converged:
        iteration += 1
        
        # Update joint angles using pseudo-inverse of body Jacobian
        J_body = mr.JacobianBody(Blist, thetalist)
        thetalist = thetalist + np.dot(np.linalg.pinv(J_body), Vb)
        
        # Normalize angles to keep them within valid range
        thetalist = np.arctan2(np.sin(thetalist), np.cos(thetalist))
        
        # Store iteration
        iter_thetas = np.vstack([iter_thetas, thetalist])
        
        # Recompute forward kinematics and error
        T_act = mr.FKinBody(M, Blist, thetalist)
        Vb = _compute_error_twist(T_act, T)
        angular_error, linear_error = _compute_error_magnitudes(Vb)
        
        # Check convergence
        converged = angular_error <= eomg and linear_error <= ev
        
        # Store tracking data
        positions.append([T_act[0, 3], T_act[1, 3], T_act[2, 3]])
        linear_errors.append(linear_error)
        angular_errors.append(angular_error)
        
        # Print iteration info
        _print_iteration(iteration, thetalist, T_act, Vb, angular_error, linear_error)
    
    # Save iterations to CSV
    _save_iterates_to_csv(iter_thetas, csv_filename)
    
    return iter_thetas, converged, positions, linear_errors, angular_errors

def plot_results(positions_s, linear_errors_s, angular_errors_s, positions_l, linear_errors_l, angular_errors_l):
    # 1. 绘制末端执行器位置的 3D 轨迹图
    x_positions_s = [pos[0] for pos in positions_s]
    y_positions_s = [pos[1] for pos in positions_s]
    z_positions_s = [pos[2] for pos in positions_s]

    x_positions_l = [pos[0] for pos in positions_l]
    y_positions_l = [pos[1] for pos in positions_l]
    z_positions_l = [pos[2] for pos in positions_l]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(x_positions_s, y_positions_s, z_positions_s, marker='o', linestyle='-',linewidth = 3, color = 'k', label = 'Short')
    plt.plot(x_positions_l, y_positions_l, z_positions_l, marker='o', linestyle='--', color = 'b', label = 'Long')
    # 标注起点和终点
    ax.scatter(x_positions_s[0], y_positions_s[0], z_positions_s[0], color='g', marker='o', s=100, label='Start Point')  # 起点标记为绿色圈
    ax.scatter(x_positions_l[0], y_positions_l[0], z_positions_l[0], color='g', marker='o', s=100, label='Start Point')  # 起点标记为绿色圈
    ax.scatter(x_positions_s[-1], y_positions_s[-1], z_positions_s[-1], color='r', marker='x', s=100, label='End Point')  # 终点标记为红色叉

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('End-Effector Position at Each Iteration')
    plt.legend()
    plt.show()

    # 2. 绘制线性误差大小随迭代次数变化的图
    plt.figure()
    plt.plot(range(len(linear_errors_s)), linear_errors_s, marker='o', linestyle='-', color = 'k', label = 'Short')
    plt.plot(range(len(linear_errors_l)), linear_errors_l, marker='o', linestyle='--', color = 'b', label = 'Long')
    plt.xlabel('Iteration Number')
    plt.ylabel('Linear Error Magnitude ||v_b|| (m)')
    plt.title('Linear Error Magnitude vs. Iteration Number')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. 绘制角误差大小随迭代次数变化的图
    plt.figure()
    plt.plot(range(len(angular_errors_s)), angular_errors_s, marker='o', linestyle='-', color = 'k', label = 'Short')
    plt.plot(range(len(angular_errors_l)), angular_errors_l, marker='o', linestyle='--', color = 'b', label = 'Long')
    plt.xlabel('Iteration Number')
    plt.ylabel('Angular Error Magnitude ||omega_b|| (rad)')
    plt.title('Angular Error Magnitude vs. Iteration Number')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    H1 = .089 # 1519
    H2 = .095 #08535
    W1 = .109 # 11235
    W2 = .082
    L1 = .425 # 24365
    L2 = .392 # 21325
    
    B1 = np.array([0,1,0,W1+W2,0,L1+L2])
    B2 = np.array([0,0,1,H2,-L1-L2, 0])
    B3 = np.array([0,0,1,H2,-L2,0])
    B4 = np.array([0,0,1,H2,0,0])
    B5 = np.array([0,-1,0,-W2,0,0])
    B6 = np.array([0,0,1,0,0,0])
    Blist = np.array([B1,B2,B3,B4,B5,B6])
    Blist = Blist.T

    
    # Initial guesses
    # Short: slightly offset from solution (norm ~ 2.9) - should take 2-3 iterations
    thetalist0_short = np.array([1.1, -2.3, 1.4, 2.1, 0.7, 0.4])
    # Long: starting from a more extreme configuration - should take more iterations
    thetalist0_long = np.array([2.5, -2.0, -1.5, -2.2, 2.8, 2.0])


    # M
    M = np.array([[-1, 0, 0, L1+L2], 
                  [0, 0, 1, W1+W2], 
                  [0, 1, 0, H1-H2],
                  [0,0,0,1]])
    
    T = np.array([[0, 0, -1, 0],
                  [1, 0, 0, 0.6],
                  [0, -1, 0, 0],
                  [0, 0, 0, 1]])

    eomg = 0.001
    ev = 0.0001

    iter_thetas_s, err_s, positions_s, linear_errors_s, angular_errors_s = IKinBodyIterates(Blist,M,T,thetalist0_short,eomg,ev, csv_filename="short_iterates.csv")
    iter_thetas_l, err_short_l, positions_l, linear_errors_l, angular_errors_l = IKinBodyIterates(Blist,M,T,thetalist0_long,eomg,ev, csv_filename="long_iterates.csv")

    # plot result 
    plot_results(positions_s, linear_errors_s, angular_errors_s, positions_l, linear_errors_l, angular_errors_l)