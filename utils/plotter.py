import numpy as np
import matplotlib.pyplot as plt

def plot_demo(svm, X, y, filename="plot.png"):
    K = X @ X.T
    history = np.array(getattr(svm, "alpha_history", []))
    
    grid_max = 1.0
    if len(history) > 0:
        grid_max = max(grid_max, history[:,0].max()*1.2, history[:,1].max()*1.2)

    a1 = np.linspace(0, grid_max, 300)
    a2 = np.linspace(0, grid_max, 300)
    A1, A2 = np.meshgrid(a1, a2)

    # --- Dual objective ---
    y1, y2 = y[0], y[1]
    K11, K22, K12 = K[0,0], K[1,1], K[0,1]
    Z = (A1 + A2) - 0.5*(A1**2*y1**2*K11 + A2**2*y2**2*K22 + 2*A1*A2*y1*y2*K12)

    # Constraint: α1 - α2 = 0  => α1 = α2 = t
    denom = y1**2*K11 + y2**2*K22 + 2*y1*y2*K12
    t_opt = max(0.0, 2.0/denom)

    fig, ax = plt.subplots(figsize=(6,6))
    cp = ax.contourf(A1, A2, -Z, levels=25, cmap="viridis")
    fig.colorbar(cp, ax=ax, label="Dual Objective")
    
    # Constraint line
    ax.plot(a1, a1, "r--", linewidth=1.5, label=r"$\alpha_1-\alpha_2 = 0$ (constraint)")
    
    # SMO trajectory
    if len(history) > 0:
        ax.plot(history[:,0], history[:,1], "w--", markersize=5, linewidth=1.2, label="SMO trajectory")
        ax.plot(history[0,0], history[0,1], "ws", markersize=7, markeredgecolor="grey")  # start
    
    # Optimum
    ax.plot(t_opt, t_opt, "r*", markersize=15, markeredgecolor="white", label=f"Optimum t*={t_opt:.2f}")
    
    ax.set_xlabel(r"$\alpha_1$")
    ax.set_ylabel(r"$\alpha_2$")
    ax.set_title("SMO Dual Objective")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    if hasattr(svm, 'alpha_history') and len(svm.alpha_history) > 2:
        history = np.array(svm.alpha_history)
        x_star = history[-1]
        ratios = []
        for i in range(len(history) - 1):
            num = np.linalg.norm(history[i + 1] - x_star)
            den = np.linalg.norm(history[i] - x_star)
            if den > 0:
                ratios.append(num / den)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(range(len(ratios)), ratios, 'o-', linewidth=1.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel(r'$|x_{k+1}-x^*| / |x_k-x^*|$')
        ax2.set_title('Q-Convergence (p=1)')
        fig2.tight_layout()
        fig2.savefig("q_convergence.png", dpi=150)
        plt.close(fig2)


def plot_obj_func(obj_history):
    plt.plot(obj_history)
    plt.title(r"$Q(\alpha)$ vs iterations")
    plt.xlabel("Iteration")
    plt.ylabel(r"$Q(\alpha)$")
    plt.savefig("obj_func_history.png")
    plt.show()

def plot_errors(obj_history):
    Q_star = obj_history[-1]
    q_errors = [abs(q - Q_star) for q in obj_history]
    plt.plot(np.log10(q_errors))
    plt.title("Log errors vs iterations")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\log_10 |Q(\alpha_k) - Q(\alpha*)|$")
    plt.savefig("log-errors.png")
    plt.show()


def plot_Q_conv(alpha_history, p = 1):
    alpha_star = alpha_history[-1]
    ratios = []
    for i in range(len(alpha_history) - 1):
            num = np.linalg.norm(alpha_history[i + 1] - alpha_star)
            den = np.linalg.norm(alpha_history[i] - alpha_star)**p
            if den > 0:
                ratios.append(num / den)

    plt.plot(ratios)
    plt.title(f"Q-linear convergence (p = {p})")
    plt.xlabel("Iteration")
    plt.ylabel(r'$|x_{k+1}-x^*| / |x_k-x^*|^p$')
    plt.savefig("q_conv.png")
    plt.show()

def plot_timecomplexity():
    pass

def plot_spacecomplexity():
    pass

def plot_moons_boundary(X, y, solver):
    plt.figure(figsize=(10, 8))

    bias = (solver.b_up + solver.b_low) / 2.0

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = np.array([solver.predict(point, bias) for point in grid])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], alpha=0.2, colors=['blue', 'red'])
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', edgecolors='k', s=30, label='Class 1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', edgecolors='k', s=30, label='Class -1')

    plt.title(f"SMO decision boundary (C={solver._C})")
    plt.legend()
    plt.savefig("moons_boundary.png")
    plt.show()