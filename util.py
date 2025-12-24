import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd


def plot_barcode(persistence, file_name=""):
    diagrams_roll = {}
    filenames = glob.glob("Results/Roll/" + file_name + "_H2_roll_*")
    for i, fname in enumerate(filenames):
        f = np.load(fname, allow_pickle=True)
        diagrams_roll[i] = list(f["diagrams"])
        f.close()

    cs = np.repeat([[0, 0.55, 0.2]], 3).reshape(3, 3).T
    alpha = 1
    inf_delta = 0.1
    legend = True
    colormap = cs
    maxdim = len(persistence) - 1
    dims = np.arange(maxdim + 1)
    num_rolls = len(diagrams_roll)

    if num_rolls > 0:
        diagrams_all = np.copy(diagrams_roll[0])
        for i in np.arange(1, num_rolls):
            for d in dims:
                diagrams_all[d] = np.concatenate(
                    (diagrams_all[d], diagrams_roll[i][d]), 0
                )
        infs = np.isinf(diagrams_all[0])
        diagrams_all[0][infs] = 0
        diagrams_all[0][infs] = np.max(diagrams_all[0])
        infs = np.isinf(diagrams_all[0])
        diagrams_all[0][infs] = 0
        diagrams_all[0][infs] = np.max(diagrams_all[0])

    min_birth, max_death = 0, 0
    for dim in dims:
        persistence_dim = persistence[dim][~np.isinf(persistence[dim][:, 1]), :]
        min_birth = min(min_birth, np.min(persistence_dim))
        max_death = max(max_death, np.max(persistence_dim))
    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta
    axis_start = min_birth - delta
    plotind = (dims[-1] + 1) * 100 + 10 + 1
    fig = plt.figure()
    gs = grd.GridSpec(len(dims), 1)

    indsall = 0
    # Only show either the per-subplot or the global left-side label, not both.
    labels = ["$H_0$", "$H_1$", "$H_2$"]
    axes_list = []
    for dit, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes_list.append(axes)
        axes.axis("off")
        d = np.copy(persistence[dim])
        d[np.isinf(d[:, 1]), 1] = infinity
        dlife = d[:, 1] - d[:, 0]
        dinds = np.argsort(dlife)[-30:]
        if len(dinds) == 0:
            continue
        dl1, dl2 = dlife[dinds[-2:]]
        if dim > 0:
            dinds = dinds[np.flip(np.argsort(d[dinds, 0]))]
        
        y_positions = 0.5 + np.arange(len(dinds))
        axes.barh(
            y_positions,
            dlife[dinds],
            height=0.8,
            left=d[dinds, 0],
            alpha=alpha,
            color=colormap[dim],
            linewidth=0,
        )
        indsall = len(dinds)
        if num_rolls > 0:
            bins = 50
            cs = np.flip([[0.4, 0.4, 0.4], [0.6, 0.6, 0.6], [0.8, 0.8, 0.8]])
            cs = np.repeat([[1, 0.55, 0.1]], 3).reshape(3, 3).T
            cc = 0
            lives1_all = diagrams_all[dim][:, 1] - diagrams_all[dim][:, 0]
            x1 = np.linspace(
                diagrams_all[dim][:, 0].min() - 1e-5,
                diagrams_all[dim][:, 0].max() + 1e-5,
                bins - 2,
            )

            dx1 = x1[1] - x1[0]
            x1 = np.concatenate(([x1[0] - dx1], x1, [x1[-1] + dx1]))
            dx = x1[:-1] + dx1 / 2
            ytemp = np.zeros((bins - 1))
            binned_birth = np.digitize(diagrams_all[dim][:, 0], x1) - 1
            x1 = d[dinds, 0]
            ytemp = x1 + np.max(lives1_all)
            axes.fill_betweenx(
                y_positions,
                x1,
                ytemp,
                color=cs[(dim)],
                zorder=-2,
                alpha=0.3,
            )

        # Draw manual axis lines
        axes.plot([0, 0], [0, indsall + 0.5], c="k", linestyle="-", lw=1)
        axes.plot([0, infinity], [0, 0], c="k", linestyle="-", lw=1)
        
        # Set axis limits
        axes.set_xlim([0, infinity])
        axes.set_ylim([0, indsall + 0.5])
        
        # Remove per-subplot H_0/H_1/H_2 label to avoid redundant indication.
        # axes.text(-infinity * 0.05, indsall / 2, labels[dim],
        #           ha='right', va='center', fontsize=12, fontweight='bold', color='black')
    
    # Remove empty space at the bottom, and add overall axis label on the left
    plt.tight_layout()

    # Keep only the single vertical annotation for H0, H1, H2
    for dim, axes in enumerate(axes_list):
        # Place the annotation outside the main plot area
        fig.text(-0.01, 1 - (dim + 0.5) / len(axes_list), f"H{dim}", va='center', 
                 ha='right', fontsize=14, fontweight='bold', color='black', rotation=90)
