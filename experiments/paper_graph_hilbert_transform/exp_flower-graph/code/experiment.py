"""
Experiment: Diagonalization of Flower Graph
"""

import os
import json
from datetime import datetime

# Get the directory of this script for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


def run(save_results: bool = True, verbose: bool = True) -> dict:
    """
    Run the basic experiment.

    Parameters
    ----------
    save_results : bool
        Whether to save results to the results directory.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    dict
        Dictionary containing experiment results.
    """
    from flowgsp.utils import (
        load_json,
        configure_experiment_logging,
        set_library_log_levels,
    )

    # Set up logging
    logger = configure_experiment_logging(
        experiment_name="flower_graph",
        verbose=verbose,
        log_file=None
        if not save_results
        else os.path.join(RESULTS_DIR, "flower_graph.log"),
        results_dir=RESULTS_DIR if save_results else None,
    )

    # Suppress noisy libraries
    if not verbose:
        set_library_log_levels("ERROR")

    logger.info("=" * 60)
    logger.info("Experiment: Diagonalization of Flower Graph")
    logger.info("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config", "config.json"))

    logger.info(f"\nConfiguration: {config}")

    experiments = Experiments(config, verbose=verbose, logger=logger)
    # Plot 1: Create flower graph and plot
    fig = experiments.run_experiment1()

    # Plot 2: Plot flower graph with original signal
    (
        fig2,
        fig3,
        fig4,
        (graphsig, amp, ang, Nr, Nc, frequencies),
    ) = experiments.run_experiment2()

    # Plot 3: Plot flower graph and Hilbert transform
    fig5, fig6, fig7 = experiments.run_experiment3(
        graphsig=graphsig, amp=amp, ang=ang, Nr=Nr, Nc=Nc, frequencies=frequencies
    )

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig.savefig(
            os.path.join(RESULTS_DIR, "schematic_flower_graph.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(RESULTS_DIR, "rosace-original-signal.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig3.savefig(
            os.path.join(RESULTS_DIR, "rosace-hilbert-transform.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig4.savefig(
            os.path.join(RESULTS_DIR, "rosace-hilbert-transform-jordan.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig5.savefig(os.path.join(RESULTS_DIR, "central-cycle-instants.png"), dpi=300)
        fig6.savefig(
            os.path.join(RESULTS_DIR, "flower-instant-frequencies.png"), dpi=300
        )
        fig7.savefig(os.path.join(RESULTS_DIR, "flower-instant-amplitude.png"), dpi=300)

        results_file = os.path.join(RESULTS_DIR, "experiment_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {RESULTS_DIR}")

    logger.info("\n" + "=" * 60)
    logger.info("Experiment completed successfully!")
    logger.info("=" * 60)


class Experiments:
    def __init__(self, config: dict, verbose: bool = True, logger=None):
        self.config = config
        self.verbose = verbose
        self.logger = logger

    def run_experiment1(self):
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        from flowgsp.graphs.basic_graphs import create_flower_graph
        from flowgsp.operators.jordan_destroy import destroy_jordan_blocks

        Nr, Nc = 4, 4
        G, _ = create_flower_graph(Nr, Nc, diagonalizable=False)
        A = nx.to_numpy_array(G)
        pos = {
            3: (-4, 0),
            2: (-3, 0),
            1: (-2, 0),
            0: (-1, 0),
            4: (0, 1),
            5: (0, 2),
            6: (0, 3),
            7: (0, 4),
            11: (4, 0),
            10: (3, 0),
            9: (2, 0),
            8: (1, 0),
            12: (0, -1),
            13: (0, -2),
            14: (0, -3),
            15: (0, -4),
        }
        newA = destroy_jordan_blocks(A, prefer_nodes=[0 + Nr * k for k in range(Nc)])
        Gnew = nx.from_numpy_array(np.abs(newA - A), create_using=nx.DiGraph)

        fig, ax = plt.subplots(1, figsize=(3, 3))

        nx.draw_networkx_nodes(
            G,
            pos,
            alpha=0.8,
            node_size=50,
            node_color="gray",
            cmap="bwr",
            ax=ax,
            edgecolors="black",
        )
        nx.draw_networkx_edges(
            G, pos, alpha=1, ax=ax, edge_color="black", width=1, arrowsize=7
        )
        nx.draw_networkx_edges(
            Gnew,
            pos,
            alpha=1,
            ax=ax,
            edge_color="forestgreen",
            width=1,
            connectionstyle="arc3,rad=0.7",
            label="Added Edges",
        )

        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=True)
        ax.axis("off")

        if not self.verbose:
            plt.close()
        plt.show()

        return fig

    def run_experiment2(self):
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        from flowgsp.filters import HilbertFilter
        from flowgsp.graphs import Graph
        from flowgsp.graphs.basic_graphs import create_flower_graph
        from flowgsp.operators.jordan_destroy import destroy_jordan_blocks

        Nr, Nc = self.config["Nr"], self.config["Nc"]
        G, pos = create_flower_graph(Nr, Nc, diagonalizable=False)
        A = nx.to_numpy_array(G)
        newA = destroy_jordan_blocks(A, prefer_nodes=[0 + Nr * k for k in range(Nc)])
        Gnew = nx.from_numpy_array(np.abs(newA - A), create_using=nx.MultiDiGraph())
        graph = Graph(adj_matrix=newA)
        graph.set_operator("adjacency")

        circle = np.linspace(0, 2 * np.pi, Nr)

        frequencies = [(k / 2) for k in range(Nc)]
        phase = [(Nc + 1 - 0.2) / 2 * np.pi / Nc * k for k in range(Nc)]
        graphsig = np.concatenate(
            [
                (2 * v + 1) * np.sin(freq * circle + phase[v])
                for v, freq in enumerate(frequencies)
            ]
        )

        hfilter = HilbertFilter(graph)
        xa = hfilter.analytical_signal(graphsig)
        hx = hfilter.hilbert_transform(graphsig)
        amp = np.abs(xa)
        ang = np.arctan2(xa.imag, xa.real)

        # Initial signal plot
        nd_values = graphsig + 1e-2
        scolor = ["red", "blue"]
        scale = 4
        nd_color = [scolor[0] if nd > 0 else scolor[1] for nd in nd_values]
        node_values = scale * np.abs(nd_values) ** (6 / 8)

        fig1, ax = plt.subplots(1, figsize=(3, 3))

        nx.draw_networkx_nodes(
            G,
            pos,
            alpha=1,
            node_size=node_values,
            node_color=nd_color,
            ax=ax,
            edgecolors="black",
            linewidths=0.5,
        )
        nx.draw_networkx_edges(
            Gnew,
            pos,
            alpha=1,
            node_size=node_values,
            ax=ax,
            edge_color="forestgreen",
            width=1.2,
            connectionstyle="arc3,rad=-0.15",
            label="Added Edges",
        )

        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=True)
        ax.axis("off")
        if not self.verbose:
            plt.close()
        plt.show()

        # Hilbert transform plot
        nd_values = hx - 5e-1
        nd_values[np.arange(0, 1 * Nr)] = 1e-2  # In order to make the nodes visible
        nd_color = [scolor[0] if nd > 0 else scolor[1] for nd in nd_values]
        node_values = scale * np.abs(nd_values) ** (5.5 / 8)
        fig2, ax = plt.subplots(1, figsize=(3, 3))

        nx.draw_networkx_nodes(
            G,
            pos,
            alpha=1,
            node_size=node_values,
            node_color=nd_color,
            ax=ax,
            edgecolors="black",
            linewidths=0.5,
        )
        nx.draw_networkx_edges(
            Gnew,
            pos,
            alpha=1,
            node_size=node_values,
            ax=ax,
            edge_color="forestgreen",
            width=1.2,
            connectionstyle="arc3,rad=-0.15",
            label="Added Edges",
        )

        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=True)
        ax.axis("off")
        if not self.verbose:
            plt.close()
        plt.show()

        # Hilbert transform Jordan plot
        if self.config["Nr"] <= 10:
            graph_jord = Graph(adj_matrix=A)
            graph_jord.set_operator("adjacency", decomposition="jord")
            hfilter_jord = HilbertFilter(graph_jord)
            hx_jord = hfilter_jord.hilbert_transform(graphsig)
            nd_values = hx_jord - 5e-1
            nd_values[np.arange(0, 1 * Nr)] = 1e-2  # In order to make the nodes visible
            nd_color = [scolor[0] if nd > 0 else scolor[1] for nd in nd_values]
            node_values = scale * np.abs(nd_values) ** (5.5 / 8)

        fig3, ax = plt.subplots(1, figsize=(3, 3))

        nx.draw_networkx_nodes(
            G,
            pos,
            alpha=1,
            node_size=node_values,
            node_color=nd_color,
            ax=ax,
            edgecolors="black",
            linewidths=0.5,
        )

        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=True)
        ax.axis("off")
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2, fig3, (graphsig, amp, ang, Nr, Nc, frequencies)

    def run_experiment3(self, ang, amp, graphsig, Nr, Nc, frequencies):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

        # freqs
        unwraped = [np.zeros(Nr)] + [
            np.unwrap(ang[np.arange(0 + k * Nr, Nr + k * Nr)]) for k in range(1, Nc)
        ]
        freqs_est = [
            np.concatenate(
                [np.abs(np.diff(unwraped[k])), [np.abs(np.diff(unwraped[k])).mean()]]
            )
            for k in range(Nc)
        ]
        # amps
        amps_cycle = np.array(
            [amp[np.arange(0 + k * Nr, Nr + k * Nr)] for k in range(Nc)]
        )
        hfont = {"fontname": "Helvetica"}
        amp_center = amp[np.arange(0, Nr * Nc, Nr)]

        # Plot frequencies
        fig1, ax = plt.subplots(1, figsize=(4, 3))

        freqs_est = np.array(freqs_est)
        ax.errorbar(
            np.arange(1, Nc + 1),
            freqs_est.mean(axis=1),
            yerr=freqs_est.std(axis=1) / 2,
            linestyle="",
            label="fan avg.+std",
            marker="o",
            markersize=4,
            linewidth=1,
        )
        ax.plot(
            np.arange(1, Nc + 1),
            np.array(frequencies) / Nr * 2 * np.pi,
            linestyle="solid",
            label="true frequencies",
            linewidth=1,
        )
        ax.set_ylabel(
            r"$\omega({\bf x}_k)$ for ${\bf x}_k\in{\mathcal{C}}_m$ ", size=12
        )
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_xticks(np.arange(0, Nc + 1, 5))
        ax.set_xticklabels(np.arange(0, Nc + 1, 5), **hfont)
        ax.set_xlabel(r"$\mathcal{C}_m$", size=12)

        ax.set_yticks(np.arange(0, 3))
        ax.set_yticklabels(np.arange(0, 3), **hfont)
        ax.legend(prop={"size": 10, "family": "Helvetica"})
        if not self.verbose:
            plt.close()
        plt.show()

        # Plot Amplitudes
        fig2, ax = plt.subplots(1, figsize=(4, 3))
        ax.errorbar(
            np.arange(1, Nc + 1),
            amps_cycle.mean(axis=1),
            yerr=amps_cycle.std(axis=1) / 2,
            linestyle="",
            label="fan avg.+std",
            marker="o",
            markersize=4,
            linewidth=1,
        )
        ax.plot(
            np.arange(1, Nc + 1),
            [(2 * v + 1) for v in range(Nc)],
            linestyle="solid",
            label="true amplitudes",
            linewidth=1,
        )
        ax.set_ylabel(
            r"$\mathcal{A}({\bf x}_k)$ for ${\bf x}_k\in{\mathcal{C}}_m$ ", size=12
        )

        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_xticks(np.arange(0, Nc + 1, 5))
        ax.set_xticklabels(np.arange(0, Nc + 1, 5), **hfont)
        ax.set_xlabel(r"$\mathcal{C}_m$", size=12)

        ax.set_yticks(np.arange(0, 42, 20))
        ax.set_yticklabels(np.arange(0, 41, 20), **hfont)
        ax.legend(prop={"size": 10, "family": "Helvetica"})
        if not self.verbose:
            plt.close()
        plt.show()

        # Plot instant amplitude and phase
        fig3, ax = plt.subplots(1, figsize=(4, 2))

        ax2 = ax.twinx()
        ax.plot(graphsig[np.arange(0, Nr * Nc, Nr)], label="signal", linewidth=1.3)
        ax.plot(amp_center, label="instant amplitude", color="b", linewidth=1.3)
        ax2.plot(
            ang[np.arange(0, Nr * Nc, Nr)],
            label="phase",
            color="r",
            linestyle="--",
            linewidth=1.3,
        )

        ybox1 = TextArea(
            r"${\bf x}[k], $",
            textprops=dict(color="k", size=12, rotation=90, ha="left", va="bottom"),
        )
        ybox2 = TextArea(
            r"$\mathcal{A}{{\bf (x})}[k]$",
            textprops=dict(color="b", size=12, rotation=90, ha="left", va="bottom"),
        )

        ybox = VPacker(children=[ybox2, ybox1], align="bottom", pad=0, sep=5)

        anchored_ybox = AnchoredOffsetbox(
            loc=6,
            child=ybox,
            pad=0.0,
            frameon=False,
            bbox_to_anchor=(-0.12, 0.43),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )

        ax.add_artist(anchored_ybox)

        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_xticks(np.arange(0, Nc + 1, 5))
        ax.set_yticklabels([r"$0$", r"$5$", r"$10$", r"$15$", r"$20$"], **hfont)

        ax.set_yticks(np.arange(-40, 41, 20))
        ax.set_yticklabels([r"$-40$", r"$-20$", r"$0$", r"$20$", r"$40$"], **hfont)

        ax2.set_yticks(np.arange(-np.pi, 7, np.pi))
        ax2.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"], **hfont)

        ax2.tick_params(axis="both", which="major", labelsize=10)
        ax.legend(prop={"size": 8, "family": "Helvetica"}, loc=(0.05, 0.75))
        ax2.legend(prop={"size": 8, "family": "Helvetica"}, loc=(0.05, 0.65))
        ax.set_xlabel(r"$k$", size=12, **hfont)
        ax2.set_ylabel(
            r"$\varphi({\bf x})[k]$", size=12, rotation=270, labelpad=20, color="red"
        )
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2, fig3


if __name__ == "__main__":
    run()
