"""
Experiment: Hilbert Transform on Manhattan Graph
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Get the directory of this script for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(EXPERIMENT_DIR, "data")
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
        experiment_name="manhattan_graph",
        verbose=verbose,
        log_file=None
        if not save_results
        else os.path.join(RESULTS_DIR, "manhattan_graph.log"),
        results_dir=RESULTS_DIR if save_results else None,
    )
    # Suppress noisy libraries
    if not verbose:
        # Suppress Python warnings
        set_library_log_levels("ERROR")

    logger.info("=" * 60)
    logger.info("Experiment: Hilbert Transform on Manhattan Graph")
    logger.info("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config", "config.json"))

    logger.info(f"\nConfiguration: {config}")

    experiments = Experiments(config, verbose=verbose, logger=logger)
    # Plot 1: Create flower graph and plot
    fig1, fig2, fig3 = experiments.run_experiment1()

    # Plot 2: Plot flower graph with original signal
    fig4, fig5 = experiments.run_experiment2()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig1.savefig(
            os.path.join(RESULTS_DIR, "torus_graph.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(RESULTS_DIR, "torus_graph_initial.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig3.savefig(
            os.path.join(RESULTS_DIR, "torus_graph_hilbert_transform.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig4.savefig(
            os.path.join(RESULTS_DIR, "manhattan_graph_diagonalized.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig5.savefig(
            os.path.join(RESULTS_DIR, "manhattan_graph_hilbert_transform.png"),
            dpi=300,
            bbox_inches="tight",
        )

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
        from flowgsp.graphs import Graph
        from flowgsp.filters import HilbertFilter
        from flowgsp.graphs.basic_graphs import create_directed_torus

        Nr, Nc = self.config["Nr"], self.config["Nc"]
        G, _ = create_directed_torus(Nr, Nc, directed=True)
        tor = nx.to_numpy_array(G)
        N = tor.shape[0]
        pos = {k: (-k % Nr, k // Nr) for k in range(0, Nr * Nc)}

        theta = np.radians(160)  # rotate 180 degrees
        rot_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        coords = np.array(list(pos.values()))
        rotpos = (rot_matrix @ coords.T).T
        rotpos = {k: rotpos[k] for k in range(len(rotpos))}

        nodesize = 1
        # Plot 1: Plot the directed torus graph
        fig1, ax = plt.subplots(1, figsize=(3, 3))

        Gund = nx.from_numpy_array(tor.astype(int), create_using=nx.Graph())
        nx.draw_networkx_nodes(
            G,
            rotpos,
            alpha=0.2,
            node_size=nodesize,
            node_color=np.ones(N),
            cmap="winter",
            ax=ax,
        )
        nx.draw_networkx_edges(
            Gund,
            rotpos,
            node_size=nodesize,
            alpha=1,
            ax=ax,
            edge_color="black",
            width=1,
        )

        x1, x2, y1, y2 = -12, -9, -6, -2
        axins = ax.inset_axes(
            [0.75, 0, 0.3, 0.3],
            xlim=(x1, x2),
            ylim=(y1, y2),
            xticklabels=[],
            yticklabels=[],
        )

        Gund = nx.from_numpy_array(tor.astype(int), create_using=nx.DiGraph())

        nx.draw_networkx_nodes(
            G,
            rotpos,
            alpha=0.2,
            node_size=nodesize,
            node_color=np.ones(N),
            cmap="winter",
            ax=axins,
        )
        nx.draw_networkx_edges(
            Gund,
            rotpos,
            node_size=nodesize,
            alpha=1,
            ax=axins,
            edge_color="black",
            width=1,
        )

        ax.indicate_inset_zoom(axins, edgecolor="blue", alpha=1, linewidth=1)

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)
        plt.axis("off")

        if not self.verbose:
            plt.close()
        plt.show()

        graph = Graph(adj_matrix=tor)
        graph.set_operator("adjacency")
        hfilter = HilbertFilter(graph)
        # Plot 2: Plot the original signal and its Hilbert transform
        graphsig = np.sin(coords[:, 1] / coords[:, 1].max() * 2 * np.pi)
        hx = hfilter.hilbert_transform(graphsig).real

        fig2, ax = plt.subplots(1, figsize=(3, 3))
        nx.draw_networkx_nodes(
            G, rotpos, alpha=0.8, node_size=25, node_color=graphsig, cmap="bwr", ax=ax
        )
        ax.axis("off")
        if not self.verbose:
            plt.close()
        plt.show()

        fig3, ax = plt.subplots(1, figsize=(3, 3))
        nx.draw_networkx_nodes(
            G, rotpos, alpha=0.8, node_size=25, node_color=hx, cmap="bwr", ax=ax
        )
        ax.axis("off")
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2, fig3

    def run_experiment2(self):
        import osmnx as ox
        from shapely import Polygon

        from flowgsp.utils import load, save, zscore, smooth_1d, spatial_smooth
        from flowgsp.graphs import Graph
        from flowgsp.filters import HilbertFilter
        from flowgsp.operators import destroy_jordan_blocks, destroy_zero_eigenvals

        path_graph = "./data/manhattan_graph_data/"
        if os.path.exists(os.path.join(path_graph, "manhattan_graph.pkl")):
            graph_info = load(os.path.join(path_graph, "manhattan_graph.pkl"))
            adj = graph_info["adj"]
            nodes_coords = graph_info["nodes_coords"]
            graphnodes = graph_info["graphnodes"]
            graph = graph_info["graph"]
            pos = graph_info["pos"]
            G = graph_info["G"]

        else:
            coords = (
                (-73.958759, 40.727877),
                (-74.013777, 40.74),
                (-74.005, 40.772677),
                (-73.958759, 40.765),
            )
            polygon = Polygon(coords)
            place = ox.graph_from_polygon(polygon, network_type="drive")
            graph = ox.project_graph(place)
            graphnodes = np.array(graph)

            nodes_coords = np.array(
                [
                    (
                        graph.nodes[graphnodes[k]]["lon"],
                        graph.nodes[graphnodes[k]]["lat"],
                    )
                    for k in range(len(graphnodes))
                ]
            )

            adj = nx.convert_matrix.to_numpy_array(graph)
            G = nx.from_numpy_array(adj, create_using=nx.MultiDiGraph())
            pos = {k: nodes_coords[k] for k in range(len(G.nodes))}

            graph_info = {
                "adj": adj,
                "nodes_coords": nodes_coords,
                "graphnodes": graphnodes,
                "graph": graph,
                "pos": pos,
                "G": G,
            }
            save(os.path.join(path_graph, "manhattan_graph.pkl"), graph_info)

        # Diagonalization without preferred selected nodes
        if os.path.exists(os.path.join(path_graph, "midmanhattan_graph_nopref.pkl")):
            newA = load(os.path.join(path_graph, "midmanhattan_graph_nopref.pkl"))
        else:
            newA = destroy_jordan_blocks(adj.astype(float))
            newA = destroy_zero_eigenvals(newA.astype(float), eps=5e-3, verbose=False)
            save(os.path.join(path_graph, "midmanhattan_graph_nopref.pkl"), newA)

        nodesize = 1
        fig1, ax = plt.subplots(1, figsize=(3, 3))

        Gund = nx.from_numpy_array(adj.astype(int), create_using=nx.Graph())
        Gund_added = nx.from_numpy_array(
            np.abs(newA - adj), create_using=nx.MultiGraph()
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            alpha=0.2,
            node_size=nodesize,
            node_color=np.ones(len(adj)),
            cmap="winter",
            ax=ax,
        )
        nx.draw_networkx_edges(
            Gund, pos, node_size=nodesize, alpha=1, ax=ax, edge_color="black", width=1
        )
        nx.draw_networkx_edges(
            Gund_added,
            pos,
            node_size=nodesize,
            alpha=0.7,
            ax=ax,
            edge_color="forestgreen",
            width=0.4,
        )

        x1, x2, y1, y2 = (
            -73.981,
            -73.975,
            40.757,
            40.762,
        )  # subregion of the original image
        axins = ax.inset_axes(
            [0.75, 0, 0.3, 0.3],
            xlim=(x1, x2),
            ylim=(y1, y2),
            xticklabels=[],
            yticklabels=[],
        )

        Gund = nx.from_numpy_array(adj.astype(int), create_using=nx.DiGraph())
        Gund_added = nx.from_numpy_array(np.abs(newA - adj), create_using=nx.DiGraph())

        nx.draw_networkx_nodes(
            G,
            pos,
            alpha=0.2,
            node_size=nodesize,
            node_color=np.ones(len(adj)),
            cmap="winter",
            ax=axins,
        )
        nx.draw_networkx_edges(
            Gund,
            pos,
            node_size=nodesize,
            alpha=1,
            ax=axins,
            edge_color="black",
            width=1,
        )
        nx.draw_networkx_edges(
            Gund_added,
            pos,
            node_size=nodesize,
            alpha=0.5,
            ax=axins,
            edge_color="forestgreen",
            width=0.4,
        )

        ax.indicate_inset_zoom(axins, edgecolor="blue", alpha=1, linewidth=1)

        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=True)

        ax.axis("off")
        if not self.verbose:
            plt.close()
        plt.show()

        # Plot 2: Plot the original signal and its Hilbert transform
        # Create graph signal based on the space coordinates
        tmp = nodes_coords[np.argsort(nodes_coords[:, 1])]
        tmpselect = [-10, -16]

        coefficients = np.polyfit(tmp[tmpselect][:, 0], tmp[tmpselect][:, 1], 1)
        polynomial_vertical = np.poly1d(coefficients)

        frequency_vertical = 200
        manifold_vertical = np.sin(np.sort(nodes_coords[:, 0]) * frequency_vertical)
        graph_signal = np.zeros_like(manifold_vertical)

        # we order the coordinates by a certain direction
        direction_vector = np.array([1, polynomial_vertical[1] * 2])

        ranking_vector = nodes_coords @ direction_vector
        xycoords_rank = np.argsort(ranking_vector)
        frequency_vertdiagonal = 1
        manifold_vertdiagonal = np.sin(
            np.sort(zscore(ranking_vector)) * frequency_vertdiagonal
        )

        graph_signal[xycoords_rank] += manifold_vertdiagonal

        graph = Graph(adj_matrix=newA)
        graph.set_operator("adjacency")
        hfilter = HilbertFilter(graph)
        hx = hfilter.hilbert_transform(graph_signal).real
        smoothened_ht = np.zeros_like(graph_signal)
        smoothened_ht[xycoords_rank] = smooth_1d(hx[xycoords_rank], 1)
        ssize = 2e-3
        display_ht = spatial_smooth(smoothened_ht, nodes_coords, ssize)
        ssize = 2e-3
        display_ht = spatial_smooth(smoothened_ht, nodes_coords, ssize)

        fig2, ax = plt.subplots(1, 2, figsize=(8, 2.5))

        nx.draw_networkx_nodes(
            G,
            pos,
            alpha=0.8,
            node_size=25,
            node_color=graph_signal,
            cmap="bwr",
            ax=ax[0],
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            alpha=0.8,
            node_size=25,
            node_color=display_ht,
            cmap="bwr",
            ax=ax[1],
            vmin=np.percentile(display_ht, 10),
            vmax=np.percentile(display_ht, 90),
        )

        ax[0].set_axis_off()
        ax[1].set_axis_off()

        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2


if __name__ == "__main__":
    run()
