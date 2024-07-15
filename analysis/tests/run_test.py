import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_arguments():
    parser = argparse.ArgumentParser(description='Summarizes an experiment file')
    parser.add_argument('test_folder', type=str, help='folder containing the baseline and conditions file')
    parser.add_argument('-cr', '--conditions_root_folder', type=str, help='conditions root folder', required=True)
    parser.add_argument('-br', '--baseline_root_folder', type=str, help='baseline root folder', required=True)
    args = parser.parse_args()
    return args


running_processes = []


def start_process(command_line):
    result = subprocess.run(command_line, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


def plot_results(ax,
                 data,
                 x_axis,
                 y_axis,
                 fill_color,
                 edge_color,
                 groups,
                 size,
                 alpha):
    if x_axis in data and y_axis in data:
        ax.scatter(data[x_axis],
                   data[y_axis],
                   color=fill_color,
                   edgecolors=edge_color,

                   alpha=alpha[0],
                   s=size[0])
        if "groups" in data:
            for group_name, group in data["groups"].items():
                plot_results(ax=ax,
                             data=group,
                             x_axis=x_axis,
                             y_axis=y_axis,
                             edge_color=edge_color,
                             fill_color=fill_color,
                             groups=groups,
                             size=size[1:],
                             alpha=alpha[1:])


def plot_axes(title,
              baseline,
              conditions,
              x_axis,
              y_axis,
              output_file,
              baseline_groups=[],
              condition_groups=[]):
    size = 20
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title)
    ax = fig.add_subplot(111)

    plot_results(ax=ax,
                 data=baseline,
                 x_axis=x_axis,
                 y_axis=y_axis,
                 fill_color="black",
                 edge_color="black",
                 groups=baseline_groups,
                 size=[350, 50, 15, 15, 15, 15],
                 alpha=[.3, .5, 1, 1, 1, 1])

    conditions_colors = list(mcolors.TABLEAU_COLORS.keys())[:len(conditions["groups"])]

    for condition_number, (condition, condition_data) in enumerate(conditions["groups"].items()):
        plot_results(ax=ax,
                     data=condition_data,
                     x_axis=x_axis,
                     y_axis=y_axis,
                     fill_color=conditions_colors[condition_number],
                     edge_color=conditions_colors[condition_number],
                     groups=conditions_groups,
                     size=[350, 50, 15, 15, 15, 15],
                     alpha=[.3, .5, 1, 1, 1, 1])

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    labels = ['Mice'] + list(conditions["groups"].keys())
    colors = ['black'] + conditions_colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10) for label, color in zip(labels, colors)]
    plt.legend(handles=handles, title='Colors')

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

    import os

    if not os.path.isdir(args.test_folder):
        print(f"Cannot find folder {args.test_folder}")
        exit(1)

    if not os.path.isdir(args.baseline_root_folder):
        print(f"Cannot find baseline root folder {args.baseline_root_folder}")
        exit(1)

    if not os.path.isdir(args.conditions_root_folder):
        print(f"Cannot find conditions root folder {args.conditions_root_folder}")
        exit(1)

    test_file = os.path.join(args.test_folder, "config.json")
    if not os.path.isfile(test_file):
        print(f"Error: file {test_file} does not exist")
        exit(1)

    baseline_file = os.path.join(args.test_folder, "baseline.json")
    if not os.path.isfile(baseline_file):
        print(f"Error: file {baseline_file} does not exist")
        exit(1)

    conditions_file = os.path.join(args.test_folder, "conditions.json")

    if not os.path.isfile(conditions_file):
        print(f"Error: file {conditions_file} does not exist")
        exit(1)

    figures_folder = os.path.join(args.test_folder, "figures")
    os.makedirs(figures_folder, exist_ok=True)

    baseline_clusters_folder = os.path.join(args.test_folder, "figures", "baseline_clusters")
    os.makedirs(baseline_clusters_folder, exist_ok=True)

    conditions_clusters_folder = os.path.join(args.test_folder, "figures", "conditions_clusters")
    os.makedirs(conditions_clusters_folder, exist_ok=True)

    baseline_summary_file = os.path.join(args.test_folder, "baseline_summary.json")
    if not os.path.isfile(baseline_summary_file):
        print("Creating baseline summary file")
        import subprocess
        import sys
        import threading
        script = 'process_file.py'
        command = [sys.executable, script, baseline_file, "-o", baseline_summary_file, "-r", args.baseline_root_folder, "-cf", baseline_clusters_folder]
        thread = threading.Thread(target=start_process, args=(command,))
        thread.start()
        running_processes.append(thread)

    conditions_summary_file = os.path.join(args.test_folder, "conditions_summary.json")
    if not os.path.isfile(conditions_summary_file):
        print("Creating conditions summary file")
        import subprocess
        import sys
        import threading
        script = 'process_file.py'
        command = [sys.executable, script, conditions_file, "-o", conditions_summary_file, "-r", args.conditions_root_folder, "-cf", conditions_clusters_folder]
        thread = threading.Thread(target=start_process, args=(command,))
        thread.start()
        running_processes.append(thread)

    if running_processes:
        print("Waiting for processes to finish...")
        for process in running_processes:
            process.join()

    conditions_normalized_summary_file = os.path.join(args.test_folder, "conditions_normalized_summary.json")
    if not os.path.isfile(conditions_normalized_summary_file):
        print("Creating conditions normalized summary file")
        import subprocess
        import sys
        import threading
        script = 'normalize_conditions.py'
        command = [sys.executable, script, args.test_folder]
        start_process(command)

    with open(test_file, "r") as f:
        test_config = json.load(f)

    with open(baseline_summary_file, "r") as f:
        baseline_summary = json.load(f)

    with open(conditions_summary_file, "r") as f:
        conditions_summary = json.load(f)

    baseline_groups = baseline_summary["groups"]

    conditions_groups = conditions_summary["groups"]


    print("Creating figures")
    for x_axis in baseline_summary["data"]:
        for y_axis in baseline_summary["data"]:
            if x_axis == y_axis or "groups" in [x_axis, y_axis]:
                continue
            output_file = os.path.join(figures_folder, f"{x_axis}_{y_axis}.png")
            plot_axes(title=test_config["name"],
                      baseline=baseline_summary["data"],
                      conditions=conditions_summary["data"],
                      x_axis=x_axis,
                      y_axis=y_axis,
                      output_file=output_file,
                      baseline_groups=baseline_groups,
                      condition_groups=conditions_groups)
