import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analysis baseline creation tool: create a baseline file for a test')
    parser.add_argument('-o', '--output_file', type=str, help='output file', default="baseline.json")
    parser.add_argument('-w', '--world', type=str, help='world identifier', required=True)
    parser.add_argument('-p', '--phases', type=str, help='phases included in the baseline', required=False)
    parser.add_argument('-s', '--subjects', type=str, help='subjects included in the baseline', required=False)
    parser.add_argument('-g', '--group_by', type=str, help='field to group by results (default: subject)', required=False, default="subject")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

    import os
    if not os.path.isdir(args.world):
        print(f"Error: {args.world} is not a directory")
        exit(1)

    experiment_list_file = f"{args.world}/experiment_list.json"

    if not os.path.isfile(experiment_list_file):
        print(f"Error: {experiment_list_file} is not a file")
        exit(1)

    with open(experiment_list_file) as f:
        experiment_list = json.load(f)

    if args.phases:
        phases_list = args.phases.split(',')
        phases_filter = lambda x: x["phase"] in phases_list
    else:
        phases_filter = lambda x: True

    if args.subjects:
        subjects_list = args.subjects.split(',')
        subjects_filter = lambda x: x["subject"] in subjects_list
    else:
        subjects_filter = lambda x: True

    with open(experiment_list_file) as f:
        experiment_list = json.load(f)

    filter = lambda x: phases_filter(x) and subjects_filter(x)
    baseline_files = []
    group_fields = args.group_by.split(',')
    for experiment_file in experiment_list:
        if filter(experiment_file):
            new_experiment_file = {"file": experiment_file["file"]}
            for i, key in enumerate(group_fields):
                new_experiment_file[f"group_{i}"] = experiment_file[key]
            baseline_files.append(new_experiment_file)

    base_line = {"groups": group_fields, "files": baseline_files}

    with open(args.output_file, "w") as f:
        json.dump(base_line, f)
