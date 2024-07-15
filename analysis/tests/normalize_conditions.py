import json
import argparse
import cellworld as cw


def parse_arguments():
    parser = argparse.ArgumentParser(description='Normalize conditions summary using the baseline')
    parser.add_argument('test_folder', type=str, help='folder containing the summary files')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

    import os

    baseline_summary_file = os.path.join(args.test_folder, "baseline_summary.json")
    if not os.path.isfile(baseline_summary_file):
        print(f"Error: file {baseline_summary_file} does not exist")
        exit(1)

    conditions_summary_file = os.path.join(args.test_folder, "conditions_summary.json")
    if not os.path.isfile(conditions_summary_file):
        print(f"Error: file {conditions_summary_file} does not exist")
        exit(1)

    with open(conditions_summary_file) as f:
        conditions_summary = json.load(f)

    with open(baseline_summary_file) as f:
        baseline_summary = json.load(f)

    def normalize_condition(data, baseline, normalized_condition={}):
        for value in data:
            if value != "groups":
                print(value)
                normalized_condition[value] = data[value] / baseline[value]
        if "groups" in data:
            normalized_condition["groups"] = {}
            for group_name, group in data["groups"].items():
                normalized_condition["groups"][group_name] = {}
                normalize_condition(data["groups"][group_name], baseline, normalized_condition["groups"][group_name])
        return normalized_condition


    normalized_conditions_data = normalize_condition(conditions_summary["data"], baseline_summary["data"])
    normalized_conditions = {"groups": conditions_summary["groups"], "data": normalized_conditions_data}

    normalized_conditions_summary_file = os.path.join(args.test_folder, "conditions_normalized_summary.json")
    with open(normalized_conditions_summary_file, "w") as f:
        json.dump(normalized_conditions, f)

