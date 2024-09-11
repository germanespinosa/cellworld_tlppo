import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analysis conditions creation tool: create a conditions file for a test')
    parser.add_argument('-o', '--output_file', type=str, help='output file', default="conditions.json")
    parser.add_argument('-r', '--result_folder', type=str, help='result folder', required=True)
    parser.add_argument('-c', '--conditions', type=str, help='conditions included in the test (default: 1,2,3,4,5)', required=False, default="1,2,3,4,5")
    parser.add_argument('-d', '--depths', type=str, help='depths included in the test (default: 1,2,3,4,5)', required=False, default="1,2,3,4,5")
    parser.add_argument('-b', '--budgets', type=str, help='budgets included in the baseline (default: 20,50,100,200,500)', required=False, default="20,50,100,200,500")
    parser.add_argument('-g', '--group_by', type=str, help='field to group by results (default: condition,depth,budget)', required=False, default="condition,depth,budget")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

    import os
    if not os.path.isdir(args.result_folder):
        print(f"Error: {args.result_folder} is not a directory")
        exit(1)

    conditions = args.conditions.split(',')
    depths = args.depths.split(',')
    budgets = args.budgets.split(',')
    group_by = args.group_by.split(',')
    conditions_file_list = []

    for entropy in ["00", "03", "08"]:
        for world_number in ["00", "01", "02", "03"]:
            world = f"{world_number}_{entropy}"
            condition_folder = os.path.join(args.result_folder, 'logs', f"{world}_condition_4")
            if os.path.isdir(condition_folder):
                for depth in depths:
                    depth_folder = os.path.join(condition_folder, f"depth_{depth}")
                    if os.path.isdir(depth_folder):
                        for budget in budgets:
                            budget_folder = os.path.join(depth_folder, f"budget_{budget}")
                            if os.path.isdir(budget_folder):
                                experiment_file = os.path.join(budget_folder, f"{world}_condition_4_{depth}_{budget}.json")
                                file_data = {"condition": f"entropy_{entropy}", "depth": depth, "budget": budget}
                                condition_file = {"file": experiment_file}
                                for i, group in enumerate(group_by):
                                    condition_file[f"group_{i}"] = file_data[group]
                                conditions_file_list.append(condition_file)

    conditions = {"groups": group_by, "files": conditions_file_list}
    with open(args.output_file, "w") as f:
        json.dump(conditions, f)
