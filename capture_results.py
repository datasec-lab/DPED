# capture_results.py
# Script to run experiments, capture results, and save them to CSV

import os
import subprocess
import time
import csv
import re
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments and capture results")
    
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["glue", "conll", "privacy_utility", "ablation"],
                        help="Type of experiment to run")
    
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output from experiments")
    
    return parser.parse_args()

def extract_metrics_from_stdout(stdout):
    """
    Extract metrics from command output
    """
    metrics = {}
    
    # Look for test metrics
    for line in stdout.split('\n'):
        if line.startswith("Test metrics:"):
            metrics_str = line.replace("Test metrics:", "").strip()
            # Try different parsing methods
            try:
                # Try parsing as dict
                metrics = eval(metrics_str)
            except:
                # Try regex pattern
                pattern = r"'(\w+)':\s*([\d\.]+)"
                matches = re.findall(pattern, metrics_str)
                for key, value in matches:
                    try:
                        metrics[key] = float(value)
                    except:
                        metrics[key] = value
    
    return metrics

def run_glue_experiments():
    """
    Run GLUE benchmark experiments and capture results
    """
    tasks = ["sst2", "qqp", "mnli", "cola"]
    methods = ["non-private", "dp-sgd", "pate-distill", "dp-distill"]
    epsilon = 8.0
    
    # Create results directory
    os.makedirs("results/csv", exist_ok=True)
    
    # Create CSV file for combined results
    csv_path = "results/csv/glue_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Task", "Method", "Accuracy", "F1", "Precision", "Recall", "Time(s)"])
    
        # Run experiments for each task and method
        for task in tasks:
            print(f"\n{'='*80}")
            print(f"Running experiments for {task}")
            print(f"{'='*80}")
            
            for method in methods:
                print(f"\nRunning {method} on {task}...")
                
                # Construct command
                cmd = [
                    "python", "main.py",
                    "--dataset", "glue",
                    "--task", task,
                    "--method", method,
                    "--epsilon", str(epsilon),
                    "--num_epochs", "3"
                ]
                
                # Run command and time it
                start_time = time.time()
                process = subprocess.run(cmd, capture_output=True, text=True)
                elapsed_time = time.time() - start_time
                
                # Extract metrics
                metrics = extract_metrics_from_stdout(process.stdout)
                
                # Print results
                print(f"Method: {method}")
                print(f"Task: {task}")
                print(f"Results: {metrics}")
                print(f"Time: {elapsed_time:.2f} seconds")
                
                # Write to CSV
                writer.writerow([
                    task,
                    method,
                    metrics.get("accuracy", "N/A"),
                    metrics.get("f1", "N/A"),
                    metrics.get("precision", "N/A"),
                    metrics.get("recall", "N/A"),
                    f"{elapsed_time:.2f}"
                ])
                
                # Flush to save immediately
                csvfile.flush()
    
    print(f"\nResults saved to {csv_path}")

def run_conll_experiments():
    """
    Run CoNLL-2003 NER experiments and capture results
    """
    methods = ["non-private", "dp-sgd", "pate-distill", "dp-distill"]
    epsilons = [4.0, 8.0]
    
    # Create results directory
    os.makedirs("results/csv", exist_ok=True)
    
    # Create CSV file
    csv_path = "results/csv/conll_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epsilon", "Method", "F1", "Accuracy", "Precision", "Recall", "Time(s)"])
        
        # Run experiments for each epsilon and method
        for epsilon in epsilons:
            print(f"\n{'='*80}")
            print(f"Running experiments for CoNLL-2003 with epsilon={epsilon}")
            print(f"{'='*80}")
            
            for method in methods:
                print(f"\nRunning {method} with epsilon={epsilon}...")
                
                # Construct command
                cmd = [
                    "python", "main.py",
                    "--dataset", "conll2003",
                    "--method", method,
                    "--epsilon", str(epsilon),
                    "--num_epochs", "3"
                ]
                
                # Run command and time it
                start_time = time.time()
                process = subprocess.run(cmd, capture_output=True, text=True)
                elapsed_time = time.time() - start_time
                
                # Extract metrics
                metrics = extract_metrics_from_stdout(process.stdout)
                
                # Print results
                print(f"Method: {method}")
                print(f"Epsilon: {epsilon}")
                print(f"Results: {metrics}")
                print(f"Time: {elapsed_time:.2f} seconds")
                
                # Write to CSV
                writer.writerow([
                    epsilon,
                    method,
                    metrics.get("f1", "N/A"),
                    metrics.get("accuracy", "N/A"),
                    metrics.get("precision", "N/A"),
                    metrics.get("recall", "N/A"),
                    f"{elapsed_time:.2f}"
                ])
                
                # Flush to save immediately
                csvfile.flush()
    
    print(f"\nResults saved to {csv_path}")

def run_privacy_utility_tradeoff():
    """
    Run privacy-utility tradeoff experiments and capture results
    """
    tasks = ["sst2", "conll2003"]
    
    # Create results directory
    os.makedirs("results/csv", exist_ok=True)
    
    for task in tasks:
        print(f"\n{'='*80}")
        print(f"Running privacy-utility tradeoff for {task}")
        print(f"{'='*80}")
        
        # Determine dataset argument based on task
        dataset_arg = "--dataset" if task == "conll2003" else "--task"
        dataset_val = "conll2003" if task == "conll2003" else "glue"
        
        # Construct command
        cmd = [
            "python", "main.py",
            "--dataset", dataset_val,
            dataset_arg, task,
            "--privacy_utility_tradeoff",
            "--run_all_methods",
            "--num_epochs", "3"
        ]
        
        # Run command
        process = subprocess.run(cmd, capture_output=True, text=True)
        stdout = process.stdout
        
        # Create CSV file
        csv_path = f"results/csv/privacy_utility_{task}.csv"
        
        # Extract epsilon values and metrics
        epsilon_pattern = r"Epsilon values: ([\d\., ]+)"
        epsilon_match = re.search(epsilon_pattern, stdout)
        
        method_pattern = r"([\w-]+): ([\d\., ]+)"
        method_matches = re.findall(method_pattern, stdout)
        
        if epsilon_match and method_matches:
            epsilon_values = [float(eps.strip()) for eps in epsilon_match.group(1).split(",")]
            
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                
                # Create header row
                methods = [method for method, _ in method_matches]
                writer.writerow(["Epsilon"] + methods)
                
                # Extract metrics for each epsilon and method
                method_metrics = {}
                for method, values in method_matches:
                    method_metrics[method] = [float(val.strip()) for val in values.split(",")]
                
                # Write data rows
                for i, epsilon in enumerate(epsilon_values):
                    row = [epsilon]
                    for method in methods:
                        if method in method_metrics and i < len(method_metrics[method]):
                            row.append(method_metrics[method][i])
                        else:
                            row.append("N/A")
                    writer.writerow(row)
            
            print(f"Privacy-utility tradeoff results for {task} saved to {csv_path}")
            
            # Print the results for immediate reference
            print("\nPrivacy-Utility Tradeoff Results:")
            print(f"Task: {task}")
            print("Epsilon values:", ", ".join(str(eps) for eps in epsilon_values))
            for method, values in method_matches:
                print(f"{method}: {values}")
        else:
            print("Failed to extract privacy-utility tradeoff results")

def run_ablation_studies():
    """
    Run ablation studies and capture results
    """
    # Configure ablation experiments
    task = "sst2"  # Use SST-2 for ablation
    epsilon = 8.0
    
    # Define ablation configurations
    ablation_configs = {
        "full_model": [],
        "no_multi_layer": ["--no_multi_layer_noise"],
        "no_rare_token": ["--rare_token_threshold", "0"],
        "fewer_teachers": ["--num_teachers", "3"],
        "more_noise": ["--teacher_noise", "1.0"],
    }
    
    # Create results directory
    os.makedirs("results/csv", exist_ok=True)
    
    # Create CSV file
    csv_path = "results/csv/ablation_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Configuration", "Accuracy", "F1", "Precision", "Recall", "Time(s)"])
        
        print(f"\n{'='*80}")
        print(f"Running ablation studies on {task}")
        print(f"{'='*80}")
        
        for config_name, args in ablation_configs.items():
            print(f"\nRunning configuration: {config_name}...")
            
            # Construct command
            cmd = [
                "python", "main.py",
                "--dataset", "glue",
                "--task", task,
                "--method", "dp-distill",
                "--epsilon", str(epsilon),
                "--num_epochs", "3"
            ] + args
            
            # Run command and time it
            start_time = time.time()
            process = subprocess.run(cmd, capture_output=True, text=True)
            elapsed_time = time.time() - start_time
            
            # Extract metrics
            metrics = extract_metrics_from_stdout(process.stdout)
            
            # Print results
            print(f"Configuration: {config_name}")
            print(f"Results: {metrics}")
            print(f"Time: {elapsed_time:.2f} seconds")
            
            # Write to CSV
            writer.writerow([
                config_name,
                metrics.get("accuracy", "N/A"),
                metrics.get("f1", "N/A"),
                metrics.get("precision", "N/A"),
                metrics.get("recall", "N/A"),
                f"{elapsed_time:.2f}"
            ])
            
            # Flush to save immediately
            csvfile.flush()
    
    print(f"\nAblation results saved to {csv_path}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected experiment
    if args.experiment == "glue":
        run_glue_experiments()
    elif args.experiment == "conll":
        run_conll_experiments()
    elif args.experiment == "privacy_utility":
        run_privacy_utility_tradeoff()
    elif args.experiment == "ablation":
        run_ablation_studies()
    else:
        print(f"Unknown experiment type: {args.experiment}")

if __name__ == "__main__":
    main()
