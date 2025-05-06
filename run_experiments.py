# run_experiments.py
# Script for running experiments with Differentially Private Embeddings

import os
import subprocess
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with Differentially Private Embeddings")
    
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["glue", "conll", "privacy_utility", "ablation"],
                       help="Type of experiment to run")
    
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    
    return parser.parse_args()

def run_glue_experiments():
    """Run experiments on GLUE benchmark tasks"""
    tasks = ["sst2", "qqp", "mnli", "cola"]
    methods = ["non-private", "dp-sgd", "pate-distill", "dp-distill"]
    epsilon = 8.0
    
    results = {}
    
    for task in tasks:
        print(f"\n\n{'='*80}")
        print(f"Running experiments for {task}")
        print(f"{'='*80}\n")
        
        task_results = {}
        
        for method in methods:
            start_time = time.time()
            
            cmd = [
                "python", "main.py",
                "--dataset", "glue",
                "--task", task,
                "--method", method,
                "--epsilon", str(epsilon),
                "--num_epochs", "3"
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            # Run the command and show output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Read output line by line and print it
            output_lines = []
            for line in process.stdout:
                print(line, end='')
                output_lines.append(line)
            
            process.wait()
            elapsed_time = time.time() - start_time
            
            # Extract metrics from the output
            metrics = {}
            for line in output_lines:
                if "Test Results for" in line:
                    # Look for the metrics section
                    for metric_line in output_lines[output_lines.index(line):]:
                        if "="*80 in metric_line:
                            break
                        if ":" in metric_line and "=" not in metric_line:
                            metric_name, value = metric_line.split(":")
                            metrics[metric_name.strip()] = float(value.strip())
            
            if metrics:
                task_results[method] = {
                    "metrics": metrics,
                    "time": elapsed_time
                }
                print(f"\nTime taken: {elapsed_time:.2f} seconds")
            else:
                print(f"\nWarning: No metrics found for {method}")
        
        results[task] = task_results
    
    # Save results
    os.makedirs("results/glue", exist_ok=True)
    with open("results/glue/results_summary.txt", "w") as f:
        f.write("GLUE Benchmark Results\n")
        f.write("=====================\n\n")
        
        for task, task_results in results.items():
            f.write(f"{task} Results:\n")
            f.write("-" * 80 + "\n")
            for method, result in task_results.items():
                f.write(f"{method}:\n")
                for metric_name, value in result['metrics'].items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
                f.write(f"  Time: {result['time']:.2f} seconds\n")
            f.write("\n")
    
    # Print final summary
    print("\n" + "="*80)
    print("Final Results Summary")
    print("="*80)
    for task, task_results in results.items():
        print(f"\n{task}:")
        print("-" * 40)
        for method, result in task_results.items():
            print(f"{method}:")
            for metric_name, value in result['metrics'].items():
                print(f"  {metric_name}: {value:.4f}")
            print(f"  Time: {result['time']:.2f} seconds")


def run_conll_experiments():
    """Run experiments on CoNLL-2003 NER task"""
    methods = ["non-private", "dp-sgd", "pate-distill", "dp-distill"]
    epsilons = [4.0, 8.0]
    
    results = {}
    
    for epsilon in epsilons:
        print(f"\n\n{'='*80}")
        print(f"Running experiments for CoNLL-2003 with epsilon={epsilon}")
        print(f"{'='*80}\n")
        
        epsilon_results = {}
        
        for method in methods:
            start_time = time.time()
            
            cmd = [
                "python", "main.py",
                "--dataset", "conll2003",
                "--method", method,
                "--epsilon", str(epsilon),
                "--num_epochs", "3"
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            elapsed_time = time.time() - start_time
            
            # Extract test metrics from output
            output = process.stdout
            for line in output.split('\n'):
                if line.startswith("Test metrics:"):
                    metrics_str = line.replace("Test metrics:", "").strip()
                    epsilon_results[method] = {
                        "metrics": metrics_str,
                        "time": elapsed_time
                    }
                    print(f"{method} metrics: {metrics_str}")
                    print(f"Time taken: {elapsed_time:.2f} seconds")
                    break
        
        results[epsilon] = epsilon_results
    
    # Save results
    os.makedirs("results/conll", exist_ok=True)
    with open("results/conll/results_summary.txt", "w") as f:
        f.write("CoNLL-2003 NER Results\n")
        f.write("=====================\n\n")
        
        for epsilon, epsilon_results in results.items():
            f.write(f"Epsilon = {epsilon} Results:\n")
            f.write("-" * 80 + "\n")
            for method, result in epsilon_results.items():
                f.write(f"{method}: {result['metrics']}\n")
                f.write(f"Time: {result['time']:.2f} seconds\n")
            f.write("\n")


def run_privacy_utility_tradeoff():
    """Run privacy-utility tradeoff experiments"""
    tasks = ["sst2", "conll2003"]
    
    for task in tasks:
        print(f"\n\n{'='*80}")
        print(f"Running privacy-utility tradeoff for {task}")
        print(f"{'='*80}\n")
        
        dataset_arg = "--dataset" if task == "conll2003" else "--task"
        
        cmd = [
            "python", "main.py",
            dataset_arg, task,
            "--dataset", "glue" if task != "conll2003" else "conll2003",
            "--privacy_utility_tradeoff",
            "--run_all_methods",
            "--num_epochs", "3"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Save output
        os.makedirs("results/privacy_utility", exist_ok=True)
        with open(f"results/privacy_utility/{task}_tradeoff_output.txt", "w") as f:
            f.write(process.stdout)


def run_ablation_studies():
    """Run ablation studies to analyze contributions of each component"""
    # Experiment setup
    task = "sst2"  # Use SST-2 for ablation studies
    epsilon = 8.0
    
    # Base configuration (full model)
    base_cmd = [
        "python", "main.py",
        "--dataset", "glue",
        "--task", task,
        "--method", "dp-distill",
        "--epsilon", str(epsilon),
        "--num_epochs", "3"
    ]
    
    # Configurations for ablation
    ablation_configs = {
        "full_model": base_cmd,
        "no_multi_layer": base_cmd + ["--no_multi_layer_noise"],
        "no_rare_token": base_cmd + ["--rare_token_threshold", "0"],
        "fewer_teachers": base_cmd + ["--num_teachers", "3"],
        "more_noise": base_cmd + ["--teacher_noise", "1.0"],
    }
    
    results = {}
    
    print(f"\n\n{'='*80}")
    print(f"Running ablation studies on {task}")
    print(f"{'='*80}\n")
    
    for name, cmd in ablation_configs.items():
        print(f"Running {name} configuration...")
        start_time = time.time()
        
        print(f"Command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        
        # Extract test metrics from output
        output = process.stdout
        for line in output.split('\n'):
            if line.startswith("Test metrics:"):
                metrics_str = line.replace("Test metrics:", "").strip()
                results[name] = {
                    "metrics": metrics_str,
                    "time": elapsed_time
                }
                print(f"{name} metrics: {metrics_str}")
                print(f"Time taken: {elapsed_time:.2f} seconds")
                break
    
    # Save results
    os.makedirs("results/ablation", exist_ok=True)
    with open("results/ablation/results_summary.txt", "w") as f:
        f.write("Ablation Study Results\n")
        f.write("=====================\n\n")
        
        for config_name, result in results.items():
            f.write(f"{config_name}:\n")
            f.write(f"Metrics: {result['metrics']}\n")
            f.write(f"Time: {result['time']:.2f} seconds\n")
            f.write("\n")


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