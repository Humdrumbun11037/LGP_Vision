#!/usr/bin/env python3
"""Utility script to list and search experiment runs.

Usage:
    python list_experiments.py                    # List all runs
    python list_experiments.py --best 10          # Top 10 by fitness
    python list_experiments.py --today            # Today's runs
    python list_experiments.py --status completed # Filter by status
    python list_experiments.py --show <run_id>    # Show details of one run
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_run_info(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load run_info.json from a run directory."""
    run_info_path = run_dir / "run_info.json"
    if not run_info_path.exists():
        return None
    
    try:
        with open(run_info_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_all_runs(base_dir: str = "experiments") -> List[Dict[str, Any]]:
    """Get all experiment runs from the base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    runs = []
    for run_dir in sorted(base_path.iterdir(), reverse=True):
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            run_info = load_run_info(run_dir)
            if run_info is not None:
                run_info['_dir'] = str(run_dir)
                runs.append(run_info)
    
    return runs


def format_table(runs: List[Dict[str, Any]]) -> str:
    """Format runs as a table."""
    if not runs:
        return "No experiments found."
    
    # Define columns
    headers = ["Run ID", "Status", "Best Fitness", "Gens", "Date"]
    
    # Extract data
    rows = []
    for run in runs:
        run_id = run.get('run_id', 'unknown')[:35]
        status = run.get('status', 'unknown')[:10]
        
        results = run.get('results', {})
        if results:
            best_fitness = f"{results.get('best_fitness', 0):.4f}"
            gens = str(results.get('final_generation', '?'))
        else:
            best_fitness = "?"
            gens = "?"
        
        start_time = run.get('start_time', '')
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time)
                date = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                date = start_time[:16]
        else:
            date = "?"
        
        rows.append([run_id, status, best_fitness, gens, date])
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Build table
    separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths)) + "|"
    
    lines = [separator, header_row, separator]
    for row in rows:
        line = "|" + "|".join(f" {str(c):<{w}} " for c, w in zip(row, widths)) + "|"
        lines.append(line)
    lines.append(separator)
    
    return "\n".join(lines)


def show_run_details(run_info: Dict[str, Any]) -> str:
    """Format detailed information about a single run."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"EXPERIMENT: {run_info.get('run_id', 'unknown')}")
    lines.append("=" * 70)
    
    lines.append(f"\nStatus: {run_info.get('status', 'unknown')}")
    lines.append(f"Start:  {run_info.get('start_time', 'unknown')}")
    lines.append(f"End:    {run_info.get('end_time', 'unknown')}")
    lines.append(f"Dir:    {run_info.get('_dir', 'unknown')}")
    
    # Results
    results = run_info.get('results', {})
    if results:
        lines.append(f"\n--- Results ---")
        lines.append(f"Best Fitness:    {results.get('best_fitness', '?')}")
        lines.append(f"Best Generation: {results.get('best_generation', '?')}")
        lines.append(f"Final Generation:{results.get('final_generation', '?')}")
        lines.append(f"Runtime:         {results.get('runtime_formatted', '?')}")
    
    # Config summary
    config_summary = run_info.get('config_summary', {})
    if config_summary:
        lines.append(f"\n--- Configuration ---")
        for key, value in config_summary.items():
            lines.append(f"{key}: {value}")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="List and search experiment runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dir", "-d",
        default="experiments",
        help="Base directory for experiments (default: experiments)"
    )
    parser.add_argument(
        "--best", "-b",
        type=int,
        metavar="N",
        help="Show top N runs by fitness"
    )
    parser.add_argument(
        "--today", "-t",
        action="store_true",
        help="Show only today's runs"
    )
    parser.add_argument(
        "--status", "-s",
        choices=["completed", "running", "failed", "interrupted"],
        help="Filter by status"
    )
    parser.add_argument(
        "--show",
        metavar="RUN_ID",
        help="Show details of a specific run"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    # Get all runs
    runs = get_all_runs(args.dir)
    
    if not runs:
        print(f"No experiments found in '{args.dir}'")
        return
    
    # Show details of a specific run
    if args.show:
        matching = [r for r in runs if args.show in r.get('run_id', '')]
        if not matching:
            print(f"No run found matching: {args.show}")
            return
        print(show_run_details(matching[0]))
        return
    
    # Filter by today
    if args.today:
        today = datetime.now().date()
        filtered = []
        for run in runs:
            start_time = run.get('start_time', '')
            if start_time:
                try:
                    dt = datetime.fromisoformat(start_time)
                    if dt.date() == today:
                        filtered.append(run)
                except ValueError:
                    pass
        runs = filtered
    
    # Filter by status
    if args.status:
        runs = [r for r in runs if r.get('status') == args.status]
    
    # Sort by best fitness (top N)
    if args.best:
        runs_with_fitness = []
        for run in runs:
            results = run.get('results', {})
            if results and results.get('best_fitness') is not None:
                runs_with_fitness.append((results['best_fitness'], run))
        runs_with_fitness.sort(key=lambda x: x[0], reverse=True)
        runs = [r for _, r in runs_with_fitness[:args.best]]
    
    # Output
    if args.json:
        # Remove internal fields for JSON output
        clean_runs = [{k: v for k, v in r.items() if not k.startswith('_')} for r in runs]
        print(json.dumps(clean_runs, indent=2))
    else:
        print(f"\nFound {len(runs)} experiment(s) in '{args.dir}':\n")
        print(format_table(runs))
        print()


if __name__ == "__main__":
    main()

