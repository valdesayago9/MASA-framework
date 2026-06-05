"""
MASA — Multi-Agent System for Adaptive Alignment
main.py — Main Execution Engine (CLI)

Usage:
    python main.py --api-key YOUR_API_KEY
    or with environment variable:
    ANTHROPIC_API_KEY=sk-... python main.py
"""

import os
import sys
import argparse
from rich.console import Console
from rich.panel import Panel

# Adjust path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import build_masa_graph
from agents.agents import init_agents  # <-- LÍNEA AGREGADA

console = Console()

def print_header():
    """Prints the MASA Framework clinical interface header."""
    console.print("\n[bold cyan]MASA Framework (Mode A)[/bold cyan]")
    console.print("[dim]Multi-Agent System for Adaptive Alignment[/dim]")
    console.print("[dim]Neuropsychological telemetry active (Proxy Mode)[/dim]\n")

def main():
    parser = argparse.ArgumentParser(
        description="MASA Mode A — Neuropsychological testing environment"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    args = parser.parse_args()

    if not args.api_key:
        console.print(
            "[red]Error: API key required.[/]\n"
            "Usage: [bold]python main.py --api-key sk-...[/]\n"
            "Or:    [bold]export ANTHROPIC_API_KEY=sk-...[/]"
        )
        sys.exit(1)

    print_header()
    
    # Inicializar los agentes con la API key antes de construir el grafo
    init_agents(args.api_key)  # <-- LÍNEA AGREGADA
    
    # Initialize the orchestration graph
    masa_pipeline = build_masa_graph()
    
    console.print("[bold green]System initialized successfully.[/bold green]")
    console.print("Orchestration pipeline compiled: [ Detector -> Interpretive -> Regulator -> Values -> Auditor ]\n")
    
    console.print(Panel.fit(
        "To run the empirical validation suite (Clinical-Adversarial Tests), please execute:\n\n"
        "[bold yellow]python tests/stress_tests.py[/bold yellow]",
        title="Quick Start",
        border_style="cyan"
    ))

if __name__ == "__main__":
    main()
