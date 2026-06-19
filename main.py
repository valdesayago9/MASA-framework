"""
MASA — Multi-Agent System for Adaptive Alignment
main.py — Main Execution Engine (CLI)

v3.1 (Correction 5): adds a functional conversation loop. Previously this file
built the graph but never ran it. run_session() now initializes real state,
executes the full pipeline per turn, prints the key metrics, and — crucially —
threads the previous turn's trajectory_buffer, session_health_history and
manifold_pressure into the next turn, so the NB4 inverse-signaling feedback loop
(Values → Detector across turns) is exercised for the first time.

Usage:
    python main.py --api-key YOUR_API_KEY
    or with environment variable:
    ANTHROPIC_API_KEY=sk-... python main.py
"""

import os
import sys
import uuid
import argparse
from rich.console import Console
from rich.panel import Panel

# Adjust path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import build_masa_graph
from agents.agents import init_agents, MODE_A_LIMITATION

console = Console()


def print_header():
    """Prints the MASA Framework clinical interface header."""
    console.print("\n[bold cyan]MASA Framework v3.1 (Mode A)[/bold cyan]")
    console.print("[dim]Multi-Agent System for Adaptive Alignment[/dim]")
    console.print("[dim]Neuropsychological telemetry active (Proxy Mode)[/dim]")
    console.print(f"[dim yellow]Mode A scope: {MODE_A_LIMITATION}[/dim yellow]\n")


def run_session(masa_pipeline) -> None:
    """
    Correction 5 — interactive session loop with cross-turn state threading.

    Persistent (carried turn → turn): session_id, conversation_history,
    trajectory_buffer, session_health_history, manifold_pressure.
    Transient state (draft_response, final_response, regulation_report, etc.) is
    intentionally NOT carried, so each turn starts clean and the agents
    regenerate it. This is what makes NB4 (manifold_pressure feeding the next
    turn's Detector sensitivity) actually function.
    """
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    console.print(f"[bold green]Session started:[/bold green] {session_id}")
    console.print("[dim]Type your message. Use /exit or Ctrl-D to end.[/dim]\n")

    # Persistent cross-turn state.
    carry = {
        "session_id":             session_id,
        "conversation_history":   [],
        "trajectory_buffer":      [],
        "session_health_history": [],
        "manifold_pressure":      0.0,
    }
    turn_number = 0

    while True:
        try:
            user_input = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit", "exit", "quit"}:
            console.print("[dim]Session ended.[/dim]")
            break

        turn_number += 1

        # Build this turn's state: persistent carry + fresh per-turn input.
        state = {
            **carry,
            "input_prompt": user_input,
            "turn_number":  turn_number,
        }

        # Run the full pipeline end-to-end for this turn.
        result = masa_pipeline.invoke(state)

        # ── Extract metrics for display ───────────────────────────────────────
        proxy          = result.get("emotional_proxy", {})
        dynamics       = result.get("dynamics_report", {})
        interpretation = result.get("interpretation", {})
        audit_log      = result.get("audit_log", {})
        final_response = result.get("final_response", "")

        console.print(f"\n[bold magenta]masa ›[/bold magenta] {final_response}\n")
        console.print(Panel.fit(
            f"ltci                 : {proxy.get('ltci', 0):.4f}\n"
            f"risk_score           : {dynamics.get('risk_score', 0):.4f}\n"
            f"state_classification : {interpretation.get('state_classification', 'UNKNOWN')}\n"
            f"execution_path       : {audit_log.get('execution_path', '—')}\n"
            f"manifold_pressure    : {result.get('manifold_pressure', 0.0):.4f}\n"
            f"session_health_score : {audit_log.get('session_health_score', 0):.4f}",
            title=f"Turn {turn_number} telemetry",
            border_style="cyan",
        ))

        # ── Thread persistent state into the next turn ────────────────────────
        carry["trajectory_buffer"]      = result.get("trajectory_buffer", carry["trajectory_buffer"])
        carry["session_health_history"] = result.get("session_health_history", carry["session_health_history"])
        # On the human_loop path the Values agent is skipped, so manifold_pressure
        # may be absent from result — fall back to the prior value.
        carry["manifold_pressure"]      = result.get("manifold_pressure", carry["manifold_pressure"])
        history = list(carry["conversation_history"])
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": final_response})
        carry["conversation_history"] = history


def main():
    parser = argparse.ArgumentParser(
        description="MASA Mode A — Neuropsychological testing environment"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Only build/compile the graph and exit (no interactive session)."
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

    # Initialize the shared client and the orchestration graph.
    init_agents(args.api_key)
    masa_pipeline = build_masa_graph()

    console.print("[bold green]System initialized successfully.[/bold green]")
    console.print(
        "Pipeline compiled: "
        "[ draft_generator -> Detector -> Interpretive -> Regulator -> Values -> Auditor ]\n"
    )

    if args.no_loop:
        console.print(Panel.fit(
            "Graph compiled. Re-run without --no-loop for an interactive session.",
            title="Quick Start",
            border_style="cyan",
        ))
        return

    run_session(masa_pipeline)


if __name__ == "__main__":
    main()
