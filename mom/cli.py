"""Command-line interface for MoM Phase 1 prototype."""

from .core.orchestrator import Orchestrator


def main():
    print("Mixture of Models (MoM) Phase 1 prototype CLI")
    orchestrator = Orchestrator()
    try:
        while True:
            text = input("You: ")
            if not text.strip():
                continue
            if text.strip().lower() in ("exit", "quit"):
                break
            result = orchestrator.handle_request(text)
            print("\n--- MoM Output ---")
            print(result.get("answer"))
            print("--- metadata ---")
            print(result.get("metadata", {}))
            print("-------------------\n")
    except (KeyboardInterrupt, EOFError):
        print("Exiting CLI")


if __name__ == "__main__":
    main()
