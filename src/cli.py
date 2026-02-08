import argparse
import sys
from .controllers.cli_controller import CLIController

def main():
    parser = argparse.ArgumentParser(description="OWASP AIBOM Generator CLI")
    parser.add_argument("model_id", help="Hugging Face Model ID (e.g. 'owner/model')")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--inference", "-i", action="store_true", help="Use AI inference for enhanced metadata (requires configured valid endpoint)")
    parser.add_argument("--summarize", "-s", action="store_true", help="Enable intelligent description summarization (requires model download)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--name", "-n", help="Component name in metadata")
    parser.add_argument("--version", "-v", help="Component version in metadata")
    parser.add_argument("--manufacturer", "-m", help="Component manufacturer/supplier in metadata")
    
    args = parser.parse_args()
    
    controller = CLIController()
    controller.generate(
        model_id=args.model_id,
        output_file=args.output,
        include_inference=args.inference,
        enable_summarization=args.summarize,
        verbose=args.verbose,
        name=args.name,
        version=args.version,
        manufacturer=args.manufacturer
    )

if __name__ == "__main__":
    main()
