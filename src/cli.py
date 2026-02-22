import argparse
import sys
from .controllers.cli_controller import CLIController

def main():
    parser = argparse.ArgumentParser(description="OWASP AIBOM Generator CLI")
    parser.add_argument("model_id", nargs="?", help="Hugging Face Model ID (e.g. 'owner/model')")
    parser.add_argument("--test", "-t", action="store_true", help="Run test mode for multiple predefined models to verify description generation")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--inference", "-i", action="store_true", help="Use AI inference for enhanced metadata (requires configured valid endpoint)")
    parser.add_argument("--summarize", "-s", action="store_true", help="Enable intelligent description summarization (requires model download)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--name", "-n", help="Component name in metadata")
    parser.add_argument("--version", "-v", help="Component version in metadata")
    parser.add_argument("--manufacturer", "-m", help="Component manufacturer/supplier in metadata")
    
    args = parser.parse_args()
    
    controller = CLIController()

    if args.test:
        test_models = [
            "Qwen/Qwen3.5-397B-A17B",
            "nvidia/personaplex-7b-v1",
            "Nanbeige/Nanbeige4.1-3B",
            "zai-org/GLM-5",
            "MiniMaxAI/MiniMax-M2.5",
            "unsloth/Qwen3.5-397B-A17B-GGUF",
            "FireRedTeam/FireRed-Image-Edit-1.0",
            "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese",
            "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF"
        ]
        
        print(f"Running test mode against {len(test_models)} models...")
        for model in test_models:
            print(f"\n{'='*50}\nTesting model: {model}\n{'='*50}")
            try:
                controller.generate(
                    model_id=model,
                    output_file=args.output,
                    include_inference=args.inference,
                    enable_summarization=True,  # Ensure summarization is on for testing description 
                    verbose=args.verbose,
                    name=args.name,
                    version=args.version,
                    manufacturer=args.manufacturer
                )
            except Exception as e:
                print(f"Error testing {model}: {e}")
        sys.exit(0)
    
    if not args.model_id:
        parser.error("model_id is required unless --test is specified")

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
