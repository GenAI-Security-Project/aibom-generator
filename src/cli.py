import argparse
import logging
import sys
from .controllers.cli_controller import CLIController
from .models import owasp_risks

logger = logging.getLogger(__name__)

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

    risks = parser.add_argument_group(
        "OWASP GenAI risks (off by default)",
        "Add candidate OWASP GenAI risks and the framework controls mapped to them, from the "
        "OWASP GenAI Crosswalk. Candidates only: nothing is assessed. Any option below turns the section on.")
    risks.add_argument("--owasp-risks", action="store_true",
                       help="Add the section; the LLM Top 10 applies when the model's task is generative")
    risks.add_argument("--llm", action="store_true", help="Add the LLM Top 10 whatever the task")
    risks.add_argument("--agentic", action="store_true", help="The model runs in an agent (Agentic Top 10)")
    risks.add_argument("--rag", action="store_true", help="The model uses retrieval (GenAI Data Security)")
    risks.add_argument("--training-data", action="store_true",
                       help="The model is trained or tuned on organisational data (GenAI Data Security)")
    risks.add_argument("--skills", action="store_true", help="The model loads agent skills (Agentic Skills Top 10)")
    risks.add_argument("--frameworks",
                       help="Comma-separated framework ids or names (default: the crosswalk's verified core)")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    risk_options = owasp_risks.RiskOptions.from_flags(
        owasp_risks=args.owasp_risks, llm=args.llm, agentic=args.agentic, rag=args.rag,
        training_data=args.training_data, skills=args.skills, frameworks=args.frameworks)
    try:
        owasp_risks.check_options(risk_options)
    except owasp_risks.OwaspRiskError as e:
        parser.error(str(e))

    controller = CLIController()

    if args.test:
        test_models = [
            "Qwen/Qwen3.5-397B-A17B",
            "nvidia/personaplex-7b-v1",
            "meta-llama/Llama-2-7b-chat-hf",
            "unsloth/Qwen3.5-35B-A3B-GGUF",
            "LocoreMind/LocoOperator-4B",
            "Nanbeige/Nanbeige4.1-3B",
            "zai-org/GLM-5",
            "MiniMaxAI/MiniMax-M2.5",
            "unsloth/Qwen3.5-397B-A17B-GGUF",
            "FireRedTeam/FireRed-Image-Edit-1.0",
            "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese",
            "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF",
            "CIRCL/vulnerability-severity-classification-roberta-base"
        ]
        
        logger.info("Running test mode against %d models...", len(test_models))
        for model in test_models:
            logger.info("Testing model: %s", model)
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
                logger.error("Error testing %s: %s", model, e)
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
        manufacturer=args.manufacturer,
        risk_options=risk_options,
    )

if __name__ == "__main__":
    main()
