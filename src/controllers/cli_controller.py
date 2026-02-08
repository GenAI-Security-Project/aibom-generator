import json
import logging
from typing import Optional
from ..models.service import AIBOMService
from ..models.scoring import calculate_completeness_score
from ..config import OUTPUT_DIR

logger = logging.getLogger(__name__)

class CLIController:
    def __init__(self):
        self.service = AIBOMService()

    def generate(self, model_id: str, output_file: Optional[str] = None, include_inference: bool = False, enable_summarization: bool = False, verbose: bool = False):
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
            
        print(f"Generating AIBOM for {model_id}...")
        try:
            aibom = self.service.generate_aibom(model_id, include_inference=include_inference, enable_summarization=enable_summarization)
            report = self.service.get_enhancement_report()
            
            if not output_file:
                normalized_id = self.service._normalise_model_id(model_id)
                # Create sboms directory if it doesn't exist
                import os
                os.makedirs("sboms", exist_ok=True)
                output_file = os.path.join("sboms", f"{normalized_id.replace('/', '_')}_ai_sbom.json")
                
            with open(output_file, 'w') as f:
                json.dump(aibom, f, indent=2)
                
            print(f"\n✅ Successfully generated AIBOM:\n   {output_file}")

            # Helper for HTML generation
            try:
                from jinja2 import Environment, FileSystemLoader, select_autoescape
                from ..config import TEMPLATES_DIR
                
                env = Environment(
                    loader=FileSystemLoader(TEMPLATES_DIR),
                    autoescape=select_autoescape(['html', 'xml'])
                )
                template = env.get_template("result.html")
                
                # Context for template
                completeness_score = None
                if report and "final_score" in report:
                    completeness_score = report["final_score"]
                else:
                    completeness_score = calculate_completeness_score(aibom)

                context = {
                    "request": None, # CLI has no request object
                    "filename": os.path.basename(output_file),
                    "download_url": "#",
                    "aibom": aibom, # Pass object, template handles serialization
                    "raw_aibom": aibom,
                    "model_id": normalized_id,
                    "sbom_count": 0, # Not available in CLI
                    "completeness_score": completeness_score,
                    "enhancement_report": report or {},
                    "result_file": "#"
                }

                html_content = template.render(context)
                
                # Save HTML file
                html_output_file = output_file.replace(".json", ".html")
                with open(html_output_file, "w") as f:
                    f.write(html_content)
                
                print(f"\n📄 HTML Report:\n   {html_output_file}")
                
                 # Model Description
                if "components" in aibom and aibom["components"]:
                    description = aibom["components"][0].get("description", "No description available")
                    # Truncate if very long for CLI readability
                    if len(description) > 500:
                        description = description[:497] + "..."
                    print(f"\n📝 Model Description:\n   {description}")

                # License
                if "components" in aibom and aibom["components"]:
                     comp = aibom["components"][0]
                     if "licenses" in comp:
                         license_list = []
                         for l in comp["licenses"]:
                             lic = l.get("license", {})
                             val = lic.get("id") or lic.get("name")
                             if val:
                                 license_list.append(val)
                         
                         if license_list:
                             print(f"\n⚖️ License:\n   {', '.join(license_list)}")
                
            except Exception as e:
                logger.warning(f"Failed to generate HTML report: {e}")
            
            # Display Detailed Score Summary
            if report and "final_score" in report:
                score = report["final_score"]
                print(f"\n📊 completeness Score: {score.get('total_score', 0)}/100")
                
                if "completeness_profile" in score:
                    profile = score["completeness_profile"]
                    print(f"   Profile: {profile.get('name')} - {profile.get('description')}")
                
                # Section Scores
                if "section_scores" in score:
                    print("\n📋 Section Breakdown:")
                    
                    # Schema Validation Status
                    if "schema_validation" in report:
                        val = report["schema_validation"]
                        status_icon = "✅" if val.get("valid") else "❌"
                        status_text = "Valid" if val.get("valid") else f"Invalid ({val.get('error_count')} errors)"
                        print(f"   - Schema Validation (CycloneDX 1.6): {status_icon} {status_text}")

                    for section, s_score in score["section_scores"].items():
                        max_s = score.get("max_scores", {}).get(section, "?")
                        print(f"   - {section.replace('_', ' ').title()}: {s_score}/{max_s}")

                # Warnings / Penalties
                if score.get("penalty_applied"):
                    print(f"\n⚠️ Penalty Applied: {score.get('penalty_reason')}")

                # Recommendations (Top 3)
                if "recommendations" in score and score["recommendations"]:
                    print("\n💡 Top Recommendations:")
                    for rec in score["recommendations"][:3]:
                        print(f"   - [{rec.get('priority', 'medium').upper()}] {rec.get('message')}")

                # Schema Validation Errors
                if "schema_validation" in report and not report["schema_validation"].get("valid"):
                    print("\n⚠️ Schema Validation Errors:")
                    for error in report["schema_validation"].get("errors", []):
                        print(f"   - {error}")

        except Exception as e:
            print(f"❌ Error: {e}")
            if verbose:
                logger.exception("Details:")
