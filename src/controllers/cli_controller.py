import json
import logging
from typing import Optional
from ..models.service import AIBOMService
from ..models.scoring import calculate_completeness_score
from ..models.scoring import calculate_completeness_score
from ..config import OUTPUT_DIR, TEMPLATES_DIR
import os
import shutil

logger = logging.getLogger(__name__)

class CLIController:
    def __init__(self):
        self.service = AIBOMService()

    def _validate_spdx_schema_version(self, aibom_data: dict, spec_version: str):
        """
        TODO: Implement SPDX schema validation.
        """
        pass

    def generate(self, model_id: str, output_file: Optional[str] = None, include_inference: bool = False, 
                 enable_summarization: bool = False, verbose: bool = False,
                 name: Optional[str] = None, version: Optional[str] = None, manufacturer: Optional[str] = None):
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
            
        print(f"Generating AIBOM for {model_id}...")
        
        versions_to_generate = ["1.6", "1.7"]
        reports = []
        generated_aiboms = {} 
        
        for spec_version in versions_to_generate:
            print(f"  - Generating CycloneDX {spec_version}...")
            try:
                aibom = self.service.generate_aibom(
                    model_id, 
                    include_inference=include_inference, 
                    enable_summarization=enable_summarization,
                    spec_version=spec_version,
                    metadata_overrides={
                        "name": name,
                        "version": version,
                        "manufacturer": manufacturer
                    }
                )
                report = self.service.get_enhancement_report()
                
                # Determine output filename
                current_output_file = output_file
                if not current_output_file:
                    normalized_id = self.service._normalise_model_id(model_id)
                    import os
                    os.makedirs("sboms", exist_ok=True)
                    suffix = f"_{spec_version.replace('.', '_')}"
                    current_output_file = os.path.join("sboms", f"{normalized_id.replace('/', '_')}_ai_sbom{suffix}.json")
                elif spec_version != "1.6":
                     import os
                     base, ext = os.path.splitext(current_output_file)
                     current_output_file = f"{base}_{spec_version.replace('.', '_')}{ext}"

                with open(current_output_file, 'w') as f:
                    json.dump(aibom, f, indent=2)
                
                # Check for validation results in report (populated by scoring mechanism)
                validation_data = report.get("final_score", {}).get("validation", {})
                is_valid = validation_data.get("valid", True) # Default to true if not found to avoid noise? Or False?
                # Actually, scoring.py populates this.
                validation_errors = [i["message"] for i in validation_data.get("issues", [])]
                
                # Ensure schema_validation key exists for our report structure
                if "schema_validation" not in report:
                    report["schema_validation"] = {}
                report["schema_validation"]["valid"] = is_valid
                report["schema_validation"]["errors"] = validation_errors
                report["schema_validation"]["error_count"] = len(validation_errors)
                report["output_file"] = current_output_file
                report["spec_version"] = spec_version
                reports.append(report)
                generated_aiboms[spec_version] = aibom

            except Exception as e:
                logger.error(f"Failed to generate {spec_version} SBOM: {e}")
                print(f"  ❌ Failed to generate {spec_version}: {e}")

        # Summary and HTML Report (using 1.6 as primary)
        if reports:
            # Find primary report (1.6)
            primary_report = next((r for r in reports if r.get("spec_version") == "1.6"), reports[0])
            primary_aibom = generated_aiboms.get(primary_report["spec_version"])
            output_file_primary = primary_report.get("output_file")
            
            # Generate HTML for primary only
            if output_file_primary:
                try:
                    from jinja2 import Environment, FileSystemLoader, select_autoescape
                    from ..config import TEMPLATES_DIR
                    import os
                    
                    env = Environment(
                        loader=FileSystemLoader(TEMPLATES_DIR),
                        autoescape=select_autoescape(['html', 'xml'])
                    )
                    template = env.get_template("result.html")
                    
                    completeness_score = primary_report.get("final_score")
                    if not completeness_score:
                         completeness_score = calculate_completeness_score(primary_aibom)

                    # Pre-serialize to preserve order
                    components_json = json.dumps(primary_aibom.get("components", []), indent=2)
                    aibom_json = json.dumps(primary_aibom, indent=2)

                    context = {
                        "request": None,
                        "filename": os.path.basename(output_file_primary),
                        "download_url": "#",
                        "aibom": primary_aibom,
                        "components_json": components_json,
                        "aibom_json": aibom_json,
                        "raw_aibom": primary_aibom,
                        "model_id": self.service._normalise_model_id(model_id),
                        "sbom_count": 0,
                        "completeness_score": completeness_score,
                        "enhancement_report": primary_report or {},
                        "result_file": "#",
                        "static_root": "static" 
                    }

                    html_content = template.render(context)
                    html_output_file = output_file_primary.replace("_1_6.json", ".html").replace(".json", ".html")
                    with open(html_output_file, "w") as f:
                        f.write(html_content)
                    
                    print(f"\n📄 HTML Report:\n   {html_output_file}")

                    # Copy static assets
                    try:
                        # output_file_primary is e.g. sboms/model_id_ai_sbom.json
                        # html_output_file is sboms/model_id_ai_sbom.html
                        output_dir = os.path.dirname(html_output_file)
                        # src/static relative to CLI execution root or module
                        # Let's use absolute path relative to this file to be safe
                        current_dir = os.path.dirname(os.path.abspath(__file__)) # src/controllers
                        src_dir = os.path.dirname(current_dir) # src
                        static_src = os.path.join(src_dir, "static")
                        static_dst = os.path.join(output_dir, "static")
                        
                        if os.path.exists(static_src):
                            if os.path.exists(static_dst):
                                shutil.rmtree(static_dst)
                            shutil.copytree(static_src, static_dst)
                            # print(f"   - Static assets copied to: {static_dst}")
                        else:
                            logger.warning(f"Static source directory not found: {static_src}")

                    except Exception as e:
                        logger.warning(f"Failed to copy static assets: {e}")

                    # Model Description
                    if "components" in primary_aibom and primary_aibom["components"]:
                        description = primary_aibom["components"][0].get("description", "No description available")
                        if len(description) > 256:
                            description = description[:253] + "..."
                        print(f"\n📝 Model Description:\n   {description}")

                    # License
                    if "components" in primary_aibom and primary_aibom["components"]:
                        comp = primary_aibom["components"][0]
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

            # Print Summary for ALL versions
            for r in reports:
                spec = r.get("spec_version", "1.6")
                print(f"\n✅ Successfully generated CycloneDX {spec} SBOM:")
                print(f"   {r.get('output_file')}")
                
                if not r["schema_validation"]["valid"]:
                    print(f"⚠️  Schema Validation Errors ({spec}):")
                    for err in r["schema_validation"]["errors"]:
                        print(f"   - {err}")
                else:
                    print(f"   - Schema Validation ({spec}): ✅ Valid")
            
            # Display Detailed Score Summary (from primary)
            if primary_report and "final_score" in primary_report:
                score = primary_report["final_score"]
                print(f"\n📊 Completeness Score: {score.get('total_score', 0)}/100")
                
                if "completeness_profile" in score:
                    profile = score["completeness_profile"]
                    print(f"   Profile: {profile.get('name')} - {profile.get('description')}")
                
                if "section_scores" in score:
                    print("\n📋 Section Breakdown:")
                    
                    for section, s_score in score["section_scores"].items():
                        max_s = score.get("max_scores", {}).get(section, "?")
                        print(f"   - {section.replace('_', ' ').title()}: {s_score}/{max_s}")

        else:
             print("\n❌ Failed to generate any SBOMs.")
