import json
import logging
from typing import Optional
from ..models.service import AIBOMService
from ..models.scoring import calculate_completeness_score
from ..config import OUTPUT_DIR, TEMPLATES_DIR
from ..utils.formatter import export_aibom
import os
import shutil

from spdx_tools.spdx.parser.jsonlikedict.json_like_dict_parser import JsonLikeDictParser
from spdx_tools.spdx.validation.document_validator import validate_full_spdx_document
from spdx_tools.spdx.parser.error import SPDXParsingError

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _strip_1_6_suffix(path_without_ext: str) -> str:
    parent, name = os.path.split(path_without_ext)
    if name.endswith("_1_6"):
        name = name[:-len("_1_6")]
    return os.path.join(parent, name) if parent else name


def _resolve_output_files(output_file: Optional[str], default_json_1_6: str) -> tuple[str, str, str]:
    requested_output = output_file or default_json_1_6
    base, ext = os.path.splitext(requested_output)
    ext_lower = ext.lower()

    if ext_lower == ".json":
        output_file_1_6 = requested_output
        output_base = _strip_1_6_suffix(base)
        return output_file_1_6, f"{output_base}_1_7.json", f"{output_base}.html"

    if ext_lower == ".html":
        return f"{base}_1_6.json", f"{base}_1_7.json", requested_output

    if not ext:
        return f"{requested_output}_1_6.json", f"{requested_output}_1_7.json", f"{requested_output}.html"

    return requested_output, f"{base}_1_7{ext}", f"{base}.html"


class CLIController:
    def __init__(self):
        self.service = AIBOMService()

    def _validate_spdx_schema_version(self, aibom_data: dict, spec_version: str):
        """
        Validates the AIBOM data using the official spdx-tools package.
        """
        try:
            # Map aibom versions to spdx_tools expected format e.g. "SPDX-2.3"
            spdx_version_str = f"SPDX-{spec_version}"

            try:
                parser = JsonLikeDictParser()
                document = parser.parse(aibom_data)
            except SPDXParsingError as e:
                logger.error("SPDX parsing failed for version %s: %s", spec_version, e)
                return False

            validation_messages = validate_full_spdx_document(document, spdx_version_str)
            if validation_messages:
                for msg in validation_messages:
                    logger.error("SPDX validation error: %s", msg.validation_message)
                return False

            logger.info("SPDX schema validation successful for version %s", spec_version)
            return True
        except Exception as e:
            logger.error("Failed to validate SPDX schema for version %s: %s", spec_version, e)
            return False

    def generate(self, model_id: str, output_file: Optional[str] = None, include_inference: bool = False, 
                 enable_summarization: bool = False, verbose: bool = False,
                 name: Optional[str] = None, version: Optional[str] = None, manufacturer: Optional[str] = None):
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info("Generating AIBOM for %s...", model_id)

        versions_to_generate = ["1.6", "1.7"]
        reports = []
        generated_aiboms = {}

        logger.info("Generating AIBOM model data...")
        try:
            primary_aibom = self.service.generate_aibom(
                model_id, 
                include_inference=include_inference, 
                enable_summarization=enable_summarization,
                metadata_overrides={
                    "name": name,
                    "version": version,
                    "manufacturer": manufacturer
                }
            )
            primary_report = self.service.get_enhancement_report()
            
            # Formatted AIBOM Strings
            json_1_6 = export_aibom(primary_aibom, bom_type="cyclonedx", spec_version="1.6")
            json_1_7 = export_aibom(primary_aibom, bom_type="cyclonedx", spec_version="1.7")

            # Determine output filenames
            normalized_id = self.service._normalise_model_id(model_id)
            os.makedirs("sboms", exist_ok=True)
            default_output_file = os.path.join("sboms", f"{normalized_id.replace('/', '_')}_ai_sbom_1_6.json")
            output_file_1_6, output_file_1_7, html_output_file = _resolve_output_files(
                output_file,
                default_output_file,
            )

            _ensure_parent_dir(output_file_1_6)
            _ensure_parent_dir(output_file_1_7)
            _ensure_parent_dir(html_output_file)

            with open(output_file_1_6, 'w', encoding="utf-8") as f:
                f.write(json_1_6)
            with open(output_file_1_7, 'w', encoding="utf-8") as f:
                f.write(json_1_7)
            
            # Check for validation results
            validation_data = primary_report.get("final_score", {}).get("validation", {})
            is_valid = validation_data.get("valid", True)
            validation_errors = [i["message"] for i in validation_data.get("issues", [])]
            
            if "schema_validation" not in primary_report:
                primary_report["schema_validation"] = {}
            primary_report["schema_validation"]["valid"] = is_valid
            primary_report["schema_validation"]["errors"] = validation_errors
            primary_report["schema_validation"]["error_count"] = len(validation_errors)

            reports = [
                {"spec_version": "1.6", "output_file": output_file_1_6, "schema_validation": primary_report["schema_validation"]},
                {"spec_version": "1.7", "output_file": output_file_1_7, "schema_validation": primary_report["schema_validation"]}
            ]
            output_file_primary = output_file_1_6

        except Exception as e:
            logger.error("Failed to generate SBOM: %s", e, exc_info=True)
            reports = []

        if reports:
            if output_file_primary:
                try:
                    from jinja2 import Environment, FileSystemLoader, select_autoescape
                    from ..config import TEMPLATES_DIR
                    
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

                    context = {
                        "request": None,
                        "filename": os.path.basename(output_file_primary),
                        "download_url": "#",
                        "aibom": primary_aibom,
                        "components_json": components_json,
                        "aibom_cdx_json_1_6": json_1_6,
                        "aibom_cdx_json_1_7": json_1_7,
                        "raw_aibom": primary_aibom,
                        "model_id": self.service._normalise_model_id(model_id),
                        "sbom_count": 0,
                        "completeness_score": completeness_score,
                        "enhancement_report": primary_report or {},
                        "result_file": "#",
                        "static_root": "static" 
                    }

                    html_content = template.render(context)
                    html_temp_file = f"{html_output_file}.tmp"
                    with open(html_temp_file, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    os.replace(html_temp_file, html_output_file)
                    
                    logger.info("HTML Report: %s", html_output_file)

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
                            shutil.copytree(static_src, static_dst, dirs_exist_ok=True)
                            logger.debug("Static assets copied to: %s", static_dst)
                        else:
                            logger.warning("Static source directory not found: %s", static_src)

                    except Exception as e:
                        logger.warning("Failed to copy static assets: %s", e)

                    # Model Description
                    if "components" in primary_aibom and primary_aibom["components"]:
                        description = primary_aibom["components"][0].get("description", "No description available")
                        if len(description) > 256:
                            description = description[:253] + "..."
                        logger.info("Model Description: %s", description)

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
                                logger.info("License: %s", ", ".join(license_list))
                                
                except Exception as e:
                    if "html_temp_file" in locals() and os.path.exists(html_temp_file):
                        try:
                            os.remove(html_temp_file)
                        except OSError:
                            logger.debug("Failed to remove temporary HTML report: %s", html_temp_file, exc_info=True)
                    logger.warning("Failed to generate HTML report: %s", e, exc_info=True)

            for r in reports:
                spec = r.get("spec_version", "1.6")
                logger.info("Successfully generated CycloneDX %s SBOM: %s", spec, r.get("output_file"))

                if not r["schema_validation"]["valid"]:
                    logger.warning("Schema Validation Errors (%s):", spec)
                    for err in r["schema_validation"]["errors"]:
                        logger.warning("  - %s", err)
                else:
                    logger.info("Schema Validation (%s): Valid", spec)

            # Display Detailed Score Summary (from primary)
            if primary_report and "final_score" in primary_report:
                score = primary_report["final_score"]
                t_score = score.get('total_score', 0)
                formatted_t_score = int(t_score) if isinstance(t_score, (int, float)) and t_score == int(t_score) else t_score
                logger.info("Completeness Score: %s/100", formatted_t_score)

                if "completeness_profile" in score:
                    profile = score["completeness_profile"]
                    logger.info("Profile: %s - %s", profile.get("name"), profile.get("description"))

                if "section_scores" in score:
                    logger.info("Section Breakdown:")
                    for section, s_score in score["section_scores"].items():
                        max_s = score.get("max_scores", {}).get(section, "?")
                        formatted_s_score = int(s_score) if isinstance(s_score, (int, float)) and s_score == int(s_score) else s_score
                        logger.info("  %s: %s/%s", section.replace("_", " ").title(), formatted_s_score, max_s)

        else:
            logger.error("Failed to generate any SBOMs.")
