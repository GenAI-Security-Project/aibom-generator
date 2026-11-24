# Model File Hashes

The generator records a SHA-256 for each file in the model repository at the commit the AIBOM describes ([issue #138](https://github.com/GenAI-Security-Project/aibom-generator/issues/138)). A consumer can then check that the files a host loads are the files the AIBOM declares.

## What is recorded

Each file is a `file` component nested under the `machine-learning-model` component:

```json
{
  "type": "file",
  "name": "config.json",
  "bom-ref": "pkg:huggingface/Qwen/Qwen3-0.6B@c1899de2#config.json",
  "hashes": [{ "alg": "SHA-256", "content": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd" }],
  "properties": [
    { "name": "owasp:aibom:file:hashSource", "value": "computed" },
    { "name": "owasp:aibom:file:size", "value": "726" }
  ]
}
```

- **Scope.** Every file in the repository except documentation (README, LICENSE, NOTICE, `.gitattributes`, Markdown and image files). This covers weights, tokenizer files, config and generation config, the chat template wherever it lives, and custom code. It also covers whatever else the repository ships, such as ONNX exports or training logs: which files a runtime loads depends on the runtime, and the verification below only checks the files a host actually has.
- **Revision.** Files are listed and downloaded at the full commit SHA from the Hub, the same commit as the model component's `version` and its commit external reference. Without a commit, no file hashes are recorded: hashes taken from a branch are not reproducible.
- **`owasp:aibom:file:hashSource`**
  - `hf-lfs`: the `lfs.sha256` the Hugging Face API reports for an LFS file. It is the SHA-256 of the file content.
  - `computed`: for a non-LFS file the API only reports the git blob SHA-1, which is not a hash of the content, so the generator downloads the file at the commit and hashes it.
  - `unknown`: no hash could be obtained. The file is still listed, without `hashes`, and `owasp:aibom:file:hashUnavailableReason` says why (for example a gated repository without access, or a failed download).

The properties use the `owasp:aibom:` namespace, like the distribution properties, because they are not part of the official `genai:aibom` taxonomy.

For a gated repository without a token, LFS files still get their hash from the API, but non-LFS files can't be downloaded. Those are often the files with the chat template (`tokenizer_config.json`, `chat_template.jinja`), so pass a token with access to get a complete set.

## Verifying files against the AIBOM

`scripts/verify_model_files.py` uses only the Python standard library, so it can run on the host that loads the model. It hashes every file in a model directory and looks each one up in the AIBOM.

```bash
# 1. Generate the AIBOM
python -m src.cli HuggingFaceTB/SmolLM2-135M-Instruct -o smollm2.json

# 2. Get the files at the commit the AIBOM records
REV=$(python -c "import json,sys; c=json.load(open(sys.argv[1]))['components'][0]; print(next(r['url'].rsplit('/',1)[1] for r in c['externalReferences'] if '/commit/' in r['url']))" smollm2.json)
hf download HuggingFaceTB/SmolLM2-135M-Instruct --revision "$REV" --local-dir smollm2 \
  --include "*.json" --include "*.txt" --include "model.safetensors"

# 3. Verify
python scripts/verify_model_files.py smollm2.json smollm2
```

Each local file gets one status:

| Status | Meaning |
|---|---|
| `MATCH` | declared, and the SHA-256 equals the recorded one |
| `MISMATCH` | declared, but the content differs from what the AIBOM records |
| `UNVERIFIED` | declared without a hash (`hashSource` unknown), so it can't be checked |
| `UNDECLARED` | not in the AIBOM at all |

Declared files that are not in the directory are only counted, because an AIBOM lists the whole repository and a host usually loads one variant (for example the safetensors weights and not the ONNX exports). Hidden files and directories, such as the `.cache` directory that `hf download --local-dir` creates, are skipped. The exit code is 0 when every local file is `MATCH` and 1 otherwise.

Example after changing the chat template in `tokenizer_config.json` and adding a file the AIBOM doesn't declare (exit code 1):

```
MATCH       all_results.json
MATCH       config.json
MATCH       eval_results.json
UNDECLARED  extra_template.jinja
MATCH       generation_config.json
MATCH       merges.txt
MATCH       model.safetensors
MATCH       special_tokens_map.json
MATCH       tokenizer.json
MISMATCH    tokenizer_config.json
MATCH       train_results.json
MATCH       trainer_state.json
MATCH       vocab.json
match 11, mismatch 1, unverified 0, undeclared 1; declared but not present 11
```

## Limits

- The check covers files on disk. It says nothing about what a runtime does after loading them, or about claims on training data.
- The recorded hashes are only as trustworthy as the AIBOM itself. Integrity of the AIBOM document is a separate concern (see #131).
