# ParseBench

[![arXiv](https://img.shields.io/badge/arXiv-2604.08538-b31b1b.svg)](https://arxiv.org/abs/2604.08538)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/llamaindex/ParseBench)
[![ParseBench 2](https://img.shields.io/badge/HuggingFace-ParseBench_2-orange)](https://huggingface.co/datasets/llamaindex/ParseBench-2)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**ParseBench** is a benchmark for document parsing tools: how well they turn PDFs into structured output that AI agents can act on. It is built by LlamaIndex:

- Upstream code: [run-llama/ParseBench](https://github.com/run-llama/ParseBench)
- ParseBench v1 dataset (~2,000 pages): [`llamaindex/ParseBench`](https://huggingface.co/datasets/llamaindex/ParseBench)
- ParseBench 2 dataset (4,317 pages): [`llamaindex/ParseBench-2`](https://huggingface.co/datasets/llamaindex/ParseBench-2)

This repository adds five **URL extract** pipelines for web extract APIs that take a public URL instead of a file upload: you.com Extract, Exa, Parallel and Tavily. This README covers how to run ParseBench evals with these and other providers.

**[Latest results](docs/results.md)** — headline accuracy for ParseBench v1 & 2 across the URL extract providers.

- [Published results for the URL extract providers](#published-results-for-the-url-extract-providers)
- [Install](#install)
- [Setup and API keys](#setup-and-api-keys)
- [Quick smoke test](#quick-smoke-test)
- [URL extract providers](#url-extract-providers)
- [Running ParseBench v1](#running-parsebench-v1)
- [Running ParseBench 2](#running-parsebench-2)
- [Rescoring without new API calls](#rescoring-without-new-api-calls)
- [How URL extract providers work](#how-url-extract-providers-work)
- [Outputs and reports](#outputs-and-reports)
- [Running other pipelines](#running-other-pipelines)
- [About the datasets](#about-the-datasets)
- [Citation](#citation) and [License](#license)

## Published results for the URL extract providers

Per-document outputs of the five URL extract pipelines added in this repository (`youcom_extract`, `youcom_extract_lite`, `tavily_extract`, `exa_contents`, `parallel_extract`) are published in the record format of the upstream results repos, so they can be inspected or re-scored without calling any provider:

- [youdotcom/parsebench-results](https://huggingface.co/datasets/youdotcom/parsebench-results): ParseBench v1 (`chart`, `table`, `text`); eval records with each pipeline's markdown under `results/`, full provider responses under `raw/`.
- [youdotcom/parsebench-2-results](https://huggingface.co/datasets/youdotcom/parsebench-2-results): ParseBench 2 (every subset except `layout`); eval records with each pipeline's markdown under `results/`.

These pipelines are not `leaderboard.csv` rows, so re-score them with `scripts/leaderboard/rescore.py` and a results map that points at the dataset (the map's top-level `results_repo` selects the repo; there is no CLI flag for it). Set `HF_TOKEN` if the results repo is private.

ParseBench v1:

```bash
cat > /tmp/youdotcom-results.yaml <<'EOF'
results_repo: youdotcom/parsebench-results
data_repo: llamaindex/ParseBench
data_revision: 2805a1d940f95a203e0ae4b88be9934f7765b3fc
scored_with: null
rows:
- {provider: You.com Extract, status: verified, pipeline: youcom_extract, results_revision: c34b1b96dbab99a92c28f2beaa256f56f035dcdf}
- {provider: You.com Extract Lite, status: verified, pipeline: youcom_extract_lite, results_revision: c34b1b96dbab99a92c28f2beaa256f56f035dcdf}
- {provider: Tavily Extract, status: verified, pipeline: tavily_extract, results_revision: c34b1b96dbab99a92c28f2beaa256f56f035dcdf}
- {provider: Exa Contents, status: verified, pipeline: exa_contents, results_revision: c34b1b96dbab99a92c28f2beaa256f56f035dcdf}
- {provider: Parallel Extract, status: verified, pipeline: parallel_extract, results_revision: c34b1b96dbab99a92c28f2beaa256f56f035dcdf}
EOF
uv run python scripts/leaderboard/rescore.py run --map /tmp/youdotcom-results.yaml \
  --pipelines youcom_extract_lite --out rescore.json
```

ParseBench 2 runs from this repository with `--board v2`:

```bash
cat > /tmp/youdotcom-results-v2.yaml <<'EOF'
results_repo: youdotcom/parsebench-2-results
data_repo: llamaindex/ParseBench-2
data_revision: a12c9785cd8cd015b5284e0ad0d74722091762f2
scored_with: null
rows:
- {provider: You.com Extract, status: verified, pipeline: youcom_extract, results_revision: e593510105fe44281564fadea48e626cb0afbd14}
- {provider: You.com Extract Lite, status: verified, pipeline: youcom_extract_lite, results_revision: e593510105fe44281564fadea48e626cb0afbd14}
- {provider: Tavily Extract, status: verified, pipeline: tavily_extract, results_revision: e593510105fe44281564fadea48e626cb0afbd14}
- {provider: Exa Contents, status: verified, pipeline: exa_contents, results_revision: e593510105fe44281564fadea48e626cb0afbd14}
- {provider: Parallel Extract, status: verified, pipeline: parallel_extract, results_revision: e593510105fe44281564fadea48e626cb0afbd14}
EOF
uv run python scripts/leaderboard/rescore.py run --board v2 --map /tmp/youdotcom-results-v2.yaml \
  --pipelines youcom_extract_lite --out rescore-v2.json
```

Pass any comma-separated subset of the five pipelines to `--pipelines`. The first run downloads the ground truth into `.cache/leaderboard-gt/`; after that, a ParseBench v1 pipeline takes roughly 10–15 minutes. Scores land in `rescore.json` under `results.<pipeline>`: `scores` (one value per leaderboard column), `overall`, and `details.<column>` (published vs. ground-truth page counts and any error). `layout` was not run for these pipelines, so v1 `Visual_Grounding` is 0, and ParseBench 2 reports `Visual_Grounding` as an error with `overall` null. Pages listed in `results/<pipeline>/failed.json` have no record and score 0. `apply.py` is not needed, since it only updates `leaderboard.csv` rows.

## Install

```bash
# From a checkout of this repo (the URL extract pipelines exist only in this repository)
uv sync --extra runners    # every provider SDK; at minimum --extra local, which provides pypdf for page counts

# Optional: the `fast` extra adds a JIT-accelerated TEDS table metric (numba). Scores are identical.
uv sync --extra runners --extra fast
```

Commands below use `uv run parse-bench ...`. `uv run parse-bench --help` lists every subcommand.

## Setup and API keys

```bash
cp .env.example .env    # then fill in the keys for the providers you plan to run
```

The CLI loads `.env` on startup, and `.env` is gitignored. You only need keys for the providers you run:

| Provider | Env var |
|---|---|
| you.com (`youcom_extract`, `youcom_extract_lite`) | `YOU_API_KEY` |
| Exa | `EXA_API_KEY` |
| Parallel | `PARALLEL_API_KEY` |
| Tavily | `TAVILY_API_KEY` |
| Everything else | see [`.env.example`](.env.example) (keys grouped by pipeline family) and [Running other pipelines](#running-other-pipelines) |

Evaluation is deterministic and rule-based by default; keys are only used to call the tool being evaluated. Two optional environment variables affect scoring:

| Variable | Default | Description |
|---|---|---|
| `PARSEBENCH_FAST_TEDS` | `1` | Fast TEDS table metric (numba JIT with the `fast` extra, else exact pure-Python fallback). `0` forces the original APTED implementation. Scores are identical. |
| `LLAMACLOUD_BENCH_LLM_NORMALIZATION` | `off` | `judge` opts into LLM-as-judge chart normalization (non-deterministic, needs `ANTHROPIC_API_KEY`). |

The local dataset is needed for ground truth and page counts even for URL providers. `run` downloads it on first use, or fetch it up front:

```bash
uv run parse-bench download                          # ParseBench v1 into ./data
uv run parse-bench download --test                   # 3 files per category into ./data/test
uv run parse-bench download --dataset parsebench-2   # ParseBench 2 into ./data_parsebench-2
uv run parse-bench status                            # also accepts --test / --dataset parsebench-2
```

## Quick smoke test

Run one provider on the small ParseBench v1 test set:

```bash
uv run parse-bench run youcom_extract --test --max_concurrent 2
```

This runs inference, scores the results and opens the report (add `--open_report False` to skip the browser). The ParseBench 2 dataset has no `--test` subset; for a quick ParseBench 2 check, run one subset with `--group` (see [Running ParseBench 2](#running-parsebench-2)).

## URL extract providers

| Pipeline | API | Env var | API request args |
|---|---|---|---|
| `youcom_extract` | you.com `POST https://ydc-index.io/v1/extract` with default `parse_effort: "standard"` | `YOU_API_KEY` | `url`, `full_page.extraction_formats: ["markdown", "html"]`, `timeout: 100`, `max_age: 0` (no `parse_effort` key sent → API default `"standard"`) |
| `youcom_extract_lite` | you.com `POST https://ydc-index.io/v1/extract` with `parse_effort: "lite"` | `YOU_API_KEY` | same as `youcom_extract` plus `parse_effort: "lite"` |
| `exa_contents` | Exa `POST https://api.exa.ai/contents` | `EXA_API_KEY` | `urls: [url]`, `text: true`, `maxAgeHours: 0`, `livecrawlTimeout: 60000` |
| `parallel_extract` | Parallel `POST https://api.parallel.ai/v1/extract` | `PARALLEL_API_KEY` | `urls: [url]`, `advanced_settings.full_content.max_chars_per_result: 200000`, `advanced_settings.fetch_policy.max_age_seconds: 600`, `advanced_settings.fetch_policy.timeout_seconds: 60` |
| `tavily_extract` | Tavily `POST https://api.tavily.com/extract` (advanced depth) | `TAVILY_API_KEY` | `urls: [url]`, `extract_depth: "advanced"`, `format: "markdown"`, `include_usage: true`, `timeout: 60` |

Tips for every URL extract provider:

- `--max_concurrent 10` is the recommended setting (the CLI default is 20). On a 429 the provider honors `Retry-After`, up to 60 seconds, before the runner's own backoff.
- Running several providers at once (one shell or background job each) is fine since they hit different APIs.
- These APIs return markdown only, with no bounding boxes, so the layout (visual grounding) dimension can't be scored. The commands below skip it.
- Reruns resume: documents that already have a `*.result.json` are skipped, so rerunning the same command retries only the failures. Use `--force True` to redo everything. Each inference pass writes failures to `_errors.json` in the pipeline output directory.

## Running ParseBench v1

ParseBench v1 has five groups: `chart`, `table`, `text_content`, `text_formatting` and `layout`. A plain `uv run parse-bench run <pipeline>` runs all of them, including `layout`. For URL extract providers, run inference one group at a time, then score everything without calling the APIs again:

```bash
P=youcom_extract
for g in chart table text_content text_formatting; do
  uv run parse-bench inference run $P --group $g --output_dir output --max_concurrent 10 --no_rich True
done
uv run parse-bench run $P --skip_inference True --open_report False
```

`text_content` and `text_formatting` share the same PDFs and results (stored under `text/`), so the second of those passes makes no new API calls.

Other useful `run` flags (all verified in `src/parse_bench/pipeline/cli.py`):

```bash
uv run parse-bench run <pipeline> --group chart                # one group: inference + scoring
uv run parse-bench run <pipeline> --input_dir data/my_sample   # custom dataset dir (no auto-download)
uv run parse-bench run <pipeline> --file path/to/doc.pdf       # one file, uses its .test.json if present
uv run parse-bench run <pipeline> --output_dir my_output       # default ./output
```

For a smoke test bigger than `--test`, copy a smaller set of documents into a separate data folder (e.g. `data/my_sample`) and run it with `--input_dir data/my_sample`. Keep the same layout as the full dataset (`<category>.jsonl` rows plus `docs/<category>/<name>.pdf`) so URL extract providers can still build the HuggingFace URL for each document.

## Running ParseBench 2

`--dataset parsebench-2` downloads [`llamaindex/ParseBench-2`](https://huggingface.co/datasets/llamaindex/ParseBench-2) to `./data_parsebench-2`, runs inference per subset and writes the combined score:

```bash
# All 13 subsets
uv run parse-bench run <pipeline> --dataset parsebench-2 --max_concurrent 10

# URL extract providers: every subset except layout
uv run parse-bench run youcom_extract --dataset parsebench-2 --max_concurrent 10 --open_report False \
  --group text,headings,strikeout,underline,tables,complex_table_headers,financial_tables,insurance_filing_tables,dense_tables,charts,forms,fillable_forms

# One or a few subsets; every subset that already has results is still scored
uv run parse-bench run <pipeline> --dataset parsebench-2 --group charts,tables
```

With `--dataset parsebench-2`, `--group` takes comma-separated **subset** names (not the v1 group names). `--test` is rejected. Results go to `output/parsebench-2/<subset>/<pipeline>/` and the score to `output/parsebench-2/_score/<pipeline>/_next_score.json`.

The overall score is the unweighted mean of six dimensions. Within a dimension, page scores are averaged per subset and subsets are combined with fixed weights:

| Dimension | Subsets (`--group` name) | Cases | Metric |
|---|---|---:|---|
| Text | `text` | 508 | `text_next` |
| Formatting | `text` (3/4), `headings` (1/8), `strikeout` (1/16), `underline` (1/16) | 508 + 353 | `formatting_v2` |
| Tables | `tables` (1/2), `complex_table_headers`, `financial_tables`, `insurance_filing_tables`, `dense_tables` (1/8 each) | 1,753 | `table_next` |
| Charts | `charts` | 568 | `chart_next` |
| Layout | `layout` | 535 | `layout_grounding_pass_rate` |
| Forms | `forms` (7/8), `fillable_forms` (1/8) | 600 | `forms_v2` |

A dimension with a missing subset is reported as missing, and so is the overall score unless all six dimensions are present. URL extract runs that skip `layout` therefore get five dimension scores and no overall score.

To score outputs from your own parser, write them in the ParseBench results layout and run `uv run parse-bench evaluation run <dir> --next=SUBSET=RESULTS_DIR::TEST_CASES_DIR,... --next_only`. ParseBench 2 scores are not comparable with ParseBench v1 scores.

## Rescoring without new API calls

`--skip_inference` scores existing results only. Add `--force True` to re-evaluate results that were already scored.

```bash
uv run parse-bench run <pipeline> --skip_inference True --open_report False                          # ParseBench v1
uv run parse-bench run <pipeline> --dataset parsebench-2 --skip_inference True --open_report False   # ParseBench 2
```

To re-normalize saved raw responses (for example after a pricing or normalization fix), then rescore:

```bash
uv run parse-bench inference renormalize output/<pipeline> --force True
```

`renormalize` rebuilds each `*.result.json` from its `*.raw.json` and recomputes cost with the pipeline's current config. It builds the provider, so the provider's API key must still be set.

### Rescore with a Metric Lab policy

Set your weights and error classes in [Metric Lab](https://www.parsebench.ai/lab/index.html),
then click **Export policy JSON**. Use the downloaded file with saved evaluation results:

```bash
# Generate extended scores and retain the counts needed for later policy changes.
uv run parse-bench evaluation run ./output/parser --next=text --next_only
uv run parse-bench rescore ./output/parser --policy parsebench-policy.json --next=text

# Combine separately evaluated subsets using the same mapping as evaluation.
uv run parse-bench rescore ./output/parser --policy parsebench-policy.json \
  --next='text=text-results,tables=table-results,charts=chart-results'
```

`rescore` reads `_evaluation_report.json` in each directory and writes `_policy_score.json`.
Use `--report_dir` to choose its destination. The output contains the policy, per-page scores,
dimension scores and overall score, all on a 0–1 scale. Original evaluation reports stay unchanged.
It makes no inference requests. Missing dimensions with positive weights leave the overall score
unavailable; dimensions weighted zero are excluded.

The versioned JSON includes severity weights, dimension weights, scaling constants and error classes.
Older reports without exact counts must be regenerated with `evaluation run --next --next_only`
using saved parser outputs. The CLI preserves exact mixed-severity counts and per-chart omission
limits, so custom scores can differ from Metric Lab's legacy data where those details were estimated.

## How URL extract providers work

The shared logic is in `src/parse_bench/inference/providers/parse/_url_extract_base.py`. Each provider module only builds the request body, picks the markdown out of the response and computes cost.

### How documents are sent

ParseBench gives every provider a local PDF path. These APIs can't accept a file, so the provider maps the path to the same file in the public Hugging Face dataset:

| Dataset | Local path | URL |
|---|---|---|
| ParseBench v1 | `.../docs/<category>/<name>.pdf` | `https://huggingface.co/datasets/llamaindex/ParseBench/resolve/<hf_revision>/docs/<category>/<name>.pdf` |
| ParseBench 2 | `.../data_parsebench-2/<subset>/<name>.pdf` | `https://huggingface.co/datasets/llamaindex/ParseBench-2/resolve/a12c9785cd8cd015b5284e0ad0d74722091762f2/<subset>/<name>.pdf` |

For ParseBench v1, the `hf_revision` pipeline config key defaults to `main`. ParseBench 2 is always pinned to the revision above (`HF_REVISION_V2`) so URLs stay reproducible; `hf_revision` is ignored there. Paths are NFD-normalized and percent-encoded, because Hugging Face matches paths byte for byte and stores accented filenames in decomposed form.

This only works if the runner passes the real dataset path, so every URL extract pipeline sets `randomize_external_filename: False`. Without it the runner copies each PDF to a randomly named temp file and the URL can't be derived. Keep the flag if you register a variant. Consequences:

- Local documents that aren't in the Hugging Face datasets can't be evaluated with these pipelines.
- A custom `--input_dir` must keep the `docs/<category>/<name>.pdf` shape (v1) or sit under a directory named `data_parsebench-2` (ParseBench 2).

### What is scored

The whole document comes back as one markdown string, stored as one page. Before evaluation, markdown pipe tables are converted to HTML with the converter the Mistral OCR provider uses (`_convert_pipe_tables_to_html` in `mistral_ocr.py`), so the table metrics see real table structure. `layout_pages` is always empty.

| ParseBench v1 group | Headline metric |
|---|---|
| `chart` | ChartDataPointMatch |
| `table` | GTRM (GriTS + TableRecordMatch) |
| `text_content` | Content Faithfulness Score |
| `text_formatting` | Semantic Formatting Score |
| `layout` | Not scored for URL extract providers |

For ParseBench 2, all dimensions except Layout are scored (see the table in [Running ParseBench 2](#running-parsebench-2)).

### Freshness

Each request asks for a fresh fetch where the API allows it, so cached copies from earlier runs don't hide changes in extraction quality:

| Provider | Setting |
|---|---|
| you.com | `max_age: 0` |
| Exa | `maxAgeHours: 0` (live crawl) |
| Parallel | `fetch_policy.max_age_seconds: 600`, the API's minimum |
| Tavily | No cache control available |

## Outputs and reports

ParseBench v1 results land in `output/<pipeline>/` (gitignored):

```
output/
├── _leaderboard.html                        # Leaderboard across all pipelines in output/
└── <pipeline>/
    ├── chart/ table/ text/ layout/           # *.raw.json (API response) and *.result.json (normalized output)
    ├── chart/ table/ text_content/ text_formatting/
    │   ├── _evaluation_report.json / .md     # Per-group scores
    │   ├── _evaluation_report_detailed.html  # Interactive per-document report
    │   └── _evaluation_results.csv
    ├── _evaluation_report_dashboard.html     # Dashboard across groups
    ├── _summary.json / _errors.json          # Inference summary and failures from the last pass
    └── _metadata.json
```

ParseBench 2 results land in `output/parsebench-2/<subset>/<pipeline>/` with the same per-document files, plus the combined score in `output/parsebench-2/_score/<pipeline>/_next_score.json`.

```bash
uv run parse-bench serve <pipeline>                     # view reports (browsers block PDFs from file:// URLs)
uv run parse-bench serve output/parsebench-2/charts/<pipeline>
uv run parse-bench compare <pipeline_a> <pipeline_b>    # side-by-side comparison
uv run parse-bench leaderboard                          # leaderboard across all pipelines in ./output
uv run parse-bench leaderboard youcom_extract exa_contents
```

<details>
<summary><strong>Advanced subcommands</strong></summary>

```bash
uv run parse-bench inference run <pipeline>                                    # inference only
uv run parse-bench evaluation run --output_dir ./output/<pipeline>             # evaluation only
uv run parse-bench analysis generate_report --evaluation_dir ./output/<pipeline>
uv run parse-bench analysis generate_dashboard --evaluation_dir ./output/<pipeline>
```

</details>

## Running other pipelines

About 260 pipelines are registered, covering LlamaParse, OpenAI, Anthropic, Gemini, Azure, Textract, Reducto, self-hosted VLMs and more. They upload the local file, so none of the URL-specific notes above apply, and a plain `run` covers every group including layout:

```bash
uv run parse-bench pipelines                                   # list every pipeline name and its provider
uv run parse-bench run llamaparse_agentic --test               # small test set
uv run parse-bench run llamaparse_agentic                      # ParseBench v1
uv run parse-bench run llamaparse_agentic --dataset parsebench-2
```

`parse-bench pipelines` prints one table per product type (parse, layout detection, ...) with each pipeline name and provider. Pass the name to `run`. Pipelines fall into four kinds:

- **Cloud APIs** need an API key in `.env`.
- **Self-hosted models** (vLLM, Modal and similar) need an endpoint URL in `.env`, such as `DOCLING_PARSE_ENDPOINT_URL` or `QWEN35_SERVER_URL`.
- **Local pipelines** need no key. Some need a binary or extra installed, for example `tesseract` for `tesseract_*`.
- **Layout detection pipelines** score layout only.

[docs/pipelines.md](docs/pipelines.md) lists the pipelines by kind, including the URL extract providers, with their env vars. `.env.example` lists every key and endpoint variable, with the pipelines that use it. The provider SDKs come from `uv sync --extra runners`, or one extra per provider (for example `--extra llamaparse`, `openai`, `anthropic`, `google`, `azure`, `aws`, `reducto`). Common examples:

| Pipeline | Kind | Env vars |
|---|---|---|
| `llamaparse_agentic`, `llamaparse_cost_effective` | Cloud API | `LLAMA_CLOUD_API_KEY` |
| `openai_gpt5_mini_reasoning_medium_parse_with_layout_file` | Cloud API | `OPENAI_API_KEY` |
| `anthropic_haiku_5_5_parse_with_layout_file` | Cloud API | `ANTHROPIC_API_KEY` |
| `google_gemini_3_flash_thinking_minimal_parse_file` | Cloud API | `GOOGLE_GEMINI_API_KEY` |
| `azure_di_layout` | Cloud API | `AZURE_DOCUMENT_INTELLIGENCE_KEY`, `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` |
| `aws_textract` | Cloud API | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_REGION` (default `us-east-1`) |
| `reducto` | Cloud API | `REDUCTO_API_KEY` |
| `docling_parse` | Self-hosted | `DOCLING_PARSE_ENDPOINT_URL`, optional `DOCLING_PARSE_API_KEY` |
| `pypdf_baseline`, `pymupdf_text` | Local | none |

To add a provider, see [docs/extending.md](docs/extending.md) or the [`/integrate-pipeline`](.claude/commands/integrate-pipeline.md) Claude Code command. Any Hugging Face dataset in the ParseBench layout runs with `--dataset org/name`; for the repetition rule, see [docs/repetition-overshoot.md](docs/repetition-overshoot.md).

## About the datasets

**ParseBench v1** ([`llamaindex/ParseBench`](https://huggingface.co/datasets/llamaindex/ParseBench)) covers ~2,000 human-verified pages from enterprise documents (insurance, finance, government) in five dimensions:

| Dimension | File | Metric | Pages | Docs | Rules |
|---|---|---|---:|---:|---:|
| Tables | `table.jsonl` | GTRM (GriTS + TableRecordMatch) | 503 | 284 | --- |
| Charts | `chart.jsonl` | ChartDataPointMatch | 568 | 99 | 4,864 |
| Content Faithfulness | `text_content.jsonl` | Content Faithfulness Score | 506 | 506 | 141,322 |
| Semantic Formatting | `text_formatting.jsonl` | Semantic Formatting Score | 476 | 476 | 5,997 |
| Visual Grounding | `layout.jsonl` | Element Pass Rate | 500 | 321 | 16,325 |
| **Total (unique)** | | | **2,078** | **1,211** | **169,011** |

- **Tables**: merged cells and hierarchical headers. A misaligned header means the agent reads the wrong column.
- **Charts**: exact data points with correct series and axis labels from bar, line, pie and compound charts.
- **Content Faithfulness**: omissions, hallucinations and reading-order violations.
- **Semantic Formatting**: formatting that carries meaning (strikethrough, superscript/subscript, bold, title hierarchy).
- **Visual Grounding**: tracing each extracted element back to its location on the page.

**ParseBench 2** ([`llamaindex/ParseBench-2`](https://huggingface.co/datasets/llamaindex/ParseBench-2)) has 4,317 page-level test cases from 2,645 documents: the revised v1 pages with more complete answer keys, a forms dimension (600 pages) and targeted sets of hard cases. A page score is the lower of a content-recovery score and a severity cap, so recovering most of a page cannot hide a serious error such as a changed number or swapped table headers. The metrics are calibrated against 330 controlled scenarios covering over 140 error types.

<p align="center">
  <img src="docs/severity_scoring.png" alt="ParseBench 2 severity-aware scoring" width="100%">
</p>

## Citation

```bibtex
@misc{zhang2026parsebench,
  title={ParseBench: A Document Parsing Benchmark for AI Agents},
  author={Boyang Zhang and Sebastián G. Acosta and Preston Carlson and Sacha Bron and Pierre-Loïc Doulcet and Daniel B. Ospina and Simon Suo},
  year={2026},
  eprint={2604.08538},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.08538},
}
```

Links: [paper (arXiv:2604.08538)](https://arxiv.org/abs/2604.08538) · [upstream code](https://github.com/run-llama/ParseBench) · [website](https://parsebench.ai) · [ExtractBench](https://github.com/run-llama/ExtractBench), the companion benchmark for schema-guided extraction.

## License

Apache 2.0, see [LICENSE](LICENSE).
