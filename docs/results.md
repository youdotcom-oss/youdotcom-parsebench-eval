# ParseBench Results

Headline accuracy for the URL extract pipelines added in this repository.

- **ParseBench v1** ([`llamaindex/ParseBench`](https://huggingface.co/datasets/llamaindex/ParseBench)): ~2,000 human-verified pages across tables, charts, content faithfulness, semantic formatting and visual grounding.
- **ParseBench 2** ([`llamaindex/ParseBench-2`](https://huggingface.co/datasets/llamaindex/ParseBench-2)): 4,317 page-level test cases from 2,645 documents, with severity-aware scoring across six dimensions.

## ParseBench v1

Full run on the 1,578 non-layout PDFs (568 chart, 503 table, 506 text_content, 476 text_formatting; the text groups share PDFs). Layout / visual grounding excluded — URL extract APIs return markdown only, no bounding boxes. Five providers run 2026-11-04; you.com lite 2026-11-09; all re-scored 2026-11-12 with harness v2.0.1.

| Pipeline | Tables | Charts | Content Faithfulness | Semantic Formatting | Mean of 4 groups |
|---|---:|---:|---:|---:|---:|
| `youcom_extract` | 77.4 | 76.2 | 87.3 | 63.5 | **76.1** |
| `youcom_extract_lite` | 26.9 | 1.6 | 78.1 | 43.4 | 37.5 |
| `exa_contents` | 26.8 | 0.8 | 79.8 | 11.7 | 29.7 |
| `parallel_extract` | 0.0 | 0.0 | 77.3 | 41.1 | 29.6 |
| `tavily_extract` | 0.0 | 0.0 | 62.3 | 12.7 | 18.8 |

Scores are group means over all docs in the group, errored docs counted as 0. Mean of 4 groups = unweighted average of the four headline scores.

## ParseBench 2

Full run on 3,782 unique PDFs across 12 of the 13 subsets (layout excluded), run 2026-11-12, dataset pinned at revision `a12c9785cd8c`. Scored on five dimensions; the official six-dimension overall needs Layout and is not reported.

| Pipeline | Text | Formatting | Tables | Charts | Forms | Mean of 5 dims |
|---|---:|---:|---:|---:|---:|---:|
| `youcom_extract` | 79.9 | 86.2 | 74.9 | 76.6 | 75.5 | **78.6** |
| `youcom_extract_lite` | 61.0 | 80.7 | 23.9 | 1.5 | 65.4 | 46.5 |
| `exa_contents` | 60.1 | 69.7 | 22.8 | 0.9 | 62.6 | 43.2 |
| `parallel_extract` | 61.1 | 69.3 | 0.1 | 0.0 | 61.0 | 38.3 |
| `tavily_extract` | 52.0 | 56.0 | 0.1 | 0.0 | 60.6 | 33.7 |

Errored docs scored as 0. ParseBench 2 scores are not comparable with ParseBench v1: different documents, metrics and dimension weights.

## Full results and reproduction

- Published per-document outputs: [`youdotcom/parsebench-results`](https://huggingface.co/datasets/youdotcom/parsebench-results) (v1) and [`youdotcom/parsebench-2-results`](https://huggingface.co/datasets/youdotcom/parsebench-2-results) (ParseBench 2).
- Reproduce or re-score: see [Published results for the URL extract providers](../README.md#published-results-for-the-url-extract-providers) in the README, [Running ParseBench v1](../README.md#running-parsebench-v1) and [Running ParseBench 2](../README.md#running-parsebench-2). Pipeline list: [pipelines.md](pipelines.md).
