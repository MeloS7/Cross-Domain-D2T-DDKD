# Data sources and attribution

The release contains prepared inputs, saved model
outputs, and evaluation annotations for *Cross-Domain, Multi-Task Data-to-Text
Generation without In-Domain Training Data* (Song, Efimov-Zhang, and Gardent,
Findings of EMNLP 2026).

## Upstream resources

| Released resource | Source and attribution | Terms |
| --- | --- | --- |
| WebNLG (used for source-domain supervision; obtain separately) | Gardent et al.; [official WebNLG corpus](https://gitlab.com/shimorina/webnlg-dataset), release 3.0 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), as stated in the [upstream README](https://gitlab.com/shimorina/webnlg-dataset/-/blob/master/README.md). WebNLG copies are omitted from this release. |
| QUINTD-1 test inputs (`data/test/quintd/`) | Kasner and Dušek; [QUINTD](https://github.com/kasnerz/quintd), introduced in [Beyond Traditional Benchmarks](https://arxiv.org/abs/2401.10186) | The upstream repository distributes its code under MIT. Underlying records originate from the providers below; that code license does not replace their data terms. |
| QUINTD-5 development inputs (`data/train/quintd5/`) | Additional records collected for this paper using the QUINTD collection procedure | The same source providers and task definitions as QUINTD-1; 500 development examples per domain. |

QUINTD source providers, following the [upstream domain list](https://github.com/kasnerz/quintd#datasets):

| Public directory name | Provider | Task |
| --- | --- | --- |
| `gsmarena` | [GSMArena](https://www.gsmarena.com/) | Product specification description |
| `ice_hockey` | [RapidAPI](https://rapidapi.com/) | Game summary |
| `owid` | [Our World in Data](https://ourworldindata.org/) | Time series caption |
| `weather` | [OpenWeather](https://openweathermap.org/) | Weather forecast reporting |
| `wikidata` | [Wikidata](https://www.wikidata.org/) | Entity description |

Consult each provider's terms and the upstream dataset documentation when
reusing or redistributing source records. No blanket code license is applied
to the data archive.

## Artifacts produced for this paper

- QUINTD-5 supplies real structured inputs; target-domain human reference texts
  and bulk teacher-generated training pairs are not supplied.
- Saved generations and LLM annotations support analysis without new API calls.
- Human annotation files use neutral numeric annotator identifiers. The released
  selection contains 60 inputs and 240 system outputs, with three annotators.
- Coverage files retain failed-parse markers from the original evaluation.
  Analysis excludes invalid values and reports valid counts.

Source-derived material retains its upstream terms. This release does not
assign a separate blanket data license to the generations or annotations.
Please cite this paper and the upstream resources used in your work.

## Packaging

The archive contains public relative paths only. `data_manifest.json` beside
`data.zip` records file sizes, record counts for JSONL files, SHA-256 checksums,
and the archive checksum. See the [data guide](DATA.md) for the inventory and
release scope. The archive supports analysis of saved results; retraining
requires separately obtained or prepared training examples.
