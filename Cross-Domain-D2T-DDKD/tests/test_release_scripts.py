from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import (  # noqa: E402
    LLM_eval_CC,
    LLM_eval_OpenRouter,
    analyze_cc,
    compute_cc_norm,
    eval_judges_agreement,
    path_utils,
    summarize_table4,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class StubClient:
    def __init__(self, result: str | Exception):
        self.result = result
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_kwargs):
        if isinstance(self.result, Exception):
            raise self.result
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.result))]
        )


class PathResolutionTests(unittest.TestCase):
    def test_resolves_case_only_filename_difference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actual = root / "Outputs" / "model_32B.jsonl"
            actual.parent.mkdir()
            actual.write_text("{}\n", encoding="utf-8")
            requested = root / "outputs" / "model_32b.jsonl"
            self.assertEqual(path_utils.resolve_case_insensitive_path(requested), str(actual))

    def test_ambiguous_case_only_matches_raise(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "Model.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "model.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                path_utils.resolve_case_insensitive_path(root / "MODEL.jsonl")

    def test_judge_discovery_pairs_case_variants_and_skips_coverage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_dir = root / "data/model_outputs/quintd/gsmarena/eval_res/zero_shot"
            output_dir = root / "data/model_outputs/quintd/gsmarena/outputs/zero_shot"
            for suffix in ("gpt-5.1", "gemini2.5-pro"):
                write_jsonl(
                    eval_dir / f"gsmarena_zero_shot_qwen3_32b_{suffix}.jsonl",
                    [{"table_idx": 0, "annotations": []}],
                )
            # This must be ignored by the faithfulness discovery regex.
            write_jsonl(
                eval_dir / "gsmarena_zero_shot_qwen3_1.7b_cc_gpt5.1.jsonl",
                [{"table_idx": 0}],
            )
            output = output_dir / "gsmarena_zero_shot_qwen3_32B_responses.jsonl"
            write_jsonl(output, [{"response": "sample"}])

            systems = eval_judges_agreement.build_system_configs(str(root), "gsmarena")
            self.assertEqual(len(systems), 1)
            self.assertEqual(systems[0].outputs_path, str(output))
            self.assertTrue(Path(systems[0].judge_b_path).name.endswith("gemini2.5-pro.jsonl"))


class ApiEvaluationFailureTests(unittest.TestCase):
    def test_faithfulness_sdk_error_fails_without_saving_output(self):
        secret = "dummy-test-secret"
        client = StubClient(ValueError(f"request failed with {secret}"))
        data = [{"system_prompt": "system", "user_prompt": "prompt", "original_response": "text"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_file = root / "input.jsonl"
            table_file = root / "table.jsonl"
            output_file = root / "results/output.jsonl"
            write_jsonl(input_file, [{"response": "A response."}])
            write_jsonl(table_file, [{"formatted_input": "data"}])
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "dummy"}):
                with patch.object(LLM_eval_OpenRouter, "create_client", return_value=client):
                    with patch.object(
                        sys,
                        "argv",
                        [
                            "LLM_eval_OpenRouter.py",
                            "--input_file", str(input_file),
                            "--input_table", str(table_file),
                            "--output_file", str(output_file),
                        ],
                    ):
                        with self.assertRaisesRegex(RuntimeError, "API request failed") as raised:
                            LLM_eval_OpenRouter.main()
            self.assertNotIn(secret, str(raised.exception))
            self.assertFalse(output_file.exists())

    def test_invalid_span_is_rejected_before_annotation_is_saved(self):
        response = json.dumps(
            {"errors": [{"reason": "bad span", "text": "missing", "type": 0}]}
        )
        data = [{"system_prompt": "system", "user_prompt": "prompt", "original_response": "text"}]
        with self.assertRaisesRegex(RuntimeError, "does not occur"):
            LLM_eval_OpenRouter.evaluate_ent_desc(
                data,
                "example/model",
                "dataset",
                "model",
                client=StubClient(response),
            )

    def test_empty_or_truncated_faithfulness_response_never_writes_output(self):
        for content in ("", '{"errors": [{"reason": "incomplete"'):
            with self.subTest(content=content):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    input_file = root / "input.jsonl"
                    table_file = root / "table.jsonl"
                    output_file = root / "results/output.jsonl"
                    write_jsonl(input_file, [{"response": "A response."}])
                    write_jsonl(table_file, [{"formatted_input": "data"}])
                    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "dummy"}):
                        with patch.object(
                            LLM_eval_OpenRouter,
                            "create_client",
                            return_value=StubClient(content),
                        ):
                            with patch.object(
                                sys,
                                "argv",
                                [
                                    "LLM_eval_OpenRouter.py",
                                    "--input_file", str(input_file),
                                    "--input_table", str(table_file),
                                    "--output_file", str(output_file),
                                ],
                            ):
                                with self.assertRaises(RuntimeError):
                                    LLM_eval_OpenRouter.main()
                    self.assertFalse(output_file.exists())

    def test_cc_sdk_error_is_marked_and_secret_is_not_logged_or_saved(self):
        secret = "dummy-test-secret"
        data = [
            {
                "system_prompt": "system",
                "user_prompt": "prompt",
                "original_response": "text",
                "table_idx": 3,
            }
        ]
        with self.assertLogs(level="ERROR") as captured:
            results = LLM_eval_CC.evaluate_content_coverage(
                data,
                "example/model",
                "dataset",
                "model",
                "example/model",
                client=StubClient(ValueError(f"request failed with {secret}")),
            )
        self.assertTrue(results[0]["parse_error"])
        self.assertIsNone(results[0]["coverage_score"])
        self.assertNotIn(secret, "\n".join(captured.output))
        self.assertNotIn(secret, json.dumps(results))

    def test_cc_output_schema_is_validated(self):
        valid = json.dumps(
            {
                "coverage_score": 4,
                "estimated_coverage_ratio": 0.8,
                "missing_points": [{"point": "one fact", "status": "PARTIAL"}],
                "notes": "Some details are missing.",
            }
        )
        result = LLM_eval_CC.evaluate_content_coverage(
            [{"system_prompt": "s", "user_prompt": "u", "table_idx": 4}],
            "example/model",
            "dataset",
            "display model",
            "example/model",
            client=StubClient(valid),
        )
        self.assertEqual(result[0]["table_idx"], 4)
        self.assertFalse(result[0]["parse_error"])
        self.assertEqual(result[0]["coverage_score"], 4)

    def test_cc_analyzer_excludes_parse_error_rows_from_means(self):
        stats = analyze_cc.compute_stats(
            [
                {"coverage_score": 4, "estimated_coverage_ratio": 0.7},
                {
                    "coverage_score": None,
                    "estimated_coverage_ratio": None,
                    "parse_error": True,
                },
            ]
        )
        self.assertEqual(stats["n_valid"], 1)
        self.assertEqual(stats["n_valid_ratio"], 1)
        self.assertEqual(stats["n_parse_error"], 1)
        self.assertEqual(stats["mean_score"], 4)


class CoverageSummaryTests(unittest.TestCase):
    def test_dynamic_summary_discovers_paper_selected_files(self):
        ddkd = {
            "gsmarena": "ddkd_zero_shot_qwen3_1.7b_pert",
            "ice_hockey": "ddkd_sft_lora_qwen3_1.7b_mixed",
            "owid": "ddkd_sft_lora_qwen3_1.7b_mixed",
            "weather": "ddkd_sft_lora_qwen3_1.7b_sub",
            "wikidata": "ddkd_sft_lora_qwen3_1.7b_mixed",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for domain in compute_cc_norm.DOMAINS:
                stems = [
                    f"{domain}_zero_shot_qwen3_1.7b",
                    f"{domain}_sft_lora_on_webnlg_json_qwen3_1.7b",
                    (
                        f"{domain}_zero_shot_qwen3_32b"
                        if domain == "gsmarena"
                        else f"{domain}_sft_lora_on_webnlg_json_qwen3_32b"
                    ),
                    f"{domain}_{ddkd[domain]}",
                ]
                for stem in stems:
                    for judge in ("gpt5.1", "gemini2.5-pro"):
                        rows = [
                            {"coverage_score": 5, "estimated_coverage_ratio": 0.9},
                            {"coverage_score": 3, "estimated_coverage_ratio": 0.5},
                        ]
                        if stem.endswith("ddkd_sft_lora_qwen3_1.7b_mixed"):
                            rows.append(
                                {
                                    "coverage_score": None,
                                    "estimated_coverage_ratio": None,
                                    "parse_error": True,
                                }
                            )
                        write_jsonl(root / domain / f"{stem}_cc_{judge}.jsonl", rows)

            # A nonselected DDKD variant must not be relabeled as the SFT baseline.
            self.assertIsNone(
                compute_cc_norm.classify_system(
                    "ice_hockey", "ice_hockey_ddkd_sft_lora_qwen3_1.7b_base"
                )
            )
            summary = compute_cc_norm.summarize(root)
            self.assertEqual(len(summary), 2)
            for judge in compute_cc_norm.JUDGES:
                score_rows = summary[judge]["score"]
                ddkd_system = compute_cc_norm.SYSTEMS[3]
                self.assertEqual(
                    score_rows[ddkd_system]["ice_hockey"]["n_valid"], 2
                )
                self.assertEqual(
                    score_rows[ddkd_system]["ice_hockey"]["n_parse_error"], 1
                )
                self.assertTrue(0 <= score_rows[ddkd_system]["ice_hockey"]["NormAvg"] <= 1)


class PromptAndTableSummaryTests(unittest.TestCase):
    def test_faithfulness_prompt_example_is_parseable_json(self):
        prompt_config = yaml.safe_load(
            (PROJECT_ROOT / "src/eval/eval_prompt_config.yaml").read_text(encoding="utf-8")
        )
        rendered = prompt_config["prompt_template"].format(data="example data", text="example text")
        match = re.search(r"\boutput:\s*```\s*(.*?)\s*```", rendered, flags=re.DOTALL)
        self.assertIsNotNone(match)
        example = json.loads(match.group(1))
        self.assertIsInstance(example["errors"], list)

    def test_table4_summary_uses_manifest_order_and_mean_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            manifest = {}
            for system in ("System A", "System B"):
                manifest[system] = {}
                for domain in summarize_table4.DOMAINS:
                    rel = f"data/{domain}/{system.replace(' ', '_')}.jsonl"
                    path = base / rel
                    annotations = [
                        {"type": 0, "text": "a"},
                        {"type": 3, "text": "b"},
                    ]
                    write_jsonl(
                        path,
                        [
                            {"annotations": annotations},
                            {"annotations": annotations[:1]},
                        ],
                    )
                    manifest[system][domain] = rel
            sources = base / "data/results/table4_sources.json"
            sources.parent.mkdir(parents=True, exist_ok=True)
            sources.write_text(json.dumps(manifest), encoding="utf-8")
            rows = summarize_table4.build_rows(base, sources)
            self.assertEqual([row["System"] for row in rows], ["System A", "System B"])
            self.assertEqual(rows[0]["Wikidata"], "1.50")


class SwiftScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.args_log = self.root / "swift_args.json"
        stub = self.bin_dir / "swift"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['SWIFT_ARGS_LOG'], 'w', encoding='utf-8') as f:\n"
            "    json.dump(sys.argv[1:], f)\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)

    def run_script(self, name: str, env: dict[str, str]):
        test_env = os.environ.copy()
        test_env.update(env)
        test_env["PATH"] = f"{self.bin_dir}{os.pathsep}{os.defpath}"
        test_env["SWIFT_ARGS_LOG"] = str(self.args_log)
        return subprocess.run(
            ["bash", str(PROJECT_ROOT / "script/swift" / name)],
            cwd=PROJECT_ROOT,
            env=test_env,
            capture_output=True,
            text=True,
        )

    def test_distill_example_quotes_paths_and_runs_one_selected_branch(self):
        result = self.run_script(
            "swift_sft_distill_example.sh",
            {
                "MODE": "augmented",
                "MODEL": "/model directory/qwen",
                "TRAIN_DATASET": "/data directory/train.jsonl",
                "VAL_DATASET": "/data directory/val.jsonl",
                "OUTPUT_DIR": "/output directory/run",
                "MODEL_NAME": "test run",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.args_log.read_text(encoding="utf-8"))
        self.assertEqual(args[args.index("--model") + 1], "/model directory/qwen")
        self.assertEqual(args[args.index("--dataset") + 1], "/data directory/train.jsonl")
        self.assertEqual(args[args.index("--val_dataset") + 1], "/data directory/val.jsonl")
        self.assertEqual(args.count("--dataset_shuffle"), 1)

    def test_missing_required_swift_variable_fails_before_execution(self):
        result = self.run_script("swift_zero_shot_example.sh", {})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.args_log.exists())


if __name__ == "__main__":
    unittest.main()
