import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from eco_score import RunMetrics, build_eco_score_payload
import main
from db import queries
from pipeline import hd_search, run_optimize_pipeline


class PromptRetrievalPersistenceTests(unittest.TestCase):
    def test_hd_search_keeps_hits_and_score_fallback(self):
        class Hit:
            def __init__(self, text, similarity=None, score=None):
                self.text = text
                self.similarity = similarity
                self.score = score

        fake_hits = [
            Hit("keep similarity", similarity=0.70),
            Hit("drop low similarity", similarity=0.64),
            Hit("keep score fallback", similarity=None, score=0.65),
            Hit("drop missing score", similarity=None, score=None),
        ]

        with patch("pipeline._hd_client", return_value=MagicMock(search=MagicMock(return_value=fake_hits))):
            out = hd_search("optimize me", top_k=5)

        self.assertEqual(
            [r["retrieved_text"] for r in out],
            ["keep similarity", "drop low similarity", "keep score fallback", "drop missing score"],
        )
        self.assertEqual(out[0]["similarity"], 0.70)
        self.assertEqual(out[2]["similarity"], 0.65)
        self.assertEqual(out[3]["similarity"], 0.0)

    def test_insert_prompt_retrievals_uses_run_id_and_rank_order(self):
        fake_cur = MagicMock()
        fake_conn = MagicMock()
        fake_conn.cursor.return_value = fake_cur

        retrievals = [
            {"retrieved_text": "first text", "similarity": 0.91, "example_id": None},
            {"retrieved_text": "second text", "similarity": None, "example_id": None},
            {"retrieved_text": "third text", "similarity": 0.71, "example_id": 42},
        ]

        with patch("db.queries._db_enabled", return_value=True), patch(
            "db.queries.connect_to_database", return_value=fake_conn
        ):
            queries.insert_prompt_retrievals(run_id=123, retrievals=retrievals)

        fake_cur.executemany.assert_called_once()
        _, rows = fake_cur.executemany.call_args[0]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][0], 123)
        self.assertEqual(rows[0][2], "first text")
        self.assertEqual(rows[0][4], 1)
        self.assertEqual(rows[1][4], 2)
        self.assertEqual(rows[2][4], 3)
        self.assertEqual(rows[1][3], None)
        self.assertEqual(rows[2][1], 42)

    def test_insert_prompt_rewrite_serializes_eco_metadata(self):
        fake_cur = MagicMock()
        fake_conn = MagicMock()
        fake_conn.cursor.return_value = fake_cur

        changes_payload = {
            "tags": ["mode:precise"],
            "eco_score": 74.3,
            "eco_score_raw": 0.0009,
            "eco_breakdown": {"retrieval_count": 3, "compute_proxy": 110.1},
            "eco_version": "v1",
        }

        with patch("db.queries._db_enabled", return_value=True), patch(
            "db.queries.connect_to_database", return_value=fake_conn
        ):
            queries.insert_prompt_rewrite(
                run_id=42,
                optimized_prompt="optimized",
                changes=changes_payload,
                model_name="qwen2.5:1.5b",
                latency_ms=345,
            )

        args = fake_cur.execute.call_args[0][1]
        serialized_changes = args[2]
        self.assertIn('"eco_score": 74.3', serialized_changes)
        self.assertIn('"eco_version": "v1"', serialized_changes)

    def test_optimize_endpoint_passes_run_id_to_pipeline(self):
        client = TestClient(main.app)
        fake_result = {
            "optimized": "optimized prompt",
            "mode": "precise",
            "beforeTokens": 100.0,
            "afterTokens": 70.0,
            "efficiency": 30.0,
            "clarityScore": 90.0,
            "skeleton": {
                "intent": "how-to",
                "task": "instruction",
                "subject": "task",
                "output": "steps",
                "prompt": "Do thing",
            },
            "rules_fallback": False,
            "retrievals": [],
            "eco": {
                "eco_score": 88.0,
                "eco_score_raw": 0.0012,
                "eco_version": "v1",
                "eco_breakdown": {
                    "input_tokens": 40.0,
                    "output_tokens": 60.0,
                    "attempts": 1,
                    "latency_ms": 120.0,
                    "retrieval_count": 0,
                    "model_size": "small",
                    "quality_score": 1.0,
                    "compute_proxy": 100.2,
                },
            },
            "rewrite_metrics": {"model_name": "qwen2.5:1.5b"},
        }

        with patch("main.run_optimize_pipeline", return_value=fake_result) as mock_run_pipeline, patch(
            "main.queries.insert_prompt_run", return_value=777
        ), patch("main.queries.insert_prompt_rewrite"):
            response = client.post("/optimize", json={"prompt": "please optimize this", "mode": "precise"})

        self.assertEqual(response.status_code, 200)
        mock_run_pipeline.assert_called_once_with("please optimize this", "precise", run_id=777)
        payload = response.json()
        self.assertEqual(payload["run_id"], 777)
        self.assertEqual(payload["eco_score"], 88.0)
        self.assertIn("eco_breakdown", payload)

    def test_run_pipeline_persists_zero_retrievals_safely(self):
        fallback_skeleton = (
            "INTENT: how-to\n"
            "TASK: instruction\n"
            "SUBJECT: task\n"
            "OUTPUT: steps\n"
            "PROMPT: do thing"
        )
        with patch("pipeline.extract_skeleton_safe", return_value=(fallback_skeleton, {})), patch(
            "pipeline.hd_search", return_value=[]
        ), patch("pipeline.revise_prompt_safe", return_value=("optimized", False, {"wall_latency_ms": 250.0, "rewrite_model": "qwen2.5:1.5b"})), patch(
            "pipeline.estimate_tokens_by_model", return_value=80.0
        ), patch("pipeline.clarity_score", return_value=90.0), patch(
            "pipeline.detect_meaning_loss", return_value=False
        ), patch("pipeline.loses_constraints", return_value=False), patch(
            "pipeline.queries.insert_prompt_retrievals"
        ) as mock_insert_retrievals:
            _ = run_optimize_pipeline("please optimize this", "precise", run_id=778)

        mock_insert_retrievals.assert_called_once_with(
            run_id=778,
            retrievals=[],
            retrieval_source="human_delta",
        )
        self.assertIn("eco", _)
        self.assertEqual(_["eco"]["eco_breakdown"]["retrieval_count"], 0)

    def test_build_eco_score_prefers_lower_compute_proxy(self):
        light = build_eco_score_payload(
            RunMetrics(
                input_tokens=30,
                output_tokens=50,
                attempts=1,
                latency_ms=200,
                retrieval_count=1,
                model_size="small",
                quality_score=1.0,
            )
        )
        heavy = build_eco_score_payload(
            RunMetrics(
                input_tokens=120,
                output_tokens=180,
                attempts=1,
                latency_ms=1200,
                retrieval_count=6,
                model_size="large",
                quality_score=1.0,
            )
        )

        self.assertGreater(light["eco_score"], heavy["eco_score"])
        self.assertLess(light["eco_breakdown"]["compute_proxy"], heavy["eco_breakdown"]["compute_proxy"])


if __name__ == "__main__":
    unittest.main()
