"""
run_inference.py — CLI script for running Hybrid Temporal Model inference.

Usage:
    python run_inference.py --checkpoint ./models/phase2_step3000_mae1.87 \
                            --query "WW2 started in 1939 and lasted 6 years. When did it end?" \
                            --context "World War II began September 1939."

    python run_inference.py --checkpoint ./models/phase2_step3000_mae1.87 \
                            --input_file queries.json \
                            --output_file predictions.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_time_embedding.inference.pipeline import TemporalQAPipeline
from hybrid_time_embedding.inference.predictor import single_predict, batch_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal QA Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--query", type=str, default=None, help="Single query string")
    parser.add_argument("--context", type=str, default="", help="Context for single query")
    parser.add_argument("--input_file", type=str, default=None, help="JSON file with list of {query, context} dicts")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save predictions JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for file inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading pipeline from {args.checkpoint} ...")
    pipeline = TemporalQAPipeline(args.checkpoint, device=args.device)

    if args.query:
        result = single_predict(pipeline, args.query, args.context)
        print(f"\nQuery:        {args.query}")
        print(f"Context:      {args.context}")
        print(f"Arith pred:   {result['raw_year_arith']:.2f} years")
        print(f"Duration pred:{result['raw_duration']:.2f} years")
        print(f"Gate value:   {result['gate_value']:.4f}")
        print(f"Latency:      {result['latency_ms']:.1f} ms")

    elif args.input_file:
        with open(args.input_file) as f:
            items = json.load(f)
        queries = [item["query"] for item in items]
        contexts = [item.get("context", "") for item in items]
        results = batch_predict(pipeline, queries, contexts, batch_size=args.batch_size)

        for item, result in zip(items, results):
            item["prediction"] = result

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(items, f, indent=2)
            print(f"Predictions saved to {args.output_file}")
        else:
            print(json.dumps(results[:5], indent=2))
    else:
        print("Provide --query or --input_file.")


if __name__ == "__main__":
    main()
