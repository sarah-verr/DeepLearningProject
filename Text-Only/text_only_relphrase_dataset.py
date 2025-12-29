import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable


RELATIONS = {
    "left_of": {
        "phrase": "to the left of",
        "inverse": "right_of",
    },
    "right_of": {
        "phrase": "to the right of",
        "inverse": "left_of",
    },
    "above": {
        "phrase": "above",
        "inverse": "below",
    },
    "below": {
        "phrase": "below",
        "inverse": "above",
    },
    "inside": {
        "phrase": "inside",
        "inverse": "around",
    },
    "around": {
        "phrase": "around",
        "inverse": "inside",
    },
}


@dataclass
class Example:
    context: str
    question: str
    answer: str
    rel_phrase: str
    context_relation: str
    question_relation: str
    subject: str
    obj: str


def _build_sentence(subject: str, relation_phrase: str, obj: str) -> str:
    return f"{subject} is {relation_phrase} {obj}."


def _build_question(subject: str, relation_phrase: str, obj: str) -> str:
    return f"Is {subject} {relation_phrase} {obj}?"


def _iter_pairs(items: list[str]) -> Iterable[tuple[str, str]]:
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j:
                yield items[i], items[j]


def _make_example_pair(
    *,
    subject: str,
    obj: str,
    context_relation: str,
) -> list[Example]:
    context_phrase = RELATIONS[context_relation]["phrase"]
    inverse_relation = RELATIONS[context_relation]["inverse"]
    inverse_phrase = RELATIONS[inverse_relation]["phrase"]

    context = _build_sentence(subject, context_phrase, obj)

    yes_q = _build_question(subject, context_phrase, obj)
    no_q = _build_question(subject, inverse_phrase, obj)

    return [
        Example(
            context=context,
            question=yes_q,
            answer="yes",
            rel_phrase=context_phrase,
            context_relation=context_relation,
            question_relation=context_relation,
            subject=subject,
            obj=obj,
        ),
        Example(
            context=context,
            question=no_q,
            answer="no",
            rel_phrase=inverse_phrase,
            context_relation=context_relation,
            question_relation=inverse_relation,
            subject=subject,
            obj=obj,
        ),
    ]


def _write_jsonl(path: str, examples: list[Example]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "context": ex.context,
                        "question": ex.question,
                        "answer": ex.answer,
                        "rel_phrase": ex.rel_phrase,
                        "context_relation": ex.context_relation,
                        "question_relation": ex.question_relation,
                        "subject": ex.subject,
                        "object": ex.obj,
                    }
                )
                + "\n"
            )


def _default_output_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "text_only_relphrase_qa.jsonl")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate text-only relational QA pairs.")
    ap.add_argument(
        "--output",
        type=str,
        default=_default_output_path(),
        help="Output JSONL path.",
    )
    ap.add_argument(
        "--num_contexts",
        type=int,
        default=50,
        help="Number of context statements to generate (each yields 2 QAs).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    ap.add_argument(
        "--relations",
        type=str,
        nargs="+",
        default=list(RELATIONS.keys()),
        help=f"Relations to sample from. Options: {', '.join(RELATIONS.keys())}",
    )
    ap.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=[
            "red square",
            "blue square",
            "green circle",
            "yellow circle",
            "purple triangle",
            "orange triangle",
        ],
        help="Object strings to use as A/B.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    relations = [r for r in args.relations if r in RELATIONS]
    if not relations:
        raise SystemExit("No valid relations provided.")

    pairs = list(_iter_pairs(args.objects))
    if not pairs:
        raise SystemExit("Need at least two distinct objects.")

    examples: list[Example] = []
    for _ in range(args.num_contexts):
        subject, obj = rng.choice(pairs)
        context_relation = rng.choice(relations)
        examples.extend(
            _make_example_pair(
                subject=subject,
                obj=obj,
                context_relation=context_relation,
            )
        )

    _write_jsonl(args.output, examples)
    print(f"Wrote {len(examples)} QA items to: {args.output}")


if __name__ == "__main__":
    main()
