import json
from pathlib import Path
import tempfile
import unittest

from eval.protocol import validate_prompts, expected_filenames
from eval.summarize import aggregate, NORMALIZATION

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_fixed_protocol_has_944_unique_prompts_and_4720_outputs(self):
        original = ROOT / 'assets/prompts/all_dimension.txt'
        expanded = ROOT / 'assets/prompts/all_dimension_extended.txt'
        with tempfile.TemporaryDirectory() as temporary:
            info = Path(temporary) / 'info.json'
            info.write_text(json.dumps([dict(prompt_en=p) for p in original.read_text().splitlines()]))
            prompts, conditioned = validate_prompts(original, expanded, info)
            self.assertEqual(len(prompts), 944)
            self.assertEqual(len(conditioned), 944)
            self.assertEqual(len(set(expected_filenames(prompts))), 4720)

    def test_summary_uses_official_normalization_and_requires_all_dimensions(self):
        maxima = {k: v[1] for k, v in NORMALIZATION.items()}
        scores = aggregate(maxima)
        for key in ('total', 'quality', 'semantic'):
            self.assertAlmostEqual(scores[key], 1.)
        reduced = dict(maxima, dynamic_degree=0.)
        scores = aggregate(reduced)
        self.assertAlmostEqual(scores['quality'], 6 / 6.5)
        self.assertAlmostEqual(scores['total'], (4 * 6 / 6.5 + 1) / 5)
        del reduced['scene']
        self.assertIsNone(aggregate(reduced)['total'])


if __name__ == '__main__':
    unittest.main()
