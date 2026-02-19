#!/usr/bin/env python3
"""
Regression tests for mol_repl rewrite helpers.

Run (recommended):
    python -m unittest discover -s tests -p "test*.py" -v
"""
import unittest

from script.mol_repl import (
    repair_sel_all_to_last,
    repair_unknown_all_sel_to_last,
    ensure_user_mentioned_structures_loaded,
    drop_unknown_selections_in_dev_loose,
)


class TestMolReplRewrites(unittest.TestCase):
    def test_repair_sel_all_to_last_does_not_corrupt_color_by_chain(self):
        dsl = 'color_by_chain(sel="all", target="chain")\nshow(sel="all", rep="cartoon")\nhide(sel="all")'
        out = repair_sel_all_to_last(dsl, "all_1crn")
        self.assertIn('color_by_chain(sel="all_1crn", target="chain")', out)
        self.assertIn('show(sel="all_1crn", rep="cartoon")', out)
        self.assertIn('hide(sel="all_1crn")', out)

    def test_repair_unknown_all_sel_to_last(self):
        nl = "show surface"
        dsl = 'show(sel="all_3eam", rep="surface")'
        out = repair_unknown_all_sel_to_last(
            nl_query=nl,
            dsl_text=dsl,
            last_all_sel="all_1crn",
            known_sels={"all_1crn"},
        )
        self.assertIn('show(sel="all_1crn", rep="surface")', out)

    def test_repair_unknown_all_sel_to_last_does_not_override_user_mention(self):
        # If the user explicitly said 3eam, do NOT rewrite; dev-loose can auto-load it.
        nl = "show 3eam surface"
        dsl = 'show(sel="all_3eam", rep="surface")'
        out = repair_unknown_all_sel_to_last(
            nl_query=nl,
            dsl_text=dsl,
            last_all_sel="all_1crn",
            known_sels={"all_1crn"},
        )
        self.assertEqual(out, dsl)

    def test_ensure_user_mentioned_structures_loaded_only_when_user_mentioned(self):
        nl = "please load 1crn and show it"
        dsl = 'show(sel="all_1crn", rep="cartoon")'
        out = ensure_user_mentioned_structures_loaded(nl, dsl, known_sels=set())
        self.assertTrue(out.splitlines()[0].startswith('add_structure(PDBID="1crn")'))

        nl2 = "show it"
        out2 = ensure_user_mentioned_structures_loaded(nl2, dsl, known_sels=set())
        self.assertEqual(out2, dsl)

    def test_drop_unknown_selections_in_dev_loose(self):
        known = {"all_1crn"}
        dsl = "\n".join([
            'show(sel="all_1crn", rep="cartoon")',
            'show(sel="all_2xyz", rep="cartoon")',
        ])
        out = drop_unknown_selections_in_dev_loose(dsl, known)
        self.assertIn('all_1crn', out)
        self.assertNotIn('all_2xyz', out)


if __name__ == "__main__":
    unittest.main()
