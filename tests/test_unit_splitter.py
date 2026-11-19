import unittest

from audiohelix.build.unit_splitter import split_into_units


class SplitIntoUnitsTests(unittest.TestCase):
    def test_caption_decimal_stays_in_same_evidence_unit(self):
        self.assertEqual(
            split_into_units("The readout says 21.5 degrees. A fan starts."),
            ["The readout says 21.5 degrees.", "A fan starts."],
        )

    def test_decimal_at_sentence_end_does_not_hide_next_sentence(self):
        self.assertEqual(
            split_into_units("The meter reads 21.5. A tone rises to 23.75."),
            ["The meter reads 21.5.", "A tone rises to 23.75."],
        )

    def test_unicode_digits_preserve_embedded_decimal(self):
        self.assertEqual(
            split_into_units("レベルは２１.５です. 音が止まります."),
            ["レベルは２１.５です.", "音が止まります."],
        )

    def test_integer_at_sentence_end_is_still_a_boundary(self):
        self.assertEqual(
            split_into_units("The meter reads 21. A tone rises."),
            ["The meter reads 21.", "A tone rises."],
        )

    def test_existing_whitespace_and_terminal_period_behavior(self):
        self.assertEqual(
            split_into_units("...  A dog barks. Music plays softly  "),
            ["A dog barks.", "Music plays softly."],
        )
        self.assertEqual(split_into_units("   "), [])


if __name__ == "__main__":
    unittest.main()
