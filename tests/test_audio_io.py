"""Synthetic WAV tests; no benchmark downloads or model weights required."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audiohelix.audio_io import (
    TARGET_SR, ChunkRecord, chunk_audio_to_records, make_unit_wavs,
)


class MakeUnitWavsTests(unittest.TestCase):
    def test_empty_records_do_not_load_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "units"
            with patch("audiohelix.audio_io.librosa.load") as load:
                self.assertEqual(make_unit_wavs("not-used.wav", [], str(target)), [])
            load.assert_not_called()
            self.assertFalse(target.exists())

    def test_short_source_with_no_retained_chunks_has_no_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "short.wav"
            sf.write(source, np.zeros(TARGET_SR // 2), TARGET_SR)
            chunks = chunk_audio_to_records(str(source), "short", str(Path(tmp) / "chunks"))
            self.assertEqual(chunks, [])
            self.assertEqual(make_unit_wavs(str(source), chunks, str(Path(tmp) / "units")), [])

    def test_nonempty_records_preserve_global_boundaries_and_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            waveform = np.linspace(-0.5, 0.5, TARGET_SR * 4, dtype=np.float32)
            sf.write(source, waveform, TARGET_SR, subtype="FLOAT")
            chunks = [ChunkRecord("sample_ch0001", "sample", 1, 1.0, 3.5, "unused.wav")]
            units = make_unit_wavs(str(source), chunks, str(Path(tmp) / "units"),
                                   unit_sec=1.0, min_last_unit_sec=0.25)
            self.assertEqual([(u.start_sec, u.end_sec) for u in units],
                             [(1.0, 2.0), (2.0, 3.0), (3.0, 3.5)])
            self.assertEqual([u.unit_id for u in units],
                             ["sample_ch0001_a00", "sample_ch0001_a01", "sample_ch0001_a02"])
            for unit in units:
                actual, sr = sf.read(unit.audio_path, dtype="float32")
                self.assertEqual(sr, TARGET_SR)
                expected = waveform[int(unit.start_sec * sr):int(unit.end_sec * sr)]
                np.testing.assert_allclose(actual, expected, atol=1 / 32768)

    def test_existing_unit_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            sf.write(source, np.zeros(TARGET_SR), TARGET_SR)
            chunks = [ChunkRecord("sample_ch0000", "sample", 0, 0.0, 1.0, "unused.wav")]
            first = make_unit_wavs(str(source), chunks, str(Path(tmp) / "units"))
            with patch("audiohelix.audio_io.sf.write") as write:
                self.assertEqual(make_unit_wavs(str(source), chunks, str(Path(tmp) / "units")), first)
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
