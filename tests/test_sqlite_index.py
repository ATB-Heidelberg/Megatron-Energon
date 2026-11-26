# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests the SQLite index of webdatasets"""

import tempfile
import unittest

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.indexing import SqliteIndexReader, SqliteIndexWriter


class TestSqliteIndex(unittest.TestCase):
    def test_index_in_a_folder_named_with_uri_delimiters(self):
        """An index is read back from a folder whose name holds characters a URI would cut or decode"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = EPath(tmp_dir) / "run #1 (100%)" / ".nv-meta" / "index.sqlite"
            with SqliteIndexWriter(index_path) as writer:
                writer.append_sample(
                    tar_file_id=0, sample_key="a", sample_index=0, byte_offset=0, byte_size=512
                )

            with SqliteIndexReader(index_path) as reader:
                sample_count = reader.get_sample_count()

            assert sample_count == 1


if __name__ == "__main__":
    unittest.main()
