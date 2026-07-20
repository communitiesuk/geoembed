"""Tests for the naming module — Unity Catalog table name quoting."""

import warnings

import pytest

from geoembed.core.naming import quote_table_name


class TestQuoteTableName:
    """Tests for quote_table_name utility."""

    def test_three_part_name(self):
        result = quote_table_name("catalog.schema.table")
        assert result == "`catalog`.`schema`.`table`"

    def test_hyphenated_catalog(self):
        result = quote_table_name("catalog-foo-bar.data.embeddings")
        assert result == "`catalog-foo-bar`.`data`.`embeddings`"

    def test_two_part_name_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = quote_table_name("schema.table")
            assert result == "`schema`.`table`"
            assert len(w) == 1
            assert "only 2 parts" in str(w[0].message)

    def test_single_part_raises(self):
        with pytest.raises(ValueError, match="must have at least 2 parts"):
            quote_table_name("just_a_table")

    def test_four_parts_raises(self):
        with pytest.raises(ValueError, match="has 4 parts"):
            quote_table_name("a.b.c.d")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must have at least 2 parts"):
            quote_table_name("")

    @pytest.mark.parametrize(
        "table_name,expected",
        [
            ("cat.sch.tbl", "`cat`.`sch`.`tbl`"),
            ("my-cat.my-sch.my-tbl", "`my-cat`.`my-sch`.`my-tbl`"),
            ("UPPER.Mixed.lower", "`UPPER`.`Mixed`.`lower`"),
            ("cat_1.sch_2.tbl_3", "`cat_1`.`sch_2`.`tbl_3`"),
        ],
    )
    def test_parametrized_valid_names(self, table_name, expected):
        assert quote_table_name(table_name) == expected
