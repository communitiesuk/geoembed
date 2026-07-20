"""Unity Catalog naming utilities.

Databricks Unity Catalog has two addressing modes:

1. **Delta tables** (SQL identifiers): catalog.schema.table
   - Identifiers with special characters (hyphens, etc.) MUST be backtick-quoted
   - Example: `catalog-foo`.`schema`.`table`
   - Used in: saveAsTable(), read.table(), spark.sql("SELECT ... FROM ...")
   - MUST be three-part: catalog.schema.table

2. **Volumes** (file paths): /Volumes/catalog/schema/volume/path/file.tif
   - These are filesystem paths — NO backticks, no quoting
   - Used in: spark.read.load(), rasterio.open(), os.path operations

This module provides `quote_table_name()` to handle case 1.
Volume paths (case 2) should be used as-is — never quote them.
"""


def quote_table_name(table_name: str) -> str:
    """
    Backtick-quote a Unity Catalog table name for use in Spark SQL operations.

    Handles identifiers containing hyphens or other special characters by wrapping
    each dot-separated part in backticks.

    Only use this for Delta table references (saveAsTable, read.table, SQL statements).
    Do NOT use for Volume file paths (/Volumes/...) — those are filesystem paths.

    Args:
        table_name: Three-part dot-separated table name: "catalog.schema.table".
                    Two-part names (schema.table) are also accepted but will use
                    the session's default catalog.

    Returns:
        Backtick-quoted name (e.g., "`catalog-foo`.`schema`.`table`").

    Raises:
        ValueError: If table_name has fewer than 2 parts or more than 3 parts.

    Examples:
        >>> quote_table_name("my_catalog.my_schema.my_table")
        '`my_catalog`.`my_schema`.`my_table`'

        >>> quote_table_name("catalog-with-hyphens.data.embeddings")
        '`catalog-with-hyphens`.`data`.`embeddings`'
    """
    parts = table_name.split(".")

    if len(parts) < 2:
        msg = (
            f"Table name '{table_name}' must have at least 2 parts (schema.table) "
            f"or preferably 3 parts (catalog.schema.table). Got {len(parts)} part(s)."
        )
        raise ValueError(msg)

    if len(parts) > 3:
        msg = (
            f"Table name '{table_name}' has {len(parts)} parts — expected at most 3 "
            f"(catalog.schema.table). Check for extra dots in the name."
        )
        raise ValueError(msg)

    if len(parts) == 2:
        import warnings

        warnings.warn(
            f"Table name '{table_name}' has only 2 parts (schema.table). "
            f"This will use the session's default catalog. "
            f"Consider using the full 3-part name: catalog.schema.table",
            UserWarning,
            stacklevel=2,
        )

    return ".".join(f"`{p}`" for p in parts)
