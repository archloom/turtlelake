"""The schema is a contract: every Arrow consumer relies on these field names
and types. Pin them here."""

from turtlelake.schema import TRIPLE_SCHEMA


def test_schema_field_names_and_order():
    assert TRIPLE_SCHEMA.names == [
        "subject",
        "predicate",
        "object",
        "object_kind",
        "object_datatype",
        "object_lang",
        "graph",
    ]


def test_only_object_metadata_and_graph_are_nullable():
    nullable = {f.name for f in TRIPLE_SCHEMA if f.nullable}
    assert nullable == {"object_datatype", "object_lang", "graph"}
