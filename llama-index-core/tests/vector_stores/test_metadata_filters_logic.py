from typing import Any, Dict

import pytest

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.utils import build_metadata_filter_fn


_METADATA_BY_ID: Dict[str, Dict[str, Any]] = {
    "n1": {
        "rank": "a",
        "weight": 1.0,
        "tags": ["alpha", "beta"],
        "identifier": "ABC123",
    },
    "n2": {
        "rank": "b",
        "weight": 2.0,
        "tags": ["beta"],
        "identifier": "abc456",
    },
    "n3": {
        "rank": "c",
        "weight": 3.0,
        "tags": [],
        "identifier": "XYZ789",
    },
}


def test_and_condition_matches_expected_nodes() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.NE, value="a"),
            MetadataFilter(
                key="weight",
                operator=FilterOperator.GTE,
                value=2.0,
            ),
        ],
        condition=FilterCondition.AND,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert not fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_none_condition_defaults_to_and() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.NE, value="a"),
            MetadataFilter(key="weight", operator=FilterOperator.GTE, value=2.0),
        ],
        condition=None,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert not fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_or_condition_matches_expected_nodes() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="a"),
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="c"),
        ],
        condition=FilterCondition.OR,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert fn("n1")
    assert not fn("n2")
    assert fn("n3")


def test_not_condition_negates_filters() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="a"),
        ],
        condition=FilterCondition.NOT,
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    assert not fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_is_empty_matches_missing_and_empty_values() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="missing",
                operator=FilterOperator.IS_EMPTY,
                value=None,
            )
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # All nodes are missing the "missing" key, so they should all match.
    assert fn("n1")
    assert fn("n2")
    assert fn("n3")


def test_ne_and_nin_match_missing_metadata_values() -> None:
    ne_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="missing", operator=FilterOperator.NE, value="a"),
        ]
    )
    nin_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="missing", operator=FilterOperator.NIN, value=["a"]),
        ]
    )

    ne_fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, ne_filters)
    nin_fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, nin_filters)

    # Missing metadata values are not equal to, and not contained in, the
    # filtered value. This keeps NE/NIN consistent with NOT(EQ)/NOT(IN).
    assert ne_fn("n1")
    assert nin_fn("n1")


def test_text_match_insensitive_uses_case_insensitive_search() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="identifier",
                operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
                value="abc",
            )
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # Both identifiers contain "abc" ignoring case.
    assert fn("n1")
    assert fn("n2")
    assert not fn("n3")


def test_text_match_requires_string_values() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="weight",
                operator=FilterOperator.TEXT_MATCH,
                value="1",
            )
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # Non-string metadata values should raise a TypeError when used with TEXT_MATCH.
    with pytest.raises(TypeError):
        fn("n1")


def test_text_match_insensitive_requires_string_values() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="weight",
                operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
                value="1",
            )
        ]
    )
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)

    # Non-string metadata values should raise a TypeError when used with TEXT_MATCH_INSENSITIVE.
    with pytest.raises(TypeError):
        fn("n1")


def _matching_ids(filters: MetadataFilters) -> set:
    fn = build_metadata_filter_fn(_METADATA_BY_ID.__getitem__, filters)
    return {node_id for node_id in _METADATA_BY_ID if fn(node_id)}


def test_nested_metadata_filters_single_level() -> None:
    inner = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="a"),
        ]
    )
    outer = MetadataFilters(filters=[inner], condition=FilterCondition.AND)

    assert _matching_ids(outer) == {"n1"}


def test_nested_metadata_filters_or_of_ands() -> None:
    # (rank == "a" AND weight < 2) OR (rank == "c" AND weight >= 3)
    filters = MetadataFilters(
        filters=[
            MetadataFilters(
                filters=[
                    MetadataFilter(key="rank", operator=FilterOperator.EQ, value="a"),
                    MetadataFilter(key="weight", operator=FilterOperator.LT, value=2.0),
                ],
                condition=FilterCondition.AND,
            ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="rank", operator=FilterOperator.EQ, value="c"),
                    MetadataFilter(
                        key="weight", operator=FilterOperator.GTE, value=3.0
                    ),
                ],
                condition=FilterCondition.AND,
            ),
        ],
        condition=FilterCondition.OR,
    )

    assert _matching_ids(filters) == {"n1", "n3"}


def test_nested_metadata_filters_mixed_with_flat_filter() -> None:
    # weight > 1 AND (tags contains "beta" OR tags is empty)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="weight", operator=FilterOperator.GT, value=1.0),
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="tags", operator=FilterOperator.CONTAINS, value="beta"
                    ),
                    MetadataFilter(
                        key="tags", operator=FilterOperator.IS_EMPTY, value=None
                    ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )

    assert _matching_ids(filters) == {"n2", "n3"}


def test_nested_metadata_filters_not_condition() -> None:
    # NOT (rank == "a" OR rank == "b")
    filters = MetadataFilters(
        filters=[
            MetadataFilters(
                filters=[
                    MetadataFilter(key="rank", operator=FilterOperator.EQ, value="a"),
                    MetadataFilter(key="rank", operator=FilterOperator.EQ, value="b"),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.NOT,
    )

    assert _matching_ids(filters) == {"n3"}


def test_nested_metadata_filters_inner_condition_defaults_to_and() -> None:
    inner = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="b"),
            MetadataFilter(key="weight", operator=FilterOperator.EQ, value=2.0),
        ],
        condition=None,
    )
    outer = MetadataFilters(filters=[inner], condition=FilterCondition.OR)

    assert _matching_ids(outer) == {"n2"}


def test_nested_metadata_filters_three_levels() -> None:
    # rank != "c" AND (identifier text_match "abc" OR (weight == 1 AND tags contains "alpha"))
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.NE, value="c"),
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="identifier",
                        operator=FilterOperator.TEXT_MATCH,
                        value="abc",
                    ),
                    MetadataFilters(
                        filters=[
                            MetadataFilter(
                                key="weight", operator=FilterOperator.EQ, value=1.0
                            ),
                            MetadataFilter(
                                key="tags",
                                operator=FilterOperator.CONTAINS,
                                value="alpha",
                            ),
                        ],
                        condition=FilterCondition.AND,
                    ),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )

    assert _matching_ids(filters) == {"n1", "n2"}


def test_empty_nested_metadata_filters() -> None:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="rank", operator=FilterOperator.EQ, value="a"),
            MetadataFilters(filters=[], condition=FilterCondition.AND),
        ],
        condition=FilterCondition.AND,
    )

    assert _matching_ids(filters) == {"n1"}
