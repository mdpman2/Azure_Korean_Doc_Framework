"""Resolve Azure AI Search field mappings from the live index schema at runtime.

This utility lets the framework connect to pre-existing indexes even when the
configured field names are stale. The resolved mapping is cached per index and
applied back onto Config for the current process.
"""

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from ..config import Config
from .azure_clients import AzureClientFactory


def _pick_first_available(field_names: Iterable[str], candidates: List[Optional[str]]) -> Optional[str]:
    available = set(field_names)
    for candidate in candidates:
        if candidate and candidate in available:
            return candidate
    return None


def _resolve_mapping_from_index(index: Any) -> Dict[str, Any]:
    """Infer the most suitable framework field mapping from an index definition."""
    field_names = [field.name for field in index.fields]
    semantic_configs = [cfg.name for cfg in (index.semantic_search.configurations if index.semantic_search and index.semantic_search.configurations else [])]

    resolved_id = _pick_first_available(field_names, [Config.SEARCH_ID_FIELD, "chunk_id", "id", "document_id"])
    resolved_content = _pick_first_available(field_names, [Config.SEARCH_CONTENT_FIELD, "chunk", "content", "text"])
    resolved_original = _pick_first_available(
        field_names,
        [Config.SEARCH_ORIGINAL_CONTENT_FIELD, "original_chunk", resolved_content, "chunk", "content"],
    )
    resolved_vector = _pick_first_available(field_names, [Config.SEARCH_VECTOR_FIELD, "text_vector", "content_vector", "vector"])
    resolved_title = _pick_first_available(field_names, [Config.SEARCH_TITLE_FIELD, "title", "file_name", "name"])
    resolved_parent = _pick_first_available(field_names, [Config.SEARCH_PARENT_FIELD, "parent_id", "parentId", "source_id"])
    resolved_source = _pick_first_available(
        field_names,
        [Config.SEARCH_SOURCE_FIELD, resolved_title, resolved_parent, "source_url", "source"],
    )
    resolved_citation = _pick_first_available(field_names, [Config.SEARCH_CITATION_FIELD, "citation"])
    resolved_bbox = _pick_first_available(field_names, [Config.SEARCH_BOUNDING_BOX_FIELD, "bounding_box_json", "bounding_box"])
    resolved_regions = _pick_first_available(field_names, [Config.SEARCH_SOURCE_REGIONS_FIELD, "source_regions_json", "source_regions"])
    resolved_semantic = None
    if semantic_configs:
        resolved_semantic = Config.SEARCH_SEMANTIC_CONFIG if Config.SEARCH_SEMANTIC_CONFIG in semantic_configs else semantic_configs[0]

    return {
        "index_name": index.name,
        "field_names": field_names,
        "semantic_configs": semantic_configs,
        "mapping": {
            "SEARCH_ID_FIELD": resolved_id,
            "SEARCH_CONTENT_FIELD": resolved_content,
            "SEARCH_ORIGINAL_CONTENT_FIELD": resolved_original,
            "SEARCH_VECTOR_FIELD": resolved_vector,
            "SEARCH_TITLE_FIELD": resolved_title,
            "SEARCH_PARENT_FIELD": resolved_parent,
            "SEARCH_SOURCE_FIELD": resolved_source,
            "SEARCH_CITATION_FIELD": resolved_citation,
            "SEARCH_BOUNDING_BOX_FIELD": resolved_bbox,
            "SEARCH_SOURCE_REGIONS_FIELD": resolved_regions,
            "SEARCH_SEMANTIC_CONFIG": resolved_semantic,
        },
    }


@lru_cache(maxsize=8)
def _get_cached_runtime_mapping(index_name: str) -> Dict[str, Any]:
    index_client = AzureClientFactory.get_search_index_client()
    index = index_client.get_index(index_name)
    return _resolve_mapping_from_index(index)


def get_search_runtime_mapping(index_name: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
    """Return the cached or freshly resolved runtime mapping for the target index."""
    name = index_name or Config.SEARCH_INDEX_NAME
    if not name or not Config.SEARCH_KEY or not Config.SEARCH_ENDPOINT:
        return {
            "index_name": name,
            "field_names": [],
            "semantic_configs": [],
            "mapping": {},
        }
    if refresh:
        _get_cached_runtime_mapping.cache_clear()
    return _get_cached_runtime_mapping(name)


def apply_search_runtime_mapping(index_name: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
    """Apply the resolved runtime mapping onto Config so downstream components use valid fields."""
    resolved = get_search_runtime_mapping(index_name=index_name, refresh=refresh)
    mapping = resolved.get("mapping", {})
    for attr_name, resolved_value in mapping.items():
        if resolved_value:
            setattr(Config, attr_name, resolved_value)
    return resolved