"""Stable identifiers for reaction sites and their concrete members.

The KMC runtime must be able to recognise the same scientific site after a
checkpoint round-trip or after dynamic network discovery returns a freshly
constructed Python object.  Object addresses (``id(site)``) cannot provide
that guarantee, so identifiers are derived from the site's channel metadata
and materialised graph-node membership and are then persisted on the site.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, TypeAlias


SiteId: TypeAlias = str
MemberSignature: TypeAlias = str
SiteMemberId: TypeAlias = tuple[SiteId, MemberSignature]
_MEMBER_SIGNATURES_ATTR = "_stable_member_signatures"
_MEMBER_IDENTIFIERS_ATTR = "_stable_member_identifiers"


def _normalise(value: Any) -> Any:
    """Return a deterministic, JSON-compatible representation of *value*."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalise(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_normalise(item) for item in value), key=repr)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _site_kind(site: Any) -> str:
    name = type(site).__name__.lower()
    if "bondreactionsite" in name or hasattr(site, "template"):
        return "bond"
    if "diffusionsite" in name:
        return "diffusion"
    return "adsorbate"


def _template_signature(site: Any) -> tuple[Any, ...] | None:
    template = getattr(site, "template", None)
    if template is None:
        return None
    return (
        getattr(template, "smiles_a", None),
        getattr(template, "smiles_b", None),
        getattr(template, "smiles_c", None),
        getattr(template, "bond_type", None),
        getattr(template, "source", None),
    )


def _member_payload(site: Any, member_index: int) -> Any:
    """Return the role-aware node payload for one concrete site member."""
    members = getattr(site, "member_node_ids", ()) or ()
    try:
        payload = _normalise(members[int(member_index)])
    except (IndexError, TypeError):
        # Keep the compatibility surface usable for lightweight downstream
        # site proxies.  Production sites always expose materialised member
        # nodes before entering the KMC index.
        payload = {"missing_member_index": int(member_index)}

    kind = _site_kind(site)
    if not isinstance(payload, list):
        return payload

    def _sort_key(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    if kind == "diffusion" and len(payload) == 2:
        # A diffusion hop is the same concrete pair whichever endpoint was
        # selected as A while run-local iso classes were numbered.
        return sorted(payload, key=_sort_key)

    template = getattr(site, "template", None)
    if (
        kind == "bond"
        and len(payload) >= 2
        and bool(getattr(template, "is_symmetric", False))
    ):
        # Only symmetric A + A channels may exchange the first two roles.
        return [*sorted(payload[:2], key=_sort_key), *payload[2:]]
    return payload


def _build_member_signature(site: Any, member_index: int) -> MemberSignature:
    encoded = json.dumps(
        _member_payload(site, member_index),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"member:{sha256(encoded).hexdigest()[:24]}"


def _member_signatures(site: Any) -> tuple[MemberSignature, ...]:
    """Return cached signatures for the materialised members of *site*.

    ``member_node_ids`` are immutable after a site enters the KMC index.
    Caching the complete tuple here removes repeated JSON serialisation and
    SHA-256 work from reverse-index lookups and segment-tree updates.  The
    length guard handles the normal pre-materialisation case where an empty
    member list is populated before indexing.
    """
    members = getattr(site, "member_node_ids", ()) or ()
    cached = getattr(site, _MEMBER_SIGNATURES_ATTR, None)
    if isinstance(cached, tuple) and len(cached) == len(members):
        return cached
    signatures = tuple(
        _build_member_signature(site, index)
        for index in range(len(members))
    )
    try:
        setattr(site, _MEMBER_SIGNATURES_ATTR, signatures)
        # A changed member list also invalidates the derived site/member IDs.
        setattr(site, _MEMBER_IDENTIFIERS_ATTR, ())
    except (AttributeError, TypeError):
        pass
    return signatures


def member_signature(site: Any, member_index: int) -> MemberSignature:
    """Return a stable, cached signature for one concrete site member."""
    index = int(member_index)
    signatures = _member_signatures(site)
    try:
        return signatures[index]
    except IndexError:
        # Preserve the compatibility behavior for lightweight proxy objects
        # whose member index is intentionally outside a materialised list.
        return _build_member_signature(site, index)


def build_site_identifier(site: Any) -> SiteId:
    """Derive a stable identifier without relying on Python object identity."""
    kind = _site_kind(site)
    members = getattr(site, "member_node_ids", ()) or ()
    payload = {
        "kind": kind,
        "reactant": getattr(site, "reactant", None),
        "template": _template_signature(site),
        # Iso-class numbers and member-list positions are run-local
        # bookkeeping.  The sorted concrete node signatures remain invariant
        # when either is renumbered/reordered.
        "members": sorted(_member_signatures(site)),
    }
    encoded = json.dumps(
        _normalise(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"{kind}:{sha256(encoded).hexdigest()[:24]}"


def site_identifier(site: Any) -> SiteId:
    """Return and, where possible, persist the stable ID for *site*.

    Checkpoints created before site IDs were introduced simply lack an
    instance value (or expose the dataclass's empty default).  They are
    upgraded in memory on first access.
    """
    existing = getattr(site, "site_id", "")
    if existing:
        return str(existing)
    identifier = build_site_identifier(site)
    try:
        site.site_id = identifier
    except (AttributeError, TypeError):
        # Read-only proxy objects remain usable; their ID is deterministic so
        # recomputing it has the same semantics.
        pass
    return identifier


def member_identifier(site: Any, member_index: int) -> SiteMemberId:
    """Return the stable ID of one concrete member of *site*.

    All identifiers are materialised together on first use, making repeated
    hot-loop lookups constant-time after a site has entered the index.
    """
    index = int(member_index)
    members = getattr(site, "member_node_ids", ()) or ()
    cached = getattr(site, _MEMBER_IDENTIFIERS_ATTR, None)
    if isinstance(cached, tuple) and len(cached) == len(members):
        try:
            return cached[index]
        except IndexError:
            pass

    identifier = site_identifier(site)
    identifiers = tuple(
        (identifier, signature)
        for signature in _member_signatures(site)
    )
    try:
        setattr(site, _MEMBER_IDENTIFIERS_ATTR, identifiers)
    except (AttributeError, TypeError):
        pass
    try:
        return identifiers[index]
    except IndexError:
        return identifier, member_signature(site, index)


__all__ = [
    "SiteId",
    "MemberSignature",
    "SiteMemberId",
    "build_site_identifier",
    "member_identifier",
    "member_signature",
    "site_identifier",
]
