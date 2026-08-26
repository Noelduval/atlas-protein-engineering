"""Explicit mutation strategies for the adaptive DP622 search."""

from __future__ import annotations

from atlas.adaptive.models import DesignStrategy
from atlas.design.design_space import ResidueDesignRecord


CHEMICAL_GROUPS = (
    "AILMV",
    "FYW",
    "STNQ",
    "DE",
    "KRH",
    "CGP",
)

CONTEXT_SUBSTITUTIONS = {
    91: ("F", "W", "H"),
    126: ("E", "N", "A"),
}


def strategy_for_residue(record: ResidueDesignRecord) -> DesignStrategy:
    if record.structural_region == "substrate_interface":
        return DesignStrategy.SUBSTRATE_INTERFACE
    if record.structural_region == "second_shell":
        return DesignStrategy.SECOND_SHELL_PREORGANIZATION
    if record.structural_region == "distal_stability":
        return DesignStrategy.STABILITY_SUPPORT
    return DesignStrategy.CONSERVATIVE


def substitution_order(record: ResidueDesignRecord) -> tuple[str, ...]:
    """Return a deterministic conservative-to-exploratory amino-acid order."""
    if record.position in CONTEXT_SUBSTITUTIONS:
        return CONTEXT_SUBSTITUTIONS[record.position]
    wildtype = record.wildtype
    same_group = next((group for group in CHEMICAL_GROUPS if wildtype in group), "")
    order: list[str] = []
    for amino_acid in same_group + "".join(CHEMICAL_GROUPS):
        if amino_acid != wildtype and amino_acid not in order:
            order.append(amino_acid)
    return tuple(order)


def strategy_narrative(
    strategy: DesignStrategy, record: ResidueDesignRecord, mutant: str
) -> tuple[str, str, str]:
    mutation = f"{record.wildtype}{record.position}{mutant}"
    if strategy is DesignStrategy.SUBSTRATE_INTERFACE:
        return (
            f"Test whether {mutation} can tune the resolved Aβ-contact surface while "
            "retaining the deposited cleavage pose.",
            "Improve or preserve substrate-contact complementarity.",
            "May weaken a productive contact or displace the resolved substrate fragment.",
        )
    if strategy is DesignStrategy.SECOND_SHELL_PREORGANIZATION:
        return (
            f"Test whether {mutation} can alter second-shell packing or electrostatics "
            "without mutating protected catalytic machinery.",
            "Support active-site preorganization while preserving direct Zn ligands.",
            "Context-dependent coupling could perturb catalytic geometry.",
        )
    if strategy is DesignStrategy.STABILITY_SUPPORT:
        return (
            f"Test whether {mutation} can improve distal scaffold packing or secondary-"
            "structure support.",
            "Increase stability margin available to function-oriented mutations.",
            "A distal stability proxy may not translate to catalytic improvement.",
        )
    return (
        f"Conservatively explore {mutation} in a resolved scaffold position.",
        "Retain fold-compatible chemistry while sampling an alternative side chain.",
        "The substitution may be neutral or create an unmodeled local liability.",
    )

