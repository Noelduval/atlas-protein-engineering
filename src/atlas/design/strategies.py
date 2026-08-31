"""Context-aware mutation policy for the adaptive DP622 search."""

from __future__ import annotations

from dataclasses import dataclass

from atlas.adaptive.models import DesignStrategy
from atlas.design.design_space import ResidueDesignRecord


AMINO_ACID_VOLUME_A3 = {
    "A": 88.6,
    "C": 108.5,
    "D": 111.1,
    "E": 138.4,
    "F": 189.9,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "K": 168.6,
    "L": 166.7,
    "M": 162.9,
    "N": 114.1,
    "P": 112.7,
    "Q": 143.8,
    "R": 173.4,
    "S": 89.0,
    "T": 116.1,
    "V": 140.0,
    "W": 227.8,
    "Y": 193.6,
}

CONSERVATIVE_FAMILIES = (
    "AILMV",
    "FYW",
    "STNQ",
    "DE",
    "KR",
    "H",
    "C",
    "G",
    "P",
)

CLASS_ALPHABETS = {
    "hydrogen_bond": "STNQ",
    "hydrophobic_tuning": "AILMVFYW",
    "preorganization": "STNQDEKRH",
    "packing": "AILMVFYWSTNQCG",
    "helix_propensity": "AELKMQRH",
    "charge_balance": "DEKRH",
    "surface_charge": "DEKRH",
    "solubility": "STNQDEKRH",
}

CONTEXT_SUBSTITUTIONS = {
    91: ("F", "W", "H"),
    126: ("E", "N", "A"),
}


@dataclass(frozen=True)
class SubstitutionProposal:
    mutant: str
    substitution_class: str
    intended_physical_change: str
    feature_to_preserve: str
    principal_biochemical_risk: str
    metal_liability: str = "none_identified"
    requires_candidate_geometry: bool = False


def strategy_for_residue(record: ResidueDesignRecord) -> DesignStrategy:
    if record.structural_region == "substrate_interface":
        return DesignStrategy.SUBSTRATE_INTERFACE
    if record.structural_region == "second_shell":
        return DesignStrategy.SECOND_SHELL_PREORGANIZATION
    if record.structural_region == "distal_stability":
        return DesignStrategy.STABILITY_SUPPORT
    return DesignStrategy.CONSERVATIVE


def _conservative_alphabet(wildtype: str) -> str:
    return next(family for family in CONSERVATIVE_FAMILIES if wildtype in family)


def _physical_change(record: ResidueDesignRecord, mutant: str, policy_class: str) -> str:
    volume_delta = AMINO_ACID_VOLUME_A3[mutant] - AMINO_ACID_VOLUME_A3[record.wildtype]
    if policy_class in {"packing", "hydrophobic_tuning", "helix_propensity"}:
        direction = "increase" if volume_delta > 0 else "decrease"
        return (
            f"{direction} local side-chain volume by approximately "
            f"{abs(volume_delta):.1f} Å³ to tune packing"
        )
    if policy_class in {"hydrogen_bond", "preorganization"}:
        return (
            "alter local hydrogen-bond/electrostatic preorganization with a "
            "chemically explicit side chain"
        )
    if policy_class in {"charge_balance", "surface_charge", "solubility"}:
        return "alter local charge or polarity while retaining the resolved scaffold"
    if policy_class == "experimentally_informed":
        return "test a bounded identity informed by position-specific retrospective evidence"
    return "make a conservative chemistry change within the wild-type residue family"


def _risk(record: ResidueDesignRecord, mutant: str) -> str:
    if mutant == "P":
        return "Proline can kink or disrupt the local backbone."
    if mutant == "G":
        return "Glycine can increase backbone flexibility and weaken local preorganization."
    if mutant == "C":
        return (
            "Cysteine introduces redox, disulfide, and possible metal-coordination "
            "liability."
        )
    if record.burial_class == "buried" and mutant in "DEKRH":
        return "A buried ionizable side chain may be energetically uncompensated."
    if record.relative_sasa >= 0.5 and mutant in "AILMVFYW":
        return (
            "An exposed hydrophobe may reduce solubility or create nonspecific surface "
            "interactions."
        )
    return "The local packing, electrostatics, or interaction network may regress."


def substitution_policy(record: ResidueDesignRecord) -> tuple[SubstitutionProposal, ...]:
    """Translate design-space classes into legal, ranked mutation identities."""
    if not record.allowed_substitution_classes:
        return ()
    proposals: list[SubstitutionProposal] = []
    seen: set[str] = set()
    for policy_class in record.allowed_substitution_classes:
        if policy_class == "experimentally_informed":
            alphabet = "".join(CONTEXT_SUBSTITUTIONS.get(record.position, ()))
        elif policy_class == "conservative":
            alphabet = _conservative_alphabet(record.wildtype)
        else:
            alphabet = CLASS_ALPHABETS.get(policy_class, "")
        for mutant in alphabet:
            if mutant == record.wildtype or mutant in seen:
                continue
            if mutant == "P" and record.secondary_structure in {"helix", "strand"}:
                continue
            if (
                mutant == "G"
                and record.secondary_structure in {"helix", "strand"}
                and record.burial_class != "exposed"
            ):
                continue
            if (
                mutant in "DEKRH"
                and record.burial_class == "buried"
                and record.wildtype not in "DEKRH"
            ):
                continue
            if policy_class == "packing" and abs(
                AMINO_ACID_VOLUME_A3[mutant]
                - AMINO_ACID_VOLUME_A3[record.wildtype]
            ) > 45.0:
                continue
            near_zinc_context = (
                record.structural_region == "second_shell"
                and record.min_zinc_distance_a <= 10.0
                and mutant in "HCDE"
            )
            proposals.append(
                SubstitutionProposal(
                    mutant=mutant,
                    substitution_class=policy_class,
                    intended_physical_change=_physical_change(
                        record, mutant, policy_class
                    ),
                    feature_to_preserve=(
                        "resolved Aβ pose and catalytic preorganization"
                        if record.structural_region
                        in {"substrate_interface", "second_shell"}
                        else "DP622 scaffold packing and secondary-structure integrity"
                    ),
                    principal_biochemical_risk=_risk(record, mutant),
                    metal_liability=(
                        "HIGH_RISK_METAL_SITE_HYPOTHESIS"
                        if near_zinc_context
                        else "none_identified"
                    ),
                    requires_candidate_geometry=near_zinc_context,
                )
            )
            seen.add(mutant)
    proposals.sort(
        key=lambda proposal: (
            proposal.requires_candidate_geometry,
            record.allowed_substitution_classes.index(proposal.substitution_class),
            abs(
                AMINO_ACID_VOLUME_A3[proposal.mutant]
                - AMINO_ACID_VOLUME_A3[record.wildtype]
            ),
            proposal.mutant,
        )
    )
    return tuple(proposals)


def substitution_order(record: ResidueDesignRecord) -> tuple[str, ...]:
    """Backward-compatible identity order backed by the executable policy."""
    return tuple(proposal.mutant for proposal in substitution_policy(record))


def strategy_narrative(
    strategy: DesignStrategy,
    record: ResidueDesignRecord,
    proposal: SubstitutionProposal,
) -> tuple[str, str, str]:
    mutation = f"{record.wildtype}{record.position}{proposal.mutant}"
    hypothesis = (
        f"Test whether {mutation} can {proposal.intended_physical_change.lower()} while "
        f"preserving {proposal.feature_to_preserve}."
    )
    if strategy is DesignStrategy.SUBSTRATE_INTERFACE:
        upside = "Create a physically interpretable Aβ contact/shape hypothesis."
    elif strategy is DesignStrategy.SECOND_SHELL_PREORGANIZATION:
        upside = (
            "Tune second-shell packing or electrostatics without changing protected Zn "
            "ligands."
        )
    elif strategy is DesignStrategy.STABILITY_SUPPORT:
        upside = "Increase scaffold-support margin for function-oriented perturbations."
    else:
        upside = "Sample a fold-compatible alternative with an explicit chemistry rationale."
    return hypothesis, upside, proposal.principal_biochemical_risk
