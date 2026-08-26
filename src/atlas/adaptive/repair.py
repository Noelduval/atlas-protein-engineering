"""Bounded repair/rescue generation for scientifically promising near-misses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from atlas.adaptive.critic import CriticRoute, Critique
from atlas.adaptive.models import CandidateRecord, DesignStrategy, Mutation
from atlas.design.design_space import ResidueClass, ResidueDesignRecord
from atlas.design.strategies import substitution_order


@dataclass(frozen=True)
class RepairProposal:
    parent_id: str
    child: CandidateRecord
    diagnosed_weakness: str
    repair_hypothesis: str
    change_made: str
    feature_to_preserve: str

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child.candidate_id,
            "diagnosed_weakness": self.diagnosed_weakness,
            "repair_hypothesis": self.repair_hypothesis,
            "change_made": self.change_made,
            "feature_to_preserve": self.feature_to_preserve,
        }


class RepairGenerator:
    def __init__(
        self,
        design_space: Iterable[ResidueDesignRecord],
        *,
        max_children_per_parent_round: int = 3,
        max_generations: int = 2,
    ) -> None:
        self.design_space = tuple(sorted(design_space, key=lambda record: record.position))
        self.reference_sequence = "".join(record.wildtype for record in self.design_space)
        self.max_children = max_children_per_parent_round
        self.max_generations = max_generations
        self._issued: set[tuple[str, int]] = set()

    def propose(
        self,
        parent: CandidateRecord,
        critique: Critique,
        *,
        round_index: int,
        existing_sequences: Iterable[str] = (),
    ) -> tuple[RepairProposal, ...]:
        key = (parent.candidate_id, round_index)
        if key in self._issued:
            return ()
        self._issued.add(key)
        if (
            critique.route is not CriticRoute.REVISE
            or not critique.repairable
            or parent.revision_generation >= self.max_generations
        ):
            return ()
        existing = set(existing_sequences)
        proposals: list[RepairProposal] = []
        if len(parent.mutations) == 1:
            proposals.extend(
                self._add_compensatory_mutations(parent, critique, round_index, existing)
            )
        else:
            proposals.extend(
                self._replace_double_components(parent, critique, round_index, existing)
            )
        return tuple(proposals[: self.max_children])

    def _child(
        self,
        parent: CandidateRecord,
        critique: Critique,
        mutations: tuple[Mutation, ...],
        round_index: int,
        change: str,
    ) -> RepairProposal:
        weakness = critique.diagnosed_weakness
        hypothesis = (
            f"Repair '{weakness}' via {change} while preserving "
            f"'{critique.feature_to_preserve}'."
        )
        child = CandidateRecord.create(
            reference_sequence=self.reference_sequence,
            mutations=mutations,
            parents=(parent.candidate_id,),
            strategy=DesignStrategy.REPAIR_RESCUE,
            structural_region=parent.structural_region,
            round_index=round_index,
            hypothesis=hypothesis,
            intended_upside=f"Repair {weakness.lower()}",
            expected_risk="The repair may erase the useful parent feature or add epistasis.",
            revision_generation=parent.revision_generation + 1,
        )
        return RepairProposal(
            parent_id=parent.candidate_id,
            child=child,
            diagnosed_weakness=weakness,
            repair_hypothesis=hypothesis,
            change_made=change,
            feature_to_preserve=critique.feature_to_preserve,
        )

    def _add_compensatory_mutations(
        self,
        parent: CandidateRecord,
        critique: Critique,
        round_index: int,
        existing: set[str],
    ) -> list[RepairProposal]:
        parent_position = parent.mutations[0].position
        repair_records = sorted(
            (
                record
                for record in self.design_space
                if record.residue_class is ResidueClass.DESIGNABLE
                and record.position != parent_position
            ),
            key=lambda record: (
                0 if record.structural_region == "distal_stability" else 1,
                -record.packing_neighbors,
                record.position,
            ),
        )
        proposals: list[RepairProposal] = []
        for record in repair_records:
            mutant = substitution_order(record)[0]
            added = Mutation.parse(f"{record.wildtype}{record.position}{mutant}")
            change = f"add compensatory {added.label}"
            proposal = self._child(
                parent,
                critique,
                tuple(sorted(parent.mutations + (added,))),
                round_index,
                change,
            )
            if proposal.child.sequence in existing:
                continue
            existing.add(proposal.child.sequence)
            proposals.append(proposal)
            if len(proposals) == self.max_children:
                break
        return proposals

    def _replace_double_components(
        self,
        parent: CandidateRecord,
        critique: Critique,
        round_index: int,
        existing: set[str],
    ) -> list[RepairProposal]:
        by_position = {record.position: record for record in self.design_space}
        proposals: list[RepairProposal] = []
        for mutation_index, mutation in enumerate(parent.mutations):
            record = by_position[mutation.position]
            alternatives = [
                mutant
                for mutant in substitution_order(record)
                if mutant != mutation.mutant
            ]
            for mutant in alternatives:
                replacement = Mutation.parse(f"{mutation.wildtype}{mutation.position}{mutant}")
                mutations = list(parent.mutations)
                mutations[mutation_index] = replacement
                change = f"replace {mutation.label} with {replacement.label}"
                proposal = self._child(
                    parent,
                    critique,
                    tuple(sorted(mutations)),
                    round_index,
                    change,
                )
                if proposal.child.sequence in existing:
                    continue
                existing.add(proposal.child.sequence)
                proposals.append(proposal)
                if len(proposals) == self.max_children:
                    return proposals
        return proposals

