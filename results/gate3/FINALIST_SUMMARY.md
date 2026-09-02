# Gate-3 finalist summary

## Observed funnel

| Stage | Count |
| --- | ---: |
| Generated/evaluated unique legal candidates | 5,000 |
| Broad survivors | 500 |
| Structural analyses | 146 |
| Adversarial-review candidates | 10 |
| Finalists | 5 |

## Portfolio

All five are prospective Atlas-generated variants, novel-to-this-campaign, and experimentally untested. Stability values below are model outputs, not activity predictions. Structural and substrate-interface values are plausibility/support evidence, not proof of catalysis.

| Candidate | Mutations | Design context | Stability | Structure / interface | Liability | Why retained |
| --- | --- | --- | --- | --- | --- | --- |
| [ATLAS-0215A1AE2C58FD57](dossiers/ATLAS-0215A1AE2C58FD57.md) | G65M/Q153M | Function-oriented mutation paired with distal/scaffold support; epistatic double | ThermoMPNN-D -1.49974; uncertainty 0.75 | RMSD 0.50490; interface 0.54690; catalytic-preorganization deviation 1.15349 | None triggered | Multiple independent support axes; no policy-level repair blocker |
| [ATLAS-1FB316267B702519](dossiers/ATLAS-1FB316267B702519.md) | A16C/Q153C | Orthogonal-mechanism epistatic double | ThermoMPNN-D -1.84022; uncertainty 0.75 | RMSD 0.53692; interface 0.54887; deviation 1.20286 | Cysteine introductions; soft liability 0.4 | Multiple independent support axes; no policy-level repair blocker |
| [ATLAS-2EB17A2E3DB7062E](dossiers/ATLAS-2EB17A2E3DB7062E.md) | A130L | Conservative distal-stability explorer | ThermoMPNN -0.50064; uncertainty 0.5 | RMSD 0.49441; interface 0.55555; deviation 1.21022 | None triggered | Conservative legal change with preserved required support |
| [ATLAS-39F921440EC002E5](dossiers/ATLAS-39F921440EC002E5.md) | E104V | Second-shell preorganization explorer | ThermoMPNN -1.08968; uncertainty 0.5 | RMSD 0.52637; interface 0.55655; deviation 1.21640 | Charge-class change; soft liability 0.1 | Preserved structure/interface support without a hard blocker |
| [ATLAS-66C6BB6CAA1C310A](dossiers/ATLAS-66C6BB6CAA1C310A.md) | A37Y | Substrate-interface explorer | ThermoMPNN -0.48975; uncertainty 0.5 | RMSD 0.51074; interface 0.61054; deviation 1.22881 | None triggered | Strongest retained substrate-interface support and no hard blocker |

Specificity evidence was not a discriminating axis for these mutation placements in the exported final records. Diagnostic pose robustness is not a finalist gate. The unresolved question for every candidate is experimental expression/folding and biochemical behavior; the exported dossiers define falsification assays.

## Frozen method and provenance

The Gate-2 policy froze protected catalytic/Zn chemistry, hard chemistry and structure constraints, model-aware stability screening, required substrate-interface support, and specificity where exercisable. Retrospective activity controls showed that ThermoMPNN stability and static geometry must not be treated as activity predictions; generic liabilities remain soft and pose robustness remains diagnostic. No thresholds were loosened after the prospective results were observed.

- Atlas production commit: `aa5f61d8a5fbaffb50e1c2fef3b6b3cc275f707c`
- Gate-3 preparation commit: `aa5f61d8a5fbaffb50e1c2fef3b6b3cc275f707c`
- ThermoMPNN: `2b04fd370e399911b1fa5848112cc9013f084110`
- ThermoMPNN-D: `df9a75aaddb674a7c4c193005031fc0536d325fb`
- Seed: `622`; candidate budget: `5000`; round1: `1200`; minimum doubles: `750`; broad: `500`; structure: `100`; adversarial: `10`; portfolio: `5`; repair parents: `6`

The input is the 23WN-derived `active_like_inferred` reconstruction, not an experimentally observed active DP622-S2 complex. The resolved Aβ segment is limited, static geometry does not establish cleavage kinetics, diagnostic MD was excluded after reference-validation failure, and no wet-lab validation has yet been performed.
