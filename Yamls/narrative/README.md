# Narrative Configuration — Story & Discourse Structures

This directory contains configuration for **narrative analysis**: story arcs, rhetorical devices, argumentation patterns, evidence types, and discourse structures.

## Lens Architecture

This directory supports **FOUR narrative analysis lenses** that share a unified theoretical foundation:

| Lens | Theory Domain | Config Files Used |
| ---- | ------------- | ----------------- |
| **Narrative** | Narratology, Story Structure | `narrative_structures.yaml`, `character_archetypes.yaml`, `soar_weighting.yaml`, `extraction_rules.yaml` |
| **Discourse** | RST, Text Simplification | `rst_relations.yaml`, `discourse_markers.yaml`, `complexity_rules.yaml` |
| **Argument** | Toulmin Model, IAM Dataset | `toulmin_structure.yaml`, `evidence_types.yaml`, `stance_markers.yaml` |
| **Entity** | NER for narrative elements | `narrative_entities.yaml` |

**Why unified?** These are complementary perspectives on how persuasive narratives work:

- **Narrative** provides the story arc and character framework
- **Discourse** analyzes how that story is structured rhetorically (RST relations)
- **Argument** extracts the logical claims-evidence structure
- **Entity** identifies narrative elements for NLP extraction

## Directory Contents

```
narrative/
├── README.md                    # This file
│
├── shared/                      # Shared configuration (all lenses)
│   ├── scoring.yaml             # Score bands, severity levels, thresholds
│   └── data_reqs.yaml           # Data requirements for analysis
│
├── # Core Narrative Lens
├── narrative_structures.yaml    # Story frameworks and arc patterns
├── character_archetypes.yaml    # Character roles and persona definitions
├── rhetorical_devices.yaml      # Persuasion and linguistic analysis
├── soar_weighting.yaml          # SOAR emphasis by level and sales stage
├── extraction_rules.yaml        # Content extraction heuristics
├── narrative_resolution.yaml    # Coherence and inheritance rules
├── narrative_entities.yaml      # NER entity types for narrative content
│
├── # Discourse Lens (RST, DATS)
├── rst_relations.yaml           # Seven core RST relations, nuclearity
├── discourse_markers.yaml       # Connectives (causal, adversative, additive, temporal)
├── complexity_rules.yaml        # Sentence complexity, 35 DisSim transformation rules
│
├── # Argument Lens (Toulmin, IAM)
├── toulmin_structure.yaml       # Claim, Evidence, Warrant, Backing, Rebuttal, Qualifier
├── evidence_types.yaml          # Statistic, Testimonial, Expert, Anecdote, Case Study
├── stance_markers.yaml          # Support/Contest/Neutral, Hedging, Boosting
│
└── # Multimodal Lens
    └── figure_references.yaml   # Direct/Indirect/Deictic visual references
```

**Total: 16 YAML files** (14 root + 2 shared)

---

## File Reference

### Core Narrative Lens

| File | Purpose | When to Edit |
| ---- | ------- | ------------ |
| **`narrative_structures.yaml`** | Hero's Journey, StoryBrand, SOAR, Vonnegut arcs, SCQA, Freytag | Add new narrative frameworks |
| **`character_archetypes.yaml`** | Hero/Guide/Villain, business personas, narrative tones | Update persona definitions |
| **`rhetorical_devices.yaml`** | Ethos/Pathos/Logos, linguistic devices, pacing, neurochemistry | Add rhetorical patterns |
| **`soar_weighting.yaml`** | SOAR element emphasis by content level (L1-L4) and sales stage | Adjust SOAR balance rules |
| **`extraction_rules.yaml`** | Heuristics for extracting narrative elements from documents | Improve extraction accuracy |
| **`narrative_resolution.yaml`** | Cross-hierarchy coherence and inheritance validation | Modify coherence rules |
| **`narrative_entities.yaml`** | NER entity types: claims, evidence, hooks, CTAs, transitions | Add entity detection patterns |

### Discourse Lens (RST, DATS)

| File | Purpose | When to Edit |
| ---- | ------- | ------------ |
| **`rst_relations.yaml`** | Seven core RST relations (Elaboration, Joint, Attribution, Same-unit, Contrast, Temporal, Cause/Condition), nuclearity, presentation mapping | Add new relation types or update frequency data |
| **`discourse_markers.yaml`** | Causal, Adversative, Additive, Temporal markers; ambiguity resolution | Add new discourse connectives or marker patterns |
| **`complexity_rules.yaml`** | Syntactic complexity indicators, 35 DisSim transformation rules, minimality principle | Update simplification rules or complexity thresholds |

**Theory Source:** Niklaus et al. 2023 - *Discourse-Aware Text Simplification*

### Argument Lens (Toulmin, IAM)

| File | Purpose | When to Edit |
| ---- | ------- | ------------ |
| **`toulmin_structure.yaml`** | Toulmin Model components (Claim, Evidence, Warrant, Backing, Rebuttal, Qualifier), stance classification | Update Toulmin patterns or claim detection rules |
| **`evidence_types.yaml`** | Five evidence categories (Statistic, Testimonial, Expert Opinion, Anecdote, Case Study), strength scoring | Add new evidence types or update strength criteria |
| **`stance_markers.yaml`** | Hedging markers, boosting markers, certainty level scoring | Update stance detection patterns or certainty thresholds |

**Theory Source:** Cheng et al. 2022 - *IAM: A Comprehensive and Large-Scale Dataset for Integrated Argument Mining Tasks*

---

## Entity Recognition (NER)

### `narrative_entities.yaml`

**Purpose:** NER entity definitions for extracting narrative components from content.

**Entity Types:**

| Entity | Description | Patterns |
|--------|-------------|----------|
| `claim` | Argumentative statements | Modal verbs, evaluation words, necessity markers |
| `evidence` | Supporting data/examples | Statistics, quotes, case references |
| `warrant` | Reasoning connectors | "therefore", "thus", "because" |
| `qualifier` | Certainty modifiers | Hedges, boosters, frequency markers |
| `rebuttal` | Counter-arguments | "however", "although", "critics argue" |
| `hook` | Attention-grabbing openers | Questions, provocative statements, statistics |
| `cta` | Call-to-action statements | Imperative verbs, action phrases |
| `transition` | Flow markers | Temporal, additive, contrastive connectives |
| `value_proposition` | Benefit statements | ROI language, outcome promises |
| `pain_point` | Problem descriptions | Challenge language, friction indicators |
| `outcome_statement` | Result descriptions | Achievement language, metrics |

---

## Common Tasks

| Task | File to Edit |
| ---- | ------------ |
| Add narrative framework | `narrative_structures.yaml` |
| Add business persona | `character_archetypes.yaml` |
| Add rhetorical device | `rhetorical_devices.yaml` |
| Adjust SOAR weighting | `soar_weighting.yaml` |
| Update RST relations | `rst_relations.yaml` |
| Add discourse marker | `discourse_markers.yaml` |
| Update complexity rules | `complexity_rules.yaml` |
| Add Toulmin pattern | `toulmin_structure.yaml` |
| Add evidence type | `evidence_types.yaml` |
| Update hedging/boosting | `stance_markers.yaml` |
| Add NER entity type | `narrative_entities.yaml` |

---

## Relationship to Other Config Directories

| Directory | Focus | What Moved Here/Away |
|-----------|-------|----------------------|
| **`narrative/`** | Story structure, rhetoric, argumentation | This directory |
| **`visual/`** | Design principles, accessibility, cognitive load | `clip_thresholds.yaml` moved TO visual/ |
| **`pedagogical/`** | Learning science, instructional design | `pedagogical_patterns.yaml` moved TO pedagogical/ |
| **`domain/`** | Editorial style, business terminology | — |
| **`rubrics/`** | Scoring rubrics, evaluation criteria | — |
| **`sales/`** | Buyer personas, sales plays, SOAR framework | — |

**Note:** Multimodal analysis (CLIP thresholds, figure references) moved to `visual/` as it's more concerned with visual-text alignment than pure narrative structure.

---

## Validation

Validate YAML syntax after editing:

```bash
cd narrative
python3 -c "
import yaml
from pathlib import Path

for f in Path('.').rglob('*.yaml'):
    try:
        with open(f) as fh:
            yaml.safe_load(fh)
        print(f'✓ {f}')
    except Exception as e:
        print(f'✗ {f}: {e}')
"
```

---

## Version History

| Date | Changes |
|------|---------|
| 2026-01-19 | Added `narrative_entities.yaml` for NER entity extraction |
| 2026-01-19 | Moved `clip_thresholds.yaml` to `visual/` |
| 2026-01-19 | Moved `pedagogical_patterns.yaml` to `pedagogical/` |
| 2026-01-19 | Removed duplicate `narrative_taxonomy.yaml` (canonical version in visual/) |
| 2026-01-19 | Metadata standardization: added `_meta:` blocks per `_SCHEMA.md` |
| 2026-01-18 | Added Discourse Lens: `rst_relations.yaml`, `discourse_markers.yaml`, `complexity_rules.yaml` |
| 2026-01-18 | Added Argument Lens: `toulmin_structure.yaml`, `evidence_types.yaml`, `stance_markers.yaml` |
| 2024-12-16 | Initial narrative configuration |

---

## Theoretical Foundations

**Core Narrative:**
- Campbell, J. (1949). *The Hero with a Thousand Faces*
- Miller, D. (2017). *Building a StoryBrand*
- Vonnegut, K. (1981). *The Shapes of Stories*
- Duarte, N. (2010). *Resonate*

**Discourse:**
- Niklaus et al. (2023). *Discourse-Aware Text Simplification*
- Mann & Thompson (1988). *Rhetorical Structure Theory*

**Argument:**
- Toulmin, S. (1958). *The Uses of Argument*
- Cheng et al. (2022). *IAM: Integrated Argument Mining Dataset*

---

## Related Documentation

- **Schema Definition**: `../_SCHEMA.md`
- **Root README**: `../README.md`
- **Visual Config**: `../visual/README.md` (for CLIP/multimodal)
- **Pedagogical Config**: `../pedagogical/README.md` (for pedagogical patterns)
