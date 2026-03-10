# Sales — Sales Enablement Configuration

This directory contains **sales-specific configuration** for the Design Methodology system: buyer personas, sales plays, SOAR framework mappings, and sales stage definitions.

## Directory Contents

```
sales/
├── README.md                           # This file
├── personas.yaml                       # Buyer persona definitions (v2.2.0, 15 personas)
├── play_catalog.yaml                   # FY26 play taxonomy (v3.1.0)
├── sales_stages.yaml                   # Sales funnel stages (SS0-SS10)
├── soar_mapping.yaml                   # SOAR keywords and weights
├── soar_hierarchy_rules.yaml           # SOAR requirements by level (L1-L4)
└── plays/                              # Individual play definitions
    ├── b2b_gtm.yaml
    ├── content_supply_chain.yaml
    ├── dme_end_user_productivity_canonical.yaml
    ├── dme_knowledge_based_productivity_canonical.yaml
    └── unified_cx.yaml
```

**Total: 10 YAML files** (5 root + 5 plays)

---

## File Reference

### Core Sales Configuration

| File | Version | Purpose |
|------|---------|---------|
| **`personas.yaml`** | v2.2.0 | 15 buyer personas across Executive, Senior Leadership, and Practitioner tiers |
| **`play_catalog.yaml`** | v3.1.0 | L1-L4 sales play taxonomy (Portfolio → Strategic → Tactical → Industry) |
| **`sales_stages.yaml`** | v2.1.1 | 11 sales stages (SS0-SS10) with SOAR emphasis weights per stage |
| **`soar_mapping.yaml`** | v1.0.0 | SOAR keywords and detection patterns |
| **`soar_hierarchy_rules.yaml`** | — | SOAR requirements by play level (L1-L4) |

### Sales Plays (`plays/`)

| File | Play |
|------|------|
| `b2b_gtm.yaml` | B2B Go-to-Market play |
| `content_supply_chain.yaml` | Content Supply Chain play |
| `dme_end_user_productivity_canonical.yaml` | DMe End-User Productivity (FY26) |
| `dme_knowledge_based_productivity_canonical.yaml` | DMe Knowledge-Based Productivity (FY26) |
| `unified_cx.yaml` | Unified Customer Experience play |

---

## Detailed Documentation

### `personas.yaml`

**Purpose:** Comprehensive buyer persona definitions for narrative tailoring and empathy analysis.

**Persona Structure:**

| Field | Description |
|-------|-------------|
| `persona_id` | Unique identifier (e.g., `cmo`, `cto`) |
| `label` | Display name (e.g., "Chief Marketing Officer") |
| `tier` | Executive / Senior Leadership / Practitioner |
| `pains` | Pain points and challenges |
| `kpis` | Key performance indicators they care about |
| `goals` | Business objectives |
| `objections` | Common objections to address |
| `tone_hints` | Preferred communication style |
| `preferred_terms` / `banned_terms` | Language guidance |

**Personas Included (v2.2.0):**
- **Executive:** CMO, CFO, CIO, CTO, CDO, CXO, CRO
- **Senior Leadership:** VP Marketing, VP Sales, VP IT
- **Practitioner:** Marketing Ops, Creative Director, IT Manager, Data Analyst, Content Strategist

**Use When:** Tailoring content for specific buyer roles, checking WIIFM alignment, tone matching.

---

### `play_catalog.yaml`

**Purpose:** FY26 sales play taxonomy mapping Portfolio → Strategic → Tactical → Industry levels.

**Hierarchy Levels:**

| Level | Name | Description |
|-------|------|-------------|
| L1 | Portfolio Play | Highest level, transformational vision |
| L2 | Strategic Sales Play | Core sales play, problem/solution focused |
| L3 | Application Sales Tactic | Product-specific tactics |
| L4 | Industry Adaptation | Vertical-specific variations |

**Use When:** Classifying content by play level, ensuring play alignment.

---

### `sales_stages.yaml`

**Purpose:** Adobe sales methodology defining stages from Account Planning to Renewal.

**Stages (SS0-SS10):**

| Stage | Name | Customer Journey |
|-------|------|------------------|
| SS0 | Account Planning | Discover |
| SS1 | Prospect Outreach | Discover |
| SS2 | Discovery | Explore |
| SS3 | Qualification | Explore |
| SS4 | Solution Design | Buy |
| SS5 | Proposal | Buy |
| SS6 | Negotiation | Buy |
| SS7 | Close | Buy |
| SS8 | Implementation | Implement |
| SS9 | Adoption | Adopt |
| SS10 | Renewal & Expand | Grow |

**Per-Stage Configuration:**
- `story_emphasis` — SOAR weights (S/O/A/R percentages)
- `tone` — Recommended communication tone
- `exit_criteria_customer` / `exit_criteria_internal` — Stage gates
- `key_stakeholders` — Who's involved
- `example_messages` — Sample language

**Use When:** Determining narrative emphasis by sales stage, stage-appropriate messaging.

---

### `soar_mapping.yaml`

**Purpose:** SOAR framework detection keywords and weights.

**SOAR Elements:**

| Element | Purpose |
|---------|---------|
| **S** (Situation) | Current state, context, challenges |
| **O** (Opportunity) | What's possible, vision, potential |
| **A** (Approach) | How to get there, solution, method |
| **R** (Results) | Outcomes, proof, evidence |

**Use When:** Detecting SOAR balance in content, keyword-based SOAR classification.

---

### `soar_hierarchy_rules.yaml`

**Purpose:** SOAR requirements by play level (L1-L4), defining how SOAR emphasis shifts at each level.

**Key Contents:**
- Per-level SOAR weight requirements
- Minimum element coverage by level
- Level-appropriate SOAR balance rules

**Use When:** Validating SOAR balance against play level requirements.

---

## Common Tasks

| Task | File to Edit |
|------|--------------|
| Add/modify buyer persona | `personas.yaml` |
| Update persona pain points | `personas.yaml` |
| Modify sales stage definitions | `sales_stages.yaml` |
| Adjust SOAR weights by stage | `sales_stages.yaml` |
| Update play hierarchy/catalog | `play_catalog.yaml` |
| Add SOAR detection keywords | `soar_mapping.yaml` |
| Update level-specific SOAR rules | `soar_hierarchy_rules.yaml` |
| Add new sales play | `plays/` (new file) |

---

## Relationship to Other Config Directories

| Directory | Relationship |
|-----------|--------------|
| `../domain/` | Business terminology referenced by personas; files moved FROM domain/ |
| `../narrative/` | Story structures used in sales content |
| `../rubrics/` | Scoring criteria for sales content evaluation |

---

## Version History

| Date | Changes |
|------|---------|
| 2026-01-19 | Renamed `sales_play_hierarchy.yaml` to `play_catalog.yaml` |
| 2026-01-19 | Renamed `SOAR_to_Sales_Play_hierarchy.yaml` to `soar_hierarchy_rules.yaml` |
| 2026-01-19 | Metadata standardization: added `_meta:` blocks per `_SCHEMA.md` |
| 2026-01-19 | Consolidated sales files from `domain/`: personas, soar_mapping, play hierarchy |
| 2026-01-06 | Added CXO, CRO personas; added DMe End-User Productivity L1/L2 |
| 2024-12-16 | Initial sales configuration |

---

## Related Documentation

- **Schema Definition**: `../_SCHEMA.md`
- **Root README**: `../README.md`
- **Domain Config**: `../domain/README.md` (for business terminology)
- **Narrative Config**: `../narrative/README.md` (for SOAR narrative structures)
