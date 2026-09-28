"""
Persona ablation configurations for calibration and Appendix G transfer probes.

Numbering
---------
Public paper labels are contiguous C0-C10 (see paper Table 4 and Appendix B).
Internal history preserved an older non-contiguous scheme (C0, C1, C3-C11)
in which C2 was deprecated. The `legacy_id` field records that mapping so the
archived JSONL cache filenames — which encode the legacy id — remain traceable.

  paper id → legacy id
      C0   →   C0
      C1   →   C1
      C2   →   C3
      C3   →   C4
      C4   →   C5
      C5   →   C6
      C6   →   C7      (paper's calibrated protocol config)
      C7   →   C8
      C8   →   C9
      C9   →   C10
     C10   →   C11

The Appendix G target-item holdout probes (C6-holdout-<item>) each remove one
outcome variable from the paper C6 configuration to prevent trivial-copy
leakage; they are otherwise identical to C6.
"""

from dataclasses import dataclass


# --- Excel "Variable Group" strings (verbatim from the TGSS persona dictionary) ---
# Do NOT "fix" the typo or trailing space in "Befief" — they match the source Excel.
GROUP_ALIASES: dict[str, str] = {
    "demographics": "1. Demographics",
    "belief":       "2. Befief, Ideology, Identity ",
    "economic":     "3. Economic",
    "social":       "4. Social values & attitudes",
    "social_psych": "5. Social-Psychological",
    "political":    "6. Political",
}

ALL_GROUPS: list[str] = [
    "demographics", "belief", "economic", "social", "social_psych", "political"
]


@dataclass(frozen=True)
class AblationConfig:
    config_id: str                       # paper label (C0-C10) or C6-holdout-<item>
    legacy_id: str                       # internal id used in archived JSONL filenames
    name: str
    group_aliases: list[str]
    description: str
    drop_vars: tuple[str, ...] = ()      # per-variable exclusions inside kept groups

    def excel_groups(self) -> list[str]:
        return [GROUP_ALIASES[a] for a in self.group_aliases]


# --- Paper configurations C0-C10 ---
CONFIGS: list[AblationConfig] = [
    AblationConfig(
        config_id="C0", legacy_id="C0",
        name="Demographics-only baseline",
        group_aliases=["demographics"],
        description="Lower bound: demographics only.",
    ),
    AblationConfig(
        config_id="C1", legacy_id="C1",
        name="Full profile",
        group_aliases=ALL_GROUPS,
        description="Upper bound: all six variable groups.",
    ),
    AblationConfig(
        config_id="C2", legacy_id="C3",
        name="All except Demographics",
        group_aliases=["belief", "economic", "social", "social_psych", "political"],
        description="Drops demographics; tests whether demographic anchoring matters.",
    ),
    AblationConfig(
        config_id="C3", legacy_id="C4",
        name="All except Belief/Ideology/Identity",
        group_aliases=["demographics", "economic", "social", "social_psych", "political"],
        description="Drops belief group; expected large drop on womenwork.",
    ),
    AblationConfig(
        config_id="C4", legacy_id="C5",
        name="All except Economic",
        group_aliases=["demographics", "belief", "social", "social_psych", "political"],
        description="Drops economic group; tests economic grievance channel.",
    ),
    AblationConfig(
        config_id="C5", legacy_id="C6",
        name="All except Social values",
        group_aliases=["demographics", "belief", "economic", "social_psych", "political"],
        description="Drops social-values group; isolates its contribution.",
    ),
    AblationConfig(
        config_id="C6", legacy_id="C7",
        name="All except Social-Psychological",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Drops social-psychological group. The calibrated protocol config in the paper.",
    ),
    AblationConfig(
        config_id="C7", legacy_id="C8",
        name="All except Political",
        group_aliases=["demographics", "belief", "economic", "social", "social_psych"],
        description="Drops political group; expected large drop on pacdemons.",
    ),
    AblationConfig(
        config_id="C8", legacy_id="C9",
        name="Grievance + mobilization core",
        group_aliases=["demographics", "economic", "political"],
        description="Theory-driven minimal set: economic grievance + political behavior.",
    ),
    AblationConfig(
        config_id="C9", legacy_id="C10",
        name="Identity + values core",
        group_aliases=["demographics", "belief", "social"],
        description="Theory-driven minimal set: identity and value orientations.",
    ),
    AblationConfig(
        config_id="C10", legacy_id="C11",
        name="Park-style minimal",
        group_aliases=["demographics", "political"],
        description="Literature reference: replicates Park et al. 2024 minimal format.",
    ),

    # --- Appendix G transfer-item holdout probes (paper C6 minus one outcome each) ---
    AblationConfig(
        config_id="C6-holdout-pacvolunteer", legacy_id="C7starpacvolunteer",
        name="C6 minus pacvolunteer",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Appendix G binary probe: predicts pacvolunteer with itself removed.",
        drop_vars=("pacvolunteer",),
    ),
    AblationConfig(
        config_id="C6-holdout-paccontact", legacy_id="C7starpaccontact",
        name="C6 minus paccontact",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Appendix G binary probe: predicts paccontact with itself removed.",
        drop_vars=("paccontact",),
    ),
    AblationConfig(
        config_id="C6-holdout-paccompl", legacy_id="C7starpaccompl",
        name="C6 minus paccompl",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Appendix G binary probe: predicts paccompl with itself removed.",
        drop_vars=("paccompl",),
    ),
    AblationConfig(
        config_id="C6-holdout-famroles", legacy_id="C7starfamroles",
        name="C6 minus famroles",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Appendix G ordinal probe: predicts famroles with itself removed.",
        drop_vars=("famroles",),
    ),
    AblationConfig(
        config_id="C6-holdout-satdem", legacy_id="C7starsatdem",
        name="C6 minus satdem",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Appendix G ordinal probe: predicts satdem with itself removed.",
        drop_vars=("satdem",),
    ),
    AblationConfig(
        config_id="C6-holdout-polint", legacy_id="C7starpolint",
        name="C6 minus polint",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Appendix G ordinal probe: predicts polint with itself removed.",
        drop_vars=("polint",),
    ),
]


# --- Lookup helpers ---
CONFIGS_BY_ID: dict[str, AblationConfig] = {c.config_id: c for c in CONFIGS}
CONFIGS_BY_LEGACY_ID: dict[str, AblationConfig] = {c.legacy_id: c for c in CONFIGS}

# Run order used by the calibration funnel (paper Section 3.3). Non-paper
# configs (Appendix G holdouts) run only when their outcome is targeted.
RUN_PRIORITY: list[str] = ["C0", "C1", "C3", "C4", "C7", "C8", "C9", "C5", "C6", "C2", "C10"]


def to_paper_id(any_id: str) -> str:
    """Map either a paper id or a legacy id back to the paper id."""
    if any_id in CONFIGS_BY_ID:
        return any_id
    if any_id in CONFIGS_BY_LEGACY_ID:
        return CONFIGS_BY_LEGACY_ID[any_id].config_id
    raise KeyError(f"unknown config id: {any_id}")


def to_legacy_id(any_id: str) -> str:
    """Map either a paper id or a legacy id back to the legacy id."""
    if any_id in CONFIGS_BY_ID:
        return CONFIGS_BY_ID[any_id].legacy_id
    if any_id in CONFIGS_BY_LEGACY_ID:
        return any_id
    raise KeyError(f"unknown config id: {any_id}")
