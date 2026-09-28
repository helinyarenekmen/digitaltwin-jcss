"""
Ablation configurations for Layer A persona generation.

GROUP_ALIASES maps short alias keys → verbatim Excel "Variable Group" strings.
AblationConfig.group_aliases uses alias keys only; the engine resolves to
Excel strings at runtime via GROUP_ALIASES.

11 configurations total: C0, C1, C3–C11. C2 is intentionally omitted
(deprecated random-baseline). Non-contiguous numbering preserved for version
history readability. See PERSONA_SPEC.md §6.
"""

from dataclasses import dataclass, field

# ⚠ Verbatim Excel strings — do not "fix" typos or trailing spaces.
# These must match the "Variable Group" column in TGSS2024_Persona_Variables.xlsx exactly.
GROUP_ALIASES: dict[str, str] = {
    "demographics": "1. Demographics",
    "belief":       "2. Befief, Ideology, Identity ",   # typo + trailing space preserved
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
    config_id: str          # "C0", "C1", ...
    name: str               # human-readable
    group_aliases: list[str]  # subset of ALL_GROUPS
    description: str        # one-liner for logs and paper tables
    drop_vars: tuple[str, ...] = ()  # optional per-variable exclusions inside kept groups

    def excel_groups(self) -> list[str]:
        """Resolve alias keys to verbatim Excel group strings."""
        return [GROUP_ALIASES[a] for a in self.group_aliases]


CONFIGS: list[AblationConfig] = [
    AblationConfig(
        config_id="C0",
        name="Demographics-only baseline",
        group_aliases=["demographics"],
        description="Lower bound: age, gender, education, region, ethnicity only.",
    ),
    AblationConfig(
        config_id="C1",
        name="Full set (main spec)",
        group_aliases=ALL_GROUPS,
        description="Upper bound: all six variable groups; conservative fallback for downstream persona specification",
    ),
    AblationConfig(
        config_id="C3",
        name="All except Demographics",
        group_aliases=["belief", "economic", "social", "social_psych", "political"],
        description="Drop group 1: tests whether demographic anchoring matters.",
    ),
    AblationConfig(
        config_id="C4",
        name="All except Belief/Ideology/Identity",
        group_aliases=["demographics", "economic", "social", "social_psych", "political"],
        description="Drop group 2: expected large drop on womenwork and neilang.",
    ),
    AblationConfig(
        config_id="C5",
        name="All except Economic",
        group_aliases=["demographics", "belief", "social", "social_psych", "political"],
        description="Drop group 3: tests economic grievance channel for pacdemons.",
    ),
    AblationConfig(
        config_id="C6",
        name="All except Social values",
        group_aliases=["demographics", "belief", "economic", "social_psych", "political"],
        description="Drop group 4: isolates social attitudes contribution.",
    ),
    AblationConfig(
        config_id="C7",
        name="All except Social-Psychological",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Drop group 5: removes wellbeing and out-group contact scaffolding.",
    ),
    AblationConfig(
        config_id="C8",
        name="All except Political",
        group_aliases=["demographics", "belief", "economic", "social", "social_psych"],
        description="Drop group 6: expected large drop on pacdemons.",
    ),
    AblationConfig(
        config_id="C9",
        name="Grievance + mobilization core",
        group_aliases=["demographics", "economic", "political"],
        description="Theory-driven minimal set: economic grievance + political behavior.",
    ),
    AblationConfig(
        config_id="C10",
        name="Identity + values core",
        group_aliases=["demographics", "belief", "social"],
        description="Theory-driven minimal set: identity and value orientations.",
    ),
    AblationConfig(
        config_id="C11",
        name="Park-style minimal",
        group_aliases=["demographics", "political"],
        description="Literature reference: replicates Park et al. 2024 minimal format.",
    ),
    AblationConfig(
        config_id="C7star",
        name="C7 minus past-year political participation (leakage test)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness: C7 with 9 past-year political-participation variables removed from the political group to probe outcome leakage into pacdemons.",
        drop_vars=("paccontact", "paccompl", "paccimer", "pacparty",
                   "pacvolunteer", "pacboycott", "paconline", "paccult",
                   "paccharity"),
    ),
    # Single-variable leakage-guarded C7 variants — one per new outcome probe.
    # Each drops only the outcome variable itself from the persona to prevent
    # trivial-copy leakage while otherwise matching C7 (5 groups).
    AblationConfig(
        config_id="C7starpacparty",
        name="C7 minus pacparty (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with pacparty removed to predict pacparty.",
        drop_vars=("pacparty",),
    ),
    AblationConfig(
        config_id="C7starpacboycott",
        name="C7 minus pacboycott (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with pacboycott removed to predict pacboycott.",
        drop_vars=("pacboycott",),
    ),
    AblationConfig(
        config_id="C7starpaconline",
        name="C7 minus paconline (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with paconline removed to predict paconline.",
        drop_vars=("paconline",),
    ),
    AblationConfig(
        config_id="C7starpolint",
        name="C7 minus polint (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with polint removed to predict polint.",
        drop_vars=("polint",),
    ),
    AblationConfig(
        config_id="C7starthimmig",
        name="C7 minus thimmig (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with thimmig removed to predict thimmig.",
        drop_vars=("thimmig",),
    ),
    AblationConfig(
        config_id="C7starpolminor",
        name="C7 minus polminor (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with polminor removed to predict polminor.",
        drop_vars=("polminor",),
    ),
    AblationConfig(
        config_id="C7starpolfrlim",
        name="C7 minus polfrlim (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with polfrlim removed to predict polfrlim.",
        drop_vars=("polfrlim",),
    ),
    AblationConfig(
        config_id="C7starsatdem",
        name="C7 minus satdem (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with satdem removed to predict satdem.",
        drop_vars=("satdem",),
    ),
    AblationConfig(
        config_id="C7starfamroles",
        name="C7 minus famroles (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with famroles removed to predict famroles.",
        drop_vars=("famroles",),
    ),
    AblationConfig(
        config_id="C7starpaccontact",
        name="C7 minus paccontact (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with paccontact removed to predict paccontact.",
        drop_vars=("paccontact",),
    ),
    AblationConfig(
        config_id="C7starpaccompl",
        name="C7 minus paccompl (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with paccompl removed to predict paccompl.",
        drop_vars=("paccompl",),
    ),
    AblationConfig(
        config_id="C7starpaccimer",
        name="C7 minus paccimer (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with paccimer removed to predict paccimer.",
        drop_vars=("paccimer",),
    ),
    AblationConfig(
        config_id="C7starpacvolunteer",
        name="C7 minus pacvolunteer (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with pacvolunteer removed to predict pacvolunteer.",
        drop_vars=("pacvolunteer",),
    ),
    AblationConfig(
        config_id="C7starpaccult",
        name="C7 minus paccult (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with paccult removed to predict paccult.",
        drop_vars=("paccult",),
    ),
    AblationConfig(
        config_id="C7starpaccharity",
        name="C7 minus paccharity (leakage-guarded)",
        group_aliases=["demographics", "belief", "economic", "social", "political"],
        description="Robustness probe: C7 with paccharity removed to predict paccharity.",
        drop_vars=("paccharity",),
    ),
]

# Fast lookup by config_id
CONFIGS_BY_ID: dict[str, AblationConfig] = {c.config_id: c for c in CONFIGS}

# Run priority per §6 (mandatory → theoretically loaded → minimal → remaining → reference)
RUN_PRIORITY: list[str] = ["C0", "C1", "C4", "C5", "C8", "C9", "C10", "C3", "C6", "C7", "C11"]
