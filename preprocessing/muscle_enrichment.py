"""
preprocessing/muscle_enrichment.py
-----------------------------------
Given a raw Muscle_Name from the spine/upper-body OpenSim model,
enriches each row with anatomical metadata and drops non-muscle rows
(actuators, force sensors, time stamps).

Public API:
    enrich_muscle_name(raw_name: str) -> dict | None
    filter_and_enrich(df, muscle_col="Muscle_Name") -> pd.DataFrame
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# 1. PATTERNS THAT IDENTIFY NON-MUSCLE ROWS → DROP
# ---------------------------------------------------------------------------
_DROP_PATTERNS = [
    r"^time$",
    r"_actuator$",         # elbow_L_actuator, pelvic_tilt_actuator
    r"HandForce_",         # hand_L_LeftHandForce_Fx …
    r"ScapForce_",         # scapula_R_LeftScapForce_Fx …
    r"^rib\d+[LR]act$",   # rib1Lact … rib10Ract
]
_DROP_RE = re.compile("|".join(_DROP_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# 2. MUSCLE FAMILY LOOKUP TABLE
#    key  = lowercase prefix that appears at the START of Muscle_Name
#    Sorted longest-first at runtime so more-specific keys win.
# ---------------------------------------------------------------------------
_MUSCLE_MAP = {
    # ── PSOAS ──────────────────────────────────────────────────────────────
    "ps": dict(
        full_name="Psoas Major",
        muscle_group="Hip Flexor / Lumbar",
        functional_region="Lumbar",
        primary_actions=["Trunk Flexion", "Hip Flexion", "Lumbar Stabilisation"],
        origin="Lumbar vertebral bodies & transverse processes (L1–L5)",
        insertion="Lesser trochanter of femur",
    ),
    # ── ILIOCOSTALIS ───────────────────────────────────────────────────────
    "il": dict(
        full_name="Iliocostalis",
        muscle_group="Back Extensor",
        functional_region="Lumbar / Thoracic",
        primary_actions=["Spinal Extension", "Lateral Flexion"],
        origin="Iliac crest / sacrum; lower ribs",
        insertion="Ribs; cervical transverse processes",
    ),
    # ── LONGISSIMUS LUMBORUM ───────────────────────────────────────────────
    "ltpl": dict(
        full_name="Longissimus Thoracis pars Lumborum",
        muscle_group="Back Extensor",
        functional_region="Lumbar",
        primary_actions=["Spinal Extension", "Lateral Flexion"],
        origin="Iliac crest / lumbar transverse processes",
        insertion="Thoracic transverse processes / ribs",
    ),
    # ── LONGISSIMUS THORACIS ───────────────────────────────────────────────
    "ltpt": dict(
        full_name="Longissimus Thoracis pars Thoracis",
        muscle_group="Back Extensor",
        functional_region="Thoracic",
        primary_actions=["Spinal Extension", "Lateral Flexion"],
        origin="Thoracic transverse processes",
        insertion="Ribs / thoracic transverse processes",
    ),
    # ── MULTIFIDUS (thoracic, prefix 'multifidus') ─────────────────────────
    "multifidus": dict(
        full_name="Thoracic Multifidus",
        muscle_group="Back Extensor",
        functional_region="Thoracic",
        primary_actions=["Spinal Extension", "Rotational Stabilisation"],
        origin="Thoracic spinous processes / laminae",
        insertion="Thoracic transverse / mamillary processes",
    ),
    # ── MULTIFIDUS (lumbar, prefix 'mf') ──────────────────────────────────
    "mf": dict(
        full_name="Lumbar Multifidus",
        muscle_group="Back Extensor",
        functional_region="Lumbar",
        primary_actions=["Spinal Extension", "Rotational Stabilisation"],
        origin="Lumbar spinous processes / laminae",
        insertion="Mamillary processes 2–4 levels caudal",
    ),
    # ── DEEP CERVICAL MULTIFIDUS ───────────────────────────────────────────
    "deepmult": dict(
        full_name="Deep Cervical Multifidus",
        muscle_group="Back Extensor",
        functional_region="Cervical",
        primary_actions=["Cervical Extension", "Rotational Stabilisation"],
        origin="Thoracic spinous processes",
        insertion="Cervical articular pillars",
    ),
    # ── SUPERFICIAL CERVICAL MULTIFIDUS ───────────────────────────────────
    "supmult": dict(
        full_name="Superficial Cervical Multifidus",
        muscle_group="Back Extensor",
        functional_region="Cervical",
        primary_actions=["Cervical Extension", "Lateral Flexion"],
        origin="Thoracic spinous processes",
        insertion="Cervical articular pillars",
    ),
    # ── QUADRATUS LUMBORUM ─────────────────────────────────────────────────
    "ql": dict(
        full_name="Quadratus Lumborum",
        muscle_group="Lateral Trunk",
        functional_region="Lumbar",
        primary_actions=["Lateral Flexion", "Lumbar Stabilisation", "Respiration (forced)"],
        origin="Iliac crest / iliolumbar ligament",
        insertion="12th rib; lumbar transverse processes (L1–L4)",
    ),
    # ── RECTUS ABDOMINIS ───────────────────────────────────────────────────
    "rect_abd": dict(
        full_name="Rectus Abdominis",
        muscle_group="Abdominal",
        functional_region="Trunk",
        primary_actions=["Trunk Flexion", "Abdominal Compression"],
        origin="Pubic crest & pubic symphysis",
        insertion="Xiphoid process; costal cartilages 5–7",
    ),
    # ── INTERNAL OBLIQUE ───────────────────────────────────────────────────
    "io": dict(
        full_name="Internal Oblique",
        muscle_group="Abdominal",
        functional_region="Trunk",
        primary_actions=["Trunk Flexion", "Ipsilateral Rotation", "Lateral Flexion"],
        origin="Thoracolumbar fascia; iliac crest; inguinal ligament",
        insertion="Ribs 10–12; linea alba; pubic crest",
    ),
    # ── EXTERNAL OBLIQUE (coded E0_) ───────────────────────────────────────
    "e0": dict(
        full_name="External Oblique",
        muscle_group="Abdominal",
        functional_region="Trunk",
        primary_actions=["Trunk Flexion", "Contralateral Rotation", "Lateral Flexion"],
        origin="Ribs 5–12 (external surfaces)",
        insertion="Linea alba; pubic tubercle; iliac crest",
    ),
    # ── EXTERNAL INTERCOSTALS ──────────────────────────────────────────────
    "extic": dict(
        full_name="External Intercostals",
        muscle_group="Thoracic",
        functional_region="Thoracic",
        primary_actions=["Rib Elevation", "Inspiration"],
        origin="Lower border of each rib",
        insertion="Upper border of rib below",
    ),
    # ── INTERNAL INTERCOSTALS ──────────────────────────────────────────────
    "intic": dict(
        full_name="Internal Intercostals",
        muscle_group="Thoracic",
        functional_region="Thoracic",
        primary_actions=["Rib Depression", "Forced Expiration"],
        origin="Upper border of each rib (internal surface)",
        insertion="Lower border of rib above",
    ),
    # ── TRANSVERSUS ABDOMINIS ──────────────────────────────────────────────
    "tr": dict(
        full_name="Transversus Abdominis",
        muscle_group="Abdominal",
        functional_region="Trunk",
        primary_actions=["Abdominal Compression", "Lumbar Stabilisation"],
        origin="Thoracolumbar fascia; iliac crest; costal cartilages 7–12",
        insertion="Linea alba; pubic crest",
    ),
    # ── LATISSIMUS DORSI ───────────────────────────────────────────────────
    "ld": dict(
        full_name="Latissimus Dorsi",
        muscle_group="Back / Shoulder",
        functional_region="Thoracolumbar / Shoulder",
        primary_actions=["Shoulder Extension", "Adduction", "Internal Rotation", "Spinal Extension"],
        origin="Spinous processes T7–L5; thoracolumbar fascia; iliac crest; lower ribs",
        insertion="Intertubercular sulcus of humerus",
    ),
    # ── DELTOID ────────────────────────────────────────────────────────────
    "delt1": dict(
        full_name="Deltoid (Anterior)",
        muscle_group="Shoulder",
        functional_region="Shoulder",
        primary_actions=["Shoulder Flexion", "Internal Rotation", "Horizontal Adduction"],
        origin="Lateral clavicle",
        insertion="Deltoid tuberosity of humerus",
    ),
    "delt2": dict(
        full_name="Deltoid (Middle)",
        muscle_group="Shoulder",
        functional_region="Shoulder",
        primary_actions=["Shoulder Abduction"],
        origin="Acromion of scapula",
        insertion="Deltoid tuberosity of humerus",
    ),
    "delt3": dict(
        full_name="Deltoid (Posterior)",
        muscle_group="Shoulder",
        functional_region="Shoulder",
        primary_actions=["Shoulder Extension", "External Rotation", "Horizontal Abduction"],
        origin="Scapular spine",
        insertion="Deltoid tuberosity of humerus",
    ),
    # ── ROTATOR CUFF ───────────────────────────────────────────────────────
    "infsp": dict(
        full_name="Infraspinatus",
        muscle_group="Rotator Cuff",
        functional_region="Shoulder",
        primary_actions=["External Rotation", "Shoulder Stabilisation"],
        origin="Infraspinous fossa of scapula",
        insertion="Greater tubercle of humerus (middle facet)",
    ),
    "subsc": dict(
        full_name="Subscapularis",
        muscle_group="Rotator Cuff",
        functional_region="Shoulder",
        primary_actions=["Internal Rotation", "Shoulder Stabilisation"],
        origin="Subscapular fossa",
        insertion="Lesser tubercle of humerus",
    ),
    "supsp": dict(
        full_name="Supraspinatus",
        muscle_group="Rotator Cuff",
        functional_region="Shoulder",
        primary_actions=["Shoulder Abduction initiation", "Shoulder Stabilisation"],
        origin="Supraspinous fossa of scapula",
        insertion="Greater tubercle of humerus (superior facet)",
    ),
    "tmaj": dict(
        full_name="Teres Major",
        muscle_group="Shoulder",
        functional_region="Shoulder",
        primary_actions=["Internal Rotation", "Extension", "Adduction"],
        origin="Inferior angle of scapula",
        insertion="Intertubercular sulcus of humerus",
    ),
    "tmin": dict(
        full_name="Teres Minor",
        muscle_group="Rotator Cuff",
        functional_region="Shoulder",
        primary_actions=["External Rotation", "Shoulder Stabilisation"],
        origin="Lateral border of scapula (upper 2/3)",
        insertion="Greater tubercle of humerus (inferior facet)",
    ),
    "corb": dict(
        full_name="Coracobrachialis",
        muscle_group="Shoulder",
        functional_region="Shoulder",
        primary_actions=["Shoulder Flexion", "Adduction"],
        origin="Coracoid process of scapula",
        insertion="Medial humerus (mid-shaft)",
    ),
    # ── PECTORALIS MAJOR ───────────────────────────────────────────────────
    "pecm1": dict(
        full_name="Pectoralis Major (Clavicular head)",
        muscle_group="Chest",
        functional_region="Shoulder / Chest",
        primary_actions=["Shoulder Flexion", "Horizontal Adduction", "Internal Rotation"],
        origin="Medial clavicle",
        insertion="Intertubercular sulcus of humerus",
    ),
    "pecm2": dict(
        full_name="Pectoralis Major (Sternal head)",
        muscle_group="Chest",
        functional_region="Shoulder / Chest",
        primary_actions=["Shoulder Adduction", "Internal Rotation", "Extension"],
        origin="Sternum; costal cartilages 1–6",
        insertion="Intertubercular sulcus of humerus",
    ),
    "pecm3": dict(
        full_name="Pectoralis Major (Abdominal head)",
        muscle_group="Chest",
        functional_region="Shoulder / Chest",
        primary_actions=["Shoulder Adduction", "Internal Rotation"],
        origin="Anterior rectus sheath / abdominal aponeurosis",
        insertion="Intertubercular sulcus of humerus",
    ),
    # ── SERRATUS ANTERIOR ──────────────────────────────────────────────────
    "serrant": dict(
        full_name="Serratus Anterior",
        muscle_group="Shoulder / Thoracic",
        functional_region="Thoracic / Shoulder",
        primary_actions=["Scapular Protraction", "Upward Rotation", "Rib Elevation"],
        origin="Lateral surfaces of ribs 1–9",
        insertion="Medial border & inferior angle of scapula",
    ),
    # ── TRAPEZIUS ──────────────────────────────────────────────────────────
    "trap_acr": dict(
        full_name="Trapezius (Acromial / Middle)",
        muscle_group="Back / Shoulder",
        functional_region="Thoracic / Shoulder",
        primary_actions=["Scapular Retraction", "Upward Rotation"],
        origin="Thoracic spinous processes (T1–T3) / acromion",
        insertion="Acromion / scapular spine",
    ),
    "trap_cl": dict(
        full_name="Trapezius (Clavicular / Upper)",
        muscle_group="Back / Shoulder",
        functional_region="Cervical / Shoulder",
        primary_actions=["Scapular Elevation", "Upward Rotation", "Cervical Extension"],
        origin="Occiput; cervical spinous processes",
        insertion="Lateral clavicle",
    ),
    "trap_inf": dict(
        full_name="Trapezius (Inferior)",
        muscle_group="Back / Shoulder",
        functional_region="Thoracic / Shoulder",
        primary_actions=["Scapular Depression", "Upward Rotation", "Retraction"],
        origin="Spinous processes T4–T12",
        insertion="Scapular spine",
    ),
    # ── LEVATOR SCAPULAE ───────────────────────────────────────────────────
    "levator_scap": dict(
        full_name="Levator Scapulae",
        muscle_group="Neck / Shoulder",
        functional_region="Cervical / Shoulder",
        primary_actions=["Scapular Elevation", "Cervical Lateral Flexion"],
        origin="Transverse processes C1–C4",
        insertion="Superior angle / medial border of scapula",
    ),
    # ── SCM / NECK ─────────────────────────────────────────────────────────
    "cleid_mast": dict(
        full_name="Cleidomastoid (SCM – Clavicular)",
        muscle_group="Neck",
        functional_region="Cervical",
        primary_actions=["Cervical Flexion", "Contralateral Rotation", "Lateral Flexion"],
        origin="Clavicle (medial)",
        insertion="Mastoid process",
    ),
    "cleid_occ": dict(
        full_name="Cleidooccipital (SCM – Occipital)",
        muscle_group="Neck",
        functional_region="Cervical",
        primary_actions=["Cervical Flexion", "Contralateral Rotation"],
        origin="Clavicle (medial)",
        insertion="Occipital bone (lateral)",
    ),
    "stern_mast": dict(
        full_name="Sternomastoid (SCM – Sternal)",
        muscle_group="Neck",
        functional_region="Cervical",
        primary_actions=["Cervical Flexion", "Contralateral Rotation", "Lateral Flexion"],
        origin="Manubrium of sternum",
        insertion="Mastoid process",
    ),
    # ── SCALENES ───────────────────────────────────────────────────────────
    "scalenus_ant": dict(
        full_name="Scalenus Anterior",
        muscle_group="Neck",
        functional_region="Cervical",
        primary_actions=["Cervical Flexion", "Lateral Flexion", "Rib 1 Elevation"],
        origin="Anterior tubercles C3–C6",
        insertion="1st rib (scalene tubercle)",
    ),
    "scalenus_med": dict(
        full_name="Scalenus Medius",
        muscle_group="Neck",
        functional_region="Cervical",
        primary_actions=["Cervical Lateral Flexion", "Rib 1 Elevation"],
        origin="Posterior tubercles C2–C7",
        insertion="1st rib (posterior groove)",
    ),
    "scalenus_post": dict(
        full_name="Scalenus Posterior",
        muscle_group="Neck",
        functional_region="Cervical",
        primary_actions=["Cervical Lateral Flexion", "Rib 2 Elevation"],
        origin="Posterior tubercles C4–C6",
        insertion="2nd rib (outer surface)",
    ),
    # ── SEMISPINALIS ───────────────────────────────────────────────────────
    "semi_cap": dict(
        full_name="Semispinalis Capitis",
        muscle_group="Neck / Back Extensor",
        functional_region="Cervical",
        primary_actions=["Head Extension", "Contralateral Rotation"],
        origin="Articular processes C4–C6; transverse processes T1–T6",
        insertion="Occipital bone (between nuchal lines)",
    ),
    "semi_cerv": dict(
        full_name="Semispinalis Cervicis",
        muscle_group="Back Extensor",
        functional_region="Cervical",
        primary_actions=["Cervical Extension", "Contralateral Rotation"],
        origin="Transverse processes T1–T6",
        insertion="Cervical spinous processes C2–C5",
    ),
    # ── SPLENIUS ───────────────────────────────────────────────────────────
    "splen_cap": dict(
        full_name="Splenius Capitis",
        muscle_group="Neck",
        functional_region="Cervical",
        primary_actions=["Head Extension", "Ipsilateral Rotation", "Lateral Flexion"],
        origin="Ligamentum nuchae; spinous processes C7–T3",
        insertion="Mastoid process; occipital bone",
    ),
    "splen_cerv": dict(
        full_name="Splenius Cervicis",
        muscle_group="Neck",
        functional_region="Cervical",
        primary_actions=["Cervical Extension", "Ipsilateral Rotation"],
        origin="Spinous processes T3–T6",
        insertion="Transverse processes C1–C3",
    ),
    # ── ILIOCOSTALIS CERVICIS ──────────────────────────────────────────────
    "iliocost_cerv": dict(
        full_name="Iliocostalis Cervicis",
        muscle_group="Back Extensor",
        functional_region="Cervical",
        primary_actions=["Cervical Extension", "Lateral Flexion"],
        origin="Ribs 3–6 (angles)",
        insertion="Transverse processes C4–C6",
    ),
    # ── LONGUS COLLI ───────────────────────────────────────────────────────
    "long_col": dict(
        full_name="Longus Colli",
        muscle_group="Neck",
        functional_region="Cervical",
        primary_actions=["Cervical Flexion", "Stabilisation"],
        origin="Vertebral bodies C1–T3 / transverse processes C3–C5",
        insertion="Anterior arch of atlas; vertebral bodies C1–C4",
    ),
    # ── LONGISSIMUS CERVICIS ───────────────────────────────────────────────
    "longissi_cerv": dict(
        full_name="Longissimus Cervicis",
        muscle_group="Back Extensor",
        functional_region="Cervical",
        primary_actions=["Cervical Extension", "Lateral Flexion"],
        origin="Transverse processes T4–T5",
        insertion="Transverse processes C2–C6",
    ),
}

# Pre-sort keys longest-first so more-specific prefixes win
_SORTED_KEYS = sorted(_MUSCLE_MAP.keys(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------
def _extract_side(name: str) -> str:
    """Return 'Left', 'Right', or 'Bilateral'."""
    if re.search(r"[_\-][lL]$", name):
        return "Left"
    if re.search(r"[_\-][rR]$", name):
        return "Right"
    if name.endswith("_L") or name.endswith("L"):
        return "Left"
    if name.endswith("_R") or name.endswith("R"):
        return "Right"
    return "Bilateral"


def _extract_spinal_levels(name: str) -> str:
    """Extract vertebral level tokens (e.g. L3, T6, C4) from the name."""
    levels = re.findall(r"[LTC]\d{1,2}", name, re.IGNORECASE)
    return ", ".join(sorted(set(lv.upper() for lv in levels))) if levels else ""


def _find_family(raw_name: str):
    """Return (key, info_dict) for the best matching muscle prefix, or (None, None)."""
    lower = raw_name.lower()
    for key in _SORTED_KEYS:
        if lower.startswith(key):
            return key, _MUSCLE_MAP[key]
    return None, None


# ---------------------------------------------------------------------------
# 4. MAIN PUBLIC FUNCTION
# ---------------------------------------------------------------------------
def enrich_muscle_name(raw_name: str) -> dict:
    """
    Return a dict of enrichment columns for *raw_name*, or
    {'should_drop': True} if the row is not a real muscle.
    """
    if not isinstance(raw_name, str) or _DROP_RE.search(raw_name):
        return {"should_drop": True}

    side   = _extract_side(raw_name)
    levels = _extract_spinal_levels(raw_name)
    _, info = _find_family(raw_name)

    if info is None:
        return {
            "should_drop":       False,
            "full_name":         raw_name,
            "muscle_group":      "Unknown",
            "functional_region": "Unknown",
            "side":              side,
            "spinal_level":      levels,
            "primary_actions":   "",
            "origin":            "",
            "insertion":         "",
        }

    return {
        "should_drop":       False,
        "full_name":         info["full_name"],
        "muscle_group":      info["muscle_group"],
        "functional_region": info["functional_region"],
        "side":              side,
        "spinal_level":      levels,
        "primary_actions":   ", ".join(info["primary_actions"]),
        "origin":            info["origin"],
        "insertion":         info["insertion"],
    }


# ---------------------------------------------------------------------------
# 5. CONVENIENCE: apply to a full DataFrame
# ---------------------------------------------------------------------------
def filter_and_enrich(df: pd.DataFrame, muscle_col: str = "Muscle_Name") -> pd.DataFrame:
    """
    1. Calls enrich_muscle_name() for every row.
    2. Drops rows where should_drop is True.
    3. Appends enrichment columns to the DataFrame.
    Returns the cleaned, enriched DataFrame (index reset).
    """
    if df.empty or muscle_col not in df.columns:
        return df

    enriched_records = df[muscle_col].apply(enrich_muscle_name)
    enriched_df = pd.DataFrame(enriched_records.tolist(), index=df.index)

    keep_mask  = enriched_df["should_drop"] == False  # noqa: E712
    df_clean   = df[keep_mask].copy()
    extra_cols = enriched_df[keep_mask].drop(columns=["should_drop"])

    return pd.concat(
        [df_clean.reset_index(drop=True), extra_cols.reset_index(drop=True)],
        axis=1
    )


# ---------------------------------------------------------------------------
# 6. QUICK SELF-TEST  (python -m preprocessing.muscle_enrichment)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "Ps_L1_VB_r", "rect_abd_l", "IL_L3_r", "LTpL_L2_l",
        "MF_m3s_r", "QL_ant_I_2-12_1_l", "IO3_r", "E0_R7_l",
        "ExtIC_IS5_3_R", "LD_T10_r", "DELT1_l", "INFSP",
        "trap_inf_T8_L", "SerrAnt4_1_R", "multifidus_T5_T3_L",
        "deepmult-T1-C5_L", "splen_cerv_c3_T4_L",
        "time", "elbow_L_actuator", "hand_L_LeftHandForce_Fx",
        "rib5Lact", "pelvic_tilt_actuator",
    ]
    print(f"{'Muscle_Name':<35} {'full_name':<40} {'group':<25} {'side':<10} {'levels'}")
    print("-" * 120)
    for n in samples:
        r = enrich_muscle_name(n)
        if r.get("should_drop"):
            print(f"{n:<35} [DROPPED]")
        else:
            print(f"{n:<35} {r['full_name']:<40} {r['muscle_group']:<25} {r['side']:<10} {r['spinal_level']}")
