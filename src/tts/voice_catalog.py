"""
Voice catalog, metadata index, and gender-diverse voice sampling for Piper-TTS.

Maintains registry of all official Piper voice models, classifies speaker gender,
and provides selection algorithms ensuring strictly distinct voices and balanced
gender representation in generated multi-speaker scenes.
"""

# Import Modules
import typing
from pathlib import Path
from dataclasses import field, dataclass

import json
import random
import logging
import urllib.error
import urllib.request

logger: logging.Logger = logging.getLogger(__name__)

CATALOG_URL: str = "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json"
LOCAL_CATALOG_PATH: Path = Path("data/piper_voices/voices.json")

# Known female speaker names in Piper voice repository
KNOWN_FEMALE_NAMES: set[str] = {
    "aegis_female", "alba", "alma", "amy", "anna", "berfin_renas", "berta",
    "cori", "daniela", "dii", "elena", "elsa", "eva_k", "gosia", "hfc_female",
    "irina", "isabella", "jenny_dioco", "jessica", "joy", "kasandra",
    "kathleen", "kerstin", "kristin", "lada", "lessac", "lili", "lisa",
    "ljspeech", "lucia", "maider", "maria", "marta", "marylux", "maya",
    "meera", "mls_9972", "nathalie", "natia", "padmavathi", "paola", "poppy",
    "priyamvada", "prudence", "ramona", "rapunzelina", "raya", "salka",
    "serena", "siwis", "southern_english_female", "tetiana", "tii", "ugla",
    "unmute_female", "upc_ona", "valeria", "xiao_ya",
}

# Known male speaker names in Piper voice repository
KNOWN_MALE_NAMES: set[str] = {
    "aivars", "alan", "alex", "amir", "antton", "arjun", "artur", "bass",
    "bryce", "cadu", "carlfm", "claude", "danny", "darkman", "davefx", "denis",
    "dimitar", "dmitri", "edon", "edresson", "faber", "fasih", "ganji",
    "gilles", "gor", "harri", "hfc_male", "hi_fi_captain", "imre", "jeff",
    "jirka", "joe", "john", "kareem", "karlsson", "kusal", "mihai", "mike",
    "mls_1840", "mls_10246", "mykyta", "norman", "northern_english_male",
    "obadiah", "oleksa", "pavoque", "pierre", "pim", "pratham", "rdh",
    "reza_ibrahim", "riccardo", "rohan", "ronnie", "ruslan", "ryan", "sam",
    "spike", "steinn", "thorsten", "thorsten_emotional", "tom", "upc_pau",
    "venkatesh", "vicente",
}


@dataclass
class VoiceProfile:
    """
    Metadata profile for an official Piper neural voice model.
    """

    key: str
    name: str
    language: str
    lang_family: str
    gender: str
    quality: str
    num_speakers: int
    speaker_id_map: dict[str, int] = field(default_factory=dict)


def infer_gender_from_name(name: str) -> str:
    """
    Infer vocal gender from voice model name identifier.

    Args:
        name (str): Speaker or voice name.

    Returns:
        str: 'female', 'male', or 'unspecified'.
    """

    clean_name: str = name.lower()

    if clean_name in KNOWN_FEMALE_NAMES or "female" in clean_name:
        return "female"

    if clean_name in KNOWN_MALE_NAMES or "male" in clean_name:
        return "male"

    return "unspecified"


# Built-in curated offline index of popular Piper models across languages
CURATED_VOICE_CATALOG: list[VoiceProfile] = [
    # English US
    VoiceProfile("en_US-lessac-low", "lessac", "en_US", "en", "female", "low", 1),
    VoiceProfile("en_US-lessac-medium", "lessac", "en_US", "en", "female", "medium", 1),
    VoiceProfile("en_US-amy-low", "amy", "en_US", "en", "female", "low", 1),
    VoiceProfile("en_US-amy-medium", "amy", "en_US", "en", "female", "medium", 1),
    VoiceProfile("en_US-kathleen-low", "kathleen", "en_US", "en", "female", "low", 1),
    VoiceProfile("en_US-kristin-medium", "kristin", "en_US", "en", "female", "medium", 1),
    VoiceProfile("en_US-hfc_female-medium", "hfc_female", "en_US", "en", "female", "medium", 1),
    VoiceProfile("en_US-bryce-medium", "bryce", "en_US", "en", "male", "medium", 1),
    VoiceProfile("en_US-danny-low", "danny", "en_US", "en", "male", "low", 1),
    VoiceProfile("en_US-joe-medium", "joe", "en_US", "en", "male", "medium", 1),
    VoiceProfile("en_US-john-medium", "john", "en_US", "en", "male", "medium", 1),
    VoiceProfile("en_US-ryan-low", "ryan", "en_US", "en", "male", "low", 1),
    VoiceProfile("en_US-ryan-medium", "ryan", "en_US", "en", "male", "medium", 1),
    VoiceProfile("en_US-hfc_male-medium", "hfc_male", "en_US", "en", "male", "medium", 1),
    VoiceProfile("en_US-arctic-medium", "arctic", "en_US", "en", "male", "medium", 18),
    # English GB
    VoiceProfile("en_GB-alan-low", "alan", "en_GB", "en", "male", "low", 1),
    VoiceProfile("en_GB-alan-medium", "alan", "en_GB", "en", "male", "medium", 1),
    VoiceProfile("en_GB-alba-medium", "alba", "en_GB", "en", "female", "medium", 1),
    VoiceProfile("en_GB-cori-medium", "cori", "en_GB", "en", "female", "medium", 1),
    VoiceProfile("en_GB-jenny_dioco-medium", "jenny_dioco", "en_GB", "en", "female", "medium", 1),
    VoiceProfile(
        "en_GB-northern_english_male-medium",
        "northern_male",
        "en_GB",
        "en",
        "male",
        "medium",
        1,
    ),
    # French
    VoiceProfile("fr_FR-siwis-low", "siwis", "fr_FR", "fr", "female", "low", 1),
    VoiceProfile("fr_FR-siwis-medium", "siwis", "fr_FR", "fr", "female", "medium", 1),
    VoiceProfile("fr_FR-gilles-low", "gilles", "fr_FR", "fr", "male", "low", 1),
    VoiceProfile("fr_FR-tom-medium", "tom", "fr_FR", "fr", "male", "medium", 1),
    VoiceProfile("fr_FR-mls_1840-low", "mls_1840", "fr_FR", "fr", "male", "low", 1),
    VoiceProfile(
        "fr_FR-upmc-medium",
        "upmc",
        "fr_FR",
        "fr",
        "female",
        "medium",
        2,
        {"jessica": 0, "pierre": 1},
    ),
    # Spanish
    VoiceProfile("es_ES-davefx-medium", "davefx", "es_ES", "es", "male", "medium", 1),
    VoiceProfile("es_ES-carlfm-x_low", "carlfm", "es_ES", "es", "male", "low", 1),
    VoiceProfile("es_ES-mls_10246-low", "mls_10246", "es_ES", "es", "male", "low", 1),
    VoiceProfile("es_ES-mls_9972-low", "mls_9972", "es_ES", "es", "female", "low", 1),
    VoiceProfile(
        "es_ES-sharvard-medium",
        "sharvard",
        "es_ES",
        "es",
        "male",
        "medium",
        2,
        {"F": 0, "M": 1},
    ),
    # German
    VoiceProfile("de_DE-eva_k-x_low", "eva_k", "de_DE", "de", "female", "low", 1),
    VoiceProfile("de_DE-kerstin-low", "kerstin", "de_DE", "de", "female", "low", 1),
    VoiceProfile("de_DE-ramona-low", "ramona", "de_DE", "de", "female", "low", 1),
    VoiceProfile("de_DE-karlsson-low", "karlsson", "de_DE", "de", "male", "low", 1),
    VoiceProfile("de_DE-pavoque-low", "pavoque", "de_DE", "de", "male", "low", 1),
    VoiceProfile("de_DE-thorsten-low", "thorsten", "de_DE", "de", "male", "low", 1),
    VoiceProfile("de_DE-thorsten-medium", "thorsten", "de_DE", "de", "male", "medium", 1),
]

_CACHED_CATALOG: list[VoiceProfile] | None = None


def parse_piper_voices_data(data: dict[str, typing.Any]) -> list[VoiceProfile]:
    """
    Parse voices JSON dictionary into a list of VoiceProfile records.

    Args:
        data (dict[str, typing.Any]): Raw voices dictionary.

    Returns:
        list[VoiceProfile]: Parsed voice profile records.
    """

    profiles: list[VoiceProfile] = []

    for key, info in data.items():
        name: str = str(info.get("name", "speaker"))
        lang_dict: dict[str, typing.Any] = info.get("language", {})
        lang_code: str = str(lang_dict.get("code", "en_US"))
        lang_fam: str = str(lang_dict.get("family", lang_code.split("_", maxsplit=1)[0]))
        quality: str = str(info.get("quality", "medium"))
        num_spk: int = int(info.get("num_speakers", 1))
        spk_map: dict[str, int] = info.get("speaker_id_map", {})
        gender: str = infer_gender_from_name(name)

        profile: VoiceProfile = VoiceProfile(
            key=key,
            name=name,
            language=lang_code,
            lang_family=lang_fam,
            gender=gender,
            quality=quality,
            num_speakers=num_spk,
            speaker_id_map=spk_map,
        )
        profiles.append(profile)

    return profiles


_CATALOG_CACHE: list[VoiceProfile] = []


def load_full_piper_catalog(force_reload: bool = False) -> list[VoiceProfile]:
    """
    Load complete catalog of all 176 Piper voices from local cache or remote.

    Args:
        force_reload (bool): Whether to bypass in-memory cached catalog.

    Returns:
        list[VoiceProfile]: Complete list of voice profiles.
    """

    if _CATALOG_CACHE and not force_reload:
        return list(_CATALOG_CACHE)

    loaded_profiles: list[VoiceProfile] = []

    # Check local JSON cache first
    if LOCAL_CATALOG_PATH.is_file():
        try:
            with LOCAL_CATALOG_PATH.open("r", encoding="utf-8") as file_stream:
                local_data: dict[str, typing.Any] = json.load(file_stream)
            loaded_profiles = parse_piper_voices_data(local_data)
        except (json.JSONDecodeError, OSError) as err:
            logger.warning("Failed reading local voices.json: %s", err)

    # Attempt remote download if local cache was empty
    if not loaded_profiles:
        try:
            with urllib.request.urlopen(CATALOG_URL, timeout=6) as resp:
                remote_data: dict[str, typing.Any] = json.loads(resp.read().decode("utf-8"))
            loaded_profiles = parse_piper_voices_data(remote_data)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as err:
            logger.warning("Remote catalog unreachable: %s. Using curated fallback.", err)

    # Fallback to curated catalog if both failed
    if not loaded_profiles:
        loaded_profiles = list(CURATED_VOICE_CATALOG)

    _CATALOG_CACHE.clear()
    _CATALOG_CACHE.extend(loaded_profiles)
    return list(_CATALOG_CACHE)


def filter_voices_by_language(
    catalog: list[VoiceProfile],
    languages: list[str],
) -> list[VoiceProfile]:
    """
    Filter voice profiles matching specified language codes or families.

    Args:
        catalog (list[VoiceProfile]): Source voice profiles list.
        languages (list[str]): Language codes or family prefixes.

    Returns:
        list[VoiceProfile]: Filtered profiles list.
    """

    normalized: set[str] = {lang.lower() for lang in languages}
    filtered: list[VoiceProfile] = []

    for vp in catalog:
        if vp.lang_family.lower() in normalized or vp.language.lower() in normalized:
            filtered.append(vp)

    return filtered if filtered else list(catalog)


def select_candidate_profile(
    target_gender: str,
    females: list[VoiceProfile],
    males: list[VoiceProfile],
    all_pool: list[VoiceProfile],
    used_keys: set[str],
    ensure_gender_diversity: bool,
) -> VoiceProfile:
    """
    Select an unused voice profile matching target gender with fallbacks.

    Args:
        target_gender (str): 'female' or 'male'.
        females (list[VoiceProfile]): Female profiles.
        males (list[VoiceProfile]): Male profiles.
        all_pool (list[VoiceProfile]): Complete profile pool.
        used_keys (set[str]): Already selected voice keys.
        ensure_gender_diversity (bool): Whether to enforce gender diversity.

    Returns:
        VoiceProfile: Selected voice profile.
    """

    chosen_profile: VoiceProfile | None = None

    if ensure_gender_diversity:
        primary_list: list[VoiceProfile] = females if target_gender == "female" else males
        candidates = [v for v in primary_list if v.key not in used_keys]
        if candidates:
            chosen_profile = random.choice(candidates)

    if chosen_profile is None:
        remaining = [v for v in all_pool if v.key not in used_keys]
        if remaining:
            chosen_profile = random.choice(remaining)
        else:
            chosen_profile = random.choice(all_pool)

    return chosen_profile


def sample_distinct_voice_profiles(
    count: int,
    languages: list[str],
    ensure_gender_diversity: bool = True,
    catalog: list[VoiceProfile] | None = None,
) -> list[tuple[VoiceProfile, int | None, str]]:
    """
    Sample strictly unique voice profiles ensuring balanced gender diversity.

    Args:
        count (int): Number of unique voices requested.
        languages (list[str]): Allowed languages.
        ensure_gender_diversity (bool): Whether to enforce male/female alternation.
        catalog (list[VoiceProfile] | None): Optional catalog to draw from.

    Returns:
        list[tuple[VoiceProfile, int | None, str]]: List of (VoiceProfile, speaker_id, gender).
    """

    active_catalog: list[VoiceProfile] = (
        catalog if catalog is not None else load_full_piper_catalog()
    )
    pool: list[VoiceProfile] = filter_voices_by_language(active_catalog, languages)
    random.shuffle(pool)

    females: list[VoiceProfile] = [v for v in pool if v.gender == "female"]
    males: list[VoiceProfile] = [v for v in pool if v.gender == "male"]

    selected: list[tuple[VoiceProfile, int | None, str]] = []
    used_keys: set[str] = set()

    for i in range(count):
        target_gender: str = "female" if i % 2 == 0 else "male"
        chosen_profile: VoiceProfile = select_candidate_profile(
            target_gender=target_gender,
            females=females,
            males=males,
            all_pool=pool,
            used_keys=used_keys,
            ensure_gender_diversity=ensure_gender_diversity,
        )
        used_keys.add(chosen_profile.key)

        spk_id: int | None = None
        if chosen_profile.num_speakers > 1:
            spk_id = random.randint(0, chosen_profile.num_speakers - 1)

        actual_gender: str = (
            chosen_profile.gender if chosen_profile.gender != "unspecified" else target_gender
        )
        selected.append((chosen_profile, spk_id, actual_gender))

    return selected
