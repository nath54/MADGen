"""
Procedural conversation timeline generator simulating multi-person interactions.

Coordinates parallel conversational groups, natural turn-taking dynamics,
dialogue interruptions and overlaps, and dynamic emotional vocal styles.
"""

# Import Modules
from dataclasses import dataclass

import random

from src.llm.llm_types import GeneratedTurn
from src.llm.dialogue_generator import LLMDialogueGenerator
from src.procedural.dialogue_bank import sample_dialogue_text
from src.procedural.ambiance_presets import AmbiancePreset
from src.config.models import (
    VocalStyle,
    PersonaConfig,
    UtteranceConfig,
    ConversationalStyle,
)


@dataclass
class ConversationParameters:
    """
    Configuration parameters and generators for conversational scene synthesis.
    """

    overlap_rate: float = 0.35
    shout_rate: float = 0.15
    laugh_rate: float = 0.20
    style_preset: ConversationalStyle = ConversationalStyle.MIXED
    llm_generator: LLMDialogueGenerator | None = None
    ambiance: AmbiancePreset | None = None
    constraint_words: list[str] | None = None
    duration_s: float | None = None


def partition_conversational_groups(
    personas: list[PersonaConfig],
    allow_parallel: bool = True,
) -> list[list[PersonaConfig]]:
    """
    Partition personas into small conversational cliques of two or three people.

    Args:
        personas (list[PersonaConfig]): Complete roster of scene personas.
        allow_parallel (bool): Whether parallel conversational cliques are allowed.

    Returns:
        list[list[PersonaConfig]]: List of conversational groups.
    """

    # If parallel discussions disabled or small party, everyone is in the same group
    if not allow_parallel or len(personas) <= 3:
        return [personas]

    # Shuffle copies to create organic cliques
    shuffled_pool: list[PersonaConfig] = list(personas)
    random.shuffle(shuffled_pool)

    # Split into groups of 2 or 3
    groups: list[list[PersonaConfig]] = []
    index: int = 0
    while index < len(shuffled_pool):
        remaining: int = len(shuffled_pool) - index
        group_size: int = 3 if remaining >= 5 or remaining == 3 else 2
        groups.append(shuffled_pool[index : index + group_size])
        index += group_size

    return groups


def select_vocal_style_and_category(
    shout_rate: float,
    laugh_rate: float,
    style_preset: ConversationalStyle,
) -> tuple[VocalStyle, str]:
    """
    Select vocal style and dialogue category based on emotional probabilities.

    Args:
        shout_rate (float): Probability of shouting or raised voice.
        laugh_rate (float): Probability of laughter or chuckles.
        style_preset (ConversationalStyle): Overall conversational scene style.

    Returns:
        tuple[VocalStyle, str]: Selected vocal style and dialogue bank category.
    """

    # Roll probability die
    roll: float = random.random()

    # Check shouting probability
    if roll < shout_rate or style_preset == ConversationalStyle.ARGUMENT:
        return (VocalStyle.SHOUTING, "shouting")

    # Check laughter probability
    if roll < (shout_rate + laugh_rate) or style_preset == ConversationalStyle.PARTY:
        return (VocalStyle.LAUGHTER, "laughter")

    # Check assistant command probability in assistant mode
    if style_preset == ConversationalStyle.ASSISTANT and random.random() < 0.35:
        return (VocalStyle.NORMAL, "assistant")

    return (VocalStyle.NORMAL, "general")


def estimate_utterance_duration(text: str) -> float:
    """
    Estimate acoustic speech duration based on word count.

    Args:
        text (str): Spoken phrase.

    Returns:
        float: Estimated duration in seconds.
    """

    # Average speaking speed: ~0.32 seconds per word with punctuation pauses
    word_count: int = max(1, len(text.split()))
    duration: float = word_count * 0.32 + 0.3
    return duration


def calculate_turn_timing(
    prev_end_s: float,
    overlap_rate: float,
) -> tuple[float, bool]:
    """
    Determine start timestamp for the next turn, checking for an interruption.

    Args:
        prev_end_s (float): Termination timestamp of preceding turn.
        overlap_rate (float): Probability of cutting into the preceding turn.

    Returns:
        tuple[float, bool]: Pair of (start_timestamp, is_interruption).
    """

    # Interruption cuts into ongoing speech
    if random.random() < overlap_rate and prev_end_s > 1.0:
        cut_duration_s: float = random.uniform(0.3, 1.0)
        start_time_s: float = max(0.4, prev_end_s - cut_duration_s)
        return (start_time_s, True)

    # Normal turn-taking gap
    pause_s: float = random.uniform(0.2, 0.7)
    return (prev_end_s + pause_s, False)


def generate_fallback_turns(
    group: list[PersonaConfig],
    target_turns: int,
    shout_rate: float,
    laugh_rate: float,
    style_preset: ConversationalStyle,
) -> list[GeneratedTurn]:
    """
    Generate conversational turns from offline dialogue bank matching vocal style probabilities.

    Args:
        group (list[PersonaConfig]): Personas in the group.
        target_turns (int): Target number of turns.
        shout_rate (float): Shouting probability.
        laugh_rate (float): Laughter probability.
        style_preset (ConversationalStyle): Conversational style preset.

    Returns:
        list[GeneratedTurn]: Synthesized turns list.
    """

    turns: list[GeneratedTurn] = []
    last_speaker_id: str | None = None

    for _ in range(target_turns):
        candidates: list[PersonaConfig] = [p for p in group if p.id != last_speaker_id]
        speaker: PersonaConfig = random.choice(candidates if candidates else group)
        last_speaker_id = speaker.id

        style, cat = select_vocal_style_and_category(shout_rate, laugh_rate, style_preset)
        text: str = sample_dialogue_text(language=speaker.language, category=cat)
        turns.append(GeneratedTurn(speaker_id=speaker.id, text=text, vocal_style=style.value))

    return turns


def _create_timeline_utterance(
    turn: GeneratedTurn,
    speaker: PersonaConfig,
    start_time_s: float,
    group_id: int,
    turn_order: int,
    is_cut: bool,
) -> UtteranceConfig:
    """
    Construct UtteranceConfig from generated turn and timing parameters.

    Args:
        turn (GeneratedTurn): Generated turn text and style.
        speaker (PersonaConfig): Speaking persona.
        start_time_s (float): Calculated start timestamp in seconds.
        group_id (int): Conversation clique identifier.
        turn_order (int): Turn order index.
        is_cut (bool): Whether turn is interrupted by next turn.

    Returns:
        UtteranceConfig: Assembled utterance configuration.
    """

    style_enum: VocalStyle = VocalStyle.NORMAL
    if turn.vocal_style == "shouting":
        style_enum = VocalStyle.SHOUTING
    elif turn.vocal_style == "laughter":
        style_enum = VocalStyle.LAUGHTER
    elif turn.vocal_style == "interruption" or is_cut:
        style_enum = VocalStyle.INTERRUPTION

    return UtteranceConfig(
        text=turn.text,
        start_time_s=round(start_time_s, 2),
        language=speaker.language,
        vocal_style=style_enum,
        group_id=group_id,
        turn_order=turn_order,
    )


def sequence_llm_turns_on_timeline(
    group: list[PersonaConfig],
    turns: list[GeneratedTurn],
    group_id: int = 0,
    max_duration_s: float | None = None,
    overlap_rate: float = 0.35,
    start_turn_order: int = 0,
    start_offset_s: float | None = None,
) -> float:
    """
    Chronologically sequence dialogue turns onto the group timeline.

    Args:
        group (list[PersonaConfig]): Group personas participating in dialogue.
        turns (list[GeneratedTurn]): List of dialogue turns.
        group_id (int): Conversation clique identifier.
        max_duration_s (float | None): Optional timeline boundary limit in seconds.
        overlap_rate (float): Interruption and speech overlap probability.
        start_turn_order (int): Starting turn order index.
        start_offset_s (float | None): Optional starting timestamp offset in seconds.

    Returns:
        float: Termination timestamp of the last sequenced turn.
    """

    persona_map: dict[str, PersonaConfig] = {p.id: p for p in group}
    prev_end_s: float = (
        start_offset_s if start_offset_s is not None else random.uniform(0.3, 1.0)
    )

    # Place turns along the timeline
    for turn_idx, turn in enumerate(turns):
        if max_duration_s is not None and prev_end_s >= (max_duration_s - 1.5):
            break

        speaker: PersonaConfig = persona_map.get(turn.speaker_id, group[0])
        start_t, is_cut = calculate_turn_timing(prev_end_s, overlap_rate)

        utterance: UtteranceConfig = _create_timeline_utterance(
            turn=turn,
            speaker=speaker,
            start_time_s=start_t,
            group_id=group_id,
            turn_order=start_turn_order + turn_idx,
            is_cut=is_cut,
        )
        speaker.utterances.append(utterance)
        prev_end_s = start_t + estimate_utterance_duration(turn.text)

    return prev_end_s


def generate_group_timeline(
    group: list[PersonaConfig],
    target_turns: int,
    group_id: int,
    params: ConversationParameters,
    start_turn_order: int = 0,
    start_offset_s: float | None = None,
) -> float:
    """
    Generate turn-taking dialogue exchanges for a single conversational group.

    Args:
        group (list[PersonaConfig]): Personas participating in this conversation group.
        target_turns (int): Target number of dialogue turns.
        group_id (int): Conversation clique identifier.
        params (ConversationParameters): Conversation tuning parameters and generators.
        start_turn_order (int): Starting turn order index.
        start_offset_s (float | None): Optional starting timestamp offset in seconds.

    Returns:
        float: Termination timestamp of the last sequenced turn.
    """

    turns: list[GeneratedTurn] = []

    # If LLM generator is active and ambiance is set, attempt LLM dialogue generation
    if params.llm_generator is not None and params.ambiance is not None:
        words: list[str] = (
            params.constraint_words if params.constraint_words is not None else []
        )
        turns = params.llm_generator.generate_group_dialogue(
            group=group,
            ambiance=params.ambiance,
            constraint_words=words,
            target_turns_count=target_turns,
        )

    # Fallback to rule-based dialogue bank if turns empty
    if not turns:
        turns = generate_fallback_turns(
            group=group,
            target_turns=target_turns,
            shout_rate=params.shout_rate,
            laugh_rate=params.laugh_rate,
            style_preset=params.style_preset,
        )

    # Sequence turns along the timeline
    return sequence_llm_turns_on_timeline(
        group=group,
        turns=turns,
        group_id=group_id,
        max_duration_s=params.duration_s,
        overlap_rate=params.overlap_rate,
        start_turn_order=start_turn_order,
        start_offset_s=start_offset_s,
    )


def _build_side_params(
    params: ConversationParameters,
    word_slice: list[str],
) -> ConversationParameters:
    """
    Construct parameters for a side discussion with dedicated constraint words.

    Args:
        params (ConversationParameters): Base scene parameters.
        word_slice (list[str]): Subset of constraint words for the side topic.

    Returns:
        ConversationParameters: Tailored parameters for the side group.
    """

    return ConversationParameters(
        overlap_rate=params.overlap_rate,
        shout_rate=params.shout_rate,
        laugh_rate=params.laugh_rate,
        style_preset=params.style_preset,
        llm_generator=params.llm_generator,
        ambiance=params.ambiance,
        constraint_words=word_slice,
        duration_s=params.duration_s,
    )


def _partition_parallel_rosters(
    personas: list[PersonaConfig],
) -> tuple[list[PersonaConfig], list[list[PersonaConfig]]]:
    """
    Partition personas into a core main discussion group and one or more side groups.

    Args:
        personas (list[PersonaConfig]): Complete roster of scene personas (>= 4).

    Returns:
        tuple[list[PersonaConfig], list[list[PersonaConfig]]]: Core main group and side groups.
    """

    # Determine side groups count: 1 for 4-5 personas, 2 for 6+ personas
    num_side_groups: int = 1 if len(personas) < 6 else 2
    side_groups: list[list[PersonaConfig]] = []
    used_indices: set[int] = set()

    # Extract non-overlapping cliques of 2 personas for side discussions
    for g_idx in range(num_side_groups):
        start_idx: int = 1 + g_idx * 2
        side_grp: list[PersonaConfig] = personas[start_idx : start_idx + 2]
        side_groups.append(side_grp)
        used_indices.add(start_idx)
        used_indices.add(start_idx + 1)

    # Core main discussion retains all remaining personas
    main_core: list[PersonaConfig] = [
        p for idx, p in enumerate(personas) if idx not in used_indices
    ]

    return main_core, side_groups


def _calculate_parallel_turn_splits(
    min_sentences: int,
    num_side_groups: int = 1,
) -> tuple[int, int, int, int]:
    """
    Calculate turn allocations for (phase1_main, side_turns, phase2_main, phase3_main).

    Args:
        min_sentences (int): Minimum required sentences across scene.
        num_side_groups (int): Number of active parallel side groups.

    Returns:
        tuple[int, int, int, int]: (phase1_main, side_turns, phase2_main, phase3_main).
    """

    if min_sentences < 30:
        p1: int = max(2, min_sentences // (3 + num_side_groups))
        side: int = max(2, min_sentences // (3 + num_side_groups))
        p2_main: int = side
        p3: int = max(1, min_sentences - (p1 + p2_main + side * num_side_groups))
        return (p1, side, p2_main, p3)

    p1 = 15
    side = 15
    p2_main = 15 if num_side_groups == 1 else 25
    allocated: int = p1 + p2_main + (side * num_side_groups)
    p3 = max(5, min_sentences - allocated)
    return (p1, side, p2_main, p3)


def _run_side_discussions(
    side_groups: list[list[PersonaConfig]],
    side_turns: int,
    base_offset_s: float,
    params: ConversationParameters,
) -> list[float]:
    """
    Sequentially launch staggered side discussions and collect their termination timestamps.

    Args:
        side_groups (list[list[PersonaConfig]]): Persona cliques for side conversations.
        side_turns (int): Number of turns per side group.
        base_offset_s (float): Timestamp after opening plenary concludes.
        params (ConversationParameters): Base conversation parameters.

    Returns:
        list[float]: End timestamps for each side group conversation.
    """

    ends: list[float] = []
    words: list[str] = (
        params.constraint_words if params.constraint_words is not None else []
    )

    # Launch each side group with a staggered timeline offset
    for idx, grp in enumerate(side_groups):
        stagger_s: float = base_offset_s + 1.5 + (idx * 6.0)
        sub_words: list[str] = words[2 + idx * 2 : 4 + idx * 2]
        side_params: ConversationParameters = _build_side_params(params, sub_words)
        end_s: float = generate_group_timeline(
            group=grp,
            target_turns=side_turns,
            group_id=idx + 1,
            params=side_params,
            start_turn_order=0,
            start_offset_s=stagger_s,
        )
        ends.append(end_s)

    return ends


def _generate_parallel_discussions(
    personas: list[PersonaConfig],
    min_sentences: int,
    params: ConversationParameters,
) -> None:
    """
    Generate asynchronous parallel discussions with staggered appearances and rejoined plenary.

    Args:
        personas (list[PersonaConfig]): Complete roster of personas (>= 4).
        min_sentences (int): Minimum required total sentences across scene.
        params (ConversationParameters): Conversation generation parameters.
    """

    main_core, side_groups = _partition_parallel_rosters(personas)
    p1, side, p2_main, p3 = _calculate_parallel_turn_splits(
        min_sentences=min_sentences,
        num_side_groups=len(side_groups),
    )

    # Phase 1: Opening plenary discussion (all personas)
    p1_end: float = generate_group_timeline(
        group=personas,
        target_turns=p1,
        group_id=0,
        params=params,
        start_turn_order=0,
        start_offset_s=0.5,
    )

    # Phase 2: Staggered parallel side discussions alongside ongoing main core
    side_ends: list[float] = _run_side_discussions(
        side_groups=side_groups,
        side_turns=side,
        base_offset_s=p1_end,
        params=params,
    )
    p2_main_end: float = generate_group_timeline(
        group=main_core,
        target_turns=p2_main,
        group_id=0,
        params=params,
        start_turn_order=p1,
        start_offset_s=p1_end + 0.4,
    )

    # Phase 3: Rejoined plenary discussion (all personas)
    max_phase2_end: float = max([p2_main_end] + side_ends)
    generate_group_timeline(
        group=personas,
        target_turns=p3,
        group_id=0,
        params=params,
        start_turn_order=p1 + p2_main,
        start_offset_s=max_phase2_end + 0.8,
    )


def _ensure_minimum_sentences(
    personas: list[PersonaConfig],
    min_sentences: int,
    params: ConversationParameters,
) -> None:
    """
    Append additional turns if total utterances fall below min_sentences threshold.

    Args:
        personas (list[PersonaConfig]): Complete roster of personas.
        min_sentences (int): Minimum required sentence count.
        params (ConversationParameters): Conversation generation parameters.
    """

    total: int = sum(len(p.utterances) for p in personas)
    if total < min_sentences:
        deficit: int = min_sentences - total
        extra: list[GeneratedTurn] = generate_fallback_turns(
            group=personas,
            target_turns=deficit,
            shout_rate=params.shout_rate,
            laugh_rate=params.laugh_rate,
            style_preset=params.style_preset,
        )

        # Determine timeline boundary
        max_end_s: float = 0.5
        for p in personas:
            for u in p.utterances:
                max_end_s = max(max_end_s, u.start_time_s + estimate_utterance_duration(u.text))

        sequence_llm_turns_on_timeline(
            group=personas,
            turns=extra,
            group_id=0,
            max_duration_s=params.duration_s,
            overlap_rate=params.overlap_rate,
            start_turn_order=total,
            start_offset_s=max_end_s + 0.5,
        )


def generate_conversations_for_personas(
    personas: list[PersonaConfig],
    duration_s: float | None = None,
    min_sentences: int = 100,
    overlap_rate: float = 0.35,
    shout_rate: float = 0.15,
    laugh_rate: float = 0.20,
    style_preset: ConversationalStyle = ConversationalStyle.MIXED,
    llm_generator: LLMDialogueGenerator | None = None,
    ambiance: AmbiancePreset | None = None,
    constraint_words: list[str] | None = None,
    allow_parallel: bool = True,
) -> None:
    """
    Procedurally generate conversations, interruptions, and styles for personas.

    Args:
        personas (list[PersonaConfig]): Personas to populate with utterances.
        duration_s (float | None): Optional desired audio duration in seconds.
        min_sentences (int): Minimum required total sentences across all personas.
        overlap_rate (float): Interruption probability.
        shout_rate (float): Shouting probability.
        laugh_rate (float): Laughter probability.
        style_preset (ConversationalStyle): Conversational context.
        llm_generator (LLMDialogueGenerator | None): Optional LLM dialogue generator.
        ambiance (AmbiancePreset | None): Optional ambiance preset.
        constraint_words (list[str] | None): Optional thematic constraint keywords.
        allow_parallel (bool): Whether parallel side discussions are permitted.
    """

    # Clear any preexisting utterances
    for p in personas:
        p.utterances.clear()

    params: ConversationParameters = ConversationParameters(
        overlap_rate=overlap_rate,
        shout_rate=shout_rate,
        laugh_rate=laugh_rate,
        style_preset=style_preset,
        llm_generator=llm_generator,
        ambiance=ambiance,
        constraint_words=constraint_words,
        duration_s=duration_s,
    )

    # Branch between parallel asynchronous discussions and plenary conversation
    if allow_parallel and len(personas) >= 4:
        _generate_parallel_discussions(
            personas=personas,
            min_sentences=min_sentences,
            params=params,
        )
    else:
        generate_group_timeline(
            group=personas,
            target_turns=min_sentences,
            group_id=0,
            params=params,
            start_turn_order=0,
            start_offset_s=0.5,
        )

    # Ensure minimum sentences constraint is satisfied
    _ensure_minimum_sentences(
        personas=personas,
        min_sentences=min_sentences,
        params=params,
    )
