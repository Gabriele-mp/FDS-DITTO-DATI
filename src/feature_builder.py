import numpy as np
import pandas as pd
from collections import Counter
import math

# ===========================================================================
# COSTANTI (da Cella 2 e 6 del notebook)
# ===========================================================================

gen1_type_chart = {
    'NORMAL': {'ROCK': 0.5, 'GHOST': 0.0}, 'FIRE': {'FIRE': 0.5, 'WATER': 0.5, 'GRASS': 2.0, 'ICE': 2.0, 'BUG': 2.0, 'ROCK': 0.5, 'DRAGON': 0.5},
    'WATER': {'FIRE': 2.0, 'WATER': 0.5, 'GRASS': 0.5, 'GROUND': 2.0, 'ROCK': 2.0, 'DRAGON': 0.5}, 'ELECTRIC': {'WATER': 2.0, 'ELECTRIC': 0.5, 'GRASS': 0.5, 'GROUND': 0.0, 'FLYING': 2.0, 'DRAGON': 0.5},
    'GRASS': {'FIRE': 0.5, 'WATER': 2.0, 'GRASS': 0.5, 'POISON': 0.5, 'GROUND': 2.0, 'FLYING': 0.5, 'BUG': 0.5, 'ROCK': 2.0, 'DRAGON': 0.5}, 'ICE': {'WATER': 0.5, 'GRASS': 2.0, 'ICE': 0.5, 'GROUND': 2.0, 'FLYING': 2.0, 'DRAGON': 2.0, 'FIRE': 0.5},
    'FIGHTING': {'NORMAL': 2.0, 'ICE': 2.0, 'POISON': 0.5, 'FLYING': 0.5, 'PSYCHIC': 0.5, 'BUG': 0.5, 'ROCK': 2.0, 'GHOST': 0.0}, 'POISON': {'GRASS': 2.0, 'POISON': 0.5, 'GROUND': 0.5, 'BUG': 2.0, 'ROCK': 0.5, 'GHOST': 0.5},
    'GROUND': {'FIRE': 2.0, 'ELECTRIC': 2.0, 'GRASS': 0.5, 'POISON': 2.0, 'FLYING': 0.0, 'BUG': 0.5, 'ROCK': 2.0}, 'FLYING': {'ELECTRIC': 0.5, 'GRASS': 2.0, 'FIGHTING': 2.0, 'BUG': 2.0, 'ROCK': 0.5},
    'PSYCHIC': {'FIGHTING': 2.0, 'POISON': 2.0, 'PSYCHIC': 0.5}, 'BUG': {'FIRE': 0.5, 'GRASS': 2.0, 'FIGHTING': 0.5, 'POISON': 2.0, 'FLYING': 0.5, 'PSYCHIC': 0.5, 'GHOST': 0.5},
    'ROCK': {'FIRE': 2.0, 'ICE': 2.0, 'FIGHTING': 0.5, 'GROUND': 0.5, 'FLYING': 2.0, 'BUG': 2.0}, 'GHOST': {'NORMAL': 0.0, 'PSYCHIC': 0.0, 'GHOST': 2.0}, 'DRAGON': {'DRAGON': 2.0}, 'NOTYPE': {}
}
ALL_TYPES = list(gen1_type_chart.keys())
GEN1_WEAKNESS = {
    "NORMAL": ["FIGHTING"], "FIRE": ["WATER", "GROUND", "ROCK"], "WATER": ["ELECTRIC", "GRASS"], "ELECTRIC": ["GROUND"],
    "GRASS": ["FIRE", "ICE", "POISON", "FLYING", "BUG"], "ICE": ["FIRE", "FIGHTING", "ROCK"], "FIGHTING": ["FLYING", "PSYCHIC"],
    "POISON": ["GROUND", "PSYCHIC", "BUG"], "GROUND": ["WATER", "GRASS", "ICE"], "FLYING": ["ELECTRIC", "ICE", "ROCK"],
    "PSYCHIC": ["BUG", "GHOST"], "BUG": ["FIRE", "POISON", "FLYING", "ROCK"], "ROCK": ["WATER", "GRASS", "FIGHTING", "GROUND"],
    "GHOST": ["GHOST"], "DRAGON": ["ICE", "DRAGON"], "NOTYPE": []
}
ALL_BASE_STATS = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe"]
STATUS_LIST_VALID = {"slp", "par", "frz", "brn", "psn", "tox"}
EXPLOSION_MOVES = {"explosion", "selfdestruct"}
COUNTER_MOVES = {"counter"}
RECOVERY_MOVES = {'recover', 'softboiled'}
SETUP_BOOST_MOVES = {'swords dance', 'amnesia', 'agility', 'growth', 'barrier', 'double team', 'harden', 'minimize', 'sharpen', 'withdraw'}
WALLS = {'chansey', 'snorlax'}
SETUP_SWEEPERS = {'snorlax', 'alakazam', 'starmie', 'lapras', 'jynx'}
KEY_SWEEPERS = {'tauros'}

# Costanti "Golden Moveset" (da Cella 6)
GOLDEN_RECOVERY = {'recover', 'softboiled', 'rest'}
GOLDEN_SETUP = {'swords dance', 'amnesia', 'agility', 'barrier'}
GOLDEN_STATUS = {'thunder wave', 'sleep powder', 'spore', 'lovely kiss', 'hypnosis', 'glare'}
GOLDEN_TRAP = {'wrap', 'fire spin', 'clamp', 'bind'}
GOLDEN_HAZARD = {'toxic', 'leech seed'}
GOLDEN_POWER = {'explosion', 'selfdestruct', 'hyper beam'}
GOLDEN_COMBO_TB = 'thunderbolt'
GOLDEN_COMBO_IB = 'ice beam'


# ===========================================================================
# FUNZIONI HELPER (da Cella 2)
# ===========================================================================

def get_effectiveness(attack_type, defender_types):
    attack_type = str(attack_type).upper()
    if attack_type == 'NOTYPE': return 1.0
    mult = 1.0
    attack_effectiveness_map = gen1_type_chart.get(attack_type, {})
    for def_type in [str(t).upper() for t in defender_types if str(t).upper() != 'NOTYPE']:
        mult *= attack_effectiveness_map.get(def_type, 1.0)
    return mult

def entropy_from_counts(counter: Counter) -> float:
    if not counter: return 0.0
    total_count = sum(counter.values())
    if total_count == 0: return 0.0
    entropy = 0.0
    for count in counter.values():
        if count > 0: p = count / total_count; entropy -= p * math.log2(p)
    return float(entropy)


# ===========================================================================
# FUNZIONE 1: extract_features_v8 (da Cella 4)
# ===========================================================================
def extract_features_v8(battle):
    """ Estrae ~30 feature (v8) """
    f = {
        'battle_id': battle.get('battle_id', -1),
        'delta_mean_base_spe': 0.0, 'p1_stab_advantage_vs_p2lead': 1.0, 'p1_team_weakness_entropy': 0.0, 'p1_members_threatened_by_p2lead': 0.0,
        'ko_diff': 0, 'dmg_sum_adv': 0.0, 'dmg_p90_adv': 0.0, 'hp_control_proxy': 0.0,
        'move_ratio_w30': 1.0, 'switch_ratio_w30': 1.0, 'p2_revealed_mons_count': 1,
        'SUPER_boost_advantage': 0.0, 'boost_diff_spe_sum_w30': 0.0,
        'status_adv_frz': 0, 'status_adv_slp': 0, 'status_adv_par': 0,
        'interact_ko_diff_X_dmg_adv': 0.0, 'interact_spe_base_X_boost': 0.0,
        'p1_explosion_value': 0.0,
        'counter_dmg_advantage': 0.0, 'counter_ko_advantage': 0,
        'total_hp_advantage_t20': 0.0, 'high_hp_advantage_t20': 0.0, 'status_burden_advantage_t20': 0.0,
        'total_hp_advantage_t30': 0.0, 'high_hp_advantage_t30': 0.0, 'status_burden_advantage_t30': 0.0,
        'interact_threat_X_switch': 0.0,
        'interact_pressure_X_info': 0.0,
        'interact_status_X_hp_t30': 0.0
    }
    try:
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})
        p1_stats_raw = {s: [] for s in ALL_BASE_STATS}
        p1_mean_stats = {}
        p1_team_pokemon_types = []
        p1_full_team_status = {}
        if isinstance(p1_team, list) and p1_team:
            for p in p1_team:
                if isinstance(p, dict):
                    for s in ALL_BASE_STATS: p1_stats_raw[s].append(p.get(s, 0))
                    types = [t.upper() for t in p.get('types', []) if isinstance(t, str) and t.upper() != 'NOTYPE']
                    if types: p1_team_pokemon_types.append(types)
                    if p.get('name'): p1_full_team_status[p['name']] = {'hp': 1.0, 'status': 'nostatus'}
            for s in ALL_BASE_STATS: p1_mean_stats[s] = np.mean(p1_stats_raw[s])
        if isinstance(p2_lead, dict):
            p2_lead_types = [t.upper() for t in p2_lead.get('types', []) if isinstance(t, str) and t.upper() != 'NOTYPE']
            f['delta_mean_base_spe'] = p1_mean_stats.get('base_spe', 0) - p2_lead.get('base_spe', 0)
            total_effectiveness = 0
            if p2_lead_types and p1_team_pokemon_types:
                for p1_types in p1_team_pokemon_types:
                    poke_effectiveness = sum(get_effectiveness(stab_type, p2_lead_types) for stab_type in p1_types)
                    if p1_types: total_effectiveness += poke_effectiveness / len(p1_types)
                if p1_team_pokemon_types: f['p1_stab_advantage_vs_p2lead'] = total_effectiveness / len(p1_team_pokemon_types)
            threatened_count = 0
            if p2_lead_types and p1_team_pokemon_types:
                for poke_types in p1_team_pokemon_types:
                    if any(get_effectiveness(lead_type, poke_types) >= 2.0 for lead_type in p2_lead_types): threatened_count += 1
                f['p1_members_threatened_by_p2lead'] = threatened_count
        weakness_list = []
        for poke_types in p1_team_pokemon_types:
            for t in poke_types: weakness_list.extend(GEN1_WEAKNESS.get(t, []))\
        if weakness_list: f['p1_team_weakness_entropy'] = entropy_from_counts(Counter(weakness_list))
        tl = battle.get("battle_timeline", [])
        if not isinstance(tl, list): tl = []
        p1_hp_history, p2_hp_history = [], []
        p1_dmg_drops, p2_dmg_drops = [], []
        p1_moves_used, p2_moves_used = 0, 0
        p1_switches, p2_switches = 0, 0
        p1_last_name, p2_last_name = None, None
        p1_fainted_names, p2_fainted_names = set(), set()
        p2_seen_mons = {}
        boost_sums = {f'p1_{stat}': 0 for stat in BOOST_KEYS}
        boost_sums.update({f'p2_{stat}': 0 for stat in BOOST_KEYS})
        prev_p1_hp, prev_p2_hp = 1.0, 1.0
        for i, turn_data in enumerate(tl):
            p1_state = turn_data.get("p1_pokemon_state", {})\
            p2_state = turn_data.get("p2_pokemon_state", {})\
            p1_move = turn_data.get("p1_move_details", {})\
            p2_move = turn_data.get("p2_move_details", {})\
            current_p1_hp = p1_state.get("hp_pct", prev_p1_hp)
            current_p2_hp = p2_state.get("hp_pct", prev_p2_hp)
            p1_hp_history.append(current_p1_hp)
            p2_hp_history.append(current_p2_hp)
            if i > 0:
                p1_dmg_drops.append(max(0.0, prev_p1_hp - current_p1_hp))
                p2_dmg_drops.append(max(0.0, prev_p2_hp - current_p2_hp))
            if isinstance(p1_move, dict) and p1_move: p1_moves_used += 1
            if isinstance(p2_move, dict) and p2_move: p2_moves_used += 1
            p1_name = p1_state.get("name")
            p2_name = p2_state.get("name")
            if p1_name and p1_name != p1_last_name and i > 0: p1_switches += 1
            if p2_name and p2_name != p2_last_name and i > 0: p2_switches += 1
            p1_last_name, p2_last_name = p1_name, p2_name
            if p1_state.get("status") == "fnt" and p1_name and p1_name not in p1_fainted_names: f['ko_diff'] -= 1; p1_fainted_names.add(p1_name)
            if p2_state.get("status") == "fnt" and p2_name and p2_name not in p2_fainted_names: f['ko_diff'] += 1; p2_fainted_names.add(p2_name)
            p1_status = p1_state.get("status", "nostatus")
            p2_status = p2_state.get("status", "nostatus")
            if p1_status == "frz": f['status_adv_frz'] -= 1
            if p1_status == "slp": f['status_adv_slp'] -= 1
            if p1_status == "par": f['status_adv_par'] -= 1
            if p2_status == "frz": f['status_adv_frz'] += 1
            if p2_status == "slp": f['status_adv_slp'] += 1
            if p2_status == "par": f['status_adv_par'] += 1
            b1 = p1_state.get("boosts", {})\
            b2 = p2_state.get("boosts", {})\
            for stat in BOOST_KEYS:
                boost_sums[f'p1_{stat}'] += b1.get(stat, 0)
                boost_sums[f'p2_{stat}'] += b2.get(stat, 0)
            if p1_name and p1_name in p1_full_team_status: p1_full_team_status[p1_name] = {"hp": current_p1_hp, "status": p1_status}
            if p2_name: p2_seen_mons[p2_name] = {"hp": current_p2_hp, "status": p2_status}
            if isinstance(p1_move, dict) and p1_move.get("name") in EXPLOSION_MOVES: f['p1_explosion_value'] += (prev_p2_hp - prev_p1_hp)
            if isinstance(p2_move, dict) and p2_move.get("name") in COUNTER_MOVES:
                damage_taken_by_p1 = prev_p1_hp - current_p1_hp
                if damage_taken_by_p1 > 0:
                    f['counter_dmg_advantage'] -= damage_taken_by_p1
                    if p1_state.get("status") == "fnt": f['counter_ko_advantage'] -= 1
            if isinstance(p1_move, dict) and p1_move.get("name") in COUNTER_MOVES:
                damage_taken_by_p2 = prev_p2_hp - current_p2_hp
                if damage_taken_by_p2 > 0:
                    f['counter_dmg_advantage'] += damage_taken_by_p2
                    if p2_state.get("status") == "fnt": f['counter_ko_advantage'] += 1
            prev_p1_hp, prev_p2_hp = current_p1_hp, current_p2_hp
            if i == 19:
                p1_alive_t20 = [mon for mon in p1_full_team_status.values() if mon['status'] != 'fnt']
                p2_alive_t20 = [mon for mon in p2_seen_mons.values() if mon['status'] != 'fnt']
                p1_status_count_t20 = sum(1 for mon in p1_alive_t20 if mon['status'] in STATUS_LIST_VALID)
                p2_status_count_t20 = sum(1 for mon in p2_alive_t20 if mon['status'] in STATUS_LIST_VALID)
                f['status_burden_advantage_t20'] = (p1_status_count_t20 / (len(p1_alive_t20) + 1e-6)) - (p2_status_count_t20 / (len(p2_alive_t20) + 1e-6))
                f['total_hp_advantage_t20'] = sum(mon['hp'] for mon in p1_alive_t20) - sum(mon['hp'] for mon in p2_alive_t20)
                f['high_hp_advantage_t20'] = sum(1 for mon in p1_alive_t20 if mon['hp'] >= 0.8) - sum(1 for mon in p2_alive_t20 if mon['hp'] >= 0.8)
        if p1_dmg_drops and p2_dmg_drops:
            f['dmg_sum_adv'] = np.sum(p2_dmg_drops) - np.sum(p1_dmg_drops)
            f['dmg_p90_adv'] = np.quantile(p2_dmg_drops, 0.9) - np.quantile(p1_dmg_drops, 0.9)
        if p1_hp_history and p2_hp_history:
            f['hp_control_proxy'] = ( (np.mean(p1_hp_history) - np.mean(p2_hp_history)) + (p1_hp_history[-1] - p2_hp_history[-1]) ) / 2.0
        f['move_ratio_w30'] = (p1_moves_used + 1) / (p2_moves_used + 1e-6)
        f['switch_ratio_w30'] = (p1_switches + 1) / (p2_switches + 1e-6)
        f['boost_diff_spe_sum_w30'] = boost_sums['p1_spe'] - boost_sums['p2_spe']
        f['SUPER_boost_advantage'] = sum((boost_sums[f'p1_{stat}'] - boost_sums[f'p2_{stat}']) for stat in BOOST_KEYS)
        if p2_seen_mons: f['p2_revealed_mons_count'] = len(p2_seen_mons)
        p1_alive_t30 = [mon for mon in p1_full_team_status.values() if mon['status'] != 'fnt']
        p2_alive_t30 = [mon for mon in p2_seen_mons.values() if mon['status'] != 'fnt']
        p1_status_count_t30 = sum(1 for mon in p1_alive_t30 if mon['status'] in STATUS_LIST_VALID)
        p2_status_count_t30 = sum(1 for mon in p2_alive_t30 if mon['status'] in STATUS_LIST_VALID)
        f['status_burden_advantage_t30'] = (p1_status_count_t30 / (len(p1_alive_t30) + 1e-6)) - (p2_status_count_t30 / (len(p2_alive_t30) + 1e-6))
        f['total_hp_advantage_t30'] = sum(mon['hp'] for mon in p1_alive_t30) - sum(mon['hp'] for mon in p2_alive_t30)
        f['high_hp_advantage_t30'] = sum(1 for mon in p1_alive_t30 if mon['hp'] >= 0.8) - sum(1 for mon in p2_alive_t30 if mon['hp'] >= 0.8)
        f['interact_ko_diff_X_dmg_adv'] = f['ko_diff'] * f['dmg_sum_adv']
        f['interact_spe_base_X_boost'] = f['delta_mean_base_spe'] * f['boost_diff_spe_sum_w30']
        f['interact_threat_X_switch'] = f['p1_members_threatened_by_p2lead'] * f['switch_ratio_w30']
        f['interact_pressure_X_info'] = f['move_ratio_w30'] * f['p2_revealed_mons_count']
        f['interact_status_X_hp_t30'] = f['status_burden_advantage_t30'] * f['total_hp_advantage_t30']
    except Exception as e: pass
    if 'battle_id' in f: del f['battle_id']
    return f

# ===========================================================================
# FUNZIONE 2: extract_features_v20 (da Cella 4)
# ===========================================================================
def extract_features_v20(battle, is_test_set=False):
    """ Estrae 40 feature "sweep-potate" (v20) """
    f = {
        'battle_id': battle.get('battle_id', -1),
        'p1_stab_advantage_vs_p2lead': 1.0, 'ko_diff': 0.0, 'dmg_p90_adv': 0.0, 'hp_control_proxy': 0.0,
        'p2_revealed_mons_count': 1.0, 'status_adv_frz': 0.0, 'status_adv_slp': 0.0, 'status_adv_par': 0.0,
        'p1_explosion_value': 0.0, 'total_hp_advantage_t30': 0.0, 'high_hp_advantage_t30': 0.0,
        'status_burden_advantage_t30': 0.0,
        'delta_mean_base_spe': 0.0, 'delta_mean_base_spa': 0.0, 'delta_mean_base_hp': 0.0,
        'move_diff_w30': 0.0, 'switch_diff_w30': 0.0, 'recovery_move_diff': 0.0, 'boost_move_diff': 0.0,
        'boost_diff_spa_sum_w30': 0.0, 'boost_diff_atk_sum_w30': 0.0, 'boost_diff_spe_sum_w30': 0.0,
        'double_switch_count': 0.0,
        'p1_team_has_chansey': 0.0, 'p1_team_has_snorlax': 0.0, 'p1_team_has_tauros': 0.0, 'p2_lead_is_wall': 0.0,
        'p2_sweeper_boosted': 0.0, 'p1_chansey_fainted': 0.0, 'p2_chansey_fainted': 0.0,
        'p1_snorlax_fainted': 0.0, 'p2_snorlax_fainted': 0.0,
        'ko_diff_t20': 0.0, 'total_hp_advantage_t20': 0.0, 'high_hp_advantage_t20': 0.0,
        'status_burden_advantage_t20': 0.0, 'late_game_hp_swing': 0.0, 'late_game_ko_swing': 0.0,
        'late_game_high_hp_swing': 0.0, 'late_game_status_swing': 0.0
    }
    try:
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})\
        p1_stats_raw = {s: [] for s in ALL_BASE_STATS}
        p1_mean_stats = {}
        p1_team_pokemon_types = []
        p1_full_team_status = {}
        if isinstance(p1_team, list) and p1_team:
            for p in p1_team:
                if isinstance(p, dict):
                    p_name = p.get('name', '')
                    if p_name == 'chansey': f['p1_team_has_chansey'] = 1.0
                    if p_name == 'snorlax': f['p1_team_has_snorlax'] = 1.0
                    if p_name == 'tauros': f['p1_team_has_tauros'] = 1.0
                    for s in ALL_BASE_STATS: p1_stats_raw[s].append(p.get(s, 0))
                    types = [t.upper() for t in p.get('types', []) if isinstance(t, str) and t.upper() != 'NOTYPE']
                    if types: p1_team_pokemon_types.append(types)
                    if p_name: p1_full_team_status[p_name] = {'hp': 1.0, 'status': 'nostatus'}
            for s in ALL_BASE_STATS: p1_mean_stats[s] = np.mean(p1_stats_raw[s])
        if isinstance(p2_lead, dict):
            p2_lead_name = p2_lead.get('name', '')
            if p2_lead_name in WALLS: f['p2_lead_is_wall'] = 1.0
            p2_lead_types = [t.upper() for t in p2_lead.get('types', []) if isinstance(t, str) and t.upper() != 'NOTYPE']
            f['delta_mean_base_spe'] = p1_mean_stats.get('base_spe', 0) - p2_lead.get('base_spe', 0)
            f['delta_mean_base_spa'] = p1_mean_stats.get('base_spa', 0) - p2_lead.get('base_spa', 0)
            f['delta_mean_base_hp'] = p1_mean_stats.get('base_hp', 0) - p2_lead.get('base_hp', 0)
            total_effectiveness = 0
            if p2_lead_types and p1_team_pokemon_types:
                for p1_types in p1_team_pokemon_types:
                    poke_effectiveness = sum(get_effectiveness(stab_type, p2_lead_types) for stab_type in p1_types)
                    if p1_types: total_effectiveness += poke_effectiveness / len(p1_types)
                if p1_team_pokemon_types: f['p1_stab_advantage_vs_p2lead'] = total_effectiveness / len(p1_team_pokemon_types)
        tl = battle.get("battle_timeline", [])
        if not isinstance(tl, list): tl = []
        p1_hp_history, p2_hp_history = [], []
        p1_dmg_drops, p2_dmg_drops = [], []
        p1_moves_used, p2_moves_used = 0, 0
        p1_switches, p2_switches = 0, 0
        p1_last_name, p2_last_name = None, None
        p1_fainted_names, p2_fainted_names = set(), set()
        p2_seen_mons = {}
        boost_sums = {f'p1_{stat}': 0 for stat in BOOST_KEYS}
        boost_sums.update({f'p2_{stat}': 0 for stat in BOOST_KEYS})
        prev_p1_hp, prev_p2_hp = 1.0, 1.0
        p1_rec_moves_dyn, p2_rec_moves_dyn = 0, 0
        p1_boost_moves_dyn, p2_boost_moves_dyn = 0, 0
        for i, turn_data in enumerate(tl):
            p1_state = turn_data.get("p1_pokemon_state", {})\
            p2_state = turn_data.get("p2_pokemon_state", {})\
            p1_move = turn_data.get("p1_move_details", {})\
            p2_move = turn_data.get("p2_move_details", {})\
            p1_move_name = str(p1_move.get('name','')).lower() if p1_move else None
            p2_move_name = str(p2_move.get('name','')).lower() if p2_move else None
            if i > 0 and not p1_move_name and not p2_move_name: f['double_switch_count'] += 1
            if p1_move_name in RECOVERY_MOVES: p1_rec_moves_dyn += 1
            if p1_move_name in SETUP_BOOST_MOVES: p1_boost_moves_dyn += 1
            if p2_move_name in RECOVERY_MOVES: p2_rec_moves_dyn += 1
            if p2_move_name in SETUP_BOOST_MOVES: p2_boost_moves_dyn += 1
            current_p1_hp = p1_state.get("hp_pct", prev_p1_hp)
            current_p2_hp = p2_state.get("hp_pct", prev_p2_hp)
            p1_hp_history.append(current_p1_hp)
            p2_hp_history.append(current_p2_hp)
            if i > 0:
                p1_dmg_drops.append(max(0.0, prev_p1_hp - current_p1_hp))
                p2_dmg_drops.append(max(0.0, prev_p2_hp - current_p2_hp))
            if p1_move: p1_moves_used += 1
            if p2_move: p2_moves_used += 1
            p1_name = p1_state.get("name")
            p2_name = p2_state.get("name")
            if p1_name and p1_name != p1_last_name and i > 0: p1_switches += 1
            if p2_name and p2_name != p2_last_name and i > 0: p2_switches += 1
            p1_last_name, p2_last_name = p1_name, p2_name
            if p1_state.get("status") == "fnt" and p1_name and p1_name not in p1_fainted_names: f['ko_diff'] -= 1; p1_fainted_names.add(p1_name)
            if p2_state.get("status") == "fnt" and p2_name and p2_name not in p2_fainted_names: f['ko_diff'] += 1; p2_fainted_names.add(p2_name)
            p1_status = p1_state.get("status", "nostatus")
            p2_status = p2_state.get("status", "nostatus")
            if p1_status == "frz": f['status_adv_frz'] -= 1
            if p1_status == "slp": f['status_adv_slp'] -= 1
            if p1_status == "par": f['status_adv_par'] -= 1
            if p2_status == "frz": f['status_adv_frz'] += 1
            if p2_status == "slp": f['status_adv_slp'] += 1
            if p2_status == "par": f['status_adv_par'] += 1
            b1 = p1_state.get("boosts", {})\
            b2 = p2_state.get("boosts", {})\
            for stat in BOOST_KEYS:
                boost_sums[f'p1_{stat}'] += b1.get(stat, 0)
                boost_sums[f'p2_{stat}'] += b2.get(stat, 0)
            if p2_name in SETUP_SWEEPERS and (b2.get('spa', 0) > 0 or b2.get('atk', 0) > 0): f['p2_sweeper_boosted'] = 1.0
            if p1_name and p1_name in p1_full_team_status: p1_full_team_status[p1_name] = {"hp": current_p1_hp, "status": p1_status}
            if p2_name: p2_seen_mons[p2_name] = {"hp": current_p2_hp, "status": p2_status}
            if p1_move_name in EXPLOSION_MOVES: f['p1_explosion_value'] += (prev_p2_hp - prev_p1_hp)
            prev_p1_hp, prev_p2_hp = current_p1_hp, current_p2_hp
            if i == 19:
                p1_alive_t20 = [mon for mon in p1_full_team_status.values() if mon['status'] != 'fnt']
                p2_alive_t20 = [mon for mon in p2_seen_mons.values() if mon['status'] != 'fnt']
                f['total_hp_advantage_t20'] = sum(mon['hp'] for mon in p1_alive_t20) - sum(mon['hp'] for mon in p2_alive_t20)
                f['ko_diff_t20'] = f['ko_diff']
                f['high_hp_advantage_t20'] = sum(1 for mon in p1_alive_t20 if mon['hp'] >= 0.8) - sum(1 for mon in p2_alive_t20 if mon['hp'] >= 0.8)
                p1_status_count_t20 = sum(1 for mon in p1_alive_t20 if mon['status'] in STATUS_LIST_VALID)
                p2_status_count_t20 = sum(1 for mon in p2_alive_t20 if mon['status'] in STATUS_LIST_VALID)
                f['status_burden_advantage_t20'] = (p1_status_count_t20 / (len(p1_alive_t20) + 1e-6)) - (p2_status_count_t20 / (len(p2_alive_t20) + 1e-6))
        if p1_dmg_drops and p2_dmg_drops: f['dmg_p90_adv'] = np.quantile(p2_dmg_drops, 0.9) - np.quantile(p1_dmg_drops, 0.9)
        if p1_hp_history and p2_hp_history: f['hp_control_proxy'] = ( (np.mean(p1_hp_history) - np.mean(p2_hp_history)) + (p1_hp_history[-1] - p2_hp_history[-1]) ) / 2.0
        if p2_seen_mons: f['p2_revealed_mons_count'] = len(p2_seen_mons)
        f['move_diff_w30'] = p1_moves_used - p2_moves_used
        f['switch_diff_w30'] = p1_switches - p2_switches
        f['recovery_move_diff'] = p1_rec_moves_dyn - p2_rec_moves_dyn
        f['boost_move_diff'] = p1_boost_moves_dyn - p2_boost_moves_dyn
        f['boost_diff_spa_sum_w30'] = boost_sums['p1_spa'] - boost_sums['p2_spa']
        f['boost_diff_atk_sum_w30'] = boost_sums['p1_atk'] - boost_sums['p2_atk']
        f['boost_diff_spe_sum_w30'] = boost_sums['p1_spe'] - boost_sums['p2_spe']
        p1_alive_t30 = [mon for mon in p1_full_team_status.values() if mon['status'] != 'fnt']
        p2_alive_t30 = [mon for mon in p2_seen_mons.values() if mon['status'] != 'fnt']
        if 'snorlax' in p1_full_team_status and p1_full_team_status['snorlax']['status'] == 'fnt': f['p1_snorlax_fainted'] = 1.0
        if 'chansey' in p2_seen_mons and p2_seen_mons['chansey']['status'] == 'fnt': f['p2_chansey_fainted'] = 1.0
        if 'snorlax' in p2_seen_mons and p2_seen_mons['snorlax']['status'] == 'fnt': f['p2_snorlax_fainted'] = 1.0
        if 'chansey' in p1_full_team_status and p1_full_team_status['chansey']['status'] == 'fnt': f['p1_chansey_fainted'] = 1.0
        p1_status_count_t30 = sum(1 for mon in p1_alive_t30 if mon['status'] in STATUS_LIST_VALID)
        p2_status_count_t30 = sum(1 for mon in p2_alive_t30 if mon['status'] in STATUS_LIST_VALID)
        f['status_burden_advantage_t30'] = (p1_status_count_t30 / (len(p1_alive_t30) + 1e-6)) - (p2_status_count_t30 / (len(p2_alive_t30) + 1e-6))
        f['total_hp_advantage_t30'] = sum(mon['hp'] for mon in p1_alive_t30) - sum(mon['hp'] for mon in p2_alive_t30)
        f['high_hp_advantage_t30'] = sum(1 for mon in p1_alive_t30 if mon['hp'] >= 0.8) - sum(1 for mon in p2_alive_t30 if mon['hp'] >= 0.8)
        f['late_game_hp_swing'] = f['total_hp_advantage_t30'] - f['total_hp_advantage_t20']
        f['late_game_ko_swing'] = f['ko_diff'] - f['ko_diff_t20']
        f['late_game_high_hp_swing'] = f['high_hp_advantage_t30'] - f['high_hp_advantage_t20']
        f['late_game_status_swing'] = f['status_burden_advantage_t30'] - f['status_burden_advantage_t20']
    except Exception as e: pass
    if 'battle_id' in f: del f['battle_id']
    return f

# ===========================================================================
# FUNZIONE 3: extract_features_v19 (da Cella 4)
# ===========================================================================
def extract_features_v19(battle):
    """ Estrae 41 feature (v19 modulare) """
    # Gli interruttori sono tutti True
    CALCOLA_CORE = True
    CALCOLA_RUOLO = True
    CALCOLA_DELTA = True
    CALCOLA_SWEEP = True
    
    f = {'battle_id': battle.get('battle_id', -1)}
    try:
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})\
        p1_stats_raw = {s: [] for s in ALL_BASE_STATS}
        p1_mean_stats = {}
        p1_team_pokemon_types = []
        p1_full_team_status = {}
        if CALCOLA_RUOLO:
            f['p1_team_has_chansey'] = 0.0
            f['p1_team_has_snorlax'] = 0.0
            f['p1_team_has_tauros'] = 0.0
            f['p2_lead_is_wall'] = 0.0
        if isinstance(p1_team, list) and p1_team:
            for p in p1_team:
                if isinstance(p, dict):
                    p_name = p.get('name', '')
                    if CALCOLA_RUOLO:
                        if p_name == 'chansey': f['p1_team_has_chansey'] = 1.0
                        if p_name == 'snorlax': f['p1_team_has_snorlax'] = 1.0
                        if p_name == 'tauros': f['p1_team_has_tauros'] = 1.0
                    for s in ALL_BASE_STATS: p1_stats_raw[s].append(p.get(s, 0))
                    types = [t.upper() for t in p.get('types', []) if isinstance(t, str) and t.upper() != 'NOTYPE']
                    if types: p1_team_pokemon_types.append(types)
                    if p_name: p1_full_team_status[p_name] = {'hp': 1.0, 'status': 'nostatus'}
            for s in ALL_BASE_STATS: p1_mean_stats[s] = np.mean(p1_stats_raw[s])
        if isinstance(p2_lead, dict):
            if CALCOLA_RUOLO:
                p2_lead_name = p2_lead.get('name', '')
                if p2_lead_name in WALLS: f['p2_lead_is_wall'] = 1.0
            p2_lead_types = [t.upper() for t in p2_lead.get('types', []) if isinstance(t, str) and t.upper() != 'NOTYPE']
            if CALCOLA_DELTA:
                f['delta_mean_base_spe'] = p1_mean_stats.get('base_spe', 0) - p2_lead.get('base_spe', 0)
                f['delta_mean_base_spa'] = p1_mean_stats.get('base_spa', 0) - p2_lead.get('base_spa', 0)
                f['delta_mean_base_hp'] = p1_mean_stats.get('base_hp', 0) - p2_lead.get('base_hp', 0)
            if CALCOLA_CORE:
                total_effectiveness = 0
                if p2_lead_types and p1_team_pokemon_types:
                    for p1_types in p1_team_pokemon_types:
                        poke_effectiveness = sum(get_effectiveness(stab_type, p2_lead_types) for stab_type in p1_types)
                        if p1_types: total_effectiveness += poke_effectiveness / len(p1_types)
                    if p1_team_pokemon_types: f['p1_stab_advantage_vs_p2lead'] = total_effectiveness / len(p1_team_pokemon_types)
        tl = battle.get("battle_timeline", [])
        if not isinstance(tl, list): tl = []
        p1_hp_history, p2_hp_history = [], []
        p1_dmg_drops, p2_dmg_drops = [], []
        p1_moves_used, p2_moves_used = 0, 0
        p1_switches, p2_switches = 0, 0
        p1_last_name, p2_last_name = None, None
        p1_fainted_names, p2_fainted_names = set(), set()
        p2_seen_mons = {}
        boost_sums = {f'p1_{stat}': 0 for stat in BOOST_KEYS}
        boost_sums.update({f'p2_{stat}': 0 for stat in BOOST_KEYS})
        prev_p1_hp, prev_p2_hp = 1.0, 1.0
        p1_rec_moves_dyn, p2_rec_moves_dyn = 0, 0
        p1_boost_moves_dyn, p2_boost_moves_dyn = 0, 0
        if CALCOLA_RUOLO:
            f['p1_sweeper_boosted'] = 0.0
            f['p2_sweeper_boosted'] = 0.0
        if CALCOLA_DELTA:
            f['double_switch_count'] = 0.0
        if CALCOLA_CORE:
            f['ko_diff'] = 0.0
            f['status_adv_frz'] = 0.0
            f['status_adv_slp'] = 0.0
            f['status_adv_par'] = 0.0
            f['p1_explosion_value'] = 0.0
        ko_diff_t20 = 0.0
        total_hp_t20 = 0.0
        high_hp_t20 = 0.0
        status_t20 = 0.0
        for i, turn_data in enumerate(tl):
            p1_state = turn_data.get("p1_pokemon_state", {})\
            p2_state = turn_data.get("p2_pokemon_state", {})\
            p1_move = turn_data.get("p1_move_details", {})\
            p2_move = turn_data.get("p2_move_details", {})\
            p1_move_name = str(p1_move.get('name','')).lower() if p1_move else None
            p2_move_name = str(p2_move.get('name','')).lower() if p2_move else None
            if CALCOLA_DELTA:
                if i > 0 and not p1_move_name and not p2_move_name:
                    f['double_switch_count'] += 1
            if CALCOLA_DELTA:
                if p1_move_name in RECOVERY_MOVES: p1_rec_moves_dyn += 1
                if p1_move_name in SETUP_BOOST_MOVES: p1_boost_moves_dyn += 1
            if p2_move_name in RECOVERY_MOVES: p2_rec_moves_dyn += 1
            if p2_move_name in SETUP_BOOST_MOVES: p2_boost_moves_dyn += 1
            current_p1_hp = p1_state.get("hp_pct", prev_p1_hp)
            current_p2_hp = p2_state.get("hp_pct", prev_p2_hp)
            if CALCOLA_CORE:
                p1_hp_history.append(current_p1_hp)
                p2_hp_history.append(current_p2_hp)
                if i > 0:
                    p1_dmg_drops.append(max(0.0, prev_p1_hp - current_p1_hp))
                    p2_dmg_drops.append(max(0.0, prev_p2_hp - current_p2_hp))
            if p1_move: p1_moves_used += 1
            if p2_move: p2_moves_used += 1
            p1_name = p1_state.get("name")
            p2_name = p2_state.get("name")
            if p1_name and p1_name != p1_last_name and i > 0: p1_switches += 1
            if p2_name and p2_name != p2_last_name and i > 0: p2_switches += 1
            p1_last_name, p2_last_name = p1_name, p2_name
            if CALCOLA_CORE:
                if p1_state.get("status") == "fnt" and p1_name and p1_name not in p1_fainted_names: f['ko_diff'] -= 1; p1_fainted_names.add(p1_name)
                if p2_state.get("status") == "fnt" and p2_name and p2_name not in p2_fainted_names: f['ko_diff'] += 1; p2_fainted_names.add(p2_name)
                p1_status = p1_state.get("status", "nostatus")
                p2_status = p2_state.get("status", "nostatus")
                if p1_status == "frz": f['status_adv_frz'] -= 1
                if p1_status == "slp": f['status_adv_slp'] -= 1
                if p1_status == "par": f['status_adv_par'] -= 1
                if p2_status == "frz": f['status_adv_frz'] += 1
                if p2_status == "slp": f['status_adv_slp'] += 1
                if p2_status == "par": f['status_adv_par'] += 1
            if CALCOLA_DELTA or CALCOLA_RUOLO:
                b1 = p1_state.get("boosts", {})\
                b2 = p2_state.get("boosts", {})\
                if CALCOLA_DELTA:
                    for stat in BOOST_KEYS:
                        boost_sums[f'p1_{stat}'] += b1.get(stat, 0)
                        boost_sums[f'p2_{stat}'] += b2.get(stat, 0)
                if CALCOLA_RUOLO:
                    if p2_name in SETUP_SWEEPERS and (b2.get('spa', 0) > 0 or b2.get('atk', 0) > 0): f['p2_sweeper_boosted'] = 1.0
            if p1_name and p1_name in p1_full_team_status: p1_full_team_status[p1_name] = {"hp": current_p1_hp, "status": p1_status}
            if p2_name: p2_seen_mons[p2_name] = {"hp": current_p2_hp, "status": p2_status}
            if CALCOLA_CORE and p1_move_name in EXPLOSION_MOVES: f['p1_explosion_value'] += (prev_p2_hp - prev_p1_hp)
            prev_p1_hp, prev_p2_hp = current_p1_hp, current_p2_hp
            if CALCOLA_SWEEP and i == 19:
                p1_alive_t20 = [mon for mon in p1_full_team_status.values() if mon['status'] != 'fnt']
                p2_alive_t20 = [mon for mon in p2_seen_mons.values() if mon['status'] != 'fnt']
                f['total_hp_advantage_t20'] = sum(mon['hp'] for mon in p1_alive_t20) - sum(mon['hp'] for mon in p2_alive_t20)
                f['ko_diff_t20'] = f.get('ko_diff', 0.0)
                f['high_hp_advantage_t20'] = sum(1 for mon in p1_alive_t20 if mon['hp'] >= 0.8) - sum(1 for mon in p2_alive_t20 if mon['hp'] >= 0.8)
                p1_status_count_t20 = sum(1 for mon in p1_alive_t20 if mon['status'] in STATUS_LIST_VALID)
                p2_status_count_t20 = sum(1 for mon in p2_alive_t20 if mon['status'] in STATUS_LIST_VALID)
                f['status_burden_advantage_t20'] = (p1_status_count_t20 / (len(p1_alive_t20) + 1e-6)) - (p2_status_count_t20 / (len(p2_alive_t20) + 1e-6))
        if CALCOLA_CORE:
            if p1_dmg_drops and p2_dmg_drops: f['dmg_p90_adv'] = np.quantile(p2_dmg_drops, 0.9) - np.quantile(p1_dmg_drops, 0.9)
            if p1_hp_history and p2_hp_history: f['hp_control_proxy'] = ( (np.mean(p1_hp_history) - np.mean(p2_hp_history)) + (p1_hp_history[-1] - p2_hp_history[-1]) ) / 2.0
            if p2_seen_mons: f['p2_revealed_mons_count'] = len(p2_seen_mons)
        if CALCOLA_DELTA:
            f['move_diff_w30'] = p1_moves_used - p2_moves_used
            f['switch_diff_w30'] = p1_switches - p2_switches
            f['recovery_move_diff'] = p1_rec_moves_dyn - p2_rec_moves_dyn
            f['boost_move_diff'] = p1_boost_moves_dyn - p2_boost_moves_dyn
            f['boost_diff_spa_sum_w30'] = boost_sums['p1_spa'] - boost_sums['p2_spa']
            f['boost_diff_atk_sum_w30'] = boost_sums['p1_atk'] - boost_sums['p2_atk']
            f['boost_diff_spe_sum_w30'] = boost_sums['p1_spe'] - boost_sums['p2_spe']
        p1_alive_t30 = [mon for mon in p1_full_team_status.values() if mon['status'] != 'fnt']
        p2_alive_t30 = [mon for mon in p2_seen_mons.values() if mon['status'] != 'fnt']
        if CALCOLA_RUOLO:
            f['p1_snorlax_fainted'] = 1.0 if 'snorlax' in p1_full_team_status and p1_full_team_status['snorlax']['status'] == 'fnt' else 0.0
            f['p2_chansey_fainted'] = 1.0 if 'chansey' in p2_seen_mons and p2_seen_mons['chansey']['status'] == 'fnt' else 0.0
            f['p2_snorlax_fainted'] = 1.0 if 'snorlax' in p2_seen_mons and p2_seen_mons['snorlax']['status'] == 'fnt' else 0.0
            f['p1_chansey_fainted'] = 1.0 if 'chansey' in p1_full_team_status and p1_full_team_status['chansey']['status'] == 'fnt' else 0.0
        if CALCOLA_CORE:
            p1_status_count_t30 = sum(1 for mon in p1_alive_t30 if mon['status'] in STATUS_LIST_VALID)
            p2_status_count_t30 = sum(1 for mon in p2_alive_t30 if mon['status'] in STATUS_LIST_VALID)
            f['status_burden_advantage_t30'] = (p1_status_count_t30 / (len(p1_alive_t30) + 1e-6)) - (p2_status_count_t30 / (len(p2_alive_t30) + 1e-6))
            f['total_hp_advantage_t30'] = sum(mon['hp'] for mon in p1_alive_t30) - sum(mon['hp'] for mon in p2_alive_t30)
            f['high_hp_advantage_t30'] = sum(1 for mon in p1_alive_t30 if mon['hp'] >= 0.8) - sum(1 for mon in p2_alive_t30 if mon['hp'] >= 0.8)
        if CALCOLA_SWEEP:
            f['late_game_hp_swing'] = f.get('total_hp_advantage_t30', 0.0) - f.get('total_hp_advantage_t20', 0.0)
            f['late_game_ko_swing'] = f.get('ko_diff', 0.0) - f.get('ko_diff_t20', 0.0)
            f['late_game_high_hp_swing'] = f.get('high_hp_advantage_t30', 0.0) - f.get('high_hp_advantage_t20', 0.0)
            f['late_game_status_swing'] = f.get('status_burden_advantage_t30', 0.0) - f.get('status_burden_advantage_t20', 0.0)
    except Exception as e: pass
    if 'battle_id' in f: del f['battle_id']
    return f

# ===========================================================================
# FUNZIONE 4: extract_advanced_timeline_features (da Cella 5)
# ===========================================================================
def extract_advanced_timeline_features(battle):
    """Estrae 70+ feature killer dalla timeline"""
    timeline = battle.get('battle_timeline', [])
    
    if not timeline:
        # Default values se timeline vuota
        return {k: 0.0 for k in [
            'p1_final_hp', 'p2_final_hp', 'hp_diff_final',
            'p1_mid_hp', 'p2_mid_hp', 'hp_diff_mid',
            'p1_hp_slope', 'p2_hp_slope', 'p1_hp_volatility', 'p2_hp_volatility',
            'p1_hp_lost', 'p2_hp_lost', 'hp_lost_diff',
            'p1_ko_count', 'p2_ko_count', 'ko_diff', 'first_ko_player', 'first_ko_turn',
            'p1_pokemon_used', 'p2_pokemon_used',
            'p1_status_turns', 'p2_status_turns', 'status_turns_diff',
            'p1_frozen_turns', 'p2_frozen_turns', 'frozen_diff',
            'p1_paralyzed_turns', 'p2_paralyzed_turns', 'paralyzed_diff',
            'p1_asleep_turns', 'p2_asleep_turns', 'asleep_diff',
            'p1_total_boosts', 'p2_total_boosts', 'boost_advantage',
            'p1_max_boost', 'p2_max_boost', 'p1_boost_turns', 'p2_boost_turns',
            'p1_offensive_moves', 'p2_offensive_moves', 'offensive_moves_diff',
            'p1_status_moves', 'p2_status_moves',
            'p1_recovery_count', 'p2_recovery_count', 'recovery_diff',
            'p1_setup_count', 'p2_setup_count', 'setup_diff',
            'p1_explosion_used', 'p2_explosion_used',
            'p1_avg_base_power', 'p2_avg_base_power',
            'p1_reflect_turns', 'p2_reflect_turns',
            'p1_damage_events', 'p2_damage_events',
            'p1_avg_damage', 'p2_avg_damage',
            'p1_max_single_damage', 'p2_max_single_damage',
            'early_hp_diff', 'late_hp_diff', 'momentum_shift'
        ]}
    
    features = {}
    
    # Extract HP values per turn
    p1_hp = [t.get('p1_pokemon_state', {}).get('hp_pct', 1.0) for t in timeline]
    p2_hp = [t.get('p2_pokemon_state', {}).get('hp_pct', 1.0) for t in timeline]
    
    # HP FEATURES
    features['p1_final_hp'] = p1_hp[-1]
    features['p2_final_hp'] = p2_hp[-1]
    features['hp_diff_final'] = p1_hp[-1] - p2_hp[-1]
    
    mid = len(timeline) // 2
    features['p1_mid_hp'] = p1_hp[mid]
    features['p2_mid_hp'] = p2_hp[mid]
    features['hp_diff_mid'] = p1_hp[mid] - p2_hp[mid]
    
    if len(p1_hp) > 1:
        turns = np.arange(len(p1_hp))
        features['p1_hp_slope'] = np.polyfit(turns, p1_hp, 1)[0]
        features['p2_hp_slope'] = np.polyfit(turns, p2_hp, 1)[0]
    else:
        features['p1_hp_slope'] = 0.0
        features['p2_hp_slope'] = 0.0
    
    features['p1_hp_volatility'] = np.std(p1_hp)
    features['p2_hp_volatility'] = np.std(p2_hp)
    features['p1_hp_lost'] = 1.0 - p1_hp[-1]
    features['p2_hp_lost'] = 1.0 - p2_hp[-1]
    features['hp_lost_diff'] = features['p2_hp_lost'] - features['p1_hp_lost']
    
    # KO TRACKING
    p1_ko = sum(1 for t in timeline if t.get('p1_pokemon_state', {}).get('status') == 'fnt')
    p2_ko = sum(1 for t in timeline if t.get('p2_pokemon_state', {}).get('status') == 'fnt')
    features['p1_ko_count'] = p1_ko
    features['p2_ko_count'] = p2_ko
    features['ko_diff'] = p2_ko - p1_ko
    
    first_ko_player = 0
    first_ko_turn = 0
    for i, t in enumerate(timeline):
        if t.get('p1_pokemon_state', {}).get('status') == 'fnt' and first_ko_player == 0:
            first_ko_player = 1
            first_ko_turn = i + 1
            break
        if t.get('p2_pokemon_state', {}).get('status') == 'fnt' and first_ko_player == 0:
            first_ko_player = 2
            first_ko_turn = i + 1
            break
    features['first_ko_player'] = first_ko_player
    features['first_ko_turn'] = first_ko_turn
    
    p1_mons = set(t.get('p1_pokemon_state', {}).get('name', '') for t in timeline)
    p2_mons = set(t.get('p2_pokemon_state', {}).get('name', '') for t in timeline)
    features['p1_pokemon_used'] = len(p1_mons)
    features['p2_pokemon_used'] = len(p2_mons)
    
    # STATUS CONDITIONS
    status_counts = {'p1': {'total': 0, 'frz': 0, 'par': 0, 'slp': 0},
                     'p2': {'total': 0, 'frz': 0, 'par': 0, 'slp': 0}}
    
    for t in timeline:
        p1_status = t.get('p1_pokemon_state', {}).get('status', 'nostatus')
        p2_status = t.get('p2_pokemon_state', {}).get('status', 'nostatus')
        
        if p1_status not in ['nostatus', 'fnt']:
            status_counts['p1']['total'] += 1
            if p1_status in ['frz', 'par', 'slp']:
                status_counts['p1'][p1_status] += 1
        
        if p2_status not in ['nostatus', 'fnt']:
            status_counts['p2']['total'] += 1
            if p2_status in ['frz', 'par', 'slp']:
                status_counts['p2'][p2_status] += 1
    
    features['p1_status_turns'] = status_counts['p1']['total']
    features['p2_status_turns'] = status_counts['p2']['total']
    features['status_turns_diff'] = status_counts['p2']['total'] - status_counts['p1']['total']
    features['p1_frozen_turns'] = status_counts['p1']['frz']
    features['p2_frozen_turns'] = status_counts['p2']['frz']
    features['frozen_diff'] = status_counts['p2']['frz'] - status_counts['p1']['frz']
    features['p1_paralyzed_turns'] = status_counts['p1']['par']
    features['p2_paralyzed_turns'] = status_counts['p2']['par']
    features['paralyzed_diff'] = status_counts['p2']['par'] - status_counts['p1']['par']
    features['p1_asleep_turns'] = status_counts['p1']['slp']
    features['p2_asleep_turns'] = status_counts['p2']['slp']
    features['asleep_diff'] = status_counts['p2']['slp'] - status_counts['p1']['slp']
    
    # STAT BOOSTS
    p1_boosts_total = 0
    p2_boosts_total = 0
    p1_max_boost = 0
    p2_max_boost = 0
    p1_boost_turns = 0
    p2_boost_turns = 0
    
    for t in timeline:
        p1_b = t.get('p1_pokemon_state', {}).get('boosts', {})\
        p2_b = t.get('p2_pokemon_state', {}).get('boosts', {})\
        
        p1_sum = sum(p1_b.values())
        p2_sum = sum(p2_b.values())
        
        p1_boosts_total += p1_sum
        p2_boosts_total += p2_sum
        
        if p1_b:
            p1_max_boost = max(p1_max_boost, max(p1_b.values()))
        if p2_b:
            p2_max_boost = max(p2_max_boost, max(p2_b.values()))
        
        if p1_sum > 0:
            p1_boost_turns += 1
        if p2_sum > 0:
            p2_boost_turns += 1
    
    features['p1_total_boosts'] = p1_boosts_total
    features['p2_total_boosts'] = p2_boosts_total
    features['boost_advantage'] = p1_boosts_total - p2_boosts_total
    features['p1_max_boost'] = p1_max_boost
    features['p2_max_boost'] = p2_max_boost
    features['p1_boost_turns'] = p1_boost_turns
    features['p2_boost_turns'] = p2_boost_turns
    
    # MOVE ANALYSIS
    recovery_moves = {'recover', 'rest', 'softboiled'}
    setup_moves = {'swords dance', 'amnesia', 'agility', 'reflect', 'barrier'}
    explosion_moves = {'explosion', 'selfdestruct'}
    
    p1_offensive = 0
    p2_offensive = 0
    p1_status_moves = 0
    p2_status_moves = 0
    p1_recovery = 0
    p2_recovery = 0
    p1_setup = 0
    p2_setup = 0
    p1_explosion = 0
    p2_explosion = 0
    p1_bp_total = 0
    p2_bp_total = 0
    
    for t in timeline:
        p1_move = t.get('p1_move_details')
        p2_move = t.get('p2_move_details')
        
        if p1_move:
            name = p1_move.get('name', '').lower()
            cat = p1_move.get('category', '')
            bp = p1_move.get('base_power', 0)
            
            if cat in ['PHYSICAL', 'SPECIAL']:
                p1_offensive += 1
                p1_bp_total += bp
            elif cat == 'STATUS':
                p1_status_moves += 1
            
            if name in recovery_moves:
                p1_recovery += 1
            if name in setup_moves:
                p1_setup += 1
            if name in explosion_moves:
                p1_explosion = 1
        
        if p2_move:
            name = p2_move.get('name', '').lower()
            cat = p2_move.get('category', '')
            bp = p2_move.get('base_power', 0)
            
            if cat in ['PHYSICAL', 'SPECIAL']:
                p2_offensive += 1
                p2_bp_total += bp
            elif cat == 'STATUS':
                p2_status_moves += 1
            
            if name in recovery_moves:
                p2_recovery += 1
            if name in setup_moves:
                p2_setup += 1
            if name in explosion_moves:
                p2_explosion = 1
    
    features['p1_offensive_moves'] = p1_offensive
    features['p2_offensive_moves'] = p2_offensive
    features['offensive_moves_diff'] = p1_offensive - p2_offensive
    features['p1_status_moves'] = p1_status_moves
    features['p2_status_moves'] = p2_status_moves
    features['p1_recovery_count'] = p1_recovery
    features['p2_recovery_count'] = p2_recovery
    features['recovery_diff'] = p1_recovery - p2_recovery
    features['p1_setup_count'] = p1_setup
    features['p2_setup_count'] = p2_setup
    features['setup_diff'] = p1_setup - p2_setup
    features['p1_explosion_used'] = p1_explosion
    features['p2_explosion_used'] = p2_explosion
    features['p1_avg_base_power'] = p1_bp_total / max(p1_offensive, 1)
    features['p2_avg_base_power'] = p2_bp_total / max(p2_offensive, 1)
    
    # EFFECTS
    p1_reflect = sum(1 for t in timeline if 'reflect' in t.get('p1_pokemon_state', {}).get('effects', []))
    p2_reflect = sum(1 for t in timeline if 'reflect' in t.get('p2_pokemon_state', {}).get('effects', []))
    features['p1_reflect_turns'] = p1_reflect
    features['p2_reflect_turns'] = p2_reflect
    
    # DAMAGE PATTERNS
    p1_drops = [p1_hp[i-1] - p1_hp[i] for i in range(1, len(p1_hp)) if p1_hp[i-1] - p1_hp[i] > 0]
    p2_drops = [p2_hp[i-1] - p2_hp[i] for i in range(1, len(p2_hp)) if p2_hp[i-1] - p2_hp[i] > 0]
    
    features['p1_damage_events'] = len(p1_drops)
    features['p2_damage_events'] = len(p2_drops)
    features['p1_avg_damage'] = np.mean(p1_drops) if p1_drops else 0.0
    features['p2_avg_damage'] = np.mean(p2_drops) if p2_drops else 0.0
    features['p1_max_single_damage'] = max(p1_drops) if p1_drops else 0.0
    features['p2_max_single_damage'] = max(p2_drops) if p2_drops else 0.0
    
    # EARLY vs LATE GAME
    early = min(10, len(timeline))
    late_start = max(20, len(timeline) - 10)
    
    early_p1 = np.mean(p1_hp[:early])
    early_p2 = np.mean(p2_hp[:early])
    features['early_hp_diff'] = early_p1 - early_p2
    
    late_p1 = np.mean(p1_hp[late_start:])
    late_p2 = np.mean(p2_hp[late_start:])
    features['late_hp_diff'] = late_p1 - late_p2
    
    features['momentum_shift'] = features['late_hp_diff'] - features['early_hp_diff']
    
    return features

# ===========================================================================
# FUNZIONE 5: extract_features_v21 (da Cella 5)
# ===========================================================================
def extract_features_v21(battle, *args, **kwargs):
    """
    Feature Set v21: Combina v20 + timeline avanzate
    """
    # Usa la tua migliore versione (v20)
    base_features = extract_features_v20(battle)
    
    # Aggiungi timeline features
    timeline_features = extract_advanced_timeline_features(battle)
    
    # Merge
    combined = {**base_features, **timeline_features}
    
    return combined

# ===========================================================================
# FUNZIONE 6: extract_features_CRITICAL_MISSING (da Cella 3)
# ===========================================================================
def extract_features_CRITICAL_MISSING(battle):
    """
    Feature CRITICHE - VERSIONE CORRETTA con nomi chiavi giusti
    """
    f = {}
    
    timeline = battle.get('battle_timeline', [])
    p1_team = battle.get('p1_team_details', [])
    p2_lead = battle.get('p2_lead_details', {})\
    
    if not timeline:
        return {k: 0.0 for k in [
            'freeze_advantage', 'freeze_turns_p1', 'freeze_turns_p2',
            'status_pressure_p1', 'status_pressure_p2', 'status_dominance',
            'recovery_count_p1', 'recovery_efficiency_p1', 'healing_vs_damage_ratio',
            'turn_control_total_p1', 'turn_control_streak_p1', 'momentum_swings',
            'high_crit_move_count_p1', 'critical_vulnerability_p2',
            'hp_trajectory_positive', 'hp_min_reached_p1', 'endgame_hp_advantage',
            'explosion_potential_p1', 'explosion_potential_p2',
            'supereffective_streak_p1', 'supereffective_count_p1',
            'speed_control_turns', 'speed_advantage_maintained',
            'wallbreaker_score_p1', 'wall_score_p2'
        ]}
    
    total_turns = len(timeline)
    
    # 1. FREEZE MECHANICS
    freeze_turns_p1 = 0
    freeze_turns_p2 = 0
    
    for turn in timeline:
        p1_status = ''
        p2_status = ''
        
        if 'p1_lead_status' in turn:
            p1_status = turn['p1_lead_status']
        elif 'p1_status' in turn:
            p1_status = turn['p1_status']
        elif 'p1_pokemon_state' in turn and isinstance(turn['p1_pokemon_state'], dict):
            p1_status = turn['p1_pokemon_state'].get('status', '')
            
        if 'p2_lead_status' in turn:
            p2_status = turn['p2_lead_status']
        elif 'p2_status' in turn:
            p2_status = turn['p2_status']
        elif 'p2_pokemon_state' in turn and isinstance(turn['p2_pokemon_state'], dict):
            p2_status = turn['p2_pokemon_state'].get('status', '')
        
        if 'frz' in str(p1_status).lower():
            freeze_turns_p1 += 1
        if 'frz' in str(p2_status).lower():
            freeze_turns_p2 += 1
    
    f['freeze_turns_p1'] = freeze_turns_p1
    f['freeze_turns_p2'] = freeze_turns_p2
    f['freeze_advantage'] = freeze_turns_p2 - freeze_turns_p1
    
    # 2. STATUS PRESSURE ACCUMULATION
    status_turns_p1 = 0
    status_turns_p2 = 0
    status_inflicted_by_p1 = 0
    status_inflicted_by_p2 = 0
    prev_p1_status = ''
    prev_p2_status = ''
    
    for turn in timeline:
        p1_status = ''
        p2_status = ''
        
        if 'p1_lead_status' in turn:
            p1_status = str(turn['p1_lead_status'])
        if 'p2_lead_status' in turn:
            p2_status = str(turn['p2_lead_status'])
        
        if p1_status and p1_status not in ['', 'None', 'none']:
            status_turns_p1 += 1
        if p2_status and p2_status not in ['', 'None', 'none']:
            status_turns_p2 += 1
        
        if p2_status and p2_status != prev_p2_status and not prev_p2_status:
            status_inflicted_by_p1 += 1
        if p1_status and p1_status != prev_p1_status and not prev_p1_status:
            status_inflicted_by_p2 += 1
        
        prev_p1_status = p1_status
        prev_p2_status = p2_status
    
    f['status_pressure_p1'] = status_turns_p1 / total_turns if total_turns > 0 else 0
    f['status_pressure_p2'] = status_turns_p2 / total_turns if total_turns > 0 else 0
    f['status_dominance'] = status_inflicted_by_p1 - status_inflicted_by_p2
    
    # 3. RECOVERY EFFICIENCY
    recovery_count = 0
    recovery_moves_set = {'recover', 'softboiled', 'rest', 'synthesis', 'morning sun', 'moonlight'}
    
    for turn in timeline:
        move_used = ''
        if 'p1_lead_move_used' in turn:
            move_used = str(turn['p1_lead_move_used']).lower()
        elif 'p1_move' in turn:
            move_used = str(turn['p1_move']).lower()
            
        if move_used in recovery_moves_set:
            recovery_count += 1
    
    f['recovery_count_p1'] = recovery_count
    f['recovery_efficiency_p1'] = recovery_count / total_turns if total_turns > 0 else 0
    
    damage_moves = 0
    for turn in timeline:
        move_used = ''
        if 'p1_lead_move_used' in turn:
            move_used = str(turn['p1_lead_move_used']).lower()
        elif 'p1_move' in turn:
            move_used = str(turn['p1_move']).lower()
            
        if move_used and move_used not in recovery_moves_set and move_used not in ['switch', '', 'none']:
            damage_moves += 1
    
    f['healing_vs_damage_ratio'] = recovery_count / (damage_moves + 1)
    
    # 4. TURN CONTROL & MOMENTUM
    turn_control_count = 0
    current_streak = 0
    max_streak = 0
    momentum_changes = 0
    prev_advantage = 0
    
    for turn in timeline:
        p1_hp = 0.5
        p2_hp = 0.5
        
        if 'p1_lead_hp_pct' in turn:
            p1_hp = turn['p1_lead_hp_pct']
        elif 'p1_hp' in turn:
            p1_hp = turn['p1_hp']
            
        if 'p2_lead_hp_pct' in turn:
            p2_hp = turn['p2_lead_hp_pct']
        elif 'p2_hp' in turn:
            p2_hp = turn['p2_hp']
        
        p1_status = ''
        p2_status = ''
        if 'p1_lead_status' in turn:
            p1_status = str(turn['p1_lead_status'])
        if 'p2_lead_status' in turn:
            p2_status = str(turn['p2_lead_status'])
        
        if ('frz' in p2_status.lower() or 'slp' in p2_status.lower()) and \
           'frz' not in p1_status.lower() and 'slp' not in p1_status.lower():
            turn_control_count += 1
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
        
        current_advantage = 1 if p1_hp > p2_hp else -1
        if prev_advantage != 0 and current_advantage != prev_advantage:
            momentum_changes += 1
        prev_advantage = current_advantage
    
    f['turn_control_total_p1'] = turn_control_count / total_turns if total_turns > 0 else 0
    f['turn_control_streak_p1'] = max_streak
    f['momentum_swings'] = momentum_changes
    
    # 5. CRITICAL HIT VULNERABILITY
    high_crit_moves = {'slash', 'razor leaf', 'crabhammer', 'karate chop'}
    high_crit_count = 0
    
    for poke in p1_team:
        if isinstance(poke, dict):
            moveset = poke.get('moveset', [])
            if not moveset:
                moveset = poke.get('moves', [])
            for move in moveset:
                if str(move).lower() in high_crit_moves:
                    high_crit_count += 1
    
    f['high_crit_move_count_p1'] = high_crit_count
    
    p2_speed = 100
    if isinstance(p2_lead, dict):
        p2_speed = p2_lead.get('base_spe', 100)
        if not p2_speed:
            p2_speed = p2_lead.get('speed', 100)
    f['critical_vulnerability_p2'] = 1.0 - (p2_speed / 200.0)
    
    # 6. HP TRAJECTORY ANALYSIS
    hp_advantages = []
    p1_min_hp = 1.0
    endgame_hp_diffs = []
    
    for i, turn in enumerate(timeline):
        p1_hp = 0.5
        p2_hp = 0.5
        
        if 'p1_lead_hp_pct' in turn:
            p1_hp = turn['p1_lead_hp_pct']
        elif 'p1_hp' in turn:
            p1_hp = turn['p1_hp']
            
        if 'p2_lead_hp_pct' in turn:
            p2_hp = turn['p2_lead_hp_pct']
        elif 'p2_hp' in turn:
            p2_hp = turn['p2_hp']
        
        hp_advantages.append(1 if p1_hp > p2_hp else 0)
        p1_min_hp = min(p1_min_hp, p1_hp)
        
        if i >= total_turns - 5:
            endgame_hp_diffs.append(p1_hp - p2_hp)
    
    f['hp_trajectory_positive'] = np.mean(hp_advantages) if hp_advantages else 0.5
    f['hp_min_reached_p1'] = p1_min_hp
    f['endgame_hp_advantage'] = np.mean(endgame_hp_diffs) if endgame_hp_diffs else 0
    
    # 7. EXPLOSION/SACRIFICE PLAYS
    explosion_moves_set = {'explosion', 'self-destruct', 'selfdestruct'}
    explosion_potential_p1 = 0
    explosion_potential_p2 = 0
    
    for poke in p1_team:
        if isinstance(poke, dict):
            moveset = poke.get('moveset', [])
            if not moveset:
                moveset = poke.get('moves', [])
            for move in moveset:
                if str(move).lower() in explosion_moves_set:
                    explosion_potential_p1 = 1
                    break
    
    if isinstance(p2_lead, dict):
        p2_moveset = p2_lead.get('moveset', [])
        if not p2_moveset:
            p2_moveset = p2_lead.get('moves', [])
        for move in p2_moveset:
            if str(move).lower() in explosion_moves_set:
                explosion_potential_p2 = 1
                break
    
    f['explosion_potential_p1'] = explosion_potential_p1
    f['explosion_potential_p2'] = explosion_potential_p2
    
    # 8. TYPE EFFECTIVENESS STREAKS (placeholder)
    f['supereffective_streak_p1'] = 0
    f['supereffective_count_p1'] = 0
    
    # 9. SPEED CONTROL DOMINANCE
    p1_speeds = []
    p2_speeds = []
    
    for poke in p1_team:
        if isinstance(poke, dict):
            speed = poke.get('base_spe', 100)
            if not speed:
                speed = poke.get('speed', 100)
            p1_speeds.append(speed)
    
    if isinstance(p2_lead, dict):
        p2_speed = p2_lead.get('base_spe', 100)
        if not p2_speed:
            p2_speed = p2_lead.get('speed', 100)
        p2_speeds.append(p2_speed)
    
    avg_p1_speed = np.mean(p1_speeds) if p1_speeds else 100
    avg_p2_speed = np.mean(p2_speeds) if p2_speeds else 100
    
    f['speed_control_turns'] = 1.0 if avg_p1_speed > avg_p2_speed else 0.0
    f['speed_advantage_maintained'] = max(0, avg_p1_speed - avg_p2_speed) / 200.0
    
    # 10. WALLBREAKER vs WALL DYNAMICS
    wallbreaker_count = 0
    
    for poke in p1_team:
        if isinstance(poke, dict):
            atk = poke.get('base_atk', 0)
            spa = poke.get('base_spa', 0)
            if not atk:
                atk = poke.get('attack', 0)
            if not spa:
                spa = poke.get('special_attack', 0)
            if max(atk, spa) >= 100:
                wallbreaker_count += 1
    
    f['wallbreaker_score_p1'] = wallbreaker_count / len(p1_team) if p1_team else 0
    
    wall_pokemon = {'chansey', 'snorlax', 'lapras', 'articuno', 'moltres', 'zapdos'}
    p2_name = ''
    if isinstance(p2_lead, dict):
        p2_name = str(p2_lead.get('pokemon', '')).lower()
        if not p2_name:
            p2_name = str(p2_lead.get('name', '')).lower()
    f['wall_score_p2'] = 1.0 if p2_name in wall_pokemon else 0.0
    
    return f

# ===========================================================================
# FUNZIONE 7: extract_moveset_features (da Cella 6)
# ===========================================================================
def extract_moveset_features(battle):
    features = {}
    
    p1_known_moves = {}
    p2_known_moves = {}
    
    timeline = battle.get('battle_timeline', [])
    
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state', {})\
        p1_move = turn.get('p1_move_details') or {}
        p1_name = p1_state.get('name')
        p1_move_name = p1_move.get('name', '').lower()
        
        if p1_name:
            if p1_name not in p1_known_moves: p1_known_moves[p1_name] = set()
            if p1_move_name: p1_known_moves[p1_name].add(p1_move_name)

        p2_state = turn.get('p2_pokemon_state', {})\
        p2_move = turn.get('p2_move_details') or {}
        p2_name = p2_state.get('name')
        p2_move_name = p2_move.get('name', '').lower()
        
        if p2_name:
            if p2_name not in p2_known_moves: p2_known_moves[p2_name] = set()
            if p2_move_name: p2_known_moves[p2_name].add(p2_move_name)

    p1_all_moves = set.union(*p1_known_moves.values()) if p1_known_moves else set()
    features['p1_knows_recovery'] = 1 if p1_all_moves & GOLDEN_RECOVERY else 0
    features['p1_knows_setup'] = 1 if p1_all_moves & GOLDEN_SETUP else 0
    features['p1_knows_status'] = 1 if p1_all_moves & GOLDEN_STATUS else 0
    features['p1_knows_trap'] = 1 if p1_all_moves & GOLDEN_TRAP else 0
    features['p1_knows_power'] = 1 if p1_all_moves & GOLDEN_POWER else 0
    
    p2_all_moves = set.union(*p2_known_moves.values()) if p2_known_moves else set()
    features['p2_knows_recovery'] = 1 if p2_all_moves & GOLDEN_RECOVERY else 0
    features['p2_knows_setup'] = 1 if p2_all_moves & GOLDEN_SETUP else 0
    features['p2_knows_status'] = 1 if p2_all_moves & GOLDEN_STATUS else 0
    features['p2_knows_trap'] = 1 if p2_all_moves & GOLDEN_TRAP else 0
    features['p2_knows_power'] = 1 if p2_all_moves & GOLDEN_POWER else 0

    p1_boltbeam = 0
    for moves in p1_known_moves.values():
        if GOLDEN_COMBO_TB in moves and GOLDEN_COMBO_IB in moves:
            p1_boltbeam = 1; break
            
    p2_boltbeam = 0
    for moves in p2_known_moves.values():
        if GOLDEN_COMBO_TB in moves and GOLDEN_COMBO_IB in moves:
            p2_boltbeam = 1; break
            
    features['p1_knows_boltbeam'] = p1_boltbeam
    features['p2_knows_boltbeam'] = p2_boltbeam
    
    features['recovery_adv'] = features['p1_knows_recovery'] - features['p2_knows_recovery']
    features['setup_adv'] = features['p1_knows_setup'] - features['p2_knows_setup']
    features['status_adv'] = features['p1_knows_status'] - features['p2_knows_status']
    features['boltbeam_adv'] = features['p1_knows_boltbeam'] - features['p2_knows_boltbeam']

    return features