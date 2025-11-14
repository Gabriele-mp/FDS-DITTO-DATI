import pandas as pd
import numpy as np
from collections import Counter
import math
from tqdm.notebook import tqdm
import warnings

# Costanti Generali
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


# Funzioni Helper
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

# Funzione Builder Generica
def build_feature_dataframe(df_raw, feature_extractor_func, is_test_set=False):
    """
    Applica una funzione di estrazione feature specifica a un DataFrame.
    """
    print(f"Applicazione di '{feature_extractor_func.__name__}' a {len(df_raw)} righe...")
    battles_list = df_raw.to_dict('records')
    
    feature_rows = [feature_extractor_func(battle) for battle in tqdm(battles_list)]
    
    X_final = pd.DataFrame(feature_rows)
    X_final = X_final.fillna(0)
    
    X_final = X_final.reindex(sorted(X_final.columns), axis=1)
    
    print(f"Create {len(X_final.columns)} feature totali.")
    
    if is_test_set:
        return X_final
    else:
        y_final = df_raw['player_won'].astype(int)
        return X_final, y_final


# FUNZIONE 1: extract_features_v8

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
            for t in poke_types: weakness_list.extend(GEN1_WEAKNESS.get(t, []))
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
            p1_state = turn_data.get("p1_pokemon_state", {})
            p2_state = turn_data.get("p2_pokemon_state", {})
            p1_move = turn_data.get("p1_move_details", {})
            p2_move = turn_data.get("p2_move_details", {})
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
            b1 = p1_state.get("boosts", {})
            b2 = p2_state.get("boosts", {})
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


# FUNZIONE 2: extract_features_v20

def extract_features_v20(battle):
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
        p2_lead = battle.get('p2_lead_details', {})
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
            p1_state = turn_data.get("p1_pokemon_state", {})
            p2_state = turn_data.get("p2_pokemon_state", {})
            p1_move = turn_data.get("p1_move_details", {})
            p2_move = turn_data.get("p2_move_details", {})
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
            b1 = p1_state.get("boosts", {})
            b2 = p2_state.get("boosts", {})
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


# FUNZIONE 3: extract_features_v19

def extract_features_v19(battle):
    """ Estrae 41 feature (v19 modulare) """
    CALCOLA_CORE = True
    CALCOLA_RUOLO = True
    CALCOLA_DELTA = True
    CALCOLA_SWEEP = True
    
    f = {'battle_id': battle.get('battle_id', -1)}
    try:
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})
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
            p1_state = turn_data.get("p1_pokemon_state", {})
            p2_state = turn_data.get("p2_pokemon_state", {})
            p1_move = turn_data.get("p1_move_details", {})
            p2_move = turn_data.get("p2_move_details", {})
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
                b1 = p1_state.get("boosts", {})
                b2 = p2_state.get("boosts", {})
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

print("Funzioni di feature engineering (v8, v19, v20) caricate per il Modello 1.")