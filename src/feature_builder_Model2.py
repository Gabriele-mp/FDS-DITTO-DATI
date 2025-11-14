import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import math
import os

# --- Costanti Generali ---
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


# --- Funzioni Helper ---
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
# --- INIZIO BLOCCO AGGIUNTO (da pokemon-battle-v-2) ---

# Costanti per le feature "v2"
TYPE_CHART_GEN1 = {
    "NORMAL":   {"ROCK":0.5, "GHOST":0.0},
    "FIRE":     {"FIRE":0.5, "WATER":0.5, "GRASS":2.0, "ICE":2.0, "BUG":2.0, "ROCK":0.5, "DRAGON":0.5},
    "WATER":    {"FIRE":2.0, "WATER":0.5, "GRASS":0.5, "GROUND":2.0, "ROCK":2.0, "DRAGON":0.5},
    "ELECTRIC": {"WATER":2.0, "ELECTRIC":0.5, "GRASS":0.5, "GROUND":0.0, "FLYING":2.0, "DRAGON":0.5},
    "GRASS":    {"FIRE":0.5, "WATER":2.0, "GRASS":0.5, "POISON":0.5, "GROUND":2.0, "FLYING":0.5, "BUG":0.5, "ROCK":2.0, "DRAGON":0.5},
    "ICE":      {"WATER":0.5, "GRASS":2.0, "ICE":0.5, "GROUND":2.0, "FLYING":2.0, "DRAGON":2.0},
    "FIGHTING": {"NORMAL":2.0, "ICE":2.0, "POISON":0.5, "FLYING":0.5, "PSYCHIC":0.5, "BUG":0.5, "ROCK":2.0, "GHOST":0.0},
    "POISON":   {"GRASS":2.0, "POISON":0.5, "GROUND":0.5, "ROCK":0.5, "GHOST":0.5},
    "GROUND":   {"FIRE":2.0, "ELECTRIC":2.0, "GRASS":0.5, "POISON":2.0, "FLYING":0.0, "BUG":0.5, "ROCK":2.0},
    "FLYING":   {"ELECTRIC":0.5, "GRASS":2.0, "FIGHTING":2.0, "BUG":2.0, "ROCK":0.5},
    "PSYCHIC":  {"FIGHTING":2.0, "POISON":2.0, "PSYCHIC":0.5},
    "BUG":      {"FIRE":0.5, "FIGHTING":0.5, "POISON":0.5, "FLYING":0.5, "PSYCHIC":2.0, "GHOST":0.5},
    "ROCK":     {"FIRE":2.0, "ICE":2.0, "FIGHTING":0.5, "GROUND":0.5, "FLYING":2.0, "BUG":2.0},
    "GHOST":    {"NORMAL":0.0, "PSYCHIC":0.0},
    "DRAGON":   {"DRAGON":2.0},
}

BOOST_MULTIPLIERS = {
    -6: 0.25, -5: 0.28, -4: 0.33, -3: 0.40, -2: 0.50, -1: 0.66,
    0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0
}
SETUP_MOVES = {'amnesya', 'swordsdance', 'agility'}
RECOVERY_MOVES = {'recover', 'softboiled', 'rest', 'megadrain'}
SACRIFICE_MOVES = {'explosion', 'selfdestruct'}
KEY_THREATS = {'alakazam', 'starmie', 'tauros', 'zapdos', 'jolteon'}
KEY_WALLS = {'chansey', 'snorlax', 'slowbro', 'exeggutor'}
TRAP_KO_MOVES = {'counter', 'bide'}
SNAPSHOT_TURNS_V2 = [15, 20, 23, 25, 27, 30] # Definito globalmente

print("Costanti aggiuntive per 'features v2' caricate.")
# --- FINE BLOCCO AGGIUNTO ---
print("Costanti e helper definiti.")

def build_feature_dataframe(df_raw, feature_extractor_func, is_test_set=False):
    """
    Applica una funzione di estrazione feature specifica a un DataFrame.
    """
    print(f"Applicazione di '{feature_extractor_func.__name__}' a {len(df_raw)} righe...")
    battles_list = df_raw.to_dict('records')
    
    # Applica la funzione passata come argomento
    feature_rows = [feature_extractor_func(battle) for battle in tqdm(battles_list)]
    
    X_final = pd.DataFrame(feature_rows)
    X_final = X_final.fillna(0) # Fillna finale
    
    # Assicura che le colonne siano nell'ordine corretto (utile per XGB/CAT)
    X_final = X_final.reindex(sorted(X_final.columns), axis=1)
    
    print(f"Create {len(X_final.columns)} feature totali.")
    
    if is_test_set:
        return X_final
    else:
        y_final = df_raw['player_won'].astype(int)
        return X_final, y_final

# ---
# FUNZIONE 1: extract_features_v8 (da prova-xg-vs-logistic-v2.ipynb)
# ---
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

# ---
# FUNZIONE 2: extract_features_v20 (da xg-vs-logit-con-switch-strategy.ipynb)
# ---
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

# ---
# FUNZIONE 3: extract_features_v19 (da random-forrest.ipynb / knn-vs-catboost.ipynb)
# ---
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

print("Tutte e 3 le funzioni di feature engineering sono state definite.")
# --- INIZIO BLOCCO AGGIUNTO (da pokemon-battle-v-2) ---
# Funzioni di supporto e pipeline per le feature "v2"

def build_pokemon_details_map_df(data_df: pd.DataFrame):
    """
    Versione modificata: accetta un DataFrame invece di una lista.
    Scansiona l'intero dataset per creare una mappa {nome_pokemon: {dettagli}}.
    """
    POKEDEX_GEN1 = {}
    # Modifica: itera sulla versione 'records' del DataFrame
    for battle in tqdm(data_df.to_dict('records'), desc="Building Pokédex (v2)"):
        # Scansiona la squadra di P1 (fonte completa di informazioni)
        for p_details in battle.get('p1_team_details', []):
            name = p_details.get('name')
            if name and name not in POKEDEX_GEN1:
                POKEDEX_GEN1[name] = p_details
        
        # Scansiona il lead di P2
        p2_lead = battle.get('p2_lead_details')
        if p2_lead:
            name = p2_lead.get('name')
            if name and name not in POKEDEX_GEN1:
                POKEDEX_GEN1[name] = p2_lead
    return POKEDEX_GEN1

def calculate_paralysis_advantage(timeline):
    p1_par_points = 0
    p2_par_points = 0
    p1_status_prev = {}
    p2_status_prev = {}
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        if p1_state and p2_state:
            p2_name = p2_state['name']
            p2_status_now = p2_state.get('status')
            if p2_status_now == 'par' and p2_status_prev.get(p2_name) != 'par':
                p1_par_points += 1
            p2_status_prev[p2_name] = p2_status_now
        if p1_state and p2_state:
            p1_name = p1_state['name']
            p1_status_now = p1_state.get('status')
            if p1_status_now == 'par' and p1_status_prev.get(p1_name) != 'par':
                p2_par_points += 1
            p1_status_prev[p1_name] = p1_status_now
    return {'paralysis_advantage': p1_par_points - p2_par_points}

def calculate_trap_kos(timeline):
    p1_trap_kos = 0
    p2_trap_kos = 0
    p1_fainted_names = set()
    p2_fainted_names = set()
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')
        if p1_move and p2_state and p1_move.get('name') in TRAP_KO_MOVES:
            if p2_state.get('status') == 'fnt' and p2_state['name'] not in p2_fainted_names:
                p1_trap_kos += 1
                p2_fainted_names.add(p2_state['name'])
        if p2_move and p1_state and p2_move.get('name') in TRAP_KO_MOVES:
            if p1_state.get('status') == 'fnt' and p1_state['name'] not in p1_fainted_names:
                p2_trap_kos += 1
                p1_fainted_names.add(p1_state['name'])
        if p1_state and p1_state.get('status') == 'fnt': p1_fainted_names.add(p1_state['name'])
        if p2_state and p2_state.get('status') == 'fnt': p2_fainted_names.add(p2_state['name'])
    return {'trap_ko_advantage': p1_trap_kos - p2_trap_kos}

def calculate_crippled_threat_advantage(timeline):
    p1_score = 0
    p2_score = 0
    p1_last_status = {}
    p2_last_status = {}
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        if p2_state:
            name = p2_state['name']
            status_now = p2_state.get('status')
            if status_now in ['slp', 'frz'] and p2_last_status.get(name) not in ['slp', 'frz']:
                if name in KEY_THREATS: p1_score += 2
                elif name in KEY_WALLS: p1_score += 1
                else: p1_score += 0.5
            p2_last_status[name] = status_now
        if p1_state:
            name = p1_state['name']
            status_now = p1_state.get('status')
            if status_now in ['slp', 'frz'] and p1_last_status.get(name) not in ['slp', 'frz']:
                if name in KEY_THREATS: p2_score += 2
                elif name in KEY_WALLS: p2_score += 1
                else: p2_score += 0.5
            p1_last_status[name] = status_now
    return {'crippled_threat_advantage': p1_score - p2_score}

def calculate_sacrifice_outcomes(timeline):
    p1_sac_success = 0
    p1_sac_fail = 0
    p2_sac_success = 0
    p2_sac_fail = 0
    for i in range(len(timeline)):
        turn = timeline[i]
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')
        if p1_move and p1_move.get('name') in SACRIFICE_MOVES:
            target_state_after = turn.get('p2_pokemon_state') 
            if p1_state and p1_state.get('status') == 'fnt' and target_state_after:
                if target_state_after.get('status') == 'fnt':
                    p1_sac_success += 1
                elif target_state_after.get('hp_pct', 1.0) > 0:
                    p1_sac_fail += 1
        if p2_move and p2_move.get('name') in SACRIFICE_MOVES:
            target_state_after = turn.get('p1_pokemon_state')
            if p2_state and p2_state.get('status') == 'fnt' and target_state_after:
                if target_state_after.get('status') == 'fnt':
                    p2_sac_success += 1
                elif target_state_after.get('hp_pct', 1.0) > 0:
                    p2_sac_fail += 1
    net_advantage = (p1_sac_success - p1_sac_fail) - (p2_sac_success - p2_sac_fail)
    return {'sacrifice_outcome_advantage': net_advantage}

def calculate_setup_sweep_value(timeline):
    p1_setup_pokemon = set()
    p2_setup_pokemon = set()
    p1_setup_damage = 0.0
    p2_setup_damage = 0.0
    p2_hp_prev = {}
    p1_hp_prev = {}
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')
        if p1_move and p1_state and p1_move.get('name') in SETUP_MOVES:
            p1_setup_pokemon.add(p1_state['name'])
        if p2_move and p2_state and p2_move.get('name') in SETUP_MOVES:
            p2_setup_pokemon.add(p2_state['name'])
        if p1_state and p1_state['name'] in p1_setup_pokemon:
            if p1_move and p1_move.get('category') in ['PHYSICAL', 'SPECIAL'] and p2_state:
                p2_last_hp = p2_hp_prev.get(p2_state['name'], 1.0)
                damage_dealt = p2_last_hp - p2_state.get('hp_pct', 0.0)
                if damage_dealt > 0:
                    p1_setup_damage += damage_dealt
        if p2_state and p2_state['name'] in p2_setup_pokemon:
            if p2_move and p2_move.get('category') in ['PHYSICAL', 'SPECIAL'] and p1_state:
                p1_last_hp = p1_hp_prev.get(p1_state['name'], 1.0)
                damage_dealt = p1_last_hp - p1_state.get('hp_pct', 0.0)
                if damage_dealt > 0:
                    p2_setup_damage += damage_dealt
        if p1_state: p1_hp_prev[p1_state['name']] = p1_state.get('hp_pct', 1.0)
        if p2_state: p2_hp_prev[p2_state['name']] = p2_state.get('hp_pct', 1.0)
    return {'setup_sweep_advantage': p1_setup_damage - p2_setup_damage}

def calculate_free_damage_advantage(timeline):
    p1_free_damage = 0.0
    p2_free_damage = 0.0
    p1_hp_prev = {}
    p2_hp_prev = {}
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')
        if p1_move and p1_move.get('category') in ['PHYSICAL', 'SPECIAL'] and p2_state:
            p2_last_hp = p2_hp_prev.get(p2_state['name'], 1.0)
            if p2_state.get('status') in ['slp', 'frz']:
                damage_dealt = p2_last_hp - p2_state.get('hp_pct', 0.0)
                if damage_dealt > 0:
                    p1_free_damage += damage_dealt
        if p2_move and p2_move.get('category') in ['PHYSICAL', 'SPECIAL'] and p1_state:
            p1_last_hp = p1_hp_prev.get(p1_state['name'], 1.0)
            if p1_state.get('status') in ['slp', 'frz']:
                damage_dealt = p1_last_hp - p1_state.get('hp_pct', 0.0)
                if damage_dealt > 0:
                    p2_free_damage += damage_dealt
        if p1_state: p1_hp_prev[p1_state['name']] = p1_state.get('hp_pct', 1.0)
        if p2_state: p2_hp_prev[p2_state['name']] = p2_state.get('hp_pct', 1.0)
    return {'free_damage_advantage': p1_free_damage - p2_free_damage}

def calculate_confusion_turns_advantage(timeline):
    p1_confusion_turns = 0
    p2_confusion_turns = 0
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        if p1_state and 'confusion' in p1_state.get('effects', []):
            p1_confusion_turns += 1
        p2_state = turn.get('p2_pokemon_state')
        if p2_state and 'confusion' in p2_state.get('effects', []):
            p2_confusion_turns += 1
    return {'confusion_turns_advantage': p2_confusion_turns - p1_confusion_turns}

def calculate_strategic_trade_advantage(timeline, pokedex_map):
    p1_net_gain = 0
    p2_net_gain = 0
    for i in range(1, len(timeline)):
        turn = timeline[i]
        prev_turn = timeline[i-1]
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')
        if p1_move and p1_move.get('name') in SACRIFICE_MOVES:
            sacrificer_state = prev_turn.get('p1_pokemon_state')
            target_state_before = prev_turn.get('p2_pokemon_state')
            target_state_after = turn.get('p2_pokemon_state')
            if (sacrificer_state and target_state_before and target_state_after and
                    target_state_before.get('name') != target_state_after.get('name')):
                sacrificer_details = {**pokedex_map.get(sacrificer_state['name'], {}), **sacrificer_state}
                target_details = {**pokedex_map.get(target_state_before['name'], {}), **target_state_before}
                sacrificer_threat = calculate_team_threat_score([sacrificer_details], set())
                target_threat = calculate_team_threat_score([target_details], set())
                p1_net_gain += (target_threat - sacrificer_threat)
        if p2_move and p2_move.get('name') in SACRIFICE_MOVES:
            sacrificer_state = prev_turn.get('p2_pokemon_state')
            target_state_before = prev_turn.get('p1_pokemon_state')
            target_state_after = turn.get('p1_pokemon_state')
            if (sacrificer_state and target_state_before and target_state_after and
                    target_state_before.get('name') != target_state_after.get('name')):
                sacrificer_details = {**pokedex_map.get(sacrificer_state['name'], {}), **sacrificer_state}
                target_details = {**pokedex_map.get(target_state_before['name'], {}), **target_state_before}
                sacrificer_threat = calculate_team_threat_score([sacrificer_details], set())
                target_threat = calculate_team_threat_score([target_details], set())
                p2_net_gain += (target_threat - sacrificer_threat)
    return {'strategic_trade_advantage': p1_net_gain - p2_net_gain}

def calculate_active_defense(timeline):
    p1_defense_turns = 0
    p2_defense_turns = 0
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        if p1_state and p1_state.get('effects'):
            effects = p1_state['effects']
            if 'reflect' in effects:
                p1_defense_turns += 1
        if p2_state and p2_state.get('effects'):
            effects = p2_state['effects']
            if 'reflect' in effects:
                p2_defense_turns += 1
    return {'p1_active_defense_turns': p1_defense_turns, 'p2_active_defense_turns': p2_defense_turns}

def calculate_wasted_turns(timeline):
    p1_wasted = 0
    p2_wasted = 0
    p1_hp_prev = {}
    p2_hp_prev = {}
    p1_mon_prev = None
    p2_mon_prev = None
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')
        if p1_state and p1_state['name'] not in p1_hp_prev:
            p1_hp_prev[p1_state['name']] = 1.0
        if p2_state and p2_state['name'] not in p2_hp_prev:
            p2_hp_prev[p2_state['name']] = 1.0
        if p1_state:
            if p1_move is None and p1_mon_prev == p1_state['name']:
                p1_wasted += 1
            elif p1_move and p1_move.get('category') in ['PHYSICAL', 'SPECIAL'] and p2_state:
                if p1_hp_prev.get(p2_state['name']) == p2_state['hp_pct']:
                    p1_wasted += 1
        if p2_state:
            if p2_move is None and p2_mon_prev == p2_state['name']:
                p2_wasted += 1
            elif p2_move and p2_move.get('category') in ['PHYSICAL', 'SPECIAL'] and p1_state:
                if p2_hp_prev.get(p1_state['name']) == p1_state['hp_pct']:
                    p2_wasted += 1
        if p1_state:
            p1_hp_prev[p1_state['name']] = p1_state['hp_pct']
            p1_mon_prev = p1_state['name']
        if p2_state:
            p2_hp_prev[p2_state['name']] = p2_state['hp_pct']
            p2_mon_prev = p2_state['name']
    return {'p1_wasted_turn_count': p1_wasted, 'p2_wasted_turn_count': p2_wasted}

def calculate_hax_advantage(timeline):
    p1_hax_points = 0
    p2_hax_points = 0
    p1_status_prev = None
    p2_status_prev = None
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        if p1_state and p2_state:
            p2_status_now = p2_state.get('status')
            if p2_status_now in ['slp', 'frz'] and p2_status_prev not in ['slp', 'frz']:
                p1_hax_points += 1
        if p1_state and p2_state:
            p1_status_now = p1_state.get('status')
            if p1_status_now in ['slp', 'frz'] and p1_status_prev not in ['slp', 'frz']:
                p2_hax_points += 1
        if p1_state:
            p1_status_prev = p1_state.get('status')
        if p2_state:
            p2_status_prev = p2_state.get('status')
    return {'hax_advantage': p1_hax_points - p2_hax_points}

def calculate_clutch_defense_score(timeline):
    p1_clutch_points, p2_clutch_points = 0, 0
    p1_prev_hp, p2_prev_hp = {}, {}
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        p2_state = turn.get('p2_pokemon_state')
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')
        if p1_move and p1_state and p1_move.get('name') in RECOVERY_MOVES:
            if p1_prev_hp.get(p1_state['name'], 1.0) < 0.5:
                p1_clutch_points += 1
        if p2_move and p2_state and p2_move.get('name') in RECOVERY_MOVES:
            if p2_prev_hp.get(p2_state['name'], 1.0) < 0.5:
                p2_clutch_points += 1
        if p1_state:
            if 0 < p1_state['hp_pct'] < 0.1 and p2_move:
                p1_clutch_points += 1
            p1_prev_hp[p1_state['name']] = p1_state['hp_pct']
        if p2_state:
            if 0 < p2_state['hp_pct'] < 0.1 and p1_move:
                p2_clutch_points += 1
            p2_prev_hp[p2_state['name']] = p2_state['hp_pct']
    return {'p1_clutch_score': p1_clutch_points, 'p2_clutch_score': p2_clutch_points}

def calculate_dynamic_ratios(timeline):
    p1_moves, p2_moves, p1_switches, p2_switches = 0, 0, 0, 0
    last_p1_mon, last_p2_mon = None, None
    if timeline:
        if timeline[0].get('p1_pokemon_state'): last_p1_mon = timeline[0]['p1_pokemon_state']['name']
        if timeline[0].get('p2_pokemon_state'): last_p2_mon = timeline[0]['p2_pokemon_state']['name']
    for turn in timeline:
        if turn.get('p1_move_details'): p1_moves += 1
        if turn.get('p2_move_details'): p2_moves += 1
        p1_state = turn.get('p1_pokemon_state')
        if p1_state:
            current_p1_mon = p1_state['name']
            if last_p1_mon is not None and current_p1_mon != last_p1_mon: p1_switches += 1
            last_p1_mon = current_p1_mon
        p2_state = turn.get('p2_pokemon_state')
        if p2_state:
            current_p2_mon = p2_state['name']
            if last_p2_mon is not None and current_p2_mon != last_p2_mon: p2_switches += 1
            last_p2_mon = current_p2_mon
    move_ratio = (p1_moves + 1) / (p2_moves + 1)
    switch_ratio = (p1_switches + 1) / (p2_switches + 1)
    return {'move_ratio': move_ratio, 'switch_ratio': switch_ratio}

def calculate_efficiency_metrics(timeline, p1_team_details):
    p1_hp_map = {p['name']: 1.0 for p in p1_team_details}
    p2_hp_map = {} 
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state')
        if p1_state:
            prev_hp = p1_hp_map.get(p1_state['name'], 1.0)
            current_hp = p1_state.get('hp_pct', prev_hp)
            p1_hp_map[p1_state['name']] = current_hp
        p2_state = turn.get('p2_pokemon_state')
        if p2_state:
            prev_hp = p2_hp_map.get(p2_state['name'], 1.0)
            current_hp = p2_state.get('hp_pct', prev_hp)
            p2_hp_map[p2_state['name']] = current_hp
    p1_total_damage_dealt = sum(1.0 - hp for hp in p2_hp_map.values())
    p2_total_damage_dealt = sum(1.0 - hp for hp in p1_hp_map.values())
    damage_efficiency_ratio = (p1_total_damage_dealt + 0.1) / (p2_total_damage_dealt + 0.1)
    p1_kos_made = len([hp for hp in p2_hp_map.values() if hp == 0])
    p1_hp_cost_per_ko = p2_total_damage_dealt / (p1_kos_made + 1)
    p2_kos_made = len([hp for hp in p1_hp_map.values() if hp == 0])
    p2_hp_cost_per_ko = p1_total_damage_dealt / (p2_kos_made + 1)
    return {
        'damage_efficiency_ratio': damage_efficiency_ratio,
        'p1_hp_cost_per_ko': p1_hp_cost_per_ko,
        'p2_hp_cost_per_ko': p2_hp_cost_per_ko
    }

def calculate_anchor_strength(timeline, team_details, pokedex_map, player_prefix='p1'):
    state_key = f'{player_prefix}_pokemon_state'
    if player_prefix == 'p2':
        seen_names = set()
        for turn in timeline:
            if turn.get(state_key):
                seen_names.add(turn[state_key]['name'])
        team_details = [pokedex_map[name] for name in seen_names if name in pokedex_map]
    if not team_details:
        return {f'{player_prefix}_anchor_strength_ratio': 1.0}
    base_threat_map = {p['name']: (p.get('base_atk', 50) + p.get('base_spa', 50) + p.get('base_spe', 50)) for p in team_details}
    switch_order = []
    for turn in timeline:
        state = turn.get(state_key)
        if state:
            name = state['name']
            if not switch_order or switch_order[-1] != name:
                switch_order.append(name)
    if len(switch_order) < 2:
        return {f'{player_prefix}_anchor_strength_ratio': 1.0}
    mid_point = len(switch_order) // 2
    early_game_mons = switch_order[:mid_point]
    late_game_mons = switch_order[mid_point:]
    if not early_game_mons or not late_game_mons:
        return {f'{player_prefix}_anchor_strength_ratio': 1.0}
    avg_early_threat = sum(base_threat_map.get(m, 0) for m in early_game_mons) / len(early_game_mons)
    avg_late_threat = sum(base_threat_map.get(m, 0) for m in late_game_mons) / len(late_game_mons)
    if avg_early_threat == 0:
        return {f'{player_prefix}_anchor_strength_ratio': 1.0}
    return {f'{player_prefix}_anchor_strength_ratio': avg_late_threat / avg_early_threat}

def calculate_setup_threat(timeline, pokedex_map):
    max_threat = 0
    for turn in timeline:
        move = turn.get('p1_move_details')
        p1_state = turn.get('p1_pokemon_state')
        if move and p1_state and move.get('name') in SETUP_MOVES:
            mon_name = p1_state['name']
            static_details = pokedex_map.get(mon_name, {})
            full_details = {**static_details, **p1_state}
            current_threat = calculate_team_threat_score([full_details], {mon_name} if p1_state.get('status') == 'par' else set())
            if current_threat > max_threat:
                max_threat = current_threat
    return {'p1_max_setup_threat': max_threat}

# Rinominata per evitare conflitti con la funzione in Cella 3
def get_type_effectiveness_v2_feat(attack_type, defender_types):
    lookup_attack_type = attack_type.upper()
    effectiveness = 1.0
    if lookup_attack_type in TYPE_CHART_GEN1:
        for def_type in defender_types:
            lookup_def_type = def_type.upper()
            effectiveness *= TYPE_CHART_GEN1[lookup_attack_type].get(lookup_def_type, 1)
    return effectiveness

def calculate_effective_speed(pokemon_details, is_paralyzed):
    base_speed = pokemon_details.get('base_spe', 50)
    boost_level = pokemon_details.get('boosts', {}).get('spe', 0)
    multiplier = BOOST_MULTIPLIERS[boost_level]
    effective_speed = base_speed * multiplier
    if is_paralyzed:
        effective_speed *= 0.25
    return effective_speed

def calculate_pokemon_advantage(timeline):
    p1_fainted_pokemon = set()
    p2_fainted_pokemon = set()
    for turn_data in timeline:
        p1_state = turn_data.get('p1_pokemon_state')
        if p1_state and p1_state['hp_pct'] == 0:
            p1_fainted_pokemon.add(p1_state['name'])
        p2_state = turn_data.get('p2_pokemon_state')
        if p2_state and p2_state['hp_pct'] == 0:
            p2_fainted_pokemon.add(p2_state['name'])
    return {'final_pokemon_advantage': len(p2_fainted_pokemon) - len(p1_fainted_pokemon)}

def get_status_advantage_snapshots(timeline, snapshot_turns):
    p1_paralyzed_pokemon = set()
    p2_paralyzed_pokemon = set()
    features = {f'net_{status}_adv_t{turn}': 0 for turn in snapshot_turns for status in ['par', 'slp', 'frz']}
    for turn_data in timeline:
        current_turn = turn_data['turn']
        p1_state = turn_data.get('p1_pokemon_state')
        if p1_state:
            name = p1_state['name']
            if p1_state.get('status') == 'par': p1_paralyzed_pokemon.add(name)
            elif name in p1_paralyzed_pokemon and p1_state.get('status') is None: p1_paralyzed_pokemon.remove(name)
        p2_state = turn_data.get('p2_pokemon_state')
        if p2_state:
            name = p2_state['name']
            if p2_state.get('status') == 'par': p2_paralyzed_pokemon.add(name)
            elif name in p2_paralyzed_pokemon and p2_state.get('status') is None: p2_paralyzed_pokemon.remove(name)
        if current_turn in snapshot_turns:
            features[f'net_par_adv_t{current_turn}'] = len(p2_paralyzed_pokemon) - len(p1_paralyzed_pokemon)
            net_slp_adv = (1 if p2_state and p2_state.get('status') == 'slp' else 0) - (1 if p1_state and p1_state.get('status') == 'slp' else 0)
            net_frz_adv = (1 if p2_state and p2_state.get('status') == 'frz' else 0) - (1 if p1_state and p1_state.get('status') == 'frz' else 0)
            features[f'net_slp_adv_t{current_turn}'] = net_slp_adv
            features[f'net_frz_adv_t{current_turn}'] = net_frz_adv
    return features

def calculate_comeback_potential(timeline, p1_team_details, all_p2_details):
    p1_final_states, p2_final_states, p1_paralyzed, p2_paralyzed = {}, {}, set(), set()
    for turn in timeline:
        if turn.get('p1_pokemon_state'):
            s = turn['p1_pokemon_state']; p1_final_states[s['name']] = s
            if s.get('status') == 'par': p1_paralyzed.add(s['name'])
            elif s['name'] in p1_paralyzed and s.get('status') is None: p1_paralyzed.remove(s['name'])
        if turn.get('p2_pokemon_state'):
            s = turn['p2_pokemon_state']; p2_final_states[s['name']] = s
            if s.get('status') == 'par': p2_paralyzed.add(s['name'])
            elif s['name'] in p2_paralyzed and s.get('status') is None: p2_paralyzed.remove(s['name'])
    p1_full_details = {p['name']: {**p, **p1_final_states.get(p['name'], {})} for p in p1_team_details}
    p2_full_details = {p['name']: {**p, **p2_final_states.get(p['name'], {})} for p in all_p2_details}
    p1_remaining = [p for p in p1_full_details.values() if p.get('hp_pct', 100) > 0]
    p2_remaining = [p for p in p2_full_details.values() if p.get('hp_pct', 100) > 0]
    if not p1_remaining or not p2_remaining: return {'p1_comeback_potential': 0, 'p2_comeback_potential': 0}
    
    # Nested function, uses get_type_effectiveness_v2_feat
    def calculate_team_threat_score_nested(team1, team2, team1_paralyzed, team2_paralyzed):
        total_threat, SPEED_ADVANTAGE_MULTIPLIER = 0, 1.25
        for mon1 in team1:
            mon1_threat, mon1_speed = 0, calculate_effective_speed(mon1, mon1['name'] in team1_paralyzed)
            b1 = mon1.get('boosts', {}); atk_m1, spa_m1 = BOOST_MULTIPLIERS[b1.get('atk', 0)], BOOST_MULTIPLIERS[b1.get('spa', 0)]
            for mon2 in team2:
                mon2_speed = calculate_effective_speed(mon2, mon2['name'] in team2_paralyzed)
                speed_mult = SPEED_ADVANTAGE_MULTIPLIER if mon1_speed > mon2_speed else 1.0
                b2 = mon2.get('boosts', {}); def_m2, spd_m2 = BOOST_MULTIPLIERS[b2.get('def', 0)], BOOST_MULTIPLIERS[b2.get('spd', 0)]
                # MODIFICA: Chiamata alla funzione rinominata
                type_adv = sum(get_type_effectiveness_v2_feat(t, mon2['types']) for t in mon1['types']) / len(mon1['types']) if mon1['types'] else 1.0
                boost_adv = ((atk_m1 / def_m2) + (spa_m1 / spd_m2)) / 2
                mon1_threat += type_adv * boost_adv * mon1.get('hp_pct', 1.0) * speed_mult
            total_threat += mon1_threat / len(team2)
        return total_threat
        
    p1_threat = calculate_team_threat_score_nested(p1_remaining, p2_remaining, p1_paralyzed, p2_paralyzed)
    p2_threat = calculate_team_threat_score_nested(p2_remaining, p1_remaining, p2_paralyzed, p1_paralyzed)
    pokemon_adv = len(p1_remaining) - len(p2_remaining)
    p1_potential = p1_threat if pokemon_adv < 0 else 0
    p2_potential = p2_threat if pokemon_adv > 0 else 0
    return {'p1_comeback_potential': p1_potential, 'p2_comeback_potential': p2_potential}

def calculate_potential_coverage(p1_remaining, p2_remaining):
    if not p1_remaining or not p2_remaining:
        return {'p1_potential_coverage': 0}
    potential_score = 0
    for p1_mon in p1_remaining:
        p1_types = p1_mon.get('types', [])
        for p2_mon in p2_remaining:
            p2_types = p2_mon.get('types', [])
            has_advantage = False
            for p1_type in p1_types:
                # MODIFICA: Chiamata alla funzione rinominata
                effectiveness = get_type_effectiveness_v2_feat(p1_type, p2_types)
                if effectiveness >= 2:
                    has_advantage = True
                    break
            if has_advantage:
                potential_score += 1
    return {'p1_potential_coverage': potential_score}

def calculate_team_threat_score(team_remaining, team_paralyzed):
    total_threat = 0
    if not team_remaining:
        return 0
    for mon in team_remaining:
        offense_stat = (mon.get('base_atk', 50) + mon.get('base_spa', 50)) / 2
        is_paralyzed = mon['name'] in team_paralyzed
        effective_speed = calculate_effective_speed(mon, is_paralyzed)
        boosts = mon.get('boosts', {})
        atk_mult = BOOST_MULTIPLIERS.get(boosts.get('atk', 0), 1)
        spa_mult = BOOST_MULTIPLIERS.get(boosts.get('spa', 0), 1)
        avg_offense_mult = (atk_mult + spa_mult) / 2
        hp_remaining = mon.get('hp_pct', 1.0)
        threat_score = offense_stat * effective_speed * avg_offense_mult * hp_remaining
        total_threat += threat_score
    return total_threat / 1000

def calculate_lead_matchup(timeline, p1_team_details, p2_lead_details):
    if not timeline or not p2_lead_details:
        return {'lead_matchup_advantage': 0}
    p1_lead_name = timeline[0].get('p1_pokemon_state', {}).get('name')
    p1_lead = next((p for p in p1_team_details if p['name'] == p1_lead_name), None)
    if not p1_lead:
        return {'lead_matchup_advantage': 0}
    p2_lead = p2_lead_details
    p1_speed = p1_lead.get('base_spe', 50)
    p2_speed = p2_lead.get('base_spe', 50)
    speed_adv = 1 if p1_speed > p2_speed else -1 if p2_speed > p1_speed else 0
    p1_types = p1_lead.get('types', [])
    p2_types = p2_lead.get('types', [])
    # MODIFICA: Chiamata alla funzione rinominata
    p1_eff_on_p2 = sum(get_type_effectiveness_v2_feat(t, p2_types) for t in p1_types)
    p2_eff_on_p1 = sum(get_type_effectiveness_v2_feat(t, p1_types) for t in p2_types)
    type_adv = p1_eff_on_p2 - p2_eff_on_p1
    return {'lead_matchup_advantage': speed_adv + type_adv}

# --- Funzione Estrattore Principale (da pokemon-battle-v-2) ---
def process_battle_v2(battle_data, snapshot_turns, pokedex_map):
    timeline = battle_data.get('battle_timeline')
    if not timeline: return None
    features = {'battle_id': battle_data['battle_id']}
    if 'player_won' in battle_data: features['player_won'] = battle_data['player_won']
    
    def get_state_at_turn(target_turn):
        effective_timeline = [t for t in timeline if t['turn'] <= target_turn]
        if not effective_timeline: 
            effective_timeline = [t for t in timeline if t['turn'] == 1]
            if not effective_timeline: return [], [], set(), set()
        p1_final_states, p2_final_states, p1_paralyzed, p2_paralyzed, p2_seen_names = {}, {}, set(), set(), set()
        if battle_data.get('p2_lead_details'): p2_seen_names.add(battle_data['p2_lead_details']['name'])
        for turn in effective_timeline:
            if turn.get('p1_pokemon_state'):
                s = turn['p1_pokemon_state']; p1_final_states[s['name']] = s
                if s.get('status') == 'par': p1_paralyzed.add(s['name'])
                elif s['name'] in p1_paralyzed and s.get('status') is None: p1_paralyzed.remove(s['name'])
            if turn.get('p2_pokemon_state'):
                s = turn['p2_pokemon_state']; p2_final_states[s['name']] = s; p2_seen_names.add(s['name'])
                if s.get('status') == 'par': p2_paralyzed.add(s['name'])
                elif s['name'] in p2_paralyzed and s.get('status') is None: p2_paralyzed.remove(s['name'])
        p1_full = [{**p, **p1_final_states.get(p['name'], {})} for p in battle_data['p1_team_details']]
        p2_full = [{**pokedex_map.get(name, {}), **p2_final_states.get(name, {})} for name in p2_seen_names if name in pokedex_map]
        p1_rem = [p for p in p1_full if p.get('hp_pct', 1.0) > 0]
        p2_rem = [p for p in p2_full if p.get('hp_pct', 1.0) > 0]
        return p1_rem, p2_rem, p1_paralyzed, p2_paralyzed

    base_features = calculate_pokemon_advantage(timeline)
    efficiency_metrics = calculate_efficiency_metrics(timeline, battle_data['p1_team_details'])
    dynamic_ratios = calculate_dynamic_ratios(timeline)
    clutch_metrics = calculate_clutch_defense_score(timeline)
    hax_metrics = calculate_hax_advantage(timeline)
    
    features.update(base_features)
    features.update(calculate_lead_matchup(timeline, battle_data['p1_team_details'], battle_data.get('p2_lead_details')))
    features.update(get_status_advantage_snapshots(timeline, snapshot_turns))
    features.update(calculate_hax_advantage(timeline))
    features.update(calculate_clutch_defense_score(timeline))
    features.update(calculate_wasted_turns(timeline))
    features.update(calculate_active_defense(timeline))
    features.update(calculate_strategic_trade_advantage(timeline, pokedex_map))
    features.update(calculate_sacrifice_outcomes(timeline))
    features.update(calculate_confusion_turns_advantage(timeline))
    features.update(calculate_crippled_threat_advantage(timeline))
    features.update(calculate_setup_sweep_value(timeline))
    features.update(calculate_free_damage_advantage(timeline))
    features.update(calculate_trap_kos(timeline))
    
    threat_scores = {}
    all_turns_to_analyze = [0] + snapshot_turns
    for turn in all_turns_to_analyze:
        p1_rem, p2_rem, p1_par, p2_par = get_state_at_turn(turn)
        p1_threat = calculate_team_threat_score(p1_rem, p1_par)
        p2_threat = calculate_team_threat_score(p2_rem, p2_par)
        threat_scores[turn] = p1_threat - p2_threat
    for i, current_turn in enumerate(snapshot_turns):
        previous_turn = all_turns_to_analyze[i]
        features[f'net_threat_score_t{current_turn}'] = threat_scores[current_turn]
        features[f'threat_momentum_t{current_turn}'] = threat_scores[current_turn] - threat_scores[previous_turn]
        
    p1_rem_end, p2_rem_end, _, _ = get_state_at_turn(timeline[-1]['turn'])
    features.update(calculate_potential_coverage(p1_rem_end, p2_rem_end))
    features.update(calculate_anchor_strength(timeline, battle_data['p1_team_details'], pokedex_map, 'p1'))
    features.update(calculate_anchor_strength(timeline, [], pokedex_map, 'p2'))
    features.update(calculate_setup_threat(timeline, pokedex_map))
    
    features.update(efficiency_metrics)
    features.update(dynamic_ratios)

    ko_diff = base_features.get('final_pokemon_advantage', 0)
    dmg_adv = efficiency_metrics.get('damage_efficiency_ratio', 1)
    switch_ratio = dynamic_ratios.get('switch_ratio', 1)
    features['interact_ko_X_dmg'] = ko_diff * dmg_adv
    features['interact_dmg_X_switch'] = dmg_adv * (1 / switch_ratio)
    net_clutch_score = clutch_metrics.get('p1_clutch_score', 0) - clutch_metrics.get('p2_clutch_score', 0)
    features['resilience_score'] = net_clutch_score / (dmg_adv + 0.1)
    is_in_grey_zone = 0.75 < dmg_adv < 1.25
    features['grey_zone_hax'] = hax_metrics.get('hax_advantage', 0) if is_in_grey_zone else 0
    p2_cost = efficiency_metrics.get('p2_hp_cost_per_ko', 0)
    hax_adv = hax_metrics.get('hax_advantage', 0)
    move_r = dynamic_ratios.get('move_ratio', 1)
    
    features['interact_efficiency_X_cost'] = dmg_adv * p2_cost
    features['interact_hax_X_efficiency'] = hax_adv * dmg_adv
    features['interact_move_X_switch_ratio'] = move_r * switch_ratio
    
    return features

# --- Funzione Builder (da pokemon-battle-v-2) ---
# Modificata per accettare un DataFrame
def create_feature_dataframe_v2(dataset_df, snapshot_turns, pokedex_map):
    """
    Versione modificata: accetta un DataFrame.
    """
    processed_battles = []
    # Modifica: itera sulla versione 'records' del DataFrame
    for battle_data in tqdm(dataset_df.to_dict('records'), desc="Processing battles (v2)"):
        battle_features = process_battle_v2(battle_data, snapshot_turns, pokedex_map)
        if battle_features:
            processed_battles.append(battle_features)
    
    return pd.DataFrame(processed_battles)

print("Funzioni di feature engineering 'v2' caricate.")
# --- FINE BLOCCO AGGIUNTO ---