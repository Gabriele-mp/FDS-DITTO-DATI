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
SNAPSHOT_TURNS_V2 = [15, 20, 23, 25, 27, 30] 