
import random
import json
import datetime

# --- Constants ---
TOURNAMENT_ID = "T001"
MATCH_ID = "M001"
COURT_TYPES = ["Hard", "Clay", "Grass"]
SET_ID = "S001"
GAME_ID = "G001"
SET_NUMBER = 1
GAME_NUMBER = 1
PLAYER_ID = "P001"
OPPONENT_PLAYER_ID = "P002"

# --- Helper Functions ---
def generate_point_duration():
    return round(random.uniform(3.5, 25.0), 1)

def generate_point_winner():
    return random.choice([PLAYER_ID, OPPONENT_PLAYER_ID])

def generate_point_data():
    # Simulate realistic point data
    first_serve = random.choice([True, False])
    second_serve = not first_serve
    stroke_type = random.choice(["Forehand", "Backhand"])
    service_fault = random.choice([True, False]) if second_serve else False
    break_point = random.choice([True, False])
    return first_serve, second_serve, stroke_type, service_fault, break_point

# --- Generate Points ---
num_points = random.randint(6, 12)  # one game typically has 4-10 points

match_date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
court = random.choice(COURT_TYPES)

points = []

for point_id in range(1, num_points + 1):
    first_serve, second_serve, stroke_type, service_fault, break_point = generate_point_data()

    point = {
        "tournament_id": TOURNAMENT_ID,
        "match_id": MATCH_ID,
        "match_date": match_date,
        "set_id": SET_ID,
        "game_id": GAME_ID,
        "court": court,
        "player_id": PLAYER_ID,
        "opponent_player_id": OPPONENT_PLAYER_ID,
        "game_number": GAME_NUMBER,
        "set_number": SET_NUMBER,
        "point_id": f"PT{point_id:03d}",
        "point_duration": generate_point_duration(),
        "point_winner": generate_point_winner(),
        "first_serve": first_serve,
        "second_serve": second_serve,
        "stroke_type": stroke_type,
        "service_fault": service_fault,
        "break_point": break_point
    }
    points.append(point)

# --- Save to File ---
with open("tennis_match_detailed_game.json", "w") as f:
    json.dump(points, f, indent=2)

print("Detailed tennis match game data saved to 'tennis_match_detailed_game.json'")


