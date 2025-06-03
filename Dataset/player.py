import sys
import os
import coloredlogs
import datetime
import json
import numpy as np
from faker import Factory
from faker.providers import address
import random
from typing import Dict

fake = Factory.create()
fake.seed(4231)
fake.add_provider(address)

rand = random.Random(4231)

countries = ["Spain", "Serbia", "Switzerland", "Germany", "Austria", "France", "Italy",
             "United States", "Canada", "Australia", "Argentina", "Brazil", "Chile",
             "Russia", "Poland", "Czech Republic", "Croatia", "Greece", "Denmark",
             "Norway", "Sweden", "Belgium", "Netherlands", "United Kingdom", "Japan",
             "South Korea", "China", "India", "South Africa", "Egypt", "Tunisia"]

player_counter = 1

gender = ["male", "female"]
docs = []
n = 10

def generate_physical_stats(gender: str, age: int) -> Dict:
    """Generate realistic physical statistics"""
    if gender.lower() == "male":
        height_cm = int(np.random.normal(183, 8))
        height_cm = max(165, min(215, height_cm))

        bmi = np.random.normal(22.5, 1.5)
        bmi = max(19, min(26, bmi))
        weight_kg = int(bmi * ((height_cm / 100) ** 2))

        reach_cm = int(height_cm * np.random.normal(1.02, 0.03))
    else:
        height_cm = int(np.random.normal(171, 7))
        height_cm = max(155, min(195, height_cm))

        bmi = np.random.normal(21.5, 1.5)
        bmi = max(18, min(25, bmi))
        weight_kg = int(bmi * ((height_cm / 100) ** 2))

        reach_cm = int(height_cm * np.random.normal(1.02, 0.03))

    return {
        'height_cm': height_cm,
        #'height_ft_in': f"{int(height_cm // 30.48)}'{int((height_cm % 30.48) / 2.54)}\"",
        'weight_kg': weight_kg,
        #'weight_lbs': int(weight_kg * 2.20462),
        #'reach_cm': reach_cm,
        'bmi': round(weight_kg / ((height_cm / 100) ** 2), 1)
    }

for i in range(n):
    delta = datetime.timedelta(days=i)
    for v_gender in [rand.choice(gender) for _ in range(rand.randint(2, len(gender)))]:
        for category in [rand.choice(gender) for _ in range(rand.randint(1, len(gender)))]:
            d = dict()
            d["player_id"] = f"P{player_counter:03d}"  # Format: P001, P002, ...
            #d["birthdate"] = (datetime.datetime.utcnow() - delta).strftime("%Y-%m-%d")
            #d["category"] = category
            d["gender"] = v_gender
            d["age"] = rand.randint(18, 35)
            d["ranking"] = rand.randint(1, 200)
            d["physical_stats"] = generate_physical_stats(category, d["age"])
            docs.append(d)
            player_counter += 1

output_file = "tennis_player_stats.json"

with open(output_file, "w") as f:
    json.dump(docs, f, indent=2)

print(f"Saved player stats to {output_file}")

