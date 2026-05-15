#!/usr/bin/env python
"""Debug player key matching"""

import json
import pandas as pd

# Load rosters
rosters = pd.read_csv('data/rosters_df.csv')

# Check player keys for 2025
rosters_2025 = rosters[rosters['season'] == 2025]
print(f"2025 roster sample player_keys: {rosters_2025['player_key'].head(10).tolist()}")
print(f"Total 2025 roster entries: {len(rosters_2025)}")

# Load JSON to check what format API returns
with open('data/rosters_yfpy.json', 'r') as f:
    rosters_json = json.load(f)

if '2025' in rosters_json:
    teams_2025 = rosters_json['2025']
    for team_key, roster_data in list(teams_2025.items())[:1]:
        if roster_data.get('players'):
            player = roster_data['players'][0]
            print(f"\nJSON player_key sample: {player.get('player_key')}")
            print(f"Total players in first team: {len(roster_data['players'])}")
