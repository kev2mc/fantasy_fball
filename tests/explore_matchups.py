#!/usr/bin/env python
"""Explore matchups data to see if player-level scoring is available"""

import json

with open('data/matchups_yfpy.json', 'r') as f:
    data = json.load(f)

# Check structure of matchups
for season in ['2025', '2024', '2023']:
    if season in data:
        season_data = data[season]
        for team_key, matchup_data in season_data.items():
            if matchup_data and 'matchups' in matchup_data and matchup_data['matchups']:
                matchup = matchup_data['matchups'][0]
                print(f"\nSeason {season}, Team {team_key}")
                print(f"Matchup keys: {list(matchup.keys())}")
                print(f"Full matchup: {json.dumps(matchup, indent=2)[:500]}")
                break
        break
