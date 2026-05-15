import json

# Check if player_points are populated
data = json.load(open('data/rosters_yfpy.json'))

for season in ['2022', '2023', '2024', '2025']:
    if season in data:
        roster_data = list(data[season].values())[0]
        if roster_data.get('players'):
            player = roster_data['players'][0]
            points = player.get('player_points')
            projected = player.get('player_projected_points')
            print(f"Season {season}: Points='{points}', Projected='{projected}'")
