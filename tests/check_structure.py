import json

# Check the structure of the data
data = json.load(open('data/rosters_yfpy.json'))

season = '2025'
if season in data:
    roster_data = list(data[season].values())[0]
    print(f"Season {season} roster coverage_type: {roster_data.get('coverage_type')}")
    print(f"Season {season} roster week: {roster_data.get('week')}")
    
    if roster_data.get('players'):
        player = roster_data['players'][0]
        print(f"\nFirst player keys: {list(player.keys())}")
        print(f"Player name: {player.get('name')}")
        print(f"Player points: {player.get('player_points')}")
        print(f"Player projected points: {player.get('player_projected_points')}")
