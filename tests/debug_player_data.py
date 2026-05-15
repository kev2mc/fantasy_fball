#!/usr/bin/env python
"""Debug script to check what data yfpy returns for players"""

from yfpy.query import YahooFantasySportsQuery
from pathlib import Path
import os

query = YahooFantasySportsQuery(
    league_id=884193,
    game_code="nfl",
    game_id=461,
    yahoo_consumer_key=os.getenv("YAHOO_CONSUMER_KEY"),
    yahoo_consumer_secret=os.getenv("YAHOO_CONSUMER_SECRET"),
    save_token_data_to_env_file=True,
    env_file_location=Path(".")
)

# Get a team roster to see what player data we get
try:
    roster = query.get_team_roster_by_week(team_id=3)
    if hasattr(roster, 'players') and roster.players:
        player = roster.players[0]
        print(f"Player name: {player.name if hasattr(player, 'name') else 'N/A'}")
        print(f"Has player_points: {hasattr(player, 'player_points')}")
        if hasattr(player, 'player_points'):
            print(f"  player_points: {player.player_points}")
            if player.player_points:
                print(f"  player_points attributes: {dir(player.player_points)}")
        
        print(f"\nHas player_projected_points: {hasattr(player, 'player_projected_points')}")
        if hasattr(player, 'player_projected_points'):
            print(f"  player_projected_points: {player.player_projected_points}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
