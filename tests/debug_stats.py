#!/usr/bin/env python
"""Debug script to explore player stats details"""

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

print("=== Exploring player stats objects ===\n")

try:
    players = query.get_league_players(player_count_limit=5)
    if players:
        player = players[0]
        print(f"Player: {player.full_name if hasattr(player, 'full_name') else player.name}\n")
        
        # Explore player_points
        if hasattr(player, 'player_points') and player.player_points:
            print(f"player_points type: {type(player.player_points)}")
            print(f"player_points attrs: {[a for a in dir(player.player_points) if not a.startswith('_')]}")
            print(f"player_points contents: {player.player_points}")
            if hasattr(player.player_points, 'total'):
                print(f"  player_points.total: {player.player_points.total}")
            print()
        
        # Explore player_stats  
        if hasattr(player, 'player_stats') and player.player_stats:
            print(f"player_stats type: {type(player.player_stats)}")
            print(f"player_stats: {player.player_stats}")
            if hasattr(player.player_stats, 'coverage_type'):
                print(f"  coverage_type: {player.player_stats.coverage_type}")
            if hasattr(player.player_stats, 'stats'):
                print(f"  stats: {player.player_stats.stats}")
                if player.player_stats.stats:
                    stat = player.player_stats.stats[0] if isinstance(player.player_stats.stats, list) else player.player_stats.stats
                    print(f"    stat type: {type(stat)}")
                    print(f"    stat attrs: {[a for a in dir(stat) if not a.startswith('_')][:10]}")
            print()
        
        # Check stats attribute
        if hasattr(player, 'stats'):
            print(f"stats type: {type(player.stats)}")
            print(f"stats contents: {player.stats}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
