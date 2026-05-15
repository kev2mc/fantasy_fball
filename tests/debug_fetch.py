#!/usr/bin/env python
"""Fetch player scoring stats from Yahoo Fantasy API - with better debugging"""

from yfpy.query import YahooFantasySportsQuery
from pathlib import Path
import json
import os
import sys

print("Starting player stats fetch...", flush=True)
sys.stdout.flush()

try:
    # Try 2025 first
    league_id = '884193'
    game_id = 461
    
    print(f"Creating query for league {league_id}, game {game_id}...", flush=True)
    sys.stdout.flush()
    
    query = YahooFantasySportsQuery(
        league_id=league_id,
        game_code="nfl",
        game_id=game_id,
        yahoo_consumer_key=os.getenv("YAHOO_CONSUMER_KEY"),
        yahoo_consumer_secret=os.getenv("YAHOO_CONSUMER_SECRET"),
        save_token_data_to_env_file=True,
        env_file_location=Path(".")
    )
    
    print("Fetching players...", flush=True)
    sys.stdout.flush()
    
    players = query.get_league_players(player_count_limit=5)
    
    if players:
        print(f"Got {len(players)} players", flush=True)
        player = players[0]
        
        print(f"\nPlayer: {player.name if hasattr(player, 'name') else 'N/A'}", flush=True)
        print(f"Has player_points_value: {hasattr(player, 'player_points_value')}", flush=True)
        if hasattr(player, 'player_points_value'):
            print(f"  player_points_value: {player.player_points_value}", flush=True)
        
        print(f"Has ownership: {hasattr(player, 'ownership')}", flush=True)
        if hasattr(player, 'ownership') and player.ownership:
            print(f"  ownership attrs: {[a for a in dir(player.ownership) if not a.startswith('_')][:5]}", flush=True)
            
        sys.stdout.flush()
    else:
        print("No players returned", flush=True)
        sys.stdout.flush()
        
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
