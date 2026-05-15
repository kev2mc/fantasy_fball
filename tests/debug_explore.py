#!/usr/bin/env python
"""Debug script to explore all available player data in yfpy"""

from yfpy.query import YahooFantasySportsQuery
from pathlib import Path
import os
import json

query = YahooFantasySportsQuery(
    league_id=884193,
    game_code="nfl",
    game_id=461,
    yahoo_consumer_key=os.getenv("YAHOO_CONSUMER_KEY"),
    yahoo_consumer_secret=os.getenv("YAHOO_CONSUMER_SECRET"),
    save_token_data_to_env_file=True,
    env_file_location=Path(".")
)

print("=== Exploring available player data ===\n")

# Get league info
try:
    league = query.get_league()
    if league:
        print(f"League: {league.name if hasattr(league, 'name') else 'N/A'}")
        print(f"Current week: {league.current_week if hasattr(league, 'current_week') else 'N/A'}")
        print(f"Scoring type: {league.scoring_type if hasattr(league, 'scoring_type') else 'N/A'}")
except Exception as e:
    print(f"Error getting league: {e}")

print("\n=== Checking league players with different parameters ===")
try:
    # Try different ways to get player stats
    players = query.get_league_players(player_count_limit=5)
    if players:
        player = players[0]
        print(f"Player: {player.name if hasattr(player, 'name') else 'N/A'}")
        
        # Check all attributes
        attrs = [attr for attr in dir(player) if not attr.startswith('_')]
        print(f"\nPlayer attributes: {attrs}")
        
        # Check for stats-related attributes
        stats_attrs = [a for a in attrs if 'stat' in a.lower() or 'point' in a.lower()]
        print(f"\nStats-related attributes: {stats_attrs}")
        
        for attr in stats_attrs:
            try:
                val = getattr(player, attr)
                print(f"  {attr}: {type(val).__name__} = {val if not hasattr(val, '__iter__') else '(iterable)'}")
            except:
                pass
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Checking team roster for week stats ===")
try:
    # Try getting roster with different parameters
    roster = query.get_team_roster_by_week(team_id=3, week=17)
    if hasattr(roster, 'players') and roster.players:
        player = roster.players[0]
        attrs = [attr for attr in dir(player) if not attr.startswith('_') and 'point' in attr.lower()]
        print(f"Player: {player.name if hasattr(player, 'name') else 'N/A'}")
        print(f"Points-related attributes in roster: {attrs}")
except Exception as e:
    print(f"Error: {e}")
