import json
import pandas as pd
from pathlib import Path

# Data directory
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

def clean_player_name(name_obj):
    """Extract full name from yfpy Name object or dict.
    
    Handles:
    - Name object with 'full' attribute
    - Dict with 'full' key
    - String representation of Name object
    - Plain strings
    """
    if not name_obj:
        return ''
    
    # If it's a dict, extract 'full' key
    if isinstance(name_obj, dict):
        return name_obj.get('full', '').strip()
    
    # Try to access .full attribute (yfpy Name object)
    if hasattr(name_obj, 'full'):
        return str(name_obj.full).strip()
    
    # Handle string representation - try to extract JSON
    name_str = str(name_obj)
    if 'Name({' in name_str and '"full"' in name_str:
        try:
            # Extract JSON from string like "Name({...})"
            json_start = name_str.index('{')
            json_end = name_str.rindex('}') + 1
            json_str = name_str[json_start:json_end]
            name_dict = json.loads(json_str)
            return name_dict.get('full', '').strip()
        except (ValueError, json.JSONDecodeError):
            return ''
    
    # Fall back to plain string
    return name_str.strip()

def load_standings():
    """Load standings JSON and convert to DataFrame."""
    with open(DATA_DIR / 'standings_yfpy.json', 'r') as f:
        data = json.load(f)
    
    # Flatten nested structure: season -> list of teams
    records = []
    for season, teams in data.items():
        for team in teams:
            team['season'] = season
            records.append(team)
    
    standings_df = pd.DataFrame(records)
    return standings_df

def load_rosters():
    """Load rosters JSON and convert to DataFrame."""
    with open(DATA_DIR / 'rosters_yfpy.json', 'r') as f:
        data = json.load(f)
    
    # Flatten nested structure: season -> team_id -> roster with nested players
    records = []
    for season, teams in data.items():
        for team_id, roster in teams.items():
            if roster and 'players' in roster:
                for player in roster['players']:
                    player_record = {
                        'season': season,
                        'team_id': team_id,
                        'coverage_type': roster.get('coverage_type', ''),
                        'week': roster.get('week', ''),
                    }
                    player_record.update(player)
                    records.append(player_record)
    
    rosters_df = pd.DataFrame(records)
    # Clean up name field to extract just "First Last" format
    if 'name' in rosters_df.columns:
        rosters_df['name'] = rosters_df['name'].apply(clean_player_name)
    return rosters_df

def load_draft_results():
    """Load draft results JSON and convert to a flat DataFrame."""
    with open(DATA_DIR / 'draft_results_yfpy.json', 'r') as f:
        data = json.load(f)
    
    # Flatten nested structure: season -> team_id -> draft with nested picks
    records = []
    for season, teams in data.items():
        for team_id, draft in teams.items():
            if draft and 'picks' in draft:
                for pick in draft['picks']:
                    pick_record = {
                        'season': season,
                        'team_id': team_id,
                    }
                    pick_record.update(pick)
                    
                    # Extract player data: handle nested player object or direct player_name/position fields
                    if 'player' in pick and isinstance(pick['player'], dict):
                        # If player is a dict, extract name and position from it
                        player_obj = pick['player']
                        pick_record['player_name'] = player_obj.get('name', '')
                        pick_record['player_position'] = player_obj.get('position', '')
                    # Otherwise, player_name and player_position should already be in pick_record from update()
                    
                    records.append(pick_record)
    
    draft_results_df = pd.DataFrame(records)
    
    # Clean player_name field if it exists
    if 'player_name' in draft_results_df.columns:
        draft_results_df['player_name'] = draft_results_df['player_name'].apply(clean_player_name)
    
    return draft_results_df

def load_matchups():
    """Load matchups JSON and convert to DataFrame."""
    with open(DATA_DIR / 'matchups_yfpy.json', 'r') as f:
        data = json.load(f)

    # Flatten nested structure: season -> team_id -> matchups with nested matchup data
    records = []
    for season, teams in data.items():
        for team_id, matchup_data in teams.items():
            if matchup_data and 'matchups' in matchup_data:
                for matchup in matchup_data['matchups']:
                    matchup_record = {
                        'season': season,
                        'team_id': team_id,
                    }
                    matchup_record.update(matchup)
                    records.append(matchup_record)

    matchups_df = pd.DataFrame(records)
    return matchups_df


def load_scoreboard():
    """Load scoreboard JSON and convert to a flat DataFrame.

    Columns: season, week, team_key, team_points, projected_points,
             is_winner, is_playoffs, is_consolation
    """
    filepath = DATA_DIR / 'scoreboard_yfpy.json'
    if not filepath.exists():
        return pd.DataFrame()

    with open(filepath, 'r') as f:
        data = json.load(f)

    records = []
    for season, weeks in data.items():
        for week, teams in weeks.items():
            for team_key, score_data in teams.items():
                record = {
                    'season':   season,
                    'week':     week,
                    'team_key': team_key,
                }
                record.update(score_data)
                records.append(record)

    return pd.DataFrame(records)

def load_scoreboard_season_totals():
    """Aggregate weekly scoreboard into per-team season totals.

    Columns: season, team_key, total_points, total_projected_points,
             wins, weeks_played
    """
    scoreboard_df = load_scoreboard()
    if scoreboard_df.empty:
        return pd.DataFrame()

    scoreboard_df['season']            = scoreboard_df['season'].astype(int)
    scoreboard_df['week']              = scoreboard_df['week'].astype(int)
    scoreboard_df['team_points']       = pd.to_numeric(scoreboard_df['team_points'], errors='coerce')
    scoreboard_df['projected_points']  = pd.to_numeric(scoreboard_df['projected_points'], errors='coerce')
    scoreboard_df['is_winner']         = scoreboard_df['is_winner'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)

    totals = (
        scoreboard_df
        .groupby(['season', 'team_key'])
        .agg(
            total_points       =('team_points',      'sum'),
            total_projected    =('projected_points', 'sum'),
            wins               =('is_winner',        'sum'),
            weeks_played       =('week',             'count'),
        )
        .reset_index()
    )

    # Luck proxy: how much a team out- or under-performed projections
    totals['points_vs_projected'] = totals['total_points'] - totals['total_projected']

    return totals


def load_weekly_rosters():
    """Load weekly_rosters_yfpy.json -> flat per-player-per-week DataFrame.

    Columns: season, team_key, week, player_key, name, position,
             selected_position, player_points
    """
    filepath = DATA_DIR / 'weekly_rosters_yfpy.json'
    if not filepath.exists():
        return pd.DataFrame()
    with open(filepath, 'r') as f:
        data = json.load(f)
    records = []
    for season, teams in data.items():
        if not isinstance(teams, dict):
            continue
        for team_key, weeks in teams.items():
            if not isinstance(weeks, dict):
                continue
            for week, players in weeks.items():
                if not isinstance(players, list):
                    continue
                for player in players:
                    records.append({
                        'season':            season,
                        'team_key':          team_key,
                        'week':              week,
                        'player_key':        player.get('player_key', ''),
                        'name':              player.get('name', ''),
                        'position':          player.get('position', ''),
                        'selected_position': player.get('selected_position', ''),
                        'player_points':     player.get('player_points', ''),
                    })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df['player_points'] = pd.to_numeric(df['player_points'], errors='coerce')
    return df


def load_trades():
    """Load trades_yfpy.json -> flat player-per-row DataFrame.

    Columns: season, transaction_key, transaction_id, timestamp, week,
             trader_team_key, tradee_team_key, player_key, name, position,
             source_team_key, destination_team_key
    """
    filepath = DATA_DIR / 'trades_yfpy.json'
    if not filepath.exists():
        return pd.DataFrame()
    with open(filepath, 'r') as f:
        data = json.load(f)
    records = []
    for season, trades in data.items():
        if not isinstance(trades, list):
            continue
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            meta = {
                'season':          season,
                'transaction_key': trade.get('transaction_key', ''),
                'transaction_id':  trade.get('transaction_id', ''),
                'timestamp':       trade.get('timestamp', ''),
                'week':            trade.get('week', ''),
                'trader_team_key': trade.get('trader_team_key', ''),
                'tradee_team_key': trade.get('tradee_team_key', ''),
            }
            for player in trade.get('players', []):
                records.append({
                    **meta,
                    'player_key':           player.get('player_key', ''),
                    'name':                 player.get('name', ''),
                    'position':             player.get('position', ''),
                    'source_team_key':      player.get('source_team_key', ''),
                    'destination_team_key': player.get('destination_team_key', ''),
                })
    return pd.DataFrame(records) if records else pd.DataFrame()


def load_all_csvs():
    """Load base CSVs produced by run() into DataFrames.

    Returns: standings, matchups, sb_totals, scoreboard
    """
    standings = pd.read_csv(DATA_DIR / 'standings_df.csv')
    matchups  = pd.read_csv(DATA_DIR / 'matchups_df.csv')

    def _opt(name):
        p = DATA_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    return standings, matchups, _opt('scoreboard_totals_df.csv'), _opt('scoreboard_df.csv')


def run():
    """Load all JSON files → DataFrames, display shape info, save CSVs and pickles."""
    print(f'Loading JSON files from {DATA_DIR}...\n')

    if not (DATA_DIR / 'standings_yfpy.json').exists():
        print('Error: standings_yfpy.json not found in data folder.')
        print('  Please run yahoo_api.py first to generate the data files.')
        return

    standings_df         = load_standings()
    rosters_df           = load_rosters()
    draft_results_df     = load_draft_results()
    matchups_df          = load_matchups()
    scoreboard_df        = load_scoreboard()
    scoreboard_totals_df = load_scoreboard_season_totals()
    weekly_rosters_df    = load_weekly_rosters()
    trades_df            = load_trades()

    all_dfs = [
        ('standings_df',         standings_df),
        ('rosters_df',           rosters_df),
        ('draft_results_df',     draft_results_df),
        ('matchups_df',          matchups_df),
        ('scoreboard_df',        scoreboard_df),
        ('scoreboard_totals_df', scoreboard_totals_df),
        ('weekly_rosters_df',    weekly_rosters_df),
        ('trades_df',            trades_df),
    ]
    for label, df in all_dfs:
        if df.empty:
            print(f'  {label}: (empty)')
        else:
            print(f'  {label}: {df.shape}')
            print(f'  Columns: {list(df.columns)}')
        print()

    # Save to CSV
    standings_df.to_csv(DATA_DIR / 'standings_df.csv', index=False)
    rosters_df.to_csv(DATA_DIR / 'rosters_df.csv', index=False)
    draft_results_df.to_csv(DATA_DIR / 'draft_results_df.csv', index=False)
    matchups_df.to_csv(DATA_DIR / 'matchups_df.csv', index=False)
    for name, df in [
        ('scoreboard_df',        scoreboard_df),
        ('scoreboard_totals_df', scoreboard_totals_df),
        ('weekly_rosters_df',    weekly_rosters_df),
        ('trades_df',            trades_df),
    ]:
        if not df.empty:
            df.to_csv(DATA_DIR / f'{name}.csv', index=False)
    print('All DataFrames saved to CSV (data/*_df.csv)')

    # Save to pickle
    standings_df.to_pickle(DATA_DIR / 'standings_df.pkl')
    rosters_df.to_pickle(DATA_DIR / 'rosters_df.pkl')
    draft_results_df.to_pickle(DATA_DIR / 'draft_results_df.pkl')
    matchups_df.to_pickle(DATA_DIR / 'matchups_df.pkl')
    for name, df in [
        ('scoreboard_df',        scoreboard_df),
        ('scoreboard_totals_df', scoreboard_totals_df),
        ('weekly_rosters_df',    weekly_rosters_df),
        ('trades_df',            trades_df),
    ]:
        if not df.empty:
            df.to_pickle(DATA_DIR / f'{name}.pkl')
    print('All DataFrames saved to pickle (data/*_df.pkl)')


if __name__ == '__main__':
    run()
