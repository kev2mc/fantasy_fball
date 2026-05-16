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
    """Load weekly_rosters_YEAR.json per-season files -> flat per-player-per-week DataFrame.

    Columns: season, team_key, week, player_key, name, position,
             selected_position, player_points
    """
    season_files = sorted(DATA_DIR.glob('weekly_rosters_*.json'))
    if not season_files:
        return pd.DataFrame()
    records = []
    for filepath in season_files:
        season = filepath.stem.replace('weekly_rosters_', '')
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Structure: {season: {team_key: {week: [players]}}}  OR  {team_key: {week: [players]}}
        top = data.get(season, data)
        for team_key, weeks in top.items():
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
    """Load trades_YEAR.json per-season files -> flat player-per-row DataFrame.

    Columns: season, transaction_key, transaction_id, timestamp, week,
             trader_team_key, tradee_team_key, player_key, name, position,
             source_team_key, destination_team_key
    """
    season_files = sorted(DATA_DIR.glob('trades_*.json'))
    if not season_files:
        return pd.DataFrame()
    records = []
    for filepath in season_files:
        season = filepath.stem.replace('trades_', '')
        with open(filepath, 'r') as f:
            data = json.load(f)
        trades = data if isinstance(data, list) else data.get(season, data.get('trades', []))
        for trade in (trades or []):
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


def compute_draft_overperformance(draft_results_df, weekly_rosters_df):
    """Compare drafted players' actual season points to the average for each pick slot.

    expected_pts for a pick slot = mean actual points across all players ever
    drafted at that slot in the dataset (own historical data as benchmark).
    Returns: season, team_key, total_draft_pts, expected_draft_pts,
             draft_overperformance (actual - expected, per team per season)
    """
    if draft_results_df.empty or weekly_rosters_df.empty:
        return pd.DataFrame()
    player_pts = (
        weekly_rosters_df
        .groupby(['season', 'player_key'])['player_points']
        .sum().reset_index(name='season_points')
    )
    player_pts['season'] = pd.to_numeric(player_pts['season']).astype(int)
    draft = draft_results_df.copy()
    draft['season'] = pd.to_numeric(draft['season']).astype(int)
    draft['pick']   = pd.to_numeric(draft['pick'], errors='coerce')
    draft = draft.rename(columns={'team_id': 'team_key'})
    draft = draft.merge(player_pts[['season', 'player_key', 'season_points']],
                        on=['season', 'player_key'], how='left')
    pick_avg = (
        draft.dropna(subset=['season_points'])
        .groupby('pick')['season_points']
        .mean().reset_index(name='expected_pts')
    )
    draft = draft.merge(pick_avg, on='pick', how='left')
    draft['pts_vs_expected'] = draft['season_points'] - draft['expected_pts']
    result = (
        draft.groupby(['season', 'team_key'])
        .agg(total_draft_pts       =('season_points',   'sum'),
             expected_draft_pts    =('expected_pts',    'sum'),
             draft_overperformance =('pts_vs_expected', 'sum'))
        .reset_index()
    )
    for col in ('total_draft_pts', 'expected_draft_pts', 'draft_overperformance'):
        result[col] = result[col].round(1)
    return result


def compute_optimal_lineup_stats(weekly_rosters_df):
    """Per team per season: actual starter score, optimal lineup score, pts left on bench.

    For each team-week, solves the optimal assignment of players to starting slots
    using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    pts_left_on_bench = optimal_score - actual_score.

    Returns: season, team_key, total_pts_left_on_bench, avg_pts_left_per_week
    """
    if weekly_rosters_df.empty:
        return pd.DataFrame()

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
    except ImportError:
        print('  Warning: scipy not installed; skipping optimal lineup stats.')
        return pd.DataFrame()

    FLEX_ELIGIBLE = {
        'W/R/T':   {'WR', 'RB', 'TE'},
        'W/T':     {'WR', 'TE'},
        'W/R':     {'WR', 'RB'},
        'Q/W/R/T': {'QB', 'WR', 'RB', 'TE'},
        'FLEX':    {'WR', 'RB', 'TE'},
        'UT':      {'QB', 'WR', 'RB', 'TE', 'K', 'DEF'},
    }

    def _week_scores(players):
        active = [p for p in players if p.get('selected_position') != 'IR']
        starter_slots = [
            p['selected_position'] for p in active
            if p.get('selected_position') not in ('BN', 'IR', '')
        ]
        if not starter_slots:
            return None, None

        actual = sum(
            float(p.get('player_points') or 0)
            for p in active
            if p.get('selected_position') not in ('BN', 'IR', '')
        )

        benefit = np.zeros((len(starter_slots), len(active)))
        for i, slot in enumerate(starter_slots):
            for j, player in enumerate(active):
                pts = float(player.get('player_points') or 0)
                if pts <= 0:
                    continue
                pos = player.get('position', '')
                ep_str = player.get('eligible_positions', pos)
                ep = set(ep_str.split(',')) if ep_str else {pos}
                if slot in FLEX_ELIGIBLE:
                    eligible = pos in FLEX_ELIGIBLE[slot]
                else:
                    eligible = (slot in ep) or (slot == pos)
                if eligible:
                    benefit[i, j] = pts

        row_ind, col_ind = linear_sum_assignment(benefit, maximize=True)
        optimal = float(benefit[row_ind, col_ind].sum())
        return round(actual, 2), round(optimal, 2)

    week_records = []
    for (season, team_key, week), grp in weekly_rosters_df.groupby(['season', 'team_key', 'week']):
        actual, optimal = _week_scores(grp.to_dict('records'))
        if actual is not None:
            week_records.append({
                'season':            season,
                'team_key':          team_key,
                'week':              week,
                'actual_score':      actual,
                'optimal_score':     optimal,
                'pts_left_on_bench': round(optimal - actual, 2),
            })

    if not week_records:
        return pd.DataFrame()

    week_df = pd.DataFrame(week_records)
    result = (
        week_df.groupby(['season', 'team_key'])
        .agg(
            total_pts_left_on_bench = ('pts_left_on_bench', 'sum'),
            avg_pts_left_per_week   = ('pts_left_on_bench', 'mean'),
        )
        .reset_index()
    )
    result[['total_pts_left_on_bench', 'avg_pts_left_per_week']] = \
        result[['total_pts_left_on_bench', 'avg_pts_left_per_week']].round(2)
    result['season'] = pd.to_numeric(result['season']).astype(int)
    return result


def compute_trade_value(trades_df, weekly_rosters_df):
    """Post-trade player production: points received vs points given away.

    For each traded player, sums their points in weeks AFTER the trade week.
    trade_value = post-trade points received - post-trade points given away.
    Returns: season, team_key, trade_pts_received, trade_pts_given, trade_value
    """
    if trades_df.empty or weekly_rosters_df.empty:
        return pd.DataFrame()
    trades = trades_df[['season', 'transaction_key', 'player_key',
                         'week', 'source_team_key', 'destination_team_key']].copy()
    trades = trades.rename(columns={'week': 'trade_week'})
    trades['season']     = pd.to_numeric(trades['season'])
    trades['trade_week'] = pd.to_numeric(trades['trade_week'], errors='coerce').fillna(0)
    rosters = weekly_rosters_df[['season', 'player_key', 'week', 'player_points']].copy()
    rosters = rosters.rename(columns={'week': 'roster_week'})
    rosters['season']      = pd.to_numeric(rosters['season'])
    rosters['roster_week'] = pd.to_numeric(rosters['roster_week'])
    merged = trades.merge(rosters, on=['season', 'player_key'], how='left')
    post = merged[merged['roster_week'] > merged['trade_week']].copy()
    if post.empty:
        return pd.DataFrame()
    player_post = (
        post.groupby(['season', 'player_key', 'destination_team_key', 'source_team_key'])
        ['player_points'].sum().reset_index(name='post_trade_pts')
    )
    received = (
        player_post.groupby(['season', 'destination_team_key'])['post_trade_pts']
        .sum().reset_index()
        .rename(columns={'destination_team_key': 'team_key', 'post_trade_pts': 'trade_pts_received'})
    )
    given = (
        player_post.groupby(['season', 'source_team_key'])['post_trade_pts']
        .sum().reset_index()
        .rename(columns={'source_team_key': 'team_key', 'post_trade_pts': 'trade_pts_given'})
    )
    tv = received.merge(given, on=['season', 'team_key'], how='outer').fillna(0)
    tv['trade_value'] = (tv['trade_pts_received'] - tv['trade_pts_given']).round(1)
    tv[['trade_pts_received', 'trade_pts_given']] = tv[['trade_pts_received', 'trade_pts_given']].round(1)
    tv['season'] = tv['season'].astype(int)
    return tv


if __name__ == '__main__':
    print(f'Loading JSON files from {DATA_DIR}...\n')

    if not (DATA_DIR / 'standings_yfpy.json').exists():
        print('Error: standings_yfpy.json not found in data folder.')
        print('  Please run yahoo_api.py first to generate the data files.')
        exit(1)

    # --- Core loaders ---
    standings_df         = load_standings()
    rosters_df           = load_rosters()
    draft_results_df     = load_draft_results()
    matchups_df          = load_matchups()
    scoreboard_df        = load_scoreboard()
    scoreboard_totals_df = load_scoreboard_season_totals()

    # --- New loaders (populated after weekly roster + trades API run) ---
    weekly_rosters_df = load_weekly_rosters()
    trades_df         = load_trades()

    # --- Computed metrics ---
    optimal_lineup_df  = compute_optimal_lineup_stats(weekly_rosters_df)
    draft_overperf_df  = compute_draft_overperformance(draft_results_df, weekly_rosters_df)
    trade_value_df     = compute_trade_value(trades_df, weekly_rosters_df)

    # --- Display info ---
    all_dfs = [
        ('standings_df',         standings_df),
        ('rosters_df',           rosters_df),
        ('draft_results_df',     draft_results_df),
        ('matchups_df',          matchups_df),
        ('scoreboard_df',        scoreboard_df),
        ('scoreboard_totals_df', scoreboard_totals_df),
        ('weekly_rosters_df',    weekly_rosters_df),
        ('trades_df',            trades_df),
        ('optimal_lineup_df',    optimal_lineup_df),
        ('draft_overperf_df',    draft_overperf_df),
        ('trade_value_df',       trade_value_df),
    ]
    for label, df in all_dfs:
        if df.empty:
            print(f'  {label}: (empty)')
        else:
            print(f'  {label}: {df.shape}')
            print(f'  Columns: {list(df.columns)}')
        print()

    # --- Save to CSV ---
    standings_df.to_csv(DATA_DIR / 'standings_df.csv', index=False)
    rosters_df.to_csv(DATA_DIR / 'rosters_df.csv', index=False)
    draft_results_df.to_csv(DATA_DIR / 'draft_results_df.csv', index=False)
    matchups_df.to_csv(DATA_DIR / 'matchups_df.csv', index=False)
    for name, df in [
        ('scoreboard_df',         scoreboard_df),
        ('scoreboard_totals_df',  scoreboard_totals_df),
        ('weekly_rosters_df',     weekly_rosters_df),
        ('trades_df',             trades_df),
        ('optimal_lineup_df',     optimal_lineup_df),
        ('draft_overperf_df',     draft_overperf_df),
        ('trade_value_df',        trade_value_df),
    ]:
        if not df.empty:
            df.to_csv(DATA_DIR / f'{name}.csv', index=False)

    print('All DataFrames saved to CSV (data/*_df.csv)')

    # --- Save to pickle ---
    standings_df.to_pickle(DATA_DIR / 'standings_df.pkl')
    rosters_df.to_pickle(DATA_DIR / 'rosters_df.pkl')
    draft_results_df.to_pickle(DATA_DIR / 'draft_results_df.pkl')
    matchups_df.to_pickle(DATA_DIR / 'matchups_df.pkl')
    for name, df in [
        ('scoreboard_df',         scoreboard_df),
        ('scoreboard_totals_df',  scoreboard_totals_df),
        ('weekly_rosters_df',     weekly_rosters_df),
        ('trades_df',             trades_df),
        ('optimal_lineup_df',     optimal_lineup_df),
        ('draft_overperf_df',     draft_overperf_df),
        ('trade_value_df',        trade_value_df),
    ]:
        if not df.empty:
            df.to_pickle(DATA_DIR / f'{name}.pkl')

    print('All DataFrames saved to pickle (data/*_df.pkl)')
