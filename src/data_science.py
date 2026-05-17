import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')

# Maps team names used by '--hidden--' managers to their real identities.
_HIDDEN_TEAM_TO_MANAGER = {
    'Juke Town': 'Derek',
}

# Renames Yahoo nicknames to preferred display names.
_MANAGER_RENAMES = {
    'TP1':   'Toby',
    'Chris': 'Devins',
}

_MATTHEW_TEAM_TO_MANAGER = {
    'GUCKINATORS':          'Gucker',
    'Give Me The Cheddar':  'Gucker',
    'Delawhere?':           'Endler',
    'WamBamRightInTheClam': 'Endler',
    'JoeBuckYourself':      'Endler',
    'Suh Girls One Cup':    'Endler',
    'Golden Taint':         'Endler',
    '3rd & a Deuce':        'Endler',
    'FantasyFootballTeam':  'Endler',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_name(val):
    """Strip Yahoo's b'...' / b\"...\" bytes-string artefact from team names."""
    s = str(val)
    if (s.startswith("b'") and s.endswith("'")) or \
       (s.startswith('b"') and s.endswith('"')):
        return s[2:-1]
    return s


def _to_numeric(series):
    return pd.to_numeric(series, errors='coerce')


def _make_manager_key(nickname, manager_id):
    """Return a disambiguated manager identifier.

    Used for nicknames that are shared by multiple distinct Yahoo accounts
    (including '--hidden--' privacy accounts and real-name collisions).
    Appends manager_id so each person gets a unique key.
    """
    nick = str(nickname).strip()
    mid  = int(manager_id)
    if nick == '--hidden--':
        return f'Hidden (ID:{mid})'
    return f'{nick} (ID:{mid})'


# ---------------------------------------------------------------------------
# Analytics (moved from convert_to_df)
# ---------------------------------------------------------------------------

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
                pos    = player.get('position', '')
                ep_str = player.get('eligible_positions', pos)
                ep     = set(ep_str.split(',')) if ep_str else {pos}
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
    post   = merged[merged['roster_week'] > merged['trade_week']].copy()
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
    tv[['trade_pts_received', 'trade_pts_given']] = \
        tv[['trade_pts_received', 'trade_pts_given']].round(1)
    tv['season'] = tv['season'].astype(int)
    return tv


# ---------------------------------------------------------------------------
# Expected wins
# ---------------------------------------------------------------------------

def build_expected_record(scoreboard):
    """Expected W/L based on scoring above/below the league average each week.

    For each regular-season week, a team earns an expected win if their score
    exceeds the league average that week, and an expected loss if below.
    The gap between actual and expected wins is a proxy for schedule luck.
    """
    if scoreboard.empty:
        return pd.DataFrame()

    sb = scoreboard.copy()
    sb['season']       = pd.to_numeric(sb['season'])
    sb['week']         = pd.to_numeric(sb['week'])
    sb['team_points']  = pd.to_numeric(sb['team_points'], errors='coerce')
    sb['is_playoffs']  = pd.to_numeric(sb['is_playoffs'], errors='coerce').fillna(0).astype(int)

    # Regular season only
    sb = sb[sb['is_playoffs'] == 0].copy()

    sb['week_avg'] = sb.groupby(['season', 'week'])['team_points'].transform('mean')
    sb['exp_win']  = (sb['team_points'] > sb['week_avg']).astype(int)
    sb['exp_loss'] = (sb['team_points'] < sb['week_avg']).astype(int)

    return (
        sb.groupby(['season', 'team_key'])
        .agg(exp_wins=('exp_win', 'sum'), exp_losses=('exp_loss', 'sum'))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Strength of schedule
# ---------------------------------------------------------------------------

def build_strength_of_schedule(scoreboard, matchups):
    """True SOS: mean of each opponent's avg PPW across all their regular-season games."""
    if scoreboard.empty or matchups.empty:
        return pd.DataFrame()

    sb = scoreboard.copy()
    sb['season']      = pd.to_numeric(sb['season'])
    sb['team_points'] = pd.to_numeric(sb['team_points'], errors='coerce')
    sb['is_playoffs'] = pd.to_numeric(sb['is_playoffs'], errors='coerce').fillna(0).astype(int)
    reg = sb[sb['is_playoffs'] == 0]
    team_avg = (
        reg.groupby(['season', 'team_key'])['team_points']
        .mean().round(2).reset_index(name='opp_avg_ppw')
    )
    team_avg['season'] = team_avg['season'].astype(int)

    m = matchups.copy()
    m['season']         = pd.to_numeric(m['season']).astype(int)
    m['is_playoffs']    = pd.to_numeric(m['is_playoffs']).fillna(0).astype(int)
    m['is_consolation'] = pd.to_numeric(m['is_consolation']).fillna(0).astype(int)
    m = m[(m['is_playoffs'] == 0) & (m['is_consolation'] == 0)]

    m = m.merge(
        team_avg.rename(columns={'team_key': 'opponent_team_key'}),
        on=['season', 'opponent_team_key'], how='left'
    )

    return (
        m.groupby(['season', 'team_id'])['opp_avg_ppw']
        .mean().round(2).reset_index(name='strength_of_schedule')
        .rename(columns={'team_id': 'team_key'})
    )


# ---------------------------------------------------------------------------
# Per-season manager summary
# ---------------------------------------------------------------------------

def build_season_summary(standings, matchups, sb_totals, scoreboard=None,
                         optimal_lineup_df=None, draft_overpf_df=None,
                         trade_value_df=None):
    """One row per manager per season they participated."""

    # --- Base from standings ---
    keep = [
        'season', 'team_key', 'manager_id', 'manager_nickname', 'name',
        'wins', 'losses', 'ties', 'percentage',
        'points_for', 'points_against',
        'rank', 'playoff_seed',
        'draft_position', 'draft_grade',
        'number_of_moves', 'number_of_trades',
    ]
    df = standings[keep].copy()
    df['team_name']     = df['name'].apply(_clean_name)

    # Nicknames shared by >1 manager_id in any single season are ambiguous.
    # '--hidden--' is always ambiguous. For these we append the manager_id.
    per_season_counts = (
        df.groupby(['season', 'manager_nickname'])['manager_id']
        .nunique()
        .reset_index(name='n')
    )
    ambiguous_nicks = set(
        per_season_counts.loc[per_season_counts['n'] > 1, 'manager_nickname']
        .str.strip()
    ) | {'--hidden--'}

    df['manager'] = df.apply(
        lambda r: (
            _make_manager_key(r['manager_nickname'], r['manager_id'])
            if str(r['manager_nickname']).strip() in ambiguous_nicks
            else str(r['manager_nickname']).strip()
        ),
        axis=1,
    )

    # Hidden managers: group by team name rather than manager_id slot, since
    # slot numbers shift each season and team names are more stable identifiers.
    hidden_mask = df['manager_nickname'].str.strip() == '--hidden--'
    df.loc[hidden_mask, 'manager'] = df.loc[hidden_mask, 'team_name'].apply(
        lambda t: _HIDDEN_TEAM_TO_MANAGER.get(t, f'Hidden: {t}')
    )

    # Resolve "Matthew" rows to real names using team name as identifier.
    matthew_mask = df['manager_nickname'].str.strip() == 'Matthew'
    resolved = df.loc[matthew_mask, 'team_name'].map(_MATTHEW_TEAM_TO_MANAGER)
    df.loc[matthew_mask, 'manager'] = resolved.combine_first(df.loc[matthew_mask, 'manager'])

    # Apply display name overrides (e.g. Yahoo nickname differs from real name).
    df['manager'] = df['manager'].replace(_MANAGER_RENAMES)

    df['season']        = _to_numeric(df['season']).astype(int)
    df['wins']          = _to_numeric(df['wins']).fillna(0).astype(int)
    df['losses']        = _to_numeric(df['losses']).fillna(0).astype(int)
    df['ties']          = _to_numeric(df['ties']).fillna(0).astype(int)
    df['points_for']    = _to_numeric(df['points_for'])
    df['points_against']= _to_numeric(df['points_against'])
    df['rank']          = _to_numeric(df['rank'])
    df['playoff_seed']  = _to_numeric(df['playoff_seed'])
    df['draft_position']= _to_numeric(df['draft_position']).astype('Int64')
    df['number_of_moves'] = _to_numeric(df['number_of_moves']).fillna(0).astype(int)
    df['number_of_trades']= _to_numeric(df['number_of_trades']).fillna(0).astype(int)

    df['champion'] = (df['rank'] == 1)

    # --- Playoff wins / losses / appearance from matchups ---
    m = matchups.copy()
    m['season']         = _to_numeric(m['season']).astype(int)
    m['is_playoffs']    = _to_numeric(m['is_playoffs']).fillna(0).astype(int)
    m['is_consolation'] = _to_numeric(m['is_consolation']).fillna(0).astype(int)

    playoff_m = m[(m['is_playoffs'] == 1) & (m['is_consolation'] == 0)].copy()
    playoff_m['won'] = playoff_m['winner_team_key'] == playoff_m['team_id']
    playoff_stats = (
        playoff_m
        .groupby(['season', 'team_id'])
        .agg(playoff_wins=('won', 'sum'), playoff_games=('won', 'count'))
        .reset_index()
        .rename(columns={'team_id': 'team_key'})
    )
    playoff_stats['playoff_losses'] = (
        playoff_stats['playoff_games'] - playoff_stats['playoff_wins']
    )
    playoff_stats['made_playoffs'] = True

    df = df.merge(
        playoff_stats[['season', 'team_key', 'playoff_wins', 'playoff_losses', 'made_playoffs']],
        on=['season', 'team_key'], how='left'
    )
    df['made_playoffs']  = df['made_playoffs'].fillna(False)
    df['playoff_wins']   = df['playoff_wins'].fillna(0).astype(int)
    df['playoff_losses'] = df['playoff_losses'].fillna(0).astype(int)

    # --- Expected wins + avg_ppw_against from scoreboard ---
    if scoreboard is not None and not scoreboard.empty:
        exp = build_expected_record(scoreboard)
        exp['season'] = exp['season'].astype(int)
        df = df.merge(exp, on=['season', 'team_key'], how='left')
        df['exp_wins']   = df['exp_wins'].fillna(0).astype(int)
        df['exp_losses'] = df['exp_losses'].fillna(0).astype(int)
        df['luck_wins']  = df['wins'] - df['exp_wins']

        sb_reg = scoreboard.copy()
        sb_reg['season']     = pd.to_numeric(sb_reg['season'])
        sb_reg['is_playoffs']= pd.to_numeric(sb_reg['is_playoffs'], errors='coerce').fillna(0).astype(int)
        reg_weeks = (
            sb_reg[sb_reg['is_playoffs'] == 0]
            .groupby(['season', 'team_key'])['week']
            .count()
            .reset_index(name='reg_weeks')
        )
        reg_weeks['season'] = reg_weeks['season'].astype(int)
        df = df.merge(reg_weeks, on=['season', 'team_key'], how='left')
        df['avg_ppw_against'] = (df['points_against'] / df['reg_weeks']).round(2)

    # --- Scoring stats from scoreboard_totals ---
    if not sb_totals.empty:
        sb = sb_totals[['season', 'team_key',
                         'total_points', 'total_projected',
                         'points_vs_projected', 'weeks_played']].copy()
        sb['season'] = _to_numeric(sb['season']).astype(int)
        sb = sb.rename(columns={
            'total_points':      'sb_total_points',
            'total_projected':   'sb_total_projected',
            'points_vs_projected': 'luck_pts',
        })
        df = df.merge(sb, on=['season', 'team_key'], how='left')
        df['avg_pts_per_week'] = (
            df['sb_total_points'] / df['weeks_played']
        ).round(2)

    # --- Strength of schedule ---
    if scoreboard is not None and not scoreboard.empty:
        sos = build_strength_of_schedule(scoreboard, matchups)
        if not sos.empty:
            sos['season'] = sos['season'].astype(int)
            df = df.merge(sos, on=['season', 'team_key'], how='left')

    # --- Optimal lineup / points left on bench ---
    if optimal_lineup_df is not None and not optimal_lineup_df.empty:
        ol = optimal_lineup_df[['season', 'team_key',
                                 'total_pts_left_on_bench',
                                 'avg_pts_left_per_week']].copy()
        ol['season'] = ol['season'].astype(int)
        df = df.merge(ol, on=['season', 'team_key'], how='left')

    # --- Draft overperformance ---
    if draft_overpf_df is not None and not draft_overpf_df.empty:
        dop = draft_overpf_df[['season', 'team_key',
                                'total_draft_pts', 'expected_draft_pts',
                                'draft_overperformance']].copy()
        dop['season'] = dop['season'].astype(int)
        df = df.merge(dop, on=['season', 'team_key'], how='left')

    # --- Trade value ---
    if trade_value_df is not None and not trade_value_df.empty:
        tv = trade_value_df[['season', 'team_key',
                              'trade_pts_received', 'trade_pts_given',
                              'trade_value']].copy()
        tv['season'] = tv['season'].astype(int)
        df = df.merge(tv, on=['season', 'team_key'], how='left')

    col_order = [
        'season', 'manager', 'team_name',
        'wins', 'losses', 'ties', 'percentage',
        'points_for', 'points_against',
        'rank', 'playoff_seed', 'made_playoffs', 'champion',
        'playoff_wins', 'playoff_losses',
        'draft_position', 'draft_grade',
        'number_of_moves', 'number_of_trades',
    ]
    if 'exp_wins' in df.columns:
        col_order += ['exp_wins', 'exp_losses', 'luck_wins', 'avg_ppw_against', 'reg_weeks']
    if 'avg_pts_per_week' in df.columns:
        col_order += ['avg_pts_per_week', 'luck_pts', 'weeks_played']
    if 'strength_of_schedule' in df.columns:
        col_order += ['strength_of_schedule']
    if 'avg_pts_left_per_week' in df.columns:
        col_order += ['total_pts_left_on_bench', 'avg_pts_left_per_week']
    if 'draft_overperformance' in df.columns:
        col_order += ['total_draft_pts', 'expected_draft_pts', 'draft_overperformance']
    if 'trade_value' in df.columns:
        col_order += ['trade_pts_received', 'trade_pts_given', 'trade_value']

    return (
        df[col_order]
        .sort_values(['manager', 'season'])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Career summary
# ---------------------------------------------------------------------------

def build_career_summary(season_df):
    """One row per manager — lifetime aggregates across all seasons."""
    g = season_df.groupby('manager')

    career = g.agg(
        seasons_played  = ('season',        'count'),
        first_season    = ('season',        'min'),
        last_season     = ('season',        'max'),
        total_wins      = ('wins',          'sum'),
        total_losses    = ('losses',        'sum'),
        total_ties      = ('ties',          'sum'),
        championships   = ('champion',      'sum'),
        playoff_apps    = ('made_playoffs', 'sum'),
        playoff_wins    = ('playoff_wins',  'sum'),
        playoff_losses  = ('playoff_losses','sum'),
        best_finish     = ('rank',          'min'),
        avg_finish      = ('rank',          'mean'),
        total_moves     = ('number_of_moves',  'sum'),
        total_trades    = ('number_of_trades', 'sum'),
    ).reset_index().rename(columns={'manager': 'manager'})

    if 'exp_wins' in season_df.columns:
        exp = (
            season_df.groupby('manager')
            .agg(career_exp_wins=('exp_wins', 'sum'), career_exp_losses=('exp_losses', 'sum'))
            .reset_index()
        )
        career = career.merge(exp, on='manager', how='left')
        career['career_luck_wins'] = career['total_wins'] - career['career_exp_wins']

    if 'avg_ppw_against' in season_df.columns:
        against = (
            season_df.groupby('manager')
            .apply(lambda g: (g['points_against'].sum() / g['reg_weeks'].sum()).round(2))
            .reset_index(name='career_avg_ppw_against')
        )
        career = career.merge(against, on='manager', how='left')

    career['win_pct'] = (
        career['total_wins'] /
        (career['total_wins'] + career['total_losses'] + career['total_ties'])
    ).round(3)
    career['avg_finish'] = career['avg_finish'].round(1)

    if 'avg_pts_per_week' in season_df.columns:
        pts = (
            season_df.groupby('manager')['avg_pts_per_week']
            .mean().round(2).rename('career_avg_ppw')
        )
        career = career.merge(pts, on='manager', how='left')

        total_pts = (
            season_df.groupby('manager')['points_for']
            .sum().round(2).rename('career_total_pts')
        )
        career = career.merge(total_pts, on='manager', how='left')

    if 'strength_of_schedule' in season_df.columns:
        sos = (
            season_df.groupby('manager')['strength_of_schedule']
            .mean().round(2).rename('career_sos')
        )
        career = career.merge(sos, on='manager', how='left')

    if 'avg_pts_left_per_week' in season_df.columns:
        bench_waste = (
            season_df.groupby('manager')['avg_pts_left_per_week']
            .mean().round(2).rename('career_avg_pts_left_per_week')
        )
        career = career.merge(bench_waste, on='manager', how='left')

    if 'draft_overperformance' in season_df.columns:
        dop = (
            season_df.groupby('manager')['draft_overperformance']
            .sum().round(1).rename('career_draft_overperformance')
        )
        career = career.merge(dop, on='manager', how='left')

    if 'trade_value' in season_df.columns:
        tv = (
            season_df.groupby('manager')['trade_value']
            .sum().round(1).rename('career_trade_value')
        )
        career = career.merge(tv, on='manager', how='left')

    return (
        career
        .sort_values(['championships', 'avg_finish'], ascending=[False, True])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Keeper era split summary
# ---------------------------------------------------------------------------

KEEPER_START = 2016

def build_keeper_era_summary(season_df):
    """One row per (manager, is_keeper) for managers who played in the keeper era.

    is_keeper=1 : seasons >= 2019  |  is_keeper=0 : seasons < 2019
    Managers who never played before 2019 only appear with is_keeper=1.
    """
    keeper_managers = season_df.loc[season_df['season'] >= KEEPER_START, 'manager'].unique()
    df = season_df[season_df['manager'].isin(keeper_managers)].copy()
    df['is_keeper'] = (df['season'] >= KEEPER_START).astype(int)

    g = df.groupby(['manager', 'is_keeper'])

    summary = g.agg(
        seasons_played  = ('season',        'count'),
        first_season    = ('season',        'min'),
        last_season     = ('season',        'max'),
        total_wins      = ('wins',          'sum'),
        total_losses    = ('losses',        'sum'),
        total_ties      = ('ties',          'sum'),
        championships   = ('champion',      'sum'),
        playoff_apps    = ('made_playoffs', 'sum'),
        playoff_wins    = ('playoff_wins',  'sum'),
        playoff_losses  = ('playoff_losses','sum'),
        best_finish     = ('rank',          'min'),
        avg_finish      = ('rank',          'mean'),
        total_moves     = ('number_of_moves',  'sum'),
        total_trades    = ('number_of_trades', 'sum'),
        total_points_for= ('points_for',    'sum'),
    ).reset_index()

    summary['win_pct'] = (
        summary['total_wins'] /
        (summary['total_wins'] + summary['total_losses'] + summary['total_ties'])
    ).round(3)
    summary['avg_finish'] = summary['avg_finish'].round(1)

    if 'avg_pts_per_week' in df.columns:
        ppw = (
            df.groupby(['manager', 'is_keeper'])['avg_pts_per_week']
            .mean().round(2).rename('avg_ppw')
        )
        summary = summary.merge(ppw, on=['manager', 'is_keeper'], how='left')

    if 'exp_wins' in df.columns:
        exp = (
            df.groupby(['manager', 'is_keeper'])
            .agg(exp_wins=('exp_wins', 'sum'), exp_losses=('exp_losses', 'sum'))
            .reset_index()
        )
        summary = summary.merge(exp, on=['manager', 'is_keeper'], how='left')
        summary['luck_wins'] = summary['total_wins'] - summary['exp_wins']
        summary['exp_win_rate'] = (
            summary['exp_wins'] /
            (summary['exp_wins'] + summary['exp_losses'])
        ).round(3)

    if 'avg_pts_left_per_week' in df.columns:
        bench = (
            df.groupby(['manager', 'is_keeper'])['avg_pts_left_per_week']
            .mean().round(2).rename('avg_pts_left_per_week')
        )
        summary = summary.merge(bench, on=['manager', 'is_keeper'], how='left')

    if 'draft_overperformance' in df.columns:
        dop = (
            df.groupby(['manager', 'is_keeper'])['draft_overperformance']
            .sum().round(1).rename('draft_overperformance')
        )
        summary = summary.merge(dop, on=['manager', 'is_keeper'], how='left')

    if 'strength_of_schedule' in df.columns:
        sos = (
            df.groupby(['manager', 'is_keeper'])['strength_of_schedule']
            .mean().round(2).rename('avg_sos')
        )
        summary = summary.merge(sos, on=['manager', 'is_keeper'], how='left')

    return (
        summary
        .sort_values(['manager', 'is_keeper'])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Keeper listing
# ---------------------------------------------------------------------------

# Players confirmed by league members to NOT be keepers in an ambiguous year.
# Keyed by (season, manager); values are player names to exclude from candidates.
_KEEPER_DROPS = {
    (2016, 'Bennett'):       ['Alfred Morris'],
    (2016, 'Derek'):         ['Justin Forsett', 'Chris Ivory'],
    (2016, 'Erik'):          ['Michael Crabtree'],
    (2016, 'Gucker'):        ['Delanie Walker'],
    (2016, 'kevin mc'):      ['Latavius Murray'],
    (2017, 'Jacob'):         ['Philip Rivers', 'Corey Coleman'],
    (2017, 'John Stockton'): ['Lamar Miller', 'Rishard Matthews'],
    (2017, 'K&E'):           ['Mark Ingram II'],
    (2017, 'kevin mc'):      ['Russell Wilson', 'Zach Ertz'],
    (2018, 'Branden'):       ['Christian McCaffrey'],
    (2018, 'Jacob'):         ['Antonio Brown', 'Tevin Coleman'],
    (2018, 'John Stockton'): ['Tarik Cohen', 'Latavius Murray'],
    (2018, 'K&E'):           ['Sammy Watkins'],
    (2021, 'Endler'):        ['Ezekiel Elliott'],
    (2021, 'Gucker'):        ['Tyreek Hill'],
    (2021, 'K&E'):           ['Chris Godwin Jr.'],
    (2022, 'Endler'):        ['Robert Woods'],
    (2022, 'John Stockton'): ['Alvin Kamara'],
    (2023, 'Bennett'):       ['Stefon Diggs'],
    (2023, 'Derek'):         ['Tee Higgins'],
    (2023, 'Endler'):        ['George Pickens', 'Raheem Mostert'],
    (2023, 'John Stockton'): ['DK Metcalf', 'David Montgomery'],
    (2024, 'Jacob'):         ['James Cook III', 'George Kittle'],
}

# Keepers that the algorithm can't derive from roster/draft matching alone.
# These are manually confirmed by league members.
# Keyed by (season, manager); values are (player_name, round) tuples.
_KEEPER_FORCED = {
    (2016, 'Derek'):         [('Jordan Matthews', 4)],
    (2016, 'Erik'):          [('Carson Palmer', 3)],
    (2016, 'Jacob'):         [('Devonta Freeman', 2)],
    (2016, 'kevin mc'):      [('Todd Gurley', 2)],
    (2017, 'Bennett'):       [('David Johnson', 12)],
    (2017, 'Gucker'):        [('Jordy Nelson', 10)],
    (2017, 'Jacob'):         [('Devonta Freeman', 9)],
    (2017, 'K&E'):           [('DeSean Jackson', 10)],
    (2017, 'kevin mc'):      [('Todd Gurley', 7)],
    (2018, 'Gucker'):        [('Greg Olsen', 5)],
    (2018, 'Jacob'):         [('Corey Davis', 12)],
    (2020, 'kevin mc'):      [('Calvin Ridley', 9)],
    (2023, 'Branden'):       [('Calvin Ridley', 9)],
    (2023, 'Endler'):        [('Austin Ekeler', 1)],
    (2024, 'John Stockton'): [('Jameson Williams', 9)],
}


def _normalize_pid(player_key):
    """Strip league/game prefix from player_key, return just the numeric ID."""
    return str(player_key).split('.')[-1]


def _resolve_manager_name(nickname, team_name):
    """Return canonical manager display name."""
    nick = str(nickname).strip()
    clean = _clean_name(team_name)
    if nick == 'Matthew':
        return _MATTHEW_TEAM_TO_MANAGER.get(clean, 'Matthew')
    if nick == '--hidden--':
        return _HIDDEN_TEAM_TO_MANAGER.get(clean, f'Hidden ({clean})')
    return _MANAGER_RENAMES.get(nick, nick)


def list_keepers_by_year(standings_df, weekly_rosters_df, draft_results_df,
                         max_keepers=2):
    """Print each manager's keepers per keeper-era season (KEEPER_START onward).

    Keeper identification rules (applied in priority order):
      1. Player was on same team at end of Y-1 AND drafted by same team in Y
         at the SAME round they were drafted in Y-1 (or Rd 9/10 if they were
         a waiver pickup — not in the Y-1 draft at all).
      2. If more than mk pass rule 1, take mk with the smallest
         |curr_round - prev_round| diff, then lowest curr_round.
         (mk = 3 for 2016, max_keepers for all other years)
      3. If fewer than mk pass rule 1, expand by allowing ±2 round
         tolerance to catch slight keeper-cost adjustments.
    """
    if weekly_rosters_df.empty or standings_df.empty or draft_results_df.empty:
        print('  (weekly_rosters, standings, or draft data missing)')
        return

    wr = weekly_rosters_df.copy()
    wr['pid']  = wr['player_key'].apply(_normalize_pid)
    wr['seas'] = pd.to_numeric(wr['season']).astype(int)
    wr['wk']   = pd.to_numeric(wr['week']).astype(int)

    dr = draft_results_df.copy()
    dr['pid']    = dr['player_key'].apply(_normalize_pid)
    dr['season'] = pd.to_numeric(dr['season']).astype(int)
    dr['round']  = pd.to_numeric(dr['round'], errors='coerce').astype('Int64')
    dr = dr.rename(columns={'team_id': 'team_key'})

    st = standings_df.copy()
    st['season'] = pd.to_numeric(st['season']).astype(int)
    name_col = 'name' if 'name' in st.columns else 'team_name'

    def _mgr_map(yr):
        rows = st[st['season'] == yr]
        m = {}
        for _, r in rows.iterrows():
            mgr = _resolve_manager_name(r['manager_nickname'], r.get(name_col, ''))
            m[mgr] = r['team_key']
        return m

    max_week = wr.groupby('seas')['wk'].max().to_dict()
    keeper_seasons = [y for y in sorted(wr['seas'].unique()) if y >= KEEPER_START]

    all_rows = []
    for year in keeper_seasons:
        prev = year - 1
        if prev not in max_week:
            continue

        prev_map = _mgr_map(prev)
        curr_map = _mgr_map(year)

        end_wk      = max_week[prev]
        prev_ros    = wr[(wr['seas'] == prev) & (wr['wk'] == end_wk)]
        curr_draft  = dr[dr['season'] == year]
        prev_draft  = dr[dr['season'] == prev]

        mk = 3 if year == 2016 else max_keepers

        for manager in sorted(curr_map.keys()):
            if manager not in prev_map:
                continue
            prev_tk = prev_map[manager]
            curr_tk = curr_map[manager]

            # Players on prev end-of-season roster (excluding DEF/K)
            prev_pids = set(
                prev_ros[(prev_ros['team_key'] == prev_tk) &
                         (~prev_ros['position'].isin(['DEF', 'K']))]['pid']
            )

            # This team's draft picks in year Y that overlap prev roster
            team_picks = curr_draft[
                (curr_draft['team_key'] == curr_tk) &
                (curr_draft['pid'].isin(prev_pids)) &
                (~curr_draft['player_position'].isin(['DEF', 'K']))
            ][['pid', 'player_name', 'round']].copy()

            if team_picks.empty:
                continue

            # Apply manual overrides: drop confirmed non-keepers
            drops = _KEEPER_DROPS.get((year, manager), [])
            if drops:
                team_picks = team_picks[~team_picks['player_name'].isin(drops)]
            if team_picks.empty:
                continue

            # Prev-year draft round for each player (None = waiver pickup)
            team_prev_draft = prev_draft[prev_draft['team_key'] == prev_tk]
            pid_to_prev_rd  = team_prev_draft.set_index('pid')['round'].to_dict()

            def _rd_diff(row):
                prev_rd = pid_to_prev_rd.get(row['pid'])
                curr_rd = int(row['round']) if pd.notna(row['round']) else None
                if curr_rd is None:
                    return 999
                if prev_rd is None:
                    # Waiver pickup: "keeper cost" is Rd 9 or 10
                    return 0 if curr_rd in (9, 10) else abs(curr_rd - 9)
                return abs(curr_rd - int(prev_rd))

            team_picks['rd_diff'] = team_picks.apply(_rd_diff, axis=1)
            team_picks['curr_rd'] = team_picks['round'].apply(
                lambda x: int(x) if pd.notna(x) else 999
            )

            # Take up to max_keepers: priority = smallest rd_diff, then smallest round
            team_picks = team_picks.sort_values(['rd_diff', 'curr_rd'])

            # Include exact matches + waiver hits first; if < mk, relax to ±2
            exact = team_picks[team_picks['rd_diff'] == 0]
            if len(exact) >= mk:
                selected = exact.head(mk)
            elif len(exact) > 0:
                near = team_picks[team_picks['rd_diff'] <= 2].head(mk)
                selected = near
            else:
                selected = team_picks.head(mk)

            for _, row in selected.iterrows():
                all_rows.append({
                    'season':  year,
                    'manager': manager,
                    'player':  row['player_name'],
                    'round':   row['curr_rd'],
                })

            # Append any manually confirmed keepers not found by the algorithm
            for player_name, rd in _KEEPER_FORCED.get((year, manager), []):
                all_rows.append({
                    'season':  year,
                    'manager': manager,
                    'player':  player_name,
                    'round':   rd,
                })

    if not all_rows:
        print('  No keepers found.')
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    for year in sorted(df['season'].unique()):
        print(f'\n=== {year} Keepers ===')
        sdf = df[df['season'] == year]
        for manager in sorted(sdf['manager'].unique()):
            mdf = sdf[sdf['manager'] == manager].sort_values('round')
            parts = [f"{r['player']} (Rd {r['round']})" for _, r in mdf.iterrows()]
            print(f'  {manager:<16} {", ".join(parts)}')
    return df


# ---------------------------------------------------------------------------
# Keeper decision analysis
# ---------------------------------------------------------------------------

def _compute_season_totals(weekly_rosters_df):
    """Return (totals_df, pid→name dict). totals_df has cols: seas, pid, pts."""
    wr = weekly_rosters_df.copy()
    wr['pid'] = wr['player_key'].apply(_normalize_pid)
    wr['seas'] = pd.to_numeric(wr['season']).astype(int)
    wr['pts'] = pd.to_numeric(wr['player_points'], errors='coerce').fillna(0)
    totals = wr.groupby(['seas', 'pid'])['pts'].sum().reset_index()
    name_df = wr.sort_values('seas').drop_duplicates('pid', keep='last')[['pid', 'name']]
    return totals, dict(zip(name_df['pid'], name_df['name']))


def _compute_round_benchmarks(draft_results_df, season_totals):
    """Average total-season pts by (season, round) for all non-DEF/K picks."""
    dr = draft_results_df.copy()
    dr['pid'] = dr['player_key'].apply(_normalize_pid)
    dr['season'] = pd.to_numeric(dr['season']).astype(int)
    dr['round'] = pd.to_numeric(dr['round'], errors='coerce')
    dr = dr[~dr['player_position'].isin(['DEF', 'K'])].dropna(subset=['round'])
    dr['round'] = dr['round'].astype(int)
    merged = dr.merge(
        season_totals.rename(columns={'seas': 'season'}),
        on=['season', 'pid'], how='left'
    )
    merged['pts'] = merged['pts'].fillna(0)
    bench = merged.groupby(['season', 'round'])['pts'].mean().reset_index()
    bench.columns = ['season', 'round', 'avg_pts']
    return bench


def _compute_positional_rank_benchmarks(draft_results_df, season_totals, weekly_rosters_df):
    """Return two dicts for positional-rank keeper benchmarking.

    pos_draft_count[(year, pos, round)]:
        Number of pos-position players drafted in rounds 1..round in year N.
        The keeper's positional rank is this count + 1.

    pos_pts_rank[(year, pos, K)]:
        Season pts of the Kth-highest-scoring pos-position player in year N,
        across ALL rostered players (not just drafted) for a fair benchmark.
    """
    # --- pos_draft_count: drafted players only (determines positional rank K) ---
    dr = draft_results_df.copy()
    dr['pid'] = dr['player_key'].apply(_normalize_pid)
    dr['season'] = pd.to_numeric(dr['season']).astype(int)
    dr['round'] = pd.to_numeric(dr['round'], errors='coerce')
    dr = dr[~dr['player_position'].isin(['DEF', 'K'])].dropna(subset=['round'])
    dr['round'] = dr['round'].astype(int)
    dr['pos'] = dr['player_position'].replace({'FB': 'RB'})

    pos_draft_count = {}
    for (year, pos), grp in dr.groupby(['season', 'pos']):
        rounds = grp['round'].tolist()
        max_rd = max(rounds)
        for r in range(1, max_rd + 1):
            pos_draft_count[(year, pos, r)] = sum(1 for rd in rounds if rd <= r)

    # --- pos_pts_rank: all rostered players (broader, fairer benchmark pool) ---
    wr = weekly_rosters_df.copy()
    wr['pid'] = wr['player_key'].apply(_normalize_pid)
    wr['season'] = pd.to_numeric(wr['season']).astype(int)
    wr['pos'] = wr['position'].replace({'FB': 'RB'})
    wr = wr[~wr['pos'].isin(['DEF', 'K', ''])]
    pid_pos = (wr.groupby(['season', 'pid'])['pos']
               .first().reset_index()
               .rename(columns={'season': 'seas'}))

    all_pts = season_totals.merge(pid_pos, on=['seas', 'pid'], how='inner')

    pos_pts_rank = {}
    for (year, pos), grp in all_pts.groupby(['seas', 'pos']):
        for k, pts in enumerate(sorted(grp['pts'].tolist(), reverse=True), 1):
            pos_pts_rank[(year, pos, k)] = pts

    return pos_draft_count, pos_pts_rank


def analyze_keeper_decisions(standings_df, weekly_rosters_df, draft_results_df,
                              max_keepers=2):
    """For each keeper year and manager, show the full eligible roster ranked by
    value-over-replacement (player pts in year N minus avg pts for their keeper round
    in year N).  Optimal keepers = top mk by value, not raw pts.
    """
    keeper_df = list_keepers_by_year(
        standings_df, weekly_rosters_df, draft_results_df, max_keepers
    )
    if keeper_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    season_totals, pid_name = _compute_season_totals(weekly_rosters_df)
    round_bench = _compute_round_benchmarks(draft_results_df, season_totals)
    bench_lkp = round_bench.set_index(['season', 'round'])['avg_pts'].to_dict()
    pos_draft_count, pos_pts_rank = _compute_positional_rank_benchmarks(draft_results_df, season_totals, weekly_rosters_df)
    st_lkp = season_totals.set_index(['seas', 'pid'])['pts'].to_dict()

    dr_all = draft_results_df.copy()
    dr_all['pid'] = dr_all['player_key'].apply(_normalize_pid)
    name_to_pid = {}
    for _, r in dr_all.iterrows():
        n = r.get('player_name', '')
        if n and n not in name_to_pid:
            name_to_pid[n] = r['pid']
    for pid, name in pid_name.items():
        if name and name not in name_to_pid:
            name_to_pid[name] = pid

    wr = weekly_rosters_df.copy()
    wr['pid'] = wr['player_key'].apply(_normalize_pid)
    wr['seas'] = pd.to_numeric(wr['season']).astype(int)
    wr['wk'] = pd.to_numeric(wr['week']).astype(int)
    max_week = wr.groupby('seas')['wk'].max().to_dict()

    dr2 = dr_all.copy()
    dr2['season'] = pd.to_numeric(dr2['season']).astype(int)
    dr2['round'] = pd.to_numeric(dr2['round'], errors='coerce').astype('Int64')
    dr2 = dr2.rename(columns={'team_id': 'team_key'})

    st_df = standings_df.copy()
    st_df['season'] = pd.to_numeric(st_df['season']).astype(int)
    name_col = 'name' if 'name' in st_df.columns else 'team_name'

    def mgr_map(yr):
        m = {}
        for _, r in st_df[st_df['season'] == yr].iterrows():
            mgr = _resolve_manager_name(r['manager_nickname'], r.get(name_col, ''))
            m[mgr] = r['team_key']
        return m

    keeper_seasons = sorted(keeper_df['season'].unique())
    summary_rows = []
    detail_rows  = []

    for year in keeper_seasons:
        prev = year - 1
        if prev not in max_week:
            continue

        mk = 3 if year == 2016 else max_keepers
        prev_map = mgr_map(prev)
        curr_map = mgr_map(year)
        prev_roster = wr[(wr['seas'] == prev) & (wr['wk'] == max_week[prev])]
        prev_draft = dr2[dr2['season'] == prev]

        y2 = keeper_df[keeper_df['season'] == year - 2].groupby('manager')['player'].apply(set).to_dict()
        y1 = keeper_df[keeper_df['season'] == year - 1].groupby('manager')['player'].apply(set).to_dict()
        ineligible = {
            mgr: y2.get(mgr, set()) & y1.get(mgr, set())
            for mgr in set(y2) | set(y1)
        }

        actual_year = keeper_df[keeper_df['season'] == year]

        print(f'\n{"=" * 80}')
        print(f'  {year} KEEPER ANALYSIS  '
              f'(value = player pts minus Nth-best same-pos pts in {year}, where N = positional draft rank)')
        print(f'{"=" * 80}')

        hdr = f'  {"Player":<30} {"Pos":>3}  {"Rd":>3}  {"PosRk":>5}  {"Pts":>6}  {"Benchmark":>9}  {"Value":>7}  Status'
        sep = f'  {"-"*30}  {"-"*3}  {"-"*3}  {"-"*5}  {"-"*6}  {"-"*9}  {"-"*7}  {"-"*12}'

        for manager in sorted(curr_map.keys()):
            if manager not in prev_map:
                continue
            prev_tk = prev_map[manager]

            ros = prev_roster[
                (prev_roster['team_key'] == prev_tk) &
                (~prev_roster['position'].isin(['DEF', 'K']))
            ][['pid', 'name', 'position']].drop_duplicates('pid')
            if ros.empty:
                continue

            pid_to_prev_rd = (prev_draft[prev_draft['team_key'] == prev_tk]
                              .set_index('pid')['round'].to_dict())
            mgr_ineligible = ineligible.get(manager, set())

            rows = []
            for _, row in ros.iterrows():
                pid, pname = row['pid'], row['name']
                pos = str(row.get('position', '')).replace('FB', 'RB')
                inelig = pname in mgr_ineligible
                prev_rd = pid_to_prev_rd.get(pid)
                k_rd = 9 if (prev_rd is None or pd.isna(prev_rd)) else int(prev_rd)
                pts = st_lkp.get((year, pid), 0)
                k_drafted = pos_draft_count.get((year, pos, k_rd), 0)
                pos_rank = k_drafted + 1
                pos_avg = pos_pts_rank.get((year, pos, pos_rank),
                          bench_lkp.get((year, k_rd), 0))
                rows.append({
                    'name': pname, 'pos': pos, 'k_rd': k_rd, 'pts': pts,
                    'pos_rank': pos_rank, 'pos_avg': pos_avg,
                    'value': pts - pos_avg,
                    'ineligible': inelig,
                })

            if not rows:
                continue

            elig_df = (pd.DataFrame(rows)
                       .sort_values('value', ascending=False)
                       .reset_index(drop=True))

            actual_names = set(actual_year[actual_year['manager'] == manager]['player'])
            # Optimal = top mk by value among eligible (not ineligible)
            opt_pool = elig_df[~elig_df['ineligible']]
            optimal_names = set(opt_pool.head(mk)['name'])

            actual_value = sum(
                elig_df.loc[elig_df['name'] == p, 'value'].values[0]
                if p in elig_df['name'].values else 0
                for p in actual_names
            )
            optimal_value = opt_pool.head(mk)['value'].sum()
            gap = optimal_value - actual_value
            gap_str = f'+{gap:.1f}' if gap >= 0 else f'{gap:.1f}'

            summary_rows.append({
                'season':          year,
                'manager':         manager,
                'actual_value':    round(actual_value, 1),
                'optimal_value':   round(float(optimal_value), 1),
                'gap':             round(gap, 1),
                'is_optimal':      optimal_names == actual_names,
                'actual_keepers':  ', '.join(sorted(actual_names)),
                'optimal_keepers': ', '.join(sorted(optimal_names)),
            })

            verdict = 'OPTIMAL' if optimal_names == actual_names else f'gap {gap_str} vs optimal'
            print(f'  {manager}  [{verdict}]')
            print(hdr)
            print(sep)

            for _, r in elig_df.iterrows():
                pname = r['name']
                is_kept = pname in actual_names
                is_opt  = pname in optimal_names
                is_inelig = r['ineligible']

                if is_inelig:
                    status = 'INELIGIBLE'
                elif is_kept and is_opt:
                    status = 'KEPT  OPT'
                elif is_kept:
                    status = 'KEPT  ---'
                elif is_opt:
                    status = '---   OPT'
                else:
                    status = ''

                v_str = f'+{r["value"]:.1f}' if r['value'] >= 0 else f'{r["value"]:.1f}'
                print(f'  {pname:<30} {r["pos"]:>3}  {r["k_rd"]:>3}  {r["pos_rank"]:>5}  '
                      f'{r["pts"]:>6.1f}  {r["pos_avg"]:>9.1f}  {v_str:>7}  {status}')
                detail_rows.append({
                    'season':       year,
                    'manager':      manager,
                    'player':       pname,
                    'pos':          r['pos'],
                    'rd':           int(r['k_rd']),
                    'pos_rank':     int(r['pos_rank']),
                    'pts':          round(float(r['pts']), 1),
                    'benchmark':    round(float(r['pos_avg']), 1),
                    'value':        round(float(r['value']), 1),
                    'is_kept':      bool(is_kept),
                    'is_optimal':   bool(is_opt),
                    'is_ineligible': bool(is_inelig),
                })
            print()

    return (
        pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(),
        pd.DataFrame(detail_rows)  if detail_rows  else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    """Run the full analytics pipeline: load CSVs, compute metrics, build summaries."""
    from convert_to_df import (
        load_all_csvs,
        load_standings as _load_standings_raw,
        load_weekly_rosters,
        load_draft_results,
        load_trades,
    )

    print('Loading base CSVs...')
    standings, matchups, sb_totals, scoreboard = load_all_csvs()

    print('Loading raw data for analytics...')
    weekly_rosters_df = load_weekly_rosters()
    draft_results_df  = load_draft_results()
    trades_df         = load_trades()

    print('Computing analytics...')
    optimal_lineup = compute_optimal_lineup_stats(weekly_rosters_df)
    draft_overpf   = compute_draft_overperformance(draft_results_df, weekly_rosters_df)
    trade_value    = compute_trade_value(trades_df, weekly_rosters_df)

    for name, df in [
        ('optimal_lineup_df', optimal_lineup),
        ('draft_overperf_df', draft_overpf),
        ('trade_value_df',    trade_value),
    ]:
        if not df.empty:
            df.to_csv(DATA_DIR / f'{name}.csv', index=False)

    print('Building season summary...')
    season_df = build_season_summary(standings, matchups, sb_totals, scoreboard,
                                     optimal_lineup, draft_overpf, trade_value)

    print('Building career summary...')
    career_df = build_career_summary(season_df)

    print('Building keeper era summary...')
    keeper_era_df = build_keeper_era_summary(season_df)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', '{:.2f}'.format)

    print('\n=== Season-by-Season Summary ===')
    print(season_df.to_string(index=False))

    print('\n\n=== Career Summary ===')
    print(career_df.to_string(index=False))

    print('\n\n=== Keeper Era Split Summary ===')
    print(keeper_era_df.to_string(index=False))

    season_df.to_csv(DATA_DIR / 'manager_season_df.csv', index=False)
    career_df.to_csv(DATA_DIR / 'manager_career_df.csv', index=False)
    keeper_era_df.to_csv(DATA_DIR / 'manager_keeper_era_df.csv', index=False)
    print('\nSaved manager_season_df.csv, manager_career_df.csv, manager_keeper_era_df.csv')

    raw_standings = _load_standings_raw()
    print('\n\n=== Keeper Decision Analysis ===')
    keeper_summary_df, keeper_detail_df = analyze_keeper_decisions(
        raw_standings, weekly_rosters_df, draft_results_df
    )
    if not keeper_summary_df.empty:
        keeper_summary_df.to_csv(DATA_DIR / 'keeper_summary_df.csv', index=False)
        keeper_detail_df.to_csv(DATA_DIR / 'keeper_detail_df.csv', index=False)
        print('Saved keeper_summary_df.csv, keeper_detail_df.csv')

    return season_df, career_df, keeper_era_df


if __name__ == '__main__':
    run()
