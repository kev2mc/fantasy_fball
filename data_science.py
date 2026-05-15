import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')

# Maps team names used by "Matthew" managers to their real identities.
# Two distinct people named Matthew have been in the league; team names
# distinguish them since Yahoo manager_ids shift each season.
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
# Load
# ---------------------------------------------------------------------------

def load_data():
    standings  = pd.read_csv(DATA_DIR / 'standings_df.csv')
    matchups   = pd.read_csv(DATA_DIR / 'matchups_df.csv')
    sb_totals_path = DATA_DIR / 'scoreboard_totals_df.csv'
    sb_totals  = pd.read_csv(sb_totals_path) if sb_totals_path.exists() else pd.DataFrame()
    return standings, matchups, sb_totals


# ---------------------------------------------------------------------------
# Per-season manager summary
# ---------------------------------------------------------------------------

def build_season_summary(standings, matchups, sb_totals):
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

    # Resolve "Matthew" rows to real names using team name as identifier.
    matthew_mask = df['manager_nickname'].str.strip() == 'Matthew'
    resolved = df.loc[matthew_mask, 'team_name'].map(_MATTHEW_TEAM_TO_MANAGER)
    df.loc[matthew_mask, 'manager'] = resolved.combine_first(df.loc[matthew_mask, 'manager'])

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

    df['made_playoffs'] = df['playoff_seed'].notna() & (df['playoff_seed'] > 0)
    df['champion']      = (df['rank'] == 1)

    # --- Playoff wins / losses from matchups ---
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
    df = df.merge(
        playoff_stats[['season', 'team_key', 'playoff_wins', 'playoff_losses']],
        on=['season', 'team_key'], how='left'
    )
    df['playoff_wins']   = df['playoff_wins'].fillna(0).astype(int)
    df['playoff_losses'] = df['playoff_losses'].fillna(0).astype(int)

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

    col_order = [
        'season', 'manager', 'team_name',
        'wins', 'losses', 'ties', 'percentage',
        'points_for', 'points_against',
        'rank', 'playoff_seed', 'made_playoffs', 'champion',
        'playoff_wins', 'playoff_losses',
        'draft_position', 'draft_grade',
        'number_of_moves', 'number_of_trades',
    ]
    if 'avg_pts_per_week' in df.columns:
        col_order += ['avg_pts_per_week', 'luck_pts', 'weeks_played']

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

    career['win_pct'] = (
        career['total_wins'] /
        (career['total_wins'] + career['total_losses'] + career['total_ties'])
    ).round(3)
    career['avg_finish'] = career['avg_finish'].round(1)

    if 'avg_pts_per_week' in season_df.columns:
        pts = (
            season_df.groupby('manager')['avg_pts_per_week']
            .mean()
            .round(2)
            .rename('career_avg_ppw')
        )
        career = career.merge(pts, on='manager', how='left')

        total_pts = (
            season_df.groupby('manager')['points_for']
            .sum()
            .round(2)
            .rename('career_total_pts')
        )
        career = career.merge(total_pts, on='manager', how='left')

    return (
        career
        .sort_values(['championships', 'avg_finish'], ascending=[False, True])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading data...')
    standings, matchups, sb_totals = load_data()

    print('Building season summary...')
    season_df = build_season_summary(standings, matchups, sb_totals)

    print('Building career summary...')
    career_df = build_career_summary(season_df)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', '{:.2f}'.format)

    print('\n=== Season-by-Season Summary ===')
    print(season_df.to_string(index=False))

    print('\n\n=== Career Summary ===')
    print(career_df.to_string(index=False))

    season_df.to_csv(DATA_DIR / 'manager_season_df.csv', index=False)
    career_df.to_csv(DATA_DIR  / 'manager_career_df.csv',  index=False)
    print('\nSaved manager_season_df.csv and manager_career_df.csv')
