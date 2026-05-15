from yfpy.query import YahooFantasySportsQuery
from pathlib import Path
import json
import os
import logging
import argparse
import sys

logging.basicConfig(level=logging.WARNING)

# Maps Yahoo's short DEF nickname to city/region so draft + roster entries read
# as "Detroit Lions D/ST" rather than just "Lions".
_NFL_CITY = {
    '49ers':      'San Francisco',
    'Bears':      'Chicago',
    'Bengals':    'Cincinnati',
    'Bills':      'Buffalo',
    'Broncos':    'Denver',
    'Browns':     'Cleveland',
    'Buccaneers': 'Tampa Bay',
    'Cardinals':  'Arizona',
    'Chargers':   'Los Angeles',
    'Chiefs':     'Kansas City',
    'Colts':      'Indianapolis',
    'Commanders': 'Washington',
    'Cowboys':    'Dallas',
    'Dolphins':   'Miami',
    'Eagles':     'Philadelphia',
    'Falcons':    'Atlanta',
    'Giants':     'New York',
    'Jaguars':    'Jacksonville',
    'Jets':       'New York',
    'Lions':      'Detroit',
    'Packers':    'Green Bay',
    'Panthers':   'Carolina',
    'Patriots':   'New England',
    'Raiders':    'Las Vegas',
    'Rams':       'Los Angeles',
    'Ravens':     'Baltimore',
    'Saints':     'New Orleans',
    'Seahawks':   'Seattle',
    'Steelers':   'Pittsburgh',
    'Texans':     'Houston',
    'Titans':     'Tennessee',
    'Vikings':    'Minnesota',
}

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

# One entry per season, index-aligned: seasons[i] <-> league_keys[i] <-> game_ids[i]
league_keys = ['407818', '615456', '343595', '220398', '441156', '561510', '1097183', '165273',
               '525924', '337550', '1017531', '861488', '403490', '197495', '852144', '660402',
               '468343', '884193']
game_ids    = [199, 222, 242, 257, 273, 314, 331, 348, 359, 371, 380, 390, 399, 406, 414, 423, 449, 461]
seasons     = list(range(2008, 2026))

assert len(league_keys) == len(game_ids) == len(seasons), \
    "league_keys, game_ids, and seasons must have the same length"


def create_query(league_id_num, game_id):
    return YahooFantasySportsQuery(
        league_id=league_id_num,
        game_code="nfl",
        game_id=game_id,
        yahoo_consumer_key=os.getenv("YAHOO_CONSUMER_KEY"),
        yahoo_consumer_secret=os.getenv("YAHOO_CONSUMER_SECRET"),
        save_token_data_to_env_file=True,
        env_file_location=Path(".")
    )


def extract_player_name(player):
    """Extract a usable player name from yfpy player objects or raw dicts."""
    if not player:
        return ''
    if isinstance(player, dict):
        for key in ['full', 'name', 'full_name', 'display_name']:
            value = player.get(key)
            if value:
                return str(value).strip()
        return ''

    # yfpy Name object has a .full attribute
    if hasattr(player, 'name') and player.name:
        name_obj = player.name
        if hasattr(name_obj, 'full') and name_obj.full:
            return str(name_obj.full).strip()
        return str(name_obj).strip()
    if hasattr(player, 'full_name') and player.full_name:
        return str(player.full_name).strip()
    if hasattr(player, 'display_name') and player.display_name:
        return str(player.display_name).strip()
    if hasattr(player, 'first_name') and hasattr(player, 'last_name'):
        return f"{player.first_name} {player.last_name}".strip()
    if hasattr(player, 'full') and player.full:
        return str(player.full).strip()
    return ''


def extract_player_position(player):
    """Extract player primary position from yfpy player objects or raw dicts."""
    if not player:
        return ''
    if isinstance(player, dict):
        for key in ['primary_position', 'display_position', 'position']:
            value = player.get(key)
            if value:
                return str(value).strip()
        return ''

    for attr in ('primary_position', 'display_position', 'position'):
        val = getattr(player, attr, None)
        if val:
            return str(val).strip()

    # Fall back to first eligible position
    ep = getattr(player, 'eligible_positions', None)
    if ep:
        if isinstance(ep, (list, tuple)) and len(ep) > 0:
            return str(ep[0]).strip()
        if isinstance(ep, str) and ep:
            return ep.split(',')[0].strip()
    return ''


def _fetch_player_names_by_keys(query, player_keys, batch_size=25):
    """Fetch name and position for an explicit list of player keys.

    Uses league/players;player_keys=... which returns data for any valid
    player key regardless of whether the player is still in the league
    pool — unlike get_league_players() which only returns current members.
    Batches into groups of batch_size to stay within Yahoo URL limits.
    Returns {player_key: {'name': ..., 'position': ...}}.
    """
    result = {}
    keys = list(player_keys)
    for i in range(0, len(keys), batch_size):
        batch = keys[i:i + batch_size]
        try:
            url = (
                f"https://fantasysports.yahooapis.com/fantasy/v2/"
                f"league/{query.get_league_key()}/players;"
                f"player_keys={','.join(batch)}"
            )
            players = query.query(url, ["league", "players"])
            if not isinstance(players, list):
                players = [players]
            for player in players:
                pk = str(
                    getattr(player, 'player_key', '') or
                    getattr(player, 'editorial_player_key', '')
                )
                if pk:
                    result[pk] = {
                        'name':     extract_player_name(player),
                        'position': extract_player_position(player),
                    }
        except Exception as e:
            print(f'    Warning: could not fetch player metadata for batch starting at {i}: {e}')
    return result


def get_league_standings_yfpy(query):
    """Get standings for a specific league using an existing yfpy query object."""
    standings_obj = query.get_league_standings()
    standings = []

    teams = getattr(standings_obj, 'teams', standings_obj)

    for team in teams:
        def _s(obj, *attrs, default=''):
            for attr in attrs:
                val = getattr(obj, attr, None)
                if val is not None and val != '':
                    return str(val)
            return default

        team_data = {
            'team_id':   _s(team, 'team_id'),
            'team_key':  _s(team, 'team_key'),
            'name':      _s(team, 'name'),
            'url':       _s(team, 'url'),
        }

        mgr = team.managers[0] if getattr(team, 'managers', None) else None
        team_data['manager_id']            = _s(mgr, 'manager_id') if mgr else ''
        team_data['manager_nickname']      = _s(mgr, 'nickname') if mgr else ''
        team_data['is_comanager']          = _s(mgr, 'is_comanager') if mgr else ''
        team_data['is_commissioner']       = _s(mgr, 'is_commissioner') if mgr else ''
        team_data['manager_image_url']     = _s(mgr, 'image_url') if mgr else ''
        team_data['felo_score']            = _s(mgr, 'felo_score') if mgr else ''
        team_data['felo_tier']             = _s(mgr, 'felo_tier') if mgr else ''
        team_data['is_owned_by_current_login'] = _s(team, 'is_owned_by_current_login')

        team_data['draft_grade']      = _s(team, 'draft_grade')
        team_data['draft_position']   = _s(team, 'draft_position')
        team_data['draft_recap_url']  = _s(team, 'draft_recap_url')
        team_data['has_draft_grade']  = _s(team, 'has_draft_grade')

        tp = getattr(team, 'team_points', None)
        team_data['season']           = _s(tp, 'season') if tp else ''
        team_data['total_points']     = _s(tp, 'total') if tp else ''
        tpp = getattr(team, 'team_projected_points', None)
        team_data['projected_points'] = _s(tpp, 'total') if tpp else ''

        ts = getattr(team, 'team_standings', None)
        if ts:
            ot = getattr(ts, 'outcome_totals', None)
            dot = getattr(ts, 'divisional_outcome_totals', None)
            streak = getattr(ts, 'streak', None)
            team_data['rank']              = _s(ts, 'rank')
            team_data['playoff_seed']      = _s(ts, 'playoff_seed')
            team_data['wins']              = _s(ot, 'wins') if ot else ''
            team_data['losses']            = _s(ot, 'losses') if ot else ''
            team_data['ties']              = _s(ot, 'ties') if ot else ''
            team_data['percentage']        = _s(ot, 'percentage') if ot else ''
            team_data['points_for']        = _s(ts, 'points_for')
            team_data['points_against']    = _s(ts, 'points_against')
            team_data['divisional_wins']   = _s(dot, 'wins') if dot else ''
            team_data['divisional_losses'] = _s(dot, 'losses') if dot else ''
            team_data['divisional_ties']   = _s(dot, 'ties') if dot else ''
            team_data['streak_type']       = _s(streak, 'type') if streak else ''
            team_data['streak_length']     = _s(streak, 'value') if streak else ''
        else:
            for key in ('rank', 'playoff_seed', 'wins', 'losses', 'ties', 'percentage',
                        'points_for', 'points_against', 'divisional_wins', 'divisional_losses',
                        'divisional_ties', 'streak_type', 'streak_length'):
                team_data[key] = ''

        team_data['status']               = _s(team, 'status')
        team_data['clinched_playoffs']     = _s(team, 'clinched_playoffs')
        team_data['league_scoring_type']   = _s(team, 'league_scoring_type')
        team_data['number_of_moves']       = _s(team, 'number_of_moves')
        team_data['number_of_trades']      = _s(team, 'number_of_trades')
        team_data['team_paid']             = _s(team, 'team_paid')
        team_data['waiver_priority']       = _s(team, 'waiver_priority')
        team_data['team_logo']             = _s(team, 'team_logo')
        ra = getattr(team, 'roster_adds', None)
        team_data['roster_adds']           = _s(ra, 'value') if ra else ''
        team_data['division_id']           = _s(team, 'division_id')
        team_data['faab_balance']          = _s(team, 'faab_balance')
        team_data['previous_season_team_rank'] = _s(team, 'previous_season_team_rank')
        team_data['win_probability']       = _s(team, 'win_probability')

        standings.append(team_data)

    return standings


def get_team_roster_data(query, team_id, week=None):
    """Get roster with per-player actual fantasy points for a specific team.

    Uses get_team_roster_player_stats_by_week so player_points is populated.
    Pass the last scored week of the season so points reflect final-week
    production.

    Note: player_projected_points is not included — Yahoo's v2 API does not
    expose historical per-player projected stats via any supported type
    parameter.  Team-level projected points are available in the scoreboard.
    """
    try:
        chosen_week = week if week is not None else 'current'
        players_list = query.get_team_roster_player_stats_by_week(
            team_id=team_id, chosen_week=chosen_week
        )

        roster_data = {
            'team_id':       str(team_id),
            'coverage_type': 'week',
            'week':          str(week) if week is not None else '',
            'is_editable':   '',
            'players':       [],
        }

        if not players_list:
            return roster_data

        for player in players_list:
            player_key = str(player.player_key) if hasattr(player, 'player_key') else ''

            ep = getattr(player, 'eligible_positions', None)
            if isinstance(ep, (list, tuple)):
                eligible_str = ','.join(str(p) for p in ep)
            elif isinstance(ep, str):
                eligible_str = ep
            else:
                eligible_str = ''

            bye = getattr(player, 'bye_weeks', None)
            bye_week = str(bye.week) if bye and hasattr(bye, 'week') else ''

            pp = getattr(player, 'player_points', None)

            raw_name = extract_player_name(player)
            position  = extract_player_position(player)
            if position == 'DEF' and raw_name in _NFL_CITY:
                full_name = f"{_NFL_CITY[raw_name]} {raw_name} D/ST"
            else:
                full_name = raw_name

            player_data = {
                'player_key':         player_key,
                'player_id':          str(player.player_id) if hasattr(player, 'player_id') else '',
                'name':               full_name,
                'position':           position,
                'eligible_positions': eligible_str,
                'player_points':      str(pp.total) if pp and hasattr(pp, 'total') else '',
                'status':             str(player.status) if hasattr(player, 'status') else '',
                'status_last_updated': str(player.status_last_updated) if hasattr(player, 'status_last_updated') else '',
                'bye_weeks':          bye_week,
                'injury_note':        str(player.injury_note) if hasattr(player, 'injury_note') else '',
                'nfl_team_id':        str(player.nfl_team_id) if hasattr(player, 'nfl_team_id') else '',
                'coverage_type':      str(pp.coverage_type) if pp and hasattr(pp, 'coverage_type') else '',
            }
            roster_data['players'].append(player_data)

        return roster_data
    except Exception as e:
        print(f'  Error fetching roster for team_id {team_id}: {e}')
        return None


def get_team_draft_results(query, team_id):
    """Get draft results for a specific team.

    DraftResult objects from yfpy only carry player_key — no name or position.
    Names are resolved in process_season via _fetch_player_names_by_keys after
    all teams are collected, so player.name/position start empty here.
    """
    try:
        draft_results = query.get_team_draft_results(team_id=team_id)
        draft_data = {
            'team_id': str(team_id),
            'picks':   [],
        }

        if draft_results:
            for result in draft_results:
                player_key = str(result.player_key) if hasattr(result, 'player_key') else ''
                pick_data = {
                    'round':      str(result.round) if hasattr(result, 'round') else '',
                    'pick':       str(result.pick) if hasattr(result, 'pick') else '',
                    'cost':       str(result.cost) if hasattr(result, 'cost') else '',
                    'player_key': player_key,
                    'player': {
                        'name':     '',
                        'position': '',
                    },
                }
                draft_data['picks'].append(pick_data)

        return draft_data
    except Exception as e:
        print(f'  Error fetching draft results for team_id {team_id}: {e}')
        return None


def get_team_matchups(query, team_id):
    """Get full-season matchup history for a specific team, with correct score attribution."""
    try:
        matchups = query.get_team_matchups(team_id=team_id)
        matchup_data = {
            'team_id':  str(team_id),
            'matchups': [],
        }

        if matchups:
            for matchup in matchups:
                matchup_info = {
                    'week':              str(matchup.week) if hasattr(matchup, 'week') else '',
                    'week_start':        str(matchup.week_start) if hasattr(matchup, 'week_start') else '',
                    'week_end':          str(matchup.week_end) if hasattr(matchup, 'week_end') else '',
                    'status':            str(matchup.status) if hasattr(matchup, 'status') else '',
                    'is_playoffs':       str(matchup.is_playoffs) if hasattr(matchup, 'is_playoffs') else '',
                    'is_consolation':    str(matchup.is_consolation) if hasattr(matchup, 'is_consolation') else '',
                    'is_tied':           str(matchup.is_tied) if hasattr(matchup, 'is_tied') else '',
                    'our_points':        '',
                    'opponent_points':   '',
                    'opponent_team_key': '',
                    'opponent_team_name': '',
                    'winner_team_key':   str(matchup.winner_team_key) if hasattr(matchup, 'winner_team_key') else '',
                }

                if hasattr(matchup, 'teams') and matchup.teams:
                    for team in matchup.teams:
                        t_id = str(getattr(team, 'team_id', ''))
                        t_key = str(getattr(team, 'team_key', ''))
                        tp = getattr(team, 'team_points', None)
                        pts = str(tp.total) if tp and hasattr(tp, 'total') else ''

                        is_our_team = (t_id == str(team_id)) or \
                                      (t_key and t_key.endswith(f'.t.{team_id}'))

                        if is_our_team:
                            matchup_info['our_points'] = pts
                        else:
                            matchup_info['opponent_points']    = pts
                            matchup_info['opponent_team_key']  = t_key
                            matchup_info['opponent_team_name'] = str(getattr(team, 'name', ''))

                matchup_data['matchups'].append(matchup_info)

        return matchup_data
    except Exception as e:
        print(f'  Error fetching matchups for team_id {team_id}: {e}')
        return None


def get_weekly_scoreboard(query, num_weeks=18):
    """
    Get week-by-week team scores from the league scoreboard.
    Returns {week: {team_key: {team_points, projected_points, is_winner, is_playoffs, is_consolation}}}.
    Stops early once a week returns no data (past end of season).
    """
    weekly_data = {}
    for week in range(1, num_weeks + 1):
        try:
            scoreboard = query.get_league_scoreboard_by_week(chosen_week=week)
            week_scores = {}

            if hasattr(scoreboard, 'matchups') and scoreboard.matchups:
                for matchup in scoreboard.matchups:
                    if not hasattr(matchup, 'teams'):
                        continue
                    winner_key   = str(getattr(matchup, 'winner_team_key', ''))
                    is_playoffs  = str(getattr(matchup, 'is_playoffs', '0'))
                    is_consolation = str(getattr(matchup, 'is_consolation', '0'))

                    for team in matchup.teams:
                        t_key = str(getattr(team, 'team_key', ''))
                        if not t_key:
                            continue
                        tp  = getattr(team, 'team_points', None)
                        tpp = getattr(team, 'team_projected_points', None)
                        week_scores[t_key] = {
                            'team_points':       str(tp.total) if tp and hasattr(tp, 'total') else '',
                            'projected_points':  str(tpp.total) if tpp and hasattr(tpp, 'total') else '',
                            'is_winner':         str(winner_key == t_key),
                            'is_playoffs':       is_playoffs,
                            'is_consolation':    is_consolation,
                        }

            if week_scores:
                weekly_data[str(week)] = week_scores
                print(f'    Week {week}: {len(week_scores)} teams')
            else:
                break
        except Exception as e:
            print(f'  Warning: Could not get scoreboard for week {week}: {e}')
            break

    return weekly_data


def process_season(season, league_key, game_id):
    """Process all data for a single season using a single authenticated query."""
    print(f'\n=== Processing Season {season} ===')
    season_data = {
        'standings':     [],
        'rosters':       {},
        'draft_results': {},
        'matchups':      {},
        'scoreboard':    {},
    }

    try:
        query = create_query(league_key, game_id)

        standings = get_league_standings_yfpy(query)
        season_data['standings'] = standings
        print(f'✓ Standings: {len(standings)} teams')

        print('  Fetching weekly scoreboard...')
        scoreboard = get_weekly_scoreboard(query)
        season_data['scoreboard'] = scoreboard
        print(f'✓ Scoreboard: {len(scoreboard)} weeks')

        # Use the last scored week so player_points reflect final production
        last_week = max((int(w) for w in scoreboard.keys()), default=17)
        print(f'  Using week {last_week} for player stats...')

        for team in standings:
            team_id  = team.get('team_id', '')
            team_key = team.get('team_key', '')
            if not team_id:
                print(f'  ✗ Skipping team with missing team_id: {team}')
                continue

            roster = get_team_roster_data(query, team_id, week=last_week)
            if roster:
                season_data['rosters'][team_key or str(team_id)] = roster
                print(f'  ✓ Roster: {team_key or team_id} ({len(roster.get("players", []))} players)')

            draft = get_team_draft_results(query, team_id)
            if draft:
                season_data['draft_results'][team_key or str(team_id)] = draft
                print(f'  ✓ Draft: {team_key or team_id} ({len(draft.get("picks", []))} picks)')

            matchups = get_team_matchups(query, team_id)
            if matchups:
                season_data['matchups'][team_key or str(team_id)] = matchups
                print(f'  ✓ Matchups: {team_key or team_id} ({len(matchups.get("matchups", []))} weeks)')

        # --- Draft player name resolution (runs after all teams are collected) ---
        # Pass 1: batch-fetch regular players via league/players;player_keys=...
        missing_keys = {
            pick['player_key']
            for draft in season_data['draft_results'].values()
            for pick in draft.get('picks', [])
            if pick.get('player_key') and not pick.get('player', {}).get('name')
        }
        if missing_keys:
            print(f'  Resolving names for {len(missing_keys)} draft picks...')
            name_lookup = _fetch_player_names_by_keys(query, missing_keys)
            for draft in season_data['draft_results'].values():
                for pick in draft['picks']:
                    pk = pick.get('player_key', '')
                    if pk in name_lookup:
                        pick['player']['name']     = name_lookup[pk]['name']
                        pick['player']['position'] = name_lookup[pk]['position']
            resolved = sum(1 for v in name_lookup.values() if v.get('name'))
            print(f'  ✓ Resolved {resolved}/{len(missing_keys)} names')

        # Pass 2: DEF/ST slots — the players endpoint doesn't serve them, but
        # the roster endpoint does.  Build a player_key -> full name lookup from
        # the rosters already collected (names already include the city prefix).
        def_lookup = {
            p['player_key']: p
            for roster in season_data['rosters'].values()
            for p in roster.get('players', [])
            if p.get('position') == 'DEF' and p.get('player_key') and p.get('name')
        }
        if def_lookup:
            for draft in season_data['draft_results'].values():
                for pick in draft['picks']:
                    pk = pick.get('player_key', '')
                    if pk in def_lookup and not pick.get('player', {}).get('name'):
                        pick['player']['name']     = def_lookup[pk]['name']
                        pick['player']['position'] = def_lookup[pk]['position']

        return season_data

    except Exception as e:
        import traceback
        print(f'✗ Error for {season}: {e}')
        print(f'  Details: {traceback.format_exc()}')
        return None


def save_season_data(season, data):
    """Save each data type for a single season to its own JSON file."""
    if not data:
        return

    for filename, key in [
        (f'standings_{season}.json',    'standings'),
        (f'rosters_{season}.json',      'rosters'),
        (f'draft_results_{season}.json','draft_results'),
        (f'matchups_{season}.json',     'matchups'),
        (f'scoreboard_{season}.json',   'scoreboard'),
    ]:
        filepath = DATA_DIR / filename
        with open(filepath, 'w') as f:
            json.dump({str(season): data[key]}, f, indent=4)
        print(f'✓ Saved {filepath}')


def add_season_to_datasets(season):
    """Upsert a single season into the consolidated per-type JSON datasets."""
    if season not in seasons:
        print(f"Error: Season {season} not available. Available seasons: {seasons}")
        return

    season_idx  = seasons.index(season)
    league_key  = league_keys[season_idx]
    game_id     = game_ids[season_idx]

    data = process_season(season, league_key, game_id)
    if not data:
        print(f"Failed to process season {season}")
        return

    datasets = [
        ('standings_yfpy.json',    'standings'),
        ('rosters_yfpy.json',      'rosters'),
        ('draft_results_yfpy.json','draft_results'),
        ('matchups_yfpy.json',     'matchups'),
        ('scoreboard_yfpy.json',   'scoreboard'),
    ]

    for filename, key in datasets:
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {filename}, starting fresh")
                existing = {}
        else:
            existing = {}

        existing[str(season)] = data[key]

        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=4)
        print(f"✓ Added season {season} to {filename}")

    print(f"\n=== Season {season} added to datasets ===")


def main():
    parser = argparse.ArgumentParser(description='Extract Yahoo Fantasy Sports data using yfpy')
    parser.add_argument('--season',     type=int, help='Specific season to process (e.g., 2023)')
    parser.add_argument('--all',        action='store_true', help='Process all seasons (2008-2025)')
    parser.add_argument('--list',       action='store_true', help='List available seasons')
    parser.add_argument('--add-season', type=int, help='Upsert a single season into the consolidated datasets')

    args = parser.parse_args()

    if args.list:
        print(f"Available seasons: {seasons[0]}-{seasons[-1]}")
        return

    if args.add_season:
        add_season_to_datasets(args.add_season)
        return

    if args.season:
        if args.season not in seasons:
            print(f"Error: Season {args.season} not available. Available seasons: {seasons}")
            sys.exit(1)

        season_idx = seasons.index(args.season)
        data = process_season(args.season, league_keys[season_idx], game_ids[season_idx])
        if data:
            save_season_data(args.season, data)
            print(f'\n=== Season {args.season} complete ===')
        else:
            print(f'\n=== Season {args.season} failed ===')
            sys.exit(1)

    elif args.all:
        all_data = {k: {} for k in ('standings', 'rosters', 'draft_results', 'matchups', 'scoreboard')}

        for season, gid, lid in zip(seasons, game_ids, league_keys):
            data = process_season(season, lid, gid)
            if data:
                for key in all_data:
                    all_data[key][str(season)] = data[key]
            else:
                print(f'Skipping season {season} due to errors')

        print('\n=== Saving All Data ===')
        for filename, key in [
            ('standings_yfpy.json',    'standings'),
            ('rosters_yfpy.json',      'rosters'),
            ('draft_results_yfpy.json','draft_results'),
            ('matchups_yfpy.json',     'matchups'),
            ('scoreboard_yfpy.json',   'scoreboard'),
        ]:
            with open(DATA_DIR / filename, 'w') as f:
                json.dump(all_data[key], f, indent=4)
            print(f'✓ Saved data/{filename}')

        print('\n=== All seasons complete ===')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
