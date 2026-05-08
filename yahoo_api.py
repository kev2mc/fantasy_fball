from yfpy.query import YahooFantasySportsQuery
from pathlib import Path
import json
import os
import logging
import argparse
import sys

logging.basicConfig(level=logging.WARNING)

# Create data directory if it doesn't exist
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)


# League data from R script (known league keys through 2021; seasons 2022-2025 use added game_ids and attempt league key discovery)
league_keys = ['407818', '615456', '343595', '220398', '441156', '561510', '1097183', '165273', '525924', '337550', '1017531', '861488', '403490', '197495', '852144', '660402', '468343', '884193']
game_ids = [199, 222, 242, 257, 273, 314, 331, 348, 359, 371, 380, 390, 399, 406, 414, 423, 449, 461]
seasons = list(range(2008, 2026))

# Extend league_keys with the first league ID for new seasons (same league continuing into 2022-2025)
while len(league_keys) < len(seasons):
    league_keys.append(league_keys[0])

def create_query(league_id_num, game_id):
    """Create a YahooFantasySportsQuery instance."""
    return YahooFantasySportsQuery(
        league_id=league_id_num,
        game_code="nfl",
        game_id=game_id,
        yahoo_consumer_key=os.getenv("YAHOO_CONSUMER_KEY"),
        yahoo_consumer_secret=os.getenv("YAHOO_CONSUMER_SECRET"),
        save_token_data_to_env_file=True,
        env_file_location=Path(".")
    )

def get_league_standings_yfpy(league_id_num, game_id):
    """Get standings for a specific league using yfpy."""
    query = YahooFantasySportsQuery(
        league_id=league_id_num,  # league number only
        game_code="nfl",
        game_id=game_id,
        yahoo_consumer_key=os.getenv("YAHOO_CONSUMER_KEY"),
        yahoo_consumer_secret=os.getenv("YAHOO_CONSUMER_SECRET"),
        save_token_data_to_env_file=True,
        env_file_location=Path(".")
    )
    standings_obj = query.get_league_standings()
    standings = []

    # standings_obj is a Standings object, iterate over its teams
    if hasattr(standings_obj, 'teams'):
        teams = standings_obj.teams
    else:
        # If teams is not directly accessible, try to iterate over standings_obj itself
        teams = standings_obj

    for team in teams:
        # Basic Info
        team_data = {
            'team_id': str(team.team_id) if hasattr(team, 'team_id') and team.team_id else '',
            'team_key': str(team.team_key) if hasattr(team, 'team_key') and team.team_key else '',
            'name': str(team.name) if hasattr(team, 'name') and team.name else '',
            'url': str(team.url) if hasattr(team, 'url') and team.url else '',
        }

        # Management & Ownership
        team_data['manager_id'] = str(team.managers[0].manager_id) if team.managers else ''
        team_data['manager_nickname'] = str(team.managers[0].nickname) if team.managers else ''
        team_data['is_comanager'] = str(team.managers[0].is_comanager) if team.managers and hasattr(team.managers[0], 'is_comanager') else ''
        team_data['is_commissioner'] = str(team.managers[0].is_commissioner) if team.managers and hasattr(team.managers[0], 'is_commissioner') else ''
        team_data['manager_image_url'] = str(team.managers[0].image_url) if team.managers and hasattr(team.managers[0], 'image_url') else ''
        team_data['felo_score'] = str(team.managers[0].felo_score) if team.managers and hasattr(team.managers[0], 'felo_score') else ''
        team_data['felo_tier'] = str(team.managers[0].felo_tier) if team.managers and hasattr(team.managers[0], 'felo_tier') else ''
        team_data['is_owned_by_current_login'] = str(team.is_owned_by_current_login) if hasattr(team, 'is_owned_by_current_login') else ''

        # Draft Information
        team_data['draft_grade'] = str(team.draft_grade) if hasattr(team, 'draft_grade') and team.draft_grade else ''
        team_data['draft_position'] = str(team.draft_position) if hasattr(team, 'draft_position') and team.draft_position else ''
        team_data['draft_recap_url'] = str(team.draft_recap_url) if hasattr(team, 'draft_recap_url') and team.draft_recap_url else ''
        team_data['has_draft_grade'] = str(team.has_draft_grade) if hasattr(team, 'has_draft_grade') else ''

        # Points & Scoring
        team_data['season'] = str(team.team_points.season) if hasattr(team, 'team_points') and team.team_points and hasattr(team.team_points, 'season') else ''
        team_data['total_points'] = str(team.team_points.total) if hasattr(team, 'team_points') and team.team_points and hasattr(team.team_points, 'total') else ''
        team_data['projected_points'] = str(team.team_projected_points.total) if hasattr(team, 'team_projected_points') and team.team_projected_points and hasattr(team.team_projected_points, 'total') else ''

        # Standings & Records
        if hasattr(team, 'team_standings') and team.team_standings:
            team_data['rank'] = str(team.team_standings.rank) if hasattr(team.team_standings, 'rank') else ''
            team_data['playoff_seed'] = str(team.team_standings.playoff_seed) if hasattr(team.team_standings, 'playoff_seed') else ''
            team_data['wins'] = str(team.team_standings.outcome_totals.wins) if hasattr(team.team_standings, 'outcome_totals') and team.team_standings.outcome_totals and hasattr(team.team_standings.outcome_totals, 'wins') else ''
            team_data['losses'] = str(team.team_standings.outcome_totals.losses) if hasattr(team.team_standings, 'outcome_totals') and team.team_standings.outcome_totals and hasattr(team.team_standings.outcome_totals, 'losses') else ''
            team_data['ties'] = str(team.team_standings.outcome_totals.ties) if hasattr(team.team_standings, 'outcome_totals') and team.team_standings.outcome_totals and hasattr(team.team_standings.outcome_totals, 'ties') else ''
            team_data['percentage'] = str(team.team_standings.outcome_totals.percentage) if hasattr(team.team_standings, 'outcome_totals') and team.team_standings.outcome_totals and hasattr(team.team_standings.outcome_totals, 'percentage') else ''
            team_data['points_for'] = str(team.team_standings.points_for) if hasattr(team.team_standings, 'points_for') else ''
            team_data['points_against'] = str(team.team_standings.points_against) if hasattr(team.team_standings, 'points_against') else ''
            team_data['divisional_wins'] = str(team.team_standings.divisional_outcome_totals.wins) if hasattr(team.team_standings, 'divisional_outcome_totals') and team.team_standings.divisional_outcome_totals and hasattr(team.team_standings.divisional_outcome_totals, 'wins') else ''
            team_data['divisional_losses'] = str(team.team_standings.divisional_outcome_totals.losses) if hasattr(team.team_standings, 'divisional_outcome_totals') and team.team_standings.divisional_outcome_totals and hasattr(team.team_standings.divisional_outcome_totals, 'losses') else ''
            team_data['divisional_ties'] = str(team.team_standings.divisional_outcome_totals.ties) if hasattr(team.team_standings, 'divisional_outcome_totals') and team.team_standings.divisional_outcome_totals and hasattr(team.team_standings.divisional_outcome_totals, 'ties') else ''
            # Streak info
            team_data['streak_type'] = str(team.team_standings.streak.type) if hasattr(team.team_standings, 'streak') and team.team_standings.streak and hasattr(team.team_standings.streak, 'type') else ''
            team_data['streak_length'] = str(team.team_standings.streak.value) if hasattr(team.team_standings, 'streak') and team.team_standings.streak and hasattr(team.team_standings.streak, 'value') else ''
        else:
            team_data['rank'] = ''
            team_data['playoff_seed'] = ''
            team_data['wins'] = ''
            team_data['losses'] = ''
            team_data['ties'] = ''
            team_data['percentage'] = ''
            team_data['points_for'] = ''
            team_data['points_against'] = ''
            team_data['divisional_wins'] = ''
            team_data['divisional_losses'] = ''
            team_data['divisional_ties'] = ''
            team_data['streak_type'] = ''
            team_data['streak_length'] = ''

        # Team Status & Settings
        team_data['status'] = str(team.status) if hasattr(team, 'status') and team.status else ''
        team_data['clinched_playoffs'] = str(team.clinched_playoffs) if hasattr(team, 'clinched_playoffs') else ''
        team_data['league_scoring_type'] = str(team.league_scoring_type) if hasattr(team, 'league_scoring_type') and team.league_scoring_type else ''
        team_data['number_of_moves'] = str(team.number_of_moves) if hasattr(team, 'number_of_moves') else ''
        team_data['number_of_trades'] = str(team.number_of_trades) if hasattr(team, 'number_of_trades') else ''
        team_data['team_paid'] = str(team.team_paid) if hasattr(team, 'team_paid') else ''
        team_data['waiver_priority'] = str(team.waiver_priority) if hasattr(team, 'waiver_priority') else ''

        # Logo & Branding
        team_data['team_logo'] = str(team.team_logo) if hasattr(team, 'team_logo') and team.team_logo else ''

        # Roster Stats
        team_data['roster_adds'] = str(team.roster_adds.value) if hasattr(team, 'roster_adds') and team.roster_adds and hasattr(team.roster_adds, 'value') else ''

        # Additional attributes
        team_data['division_id'] = str(team.division_id) if hasattr(team, 'division_id') and team.division_id else ''
        team_data['faab_balance'] = str(team.faab_balance) if hasattr(team, 'faab_balance') and team.faab_balance else ''
        team_data['previous_season_team_rank'] = str(team.previous_season_team_rank) if hasattr(team, 'previous_season_team_rank') and team.previous_season_team_rank else ''
        team_data['win_probability'] = str(team.win_probability) if hasattr(team, 'win_probability') and team.win_probability else ''

        standings.append(team_data)
    return standings

def get_team_roster_data(league_id_num, game_id, team_id):
    """Get roster data for a specific team."""
    try:
        query = create_query(league_id_num, game_id)
        roster = query.get_team_roster_by_week(team_id=team_id)
        roster_data = {
            'team_id': str(team_id),
            'coverage_type': str(roster.coverage_type) if hasattr(roster, 'coverage_type') else '',
            'week': str(roster.week) if hasattr(roster, 'week') else '',
            'is_editable': str(roster.is_editable) if hasattr(roster, 'is_editable') else '',
            'players': []
        }

        if hasattr(roster, 'players') and roster.players:
            for player in roster.players:
                player_data = {
                    'player_key': str(player.player_key) if hasattr(player, 'player_key') else '',
                    'player_id': str(player.player_id) if hasattr(player, 'player_id') else '',
                    'name': str(player.name) if hasattr(player, 'name') else '',
                    'position': str(player.position) if hasattr(player, 'position') else '',
                    'eligible_positions': ','.join(str(p) for p in player.eligible_positions) if hasattr(player, 'eligible_positions') and player.eligible_positions else '',
                    'player_points': str(player.player_points.total) if hasattr(player, 'player_points') and player.player_points and hasattr(player.player_points, 'total') else '',
                    'player_projected_points': str(player.player_projected_points.total) if hasattr(player, 'player_projected_points') and player.player_projected_points and hasattr(player.player_projected_points, 'total') else '',
                    'status': str(player.status) if hasattr(player, 'status') else '',
                    'status_last_updated': str(player.status_last_updated) if hasattr(player, 'status_last_updated') else '',
                    'bye_weeks': str(player.bye_weeks.week) if hasattr(player, 'bye_weeks') and player.bye_weeks and hasattr(player.bye_weeks, 'week') else '',
                    'injury_note': str(player.injury_note) if hasattr(player, 'injury_note') else '',
                    'nfl_team_id': str(player.nfl_team_id) if hasattr(player, 'nfl_team_id') else '',
                    'coverage_type': str(player.player_points.coverage_type) if hasattr(player, 'player_points') and player.player_points and hasattr(player.player_points, 'coverage_type') else '',
                }
                roster_data['players'].append(player_data)

        return roster_data
    except Exception as e:
        print(f'Error fetching roster for team_id {team_id}: {e}')
        return None

def get_team_draft_results(league_id_num, game_id, team_id):
    """Get draft results for a specific team."""
    try:
        query = create_query(league_id_num, game_id)
        draft_results = query.get_team_draft_results(team_id=team_id)
        draft_data = {
            'team_id': str(team_id),
            'picks': []
        }

        if draft_results:
            for result in draft_results:
                pick_data = {
                    'round': str(result.round) if hasattr(result, 'round') else '',
                    'pick': str(result.pick) if hasattr(result, 'pick') else '',
                    'cost': str(result.cost) if hasattr(result, 'cost') else '',
                    'player_key': str(result.player_key) if hasattr(result, 'player_key') else '',
                    'player': {
                        'name': result.player.name if hasattr(result, 'player') and result.player and hasattr(result.player, 'name') else '',
                        'position': result.player.position if hasattr(result, 'player') and result.player and hasattr(result.player, 'position') else '',
                    }
                }
                draft_data['picks'].append(pick_data)

        return draft_data
    except Exception as e:
        print(f'Error fetching draft results for team_id {team_id}: {e}')
        return None

def get_team_matchups(league_id_num, game_id, team_id):
    """Get matchup history for a specific team."""
    try:
        query = create_query(league_id_num, game_id)
        matchups = query.get_team_matchups(team_id=team_id)
        matchup_data = {
            'team_id': str(team_id),
            'matchups': []
        }

        if matchups:
            for matchup in matchups:
                matchup_info = {
                    'week': str(matchup.week) if hasattr(matchup, 'week') else '',
                    'week_start': str(matchup.week_start) if hasattr(matchup, 'week_start') else '',
                    'week_end': str(matchup.week_end) if hasattr(matchup, 'week_end') else '',
                    'status': str(matchup.status) if hasattr(matchup, 'status') else '',
                    'is_playoffs': str(matchup.is_playoffs) if hasattr(matchup, 'is_playoffs') else '',
                    'is_consolation': str(matchup.is_consolation) if hasattr(matchup, 'is_consolation') else '',
                    'is_tied': str(matchup.is_tied) if hasattr(matchup, 'is_tied') else '',
                    'opponent_team_key': '',
                    'opponent_team_name': '',
                    'our_points': str(matchup.teams[0].team_points.total) if hasattr(matchup, 'teams') and len(matchup.teams) > 0 and hasattr(matchup.teams[0], 'team_points') and matchup.teams[0].team_points and hasattr(matchup.teams[0].team_points, 'total') else '',
                    'opponent_points': str(matchup.teams[1].team_points.total) if hasattr(matchup, 'teams') and len(matchup.teams) > 1 and hasattr(matchup.teams[1], 'team_points') and matchup.teams[1].team_points and hasattr(matchup.teams[1].team_points, 'total') else '',
                    'winner_team_key': str(matchup.winner_team_key) if hasattr(matchup, 'winner_team_key') else '',
                }

                # Identify opponent
                if hasattr(matchup, 'teams') and len(matchup.teams) > 1:
                    for team in matchup.teams:
                        if hasattr(team, 'team_id') and str(team.team_id) != str(team_id):
                            matchup_info['opponent_team_key'] = str(team.team_key) if hasattr(team, 'team_key') else ''
                            matchup_info['opponent_team_name'] = str(team.name) if hasattr(team, 'name') else ''
                        elif not hasattr(team, 'team_id') and hasattr(team, 'team_key') and team.team_key != team_id:
                            matchup_info['opponent_team_key'] = str(team.team_key)
                            matchup_info['opponent_team_name'] = str(team.name) if hasattr(team, 'name') else ''

                matchup_data['matchups'].append(matchup_info)

        return matchup_data
    except Exception as e:
        print(f'Error fetching matchups for team_id {team_id}: {e}')
        return None

def process_season(season, league_key, game_id):
    """Process all data for a single season."""
    print(f'\n=== Processing Season {season} ===')
    season_data = {
        'standings': [],
        'rosters': {},
        'draft_results': {},
        'matchups': {}
    }

    try:
        # Get standings
        standings = get_league_standings_yfpy(league_key, game_id)
        season_data['standings'] = standings
        print(f'✓ Fetched standings for {season} ({len(standings)} teams)')

        # For each team, get roster, draft, and matchup data
        for team in standings:
            team_id = team.get('team_id', '')
            team_key = team.get('team_key', '')
            if team_id:
                # Get roster
                roster = get_team_roster_data(league_key, game_id, team_id)
                if roster:
                    season_data['rosters'][team_key or str(team_id)] = roster
                    print(f'  ✓ Roster: {team_key or team_id} ({len(roster.get("players", []))} players)')

                # Get draft results
                draft = get_team_draft_results(league_key, game_id, team_id)
                if draft:
                    season_data['draft_results'][team_key or str(team_id)] = draft
                    print(f'  ✓ Draft: {team_key or team_id} ({len(draft.get("picks", []))} picks)')

                # Get matchups
                matchups = get_team_matchups(league_key, game_id, team_id)
                if matchups:
                    season_data['matchups'][team_key or str(team_id)] = matchups
                    print(f'  ✓ Matchups: {team_key or team_id} ({len(matchups.get("matchups", []))} matchups)')
            else:
                print(f'  ✗ Skipping team with missing team_id: {team}')

        return season_data

    except Exception as e:
        print(f'✗ Error for {season}: {e}')
        return None

def save_season_data(season, data):
    """Save data for a single season."""
    if not data:
        return

    # Save standings
    standings_file = DATA_DIR / f'standings_{season}.json'
    with open(standings_file, 'w') as f:
        json.dump({str(season): data['standings']}, f, indent=4)
    print(f'✓ Standings saved to {standings_file}')

    # Save rosters
    rosters_file = DATA_DIR / f'rosters_{season}.json'
    with open(rosters_file, 'w') as f:
        json.dump({str(season): data['rosters']}, f, indent=4)
    print(f'✓ Rosters saved to {rosters_file}')

    # Save draft results
    draft_file = DATA_DIR / f'draft_results_{season}.json'
    with open(draft_file, 'w') as f:
        json.dump({str(season): data['draft_results']}, f, indent=4)
    print(f'✓ Draft results saved to {draft_file}')

    # Save matchups
    matchups_file = DATA_DIR / f'matchups_{season}.json'
    with open(matchups_file, 'w') as f:
        json.dump({str(season): data['matchups']}, f, indent=4)
    print(f'✓ Matchups saved to {matchups_file}')

def main():
    parser = argparse.ArgumentParser(description='Extract Yahoo Fantasy Sports data using yfpy')
    parser.add_argument('--season', type=int, help='Specific season to process (e.g., 2023)')
    parser.add_argument('--all', action='store_true', help='Process all seasons (2008-2025)')
    parser.add_argument('--list', action='store_true', help='List available seasons')

    args = parser.parse_args()

    if args.list:
        print("Available seasons: 2008-2025")
        return

    if args.season:
        if args.season not in seasons:
            print(f"Error: Season {args.season} not available. Available seasons: {seasons}")
            sys.exit(1)

        season_idx = seasons.index(args.season)
        league_key = league_keys[season_idx]
        game_id = game_ids[season_idx]

        data = process_season(args.season, league_key, game_id)
        if data:
            save_season_data(args.season, data)
            print(f'\n=== Season {args.season} Complete ===')
        else:
            print(f'\n=== Season {args.season} Failed ===')
            sys.exit(1)

    elif args.all:
        all_data = {
            'standings': {},
            'rosters': {},
            'draft_results': {},
            'matchups': {}
        }

        for season, gid, lid in zip(seasons, game_ids, league_keys):
            data = process_season(season, lid, gid)
            if data:
                all_data['standings'][str(season)] = data['standings']
                all_data['rosters'][str(season)] = data['rosters']
                all_data['draft_results'][str(season)] = data['draft_results']
                all_data['matchups'][str(season)] = data['matchups']
            else:
                print(f'Skipping season {season} due to errors')

        # Save all data
        print('\n=== Saving All Data ===')
        with open(DATA_DIR / 'standings_yfpy.json', 'w') as f:
            json.dump(all_data['standings'], f, indent=4)
        print('✓ Standings saved to data/standings_yfpy.json')

        with open(DATA_DIR / 'rosters_yfpy.json', 'w') as f:
            json.dump(all_data['rosters'], f, indent=4)
        print('✓ Rosters saved to data/rosters_yfpy.json')

        with open(DATA_DIR / 'draft_results_yfpy.json', 'w') as f:
            json.dump(all_data['draft_results'], f, indent=4)
        print('✓ Draft results saved to data/draft_results_yfpy.json')

        with open(DATA_DIR / 'matchups_yfpy.json', 'w') as f:
            json.dump(all_data['matchups'], f, indent=4)
        print('✓ Matchups saved to data/matchups_yfpy.json')

        print('\n=== All Seasons Complete ===')

    else:
        parser.print_help()

if __name__ == '__main__':
    main()