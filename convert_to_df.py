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

if __name__ == '__main__':
    print(f'Loading JSON files from {DATA_DIR}...\\n')
    
    if not (DATA_DIR / 'standings_yfpy.json').exists():
        print('✗ Error: standings_yfpy.json not found in data folder.')
        print('  Please run yahoo_api.py first to generate the data files.')
        exit(1)
    
    # Load all data
    standings_df = load_standings()
    rosters_df = load_rosters()
    draft_results_df = load_draft_results()
    matchups_df = load_matchups()
    
    # Display info
    print('✓ standings_df:', standings_df.shape)
    print('  Columns:', list(standings_df.columns))
    print()
    
    print('✓ rosters_df:', rosters_df.shape)
    print('  Columns:', list(rosters_df.columns))
    print()
    
    print('✓ draft_results_df:', draft_results_df.shape)
    print('  Columns:', list(draft_results_df.columns))
    print()
    
    print('✓ matchups_df:', matchups_df.shape)
    print('  Columns:', list(matchups_df.columns))
    print()
    
    # Save to CSV for convenience
    standings_df.to_csv(DATA_DIR / 'standings_df.csv', index=False)
    rosters_df.to_csv(DATA_DIR / 'rosters_df.csv', index=False)
    draft_results_df.to_csv(DATA_DIR / 'draft_results_df.csv', index=False)
    matchups_df.to_csv(DATA_DIR / 'matchups_df.csv', index=False)
    
    print('✓ All DataFrames saved to CSV:')
    print('  - data/standings_df.csv')
    print('  - data/rosters_df.csv')
    print('  - data/draft_results_df.csv')
    print('  - data/matchups_df.csv')
    
    # Save to pickle for Python usage
    standings_df.to_pickle(DATA_DIR / 'standings_df.pkl')
    rosters_df.to_pickle(DATA_DIR / 'rosters_df.pkl')
    draft_results_df.to_pickle(DATA_DIR / 'draft_results_df.pkl')
    matchups_df.to_pickle(DATA_DIR / 'matchups_df.pkl')
    
    # Save nested draft results JSON for direct nested structure inspection
    print('\n✓ All DataFrames saved to pickle:')
    print('  - data/standings_df.pkl')
    print('  - data/rosters_df.pkl')
    print('  - data/draft_results_df.pkl')

    print('  - data/matchups_df.pkl')
