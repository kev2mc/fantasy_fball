"""
Interactive OAuth authentication for Yahoo Fantasy Sports API
This script will help you get fresh OAuth credentials
"""

from yfpy.query import YahooFantasySportsQuery
from pathlib import Path
import json

def authenticate():
    """Authenticate with Yahoo and save credentials to .env file"""
    
    print("=" * 60)
    print("Yahoo Fantasy Sports OAuth Authentication")
    print("=" * 60)
    print()
    print("Before continuing, ensure you have:")
    print("1. Consumer Key from Yahoo Developers")
    print("2. Consumer Secret from Yahoo Developers")
    print()
    
    consumer_key = input("Enter your Consumer Key: ").strip()
    consumer_secret = input("Enter your Consumer Secret: ").strip()
    
    if not consumer_key or not consumer_secret:
        print("Error: Both key and secret are required")
        return False
    
    try:
        print("\nAttempting authentication...")
        print("This will open a browser window for authorization.")
        print()
        
        # Create a query object to trigger OAuth flow
        # Using a dummy league_id just to authenticate
        query = YahooFantasySportsQuery(
            league_id=1,  # Dummy league ID
            game_code="nfl",
            game_id=423,  # 2025 game ID
            yahoo_consumer_key=consumer_key,
            yahoo_consumer_secret=consumer_secret,
            save_token_data_to_env_file=True,
            env_file_location=Path(".")
        )
        
        print("\n✓ Authentication successful!")
        print("✓ Credentials saved to .env file")
        print()
        print("You can now run: python yahoo_api.py --season 2025")
        return True
        
    except Exception as e:
        print(f"\n✗ Authentication failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check that your Consumer Key and Secret are correct")
        print("2. Visit https://developer.yahoo.com/apps/ to verify credentials")
        print("3. Ensure the app has permission for Fantasy Sports API")
        return False

if __name__ == '__main__':
    success = authenticate()
    exit(0 if success else 1)
