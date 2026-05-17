"""
orchestrate_data.py — Fantasy football analytics pipeline orchestrator.

Pipeline steps (run in this order):
  fetch   : Pull fresh data from Yahoo API          (yahoo_api.py)
  convert : Parse JSON files -> DataFrames -> CSVs  (convert_to_df.py)
  analyze : Compute analytics and manager summaries (data_science.py)
  predict : Feature engineering + ML prediction     (predict_2026.py)

Usage examples:
  python run.py --all                       # convert + analyze + predict
  python run.py --convert --analyze         # only these two steps
  python run.py --fetch --season 2025       # fetch one season from Yahoo API
  python run.py --fetch --season 2025 --resume   # resume interrupted fetch
  python run.py --predict                   # re-run predictions only
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path so all pipeline modules are importable
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def run_fetch(season=None, resume=False):
    import yahoo_api
    if season:
        yahoo_api.add_season_to_datasets(season, resume=resume)
    else:
        print('Specify a season with --season YEAR, or use yahoo_api.py directly:')
        print('  python yahoo_api.py --all       # all seasons')
        print('  python yahoo_api.py --season YEAR')


def run_convert():
    from convert_to_df import run
    run()


def run_analyze():
    from data_science import run
    run()


def run_predict():
    from predict_2026 import run
    run()


def main():
    parser = argparse.ArgumentParser(
        description='Fantasy football analytics pipeline orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--fetch',   action='store_true',
                        help='Fetch data from Yahoo API (requires --season)')
    parser.add_argument('--season',  type=int, metavar='YEAR',
                        help='Season year for --fetch (e.g. 2025)')
    parser.add_argument('--resume',  action='store_true',
                        help='Skip already-fetched weekly roster weeks')
    parser.add_argument('--convert', action='store_true',
                        help='Load JSON files -> DataFrames -> save CSVs')
    parser.add_argument('--analyze', action='store_true',
                        help='Compute analytics, build manager summaries, run keeper analysis')
    parser.add_argument('--predict', action='store_true',
                        help='Run ML feature engineering and 2026 predictions')
    parser.add_argument('--all',     action='store_true',
                        help='Run convert + analyze + predict (skips fetch)')

    args = parser.parse_args()

    if not any([args.fetch, args.convert, args.analyze, args.predict, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.all:
        args.convert = args.analyze = args.predict = True

    if args.fetch:
        run_fetch(season=args.season, resume=args.resume)

    if args.convert:
        print('\n' + '=' * 60)
        print('STEP: convert_to_df')
        print('=' * 60)
        run_convert()

    if args.analyze:
        print('\n' + '=' * 60)
        print('STEP: data_science')
        print('=' * 60)
        run_analyze()

    if args.predict:
        print('\n' + '=' * 60)
        print('STEP: predict_2026')
        print('=' * 60)
        run_predict()


if __name__ == '__main__':
    main()
