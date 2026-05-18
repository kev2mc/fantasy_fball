"""
dashboard.py  —  Fantasy Football League Dashboard

Run with:
    streamlit run streamlit/dashboard.py

Requires: streamlit, plotly, pandas
    pip install streamlit plotly
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'

st.set_page_config(
    page_title='Fantasy Football Dashboard',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    season_df = pd.read_csv(DATA_DIR / 'manager_season_df.csv')
    career_df = pd.read_csv(DATA_DIR / 'manager_career_df.csv')

    def _opt(name):
        p = DATA_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    keeper_era_df    = _opt('manager_keeper_era_df.csv')
    keeper_summary_df = _opt('keeper_summary_df.csv')
    keeper_detail_df  = _opt('keeper_detail_df.csv')

    # Normalize bool-ish columns that CSV round-trips as strings
    for col in ('made_playoffs', 'champion'):
        if col in season_df.columns:
            season_df[col] = season_df[col].map(
                {True: True, False: False, 'True': True, 'False': False, 1: True, 0: False}
            ).astype(bool)

    for col in ('is_optimal', 'is_kept', 'is_ineligible'):
        if col in keeper_detail_df.columns:
            keeper_detail_df[col] = keeper_detail_df[col].map(
                {True: True, False: False, 'True': True, 'False': False, 1: True, 0: False}
            ).astype(bool)
    if 'is_optimal' in keeper_summary_df.columns:
        keeper_summary_df['is_optimal'] = keeper_summary_df['is_optimal'].map(
            {True: True, False: False, 'True': True, 'False': False, 1: True, 0: False}
        ).astype(bool)

    return season_df, career_df, keeper_era_df, keeper_summary_df, keeper_detail_df


if not (DATA_DIR / 'manager_season_df.csv').exists():
    st.error('manager_season_df.csv not found. Run `python run.py --analyze` first.')
    st.stop()

season_df, career_df, keeper_era_df, keeper_summary_df, keeper_detail_df = load_data()

latest_season   = int(season_df['season'].max())
current_managers = sorted(season_df[season_df['season'] == latest_season]['manager'].unique())

# Restrict to current managers only
season_df  = season_df[season_df['manager'].isin(current_managers)].copy()
career_df  = career_df[career_df['manager'].isin(current_managers)].copy()
if not keeper_era_df.empty:
    keeper_era_df = keeper_era_df[keeper_era_df['manager'].isin(current_managers)].copy()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title('Fantasy Football League')
st.caption(
    f'{season_df["season"].min()}–{latest_season} seasons  '
    f'·  {len(current_managers)} active managers  '
    f'·  data through {latest_season}'
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_trends, tab_analytics, tab_profile, tab_keeper, tab_keeper_perf = st.tabs([
    'League Overview', 'Season Trends', 'Analytics', 'Manager Profile', 'Keeper Era', 'Keeper Performance'
])


# ===========================================================================
# TAB 1 — League Overview
# ===========================================================================

with tab_overview:
    st.subheader('Career Summary')

    tbl = career_df[[
        'manager', 'seasons_played', 'total_wins', 'total_losses',
        'win_pct', 'championships', 'playoff_apps',
        'best_finish', 'avg_finish', 'career_avg_ppw',
        'career_luck_wins', 'career_draft_overperformance',
    ]].rename(columns={
        'manager':                    'Manager',
        'seasons_played':             'Seasons',
        'total_wins':                 'W',
        'total_losses':               'L',
        'win_pct':                    'Win%',
        'championships':              'Titles',
        'playoff_apps':               'Playoffs',
        'best_finish':                'Best Finish',
        'avg_finish':                 'Avg Finish',
        'career_avg_ppw':             'Avg PPW',
        'career_luck_wins':           'Luck W',
        'career_draft_overperformance': 'Draft +/-',
    }).sort_values('Win%', ascending=False).reset_index(drop=True)

    tbl.index = tbl.index + 1

    tbl_disp = tbl.copy()
    for col, fmt in {
        'Win%': '{:.1%}', 'Avg PPW': '{:.1f}', 'Avg Finish': '{:.1f}',
        'Luck W': '{:+.1f}', 'Draft +/-': '{:+.0f}',
    }.items():
        if col in tbl_disp.columns:
            tbl_disp[col] = tbl_disp[col].map(lambda x, f=fmt: f.format(x) if pd.notna(x) else '-')
    st.dataframe(tbl_disp, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            career_df.sort_values('win_pct', ascending=True),
            x='win_pct', y='manager', orientation='h',
            title='Career Win %',
            labels={'win_pct': 'Win %', 'manager': ''},
            text=career_df.sort_values('win_pct', ascending=True)['win_pct'].map('{:.1%}'.format),
            color='win_pct', color_continuous_scale='Blues',
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            career_df.sort_values('career_avg_ppw', ascending=True),
            x='career_avg_ppw', y='manager', orientation='h',
            title='Career Avg Points Per Week',
            labels={'career_avg_ppw': 'Avg PPW', 'manager': ''},
            text=career_df.sort_values('career_avg_ppw', ascending=True)['career_avg_ppw'].map('{:.1f}'.format),
            color='career_avg_ppw', color_continuous_scale='Greens',
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.bar(
            career_df.sort_values('championships', ascending=True),
            x='championships', y='manager', orientation='h',
            title='Championships',
            labels={'championships': 'Titles', 'manager': ''},
            text='championships',
            color='championships', color_continuous_scale='Oranges',
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.bar(
            career_df.sort_values('playoff_apps', ascending=True),
            x='playoff_apps', y='manager', orientation='h',
            title='Playoff Appearances',
            labels={'playoff_apps': 'Appearances', 'manager': ''},
            text='playoff_apps',
            color='playoff_apps', color_continuous_scale='Purples',
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# TAB 2 — Season Trends
# ===========================================================================

with tab_trends:
    st.subheader('Season-by-Season Trends')

    selected = st.multiselect(
        'Managers', current_managers, default=current_managers,
        key='trends_mgrs',
    )

    if not selected:
        st.warning('Select at least one manager.')
    else:
        df_sel = season_df[season_df['manager'].isin(selected)]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                df_sel, x='season', y='points_for', color='manager',
                title='Total Points For by Season',
                labels={'points_for': 'Points', 'season': 'Season'},
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                df_sel, x='season', y='rank', color='manager',
                title='Final Rank by Season',
                labels={'rank': 'Rank (1 = best)', 'season': 'Season'},
                markers=True,
            )
            fig.update_yaxes(autorange='reversed', dtick=1)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig = px.line(
                df_sel, x='season', y='avg_pts_per_week', color='manager',
                title='Avg Points Per Week by Season',
                labels={'avg_pts_per_week': 'Avg PPW', 'season': 'Season'},
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.bar(
                df_sel, x='season', y='wins', color='manager',
                title='Wins by Season',
                labels={'wins': 'Wins', 'season': 'Season'},
                barmode='group',
            )
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# TAB 3 — Analytics
# ===========================================================================

with tab_analytics:
    st.subheader('Advanced Analytics')

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            career_df.sort_values('career_luck_wins'),
            x='career_luck_wins', y='manager', orientation='h',
            title='Career Luck Wins (Actual minus Expected)',
            labels={'career_luck_wins': 'Luck Wins', 'manager': ''},
            text=career_df.sort_values('career_luck_wins')['career_luck_wins'].map('{:+.1f}'.format),
            color='career_luck_wins',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Positive = favorable schedule; Negative = tough schedule')

    with col2:
        fig = px.bar(
            career_df.sort_values('career_draft_overperformance'),
            x='career_draft_overperformance', y='manager', orientation='h',
            title='Career Draft Overperformance',
            labels={'career_draft_overperformance': 'Points vs Expected', 'manager': ''},
            text=career_df.sort_values('career_draft_overperformance')['career_draft_overperformance'].map('{:+.0f}'.format),
            color='career_draft_overperformance',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Sum of (actual pts - expected pts for pick slot) across all drafted players')

    col3, col4 = st.columns(2)

    with col3:
        fig = px.bar(
            career_df.sort_values('career_avg_pts_left_per_week', ascending=False),
            x='career_avg_pts_left_per_week', y='manager', orientation='h',
            title='Avg Points Left on Bench Per Week',
            labels={'career_avg_pts_left_per_week': 'Pts Left on Bench', 'manager': ''},
            text=career_df.sort_values('career_avg_pts_left_per_week', ascending=False)['career_avg_pts_left_per_week'].map('{:.1f}'.format),
            color='career_avg_pts_left_per_week',
            color_continuous_scale='Reds_r',
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Higher = more points left on bench (worse lineup management)')

    with col4:
        fig = px.bar(
            career_df.sort_values('career_trade_value'),
            x='career_trade_value', y='manager', orientation='h',
            title='Career Trade Value',
            labels={'career_trade_value': 'Trade Value (pts)', 'manager': ''},
            text=career_df.sort_values('career_trade_value')['career_trade_value'].map('{:+.0f}'.format),
            color='career_trade_value',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Post-trade points received minus points given away')

    st.divider()
    st.subheader('Luck vs Scoring (all seasons)')

    fig = px.scatter(
        season_df,
        x='luck_wins', y='avg_pts_per_week',
        color='manager', size='wins',
        hover_data={'season': True, 'wins': True, 'rank': True, 'points_for': True},
        title='Schedule Luck vs Scoring Output  (each point = one manager-season, sized by wins)',
        labels={'luck_wins': 'Luck Wins', 'avg_pts_per_week': 'Avg PPW'},
    )
    fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_hline(
        y=season_df['avg_pts_per_week'].mean(),
        line_dash='dash', line_color='gray', opacity=0.5,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader('Strength of Schedule by Season')

    fig = px.box(
        season_df, x='manager', y='strength_of_schedule',
        title='Strength of Schedule Distribution (all seasons)',
        labels={'strength_of_schedule': 'Opp Avg PPW', 'manager': ''},
        color='manager',
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# TAB 4 — Manager Profile
# ===========================================================================

with tab_profile:
    st.subheader('Manager Profile')

    mgr = st.selectbox('Select manager', current_managers, key='profile_mgr')

    mgr_career  = career_df[career_df['manager'] == mgr].iloc[0]
    mgr_seasons = season_df[season_df['manager'] == mgr].sort_values('season')

    # KPI row
    k = st.columns(6)
    k[0].metric('Seasons',       int(mgr_career['seasons_played']))
    k[1].metric('Record',        f'{int(mgr_career["total_wins"])}-{int(mgr_career["total_losses"])}')
    k[2].metric('Win %',         f'{mgr_career["win_pct"]:.1%}')
    k[3].metric('Championships', int(mgr_career['championships']))
    k[4].metric('Playoff Apps',  int(mgr_career['playoff_apps']))
    k[5].metric('Avg PPW',       f'{mgr_career["career_avg_ppw"]:.1f}')

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        colors = mgr_seasons['made_playoffs'].map({True: '#2ecc71', False: '#e74c3c'})
        fig = go.Figure(go.Bar(
            x=mgr_seasons['season'],
            y=mgr_seasons['points_for'],
            marker_color=colors,
            text=mgr_seasons['points_for'].round(1),
            textposition='outside',
        ))
        fig.update_layout(
            title=f'{mgr} — Points For by Season',
            xaxis_title='Season', yaxis_title='Points',
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Green = made playoffs  |  Red = missed playoffs')

    with col2:
        colors2 = mgr_seasons['champion'].map({True: '#f39c12', False: '#3498db'})
        fig2 = go.Figure(go.Bar(
            x=mgr_seasons['season'],
            y=mgr_seasons['rank'],
            marker_color=colors2,
            text=mgr_seasons['rank'].astype(int),
            textposition='outside',
        ))
        fig2.update_layout(
            title=f'{mgr} — Final Rank by Season',
            xaxis_title='Season', yaxis_title='Rank (1 = best)',
            yaxis=dict(autorange='reversed', dtick=1),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption('Gold = champion  |  Blue = non-champion')

    # Season analytics chart
    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.bar(
            mgr_seasons, x='season', y='luck_wins',
            title=f'{mgr} — Luck Wins by Season',
            labels={'luck_wins': 'Luck Wins', 'season': 'Season'},
            color='luck_wins',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            text=mgr_seasons['luck_wins'].map('{:+.0f}'.format),
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.bar(
            mgr_seasons, x='season', y='draft_overperformance',
            title=f'{mgr} — Draft Overperformance by Season',
            labels={'draft_overperformance': 'Points vs Expected', 'season': 'Season'},
            color='draft_overperformance',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            text=mgr_seasons['draft_overperformance'].round(0).map('{:+.0f}'.format),
        )
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader('Season-by-Season Stats')

    show = [
        'season', 'team_name', 'wins', 'losses', 'points_for', 'avg_pts_per_week',
        'rank', 'made_playoffs', 'champion', 'luck_wins',
        'draft_overperformance', 'trade_value', 'avg_pts_left_per_week',
        'number_of_moves', 'number_of_trades',
    ]
    avail = [c for c in show if c in mgr_seasons.columns]
    disp  = mgr_seasons[avail].reset_index(drop=True)
    disp.index = disp.index + 1

    disp_show = disp.copy()
    for col, fmt in {
        'points_for': '{:.1f}', 'avg_pts_per_week': '{:.1f}',
        'luck_wins': '{:+.0f}', 'draft_overperformance': '{:+.0f}',
        'trade_value': '{:+.0f}', 'avg_pts_left_per_week': '{:.1f}',
    }.items():
        if col in disp_show.columns:
            disp_show[col] = disp_show[col].map(lambda x, f=fmt: f.format(x) if pd.notna(x) else '-')
    st.dataframe(disp_show, use_container_width=True)


# ===========================================================================
# TAB 5 — Keeper Era
# ===========================================================================

with tab_keeper:
    st.subheader('Keeper Era Comparison')
    st.caption('Pre-keeper: seasons before 2016  |  Keeper era: 2016 onwards')

    if keeper_era_df.empty:
        st.warning('Keeper era data not available. Run `python run.py --analyze`.')
    else:
        # Label is_keeper column
        kdf = keeper_era_df.copy()
        kdf['era'] = kdf['is_keeper'].map({1: 'Keeper Era (2016+)', 0: 'Pre-Keeper'})

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                kdf, x='manager', y='win_pct', color='era',
                barmode='group',
                title='Win % by Era',
                labels={'win_pct': 'Win %', 'manager': ''},
                text=kdf['win_pct'].map('{:.1%}'.format),
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                kdf, x='manager', y='avg_ppw', color='era',
                barmode='group',
                title='Avg Points Per Week by Era',
                labels={'avg_ppw': 'Avg PPW', 'manager': ''},
                text=kdf['avg_ppw'].round(1).map('{:.1f}'.format),
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig = px.bar(
                kdf, x='manager', y='luck_wins', color='era',
                barmode='group',
                title='Luck Wins by Era',
                labels={'luck_wins': 'Luck Wins', 'manager': ''},
                text=kdf['luck_wins'].map('{:+.1f}'.format),
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            if 'draft_overperformance' in kdf.columns:
                fig = px.bar(
                    kdf, x='manager', y='draft_overperformance', color='era',
                    barmode='group',
                    title='Draft Overperformance by Era',
                    labels={'draft_overperformance': 'Points vs Expected', 'manager': ''},
                    text=kdf['draft_overperformance'].round(0).map('{:+.0f}'.format),
                )
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader('Era Summary Table')

        era_tbl = kdf[[
            'manager', 'era', 'seasons_played', 'total_wins', 'total_losses',
            'win_pct', 'championships', 'playoff_apps', 'avg_ppw',
            'luck_wins', 'avg_pts_left_per_week',
        ]].rename(columns={
            'seasons_played':       'Seasons',
            'total_wins':           'W',
            'total_losses':         'L',
            'win_pct':              'Win%',
            'championships':        'Titles',
            'playoff_apps':         'Playoffs',
            'avg_ppw':              'Avg PPW',
            'luck_wins':            'Luck W',
            'avg_pts_left_per_week': 'Bench Waste',
        }).sort_values(['manager', 'era'])

        for col, fmt in {
            'Win%': '{:.1%}', 'Avg PPW': '{:.1f}', 'Luck W': '{:+.1f}', 'Bench Waste': '{:.1f}',
        }.items():
            if col in era_tbl.columns:
                era_tbl[col] = era_tbl[col].map(lambda x, f=fmt: f.format(x) if pd.notna(x) else '-')
        st.dataframe(era_tbl, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 6 — Keeper Performance
# ===========================================================================

with tab_keeper_perf:
    st.subheader('Keeper Optimal Performance')
    st.caption(
        'Value = player pts minus benchmark (Nth-best same-position pts in year N, '
        'where N = positional draft rank).  Gap = optimal value sum minus actual value sum.'
    )

    if keeper_summary_df.empty:
        st.warning('Keeper analysis data not found. Run `python run.py --analyze` to generate it.')
    else:
        ks = keeper_summary_df[keeper_summary_df['manager'].isin(current_managers)].copy()
        kd = keeper_detail_df[keeper_detail_df['manager'].isin(current_managers)].copy()

        # ── Career leaderboard ────────────────────────────────────────────
        st.subheader('Career Keeper Leaderboard')
        st.caption('Lower gap = manager left less value on the table by choosing sub-optimal keepers.')

        career_ks = (
            ks.groupby('manager')
            .agg(
                total_actual   = ('actual_value',  'sum'),
                total_optimal  = ('optimal_value', 'sum'),
                total_gap      = ('gap',           'sum'),
                optimal_seasons= ('is_optimal',    'sum'),
                seasons        = ('season',        'count'),
            )
            .reset_index()
        )
        career_ks['opt_rate'] = career_ks['optimal_seasons'] / career_ks['seasons']
        career_ks = career_ks.sort_values('total_gap')

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                career_ks,
                x='total_gap', y='manager', orientation='h',
                title='Total Value Gap vs Optimal (career)',
                labels={'total_gap': 'Pts Left on Table', 'manager': ''},
                text=career_ks['total_gap'].map('{:+.0f}'.format),
                color='total_gap',
                color_continuous_scale='RdYlGn_r',
                color_continuous_midpoint=career_ks['total_gap'].median(),
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                career_ks.sort_values('opt_rate', ascending=True),
                x='opt_rate', y='manager', orientation='h',
                title='Optimal Keeper Season Rate',
                labels={'opt_rate': '% Seasons Optimal', 'manager': ''},
                text=career_ks.sort_values('opt_rate')['opt_rate'].map('{:.0%}'.format),
                color='opt_rate',
                color_continuous_scale='Greens',
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)

        ks_career_disp = career_ks.rename(columns={
            'manager':        'Manager',
            'total_actual':   'Actual Value',
            'total_optimal':  'Optimal Value',
            'total_gap':      'Total Gap',
            'optimal_seasons':'Optimal Seasons',
            'seasons':        'Seasons',
            'opt_rate':       'Opt Rate',
        })
        for col, fmt in {
            'Actual Value': '{:+.1f}', 'Optimal Value': '{:+.1f}',
            'Total Gap': '{:+.1f}', 'Opt Rate': '{:.0%}',
        }.items():
            if col in ks_career_disp.columns:
                ks_career_disp[col] = ks_career_disp[col].map(lambda x, f=fmt: f.format(x) if pd.notna(x) else '-')
        st.dataframe(ks_career_disp, use_container_width=True, hide_index=True)

        st.divider()

        # ── Season-by-season heatmap ──────────────────────────────────────
        st.subheader('Gap by Manager by Season')

        pivot = ks.pivot(index='manager', columns='season', values='gap')
        fig = px.imshow(
            pivot,
            color_continuous_scale='RdYlGn_r',
            color_continuous_midpoint=0,
            text_auto='.0f',
            aspect='auto',
            title='Value Gap vs Optimal (green = smaller gap = better decisions)',
            labels={'color': 'Gap'},
        )
        fig.update_coloraxes(colorbar_title='Gap')
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Season filter: actual vs optimal by manager ───────────────────
        st.subheader('Actual vs Optimal Value — by Season')

        seasons_avail = sorted(ks['season'].unique(), reverse=True)
        sel_season = st.selectbox('Select season', seasons_avail, key='kp_season')

        ks_yr = ks[ks['season'] == sel_season].sort_values('gap')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Actual Value',
            x=ks_yr['manager'],
            y=ks_yr['actual_value'],
            marker_color='steelblue',
            text=ks_yr['actual_value'].map('{:+.1f}'.format),
            textposition='outside',
        ))
        fig.add_trace(go.Bar(
            name='Optimal Value',
            x=ks_yr['manager'],
            y=ks_yr['optimal_value'],
            marker_color='seagreen',
            text=ks_yr['optimal_value'].map('{:+.1f}'.format),
            textposition='outside',
        ))
        fig.update_layout(
            barmode='group',
            title=f'{sel_season} — Actual vs Optimal Keeper Value',
            xaxis_title='', yaxis_title='Value (pts above benchmark)',
        )
        st.plotly_chart(fig, use_container_width=True)

        # Season summary table
        ks_yr_disp = ks_yr[['manager', 'actual_keepers', 'actual_value',
                             'optimal_keepers', 'optimal_value', 'gap', 'is_optimal']].rename(columns={
            'manager':         'Manager',
            'actual_keepers':  'Kept',
            'actual_value':    'Actual Val',
            'optimal_keepers': 'Should Have Kept',
            'optimal_value':   'Optimal Val',
            'gap':             'Gap',
            'is_optimal':      'Optimal?',
        })
        for col, fmt in {'Actual Val': '{:+.1f}', 'Optimal Val': '{:+.1f}', 'Gap': '{:+.1f}'}.items():
            if col in ks_yr_disp.columns:
                ks_yr_disp[col] = ks_yr_disp[col].map(lambda x, f=fmt: f.format(x) if pd.notna(x) else '-')
        st.dataframe(ks_yr_disp, use_container_width=True, hide_index=True)

        st.divider()

        # ── Manager drilldown ─────────────────────────────────────────────
        st.subheader('Manager Drilldown — Full Eligible Roster')

        col_mgr, col_yr = st.columns(2)
        with col_mgr:
            sel_mgr = st.selectbox('Manager', current_managers, key='kp_mgr')
        with col_yr:
            mgr_seasons = sorted(ks[ks['manager'] == sel_mgr]['season'].unique(), reverse=True)
            sel_yr2 = st.selectbox('Season', mgr_seasons, key='kp_yr2')

        drill = kd[(kd['manager'] == sel_mgr) & (kd['season'] == sel_yr2)].copy()

        # Summary line
        summary_row = ks[(ks['manager'] == sel_mgr) & (ks['season'] == sel_yr2)]
        if not summary_row.empty:
            r = summary_row.iloc[0]
            verdict = 'OPTIMAL' if r['is_optimal'] else f'Gap: {r["gap"]:+.1f} pts'
            st.markdown(
                f'**{sel_mgr} — {sel_yr2}**  |  '
                f'Actual value: **{r["actual_value"]:+.1f}**  |  '
                f'Optimal value: **{r["optimal_value"]:+.1f}**  |  '
                f'**{verdict}**'
            )

        def _status(row):
            if row['is_ineligible']:
                return 'Ineligible'
            if row['is_kept'] and row['is_optimal']:
                return 'Kept + Optimal'
            if row['is_kept']:
                return 'Kept (not optimal)'
            if row['is_optimal']:
                return 'Should have kept'
            return ''

        display_drill = drill[[
            'player', 'pos', 'rd', 'pos_rank', 'pts', 'benchmark', 'value',
            'is_kept', 'is_optimal', 'is_ineligible',
        ]].copy()
        display_drill['status'] = display_drill.apply(_status, axis=1)
        display_drill = display_drill[[
            'player', 'pos', 'rd', 'pos_rank', 'pts', 'benchmark', 'value', 'status',
        ]].rename(columns={
            'player':    'Player',
            'pos':       'Pos',
            'rd':        'Rd',
            'pos_rank':  'PosRk',
            'pts':       'Pts',
            'benchmark': 'Benchmark',
            'value':     'Value',
            'status':    'Status',
        }).reset_index(drop=True)
        display_drill['Pts']       = display_drill['Pts'].map('{:.1f}'.format)
        display_drill['Benchmark'] = display_drill['Benchmark'].map('{:.1f}'.format)
        display_drill['Value']     = display_drill['Value'].map('{:+.1f}'.format)

        st.dataframe(display_drill, use_container_width=True, hide_index=True)
        st.caption(
            'Kept + Optimal = correct choice  |  '
            'Kept (not optimal) = sub-optimal keep  |  '
            'Should have kept = missed optimal  |  '
            'Ineligible = kept 2 years running'
        )
