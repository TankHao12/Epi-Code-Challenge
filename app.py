"""
EPI Coding Challenge -- Interactive Data Explorer (Task 4 + Visualisations)

Streamlit app that loads the cleaned dataset (produced by analysis.ipynb)
and provides an interactive filtering interface with summary statistics
and visualisations.

Run with:
    streamlit run app.py
"""

import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

#### Page config ####

st.set_page_config(
    page_title="World Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

#### Load cleaned data ####

DATA_PATH = Path(__file__).parent / "cleaned_worldData.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the cleaned dataset produced by the analysis notebook."""
    return pd.read_csv(DATA_PATH)


df = load_data()

#### Title ####

st.title("World Data Explorer")
st.markdown(
    "Interactive filtering interface for the cleaned World dataset."
    "Use the sidebar to filter, or leave empty to show all."
)

#### Sidebar filters ####

st.sidebar.title("Filters")

# Generation counter for filter widget keys.
if "filter_gen" not in st.session_state:
    st.session_state.filter_gen = 0
_gen = st.session_state.filter_gen

st.sidebar.caption("Leave empty to show all data.")

# --- Continent pills ---
all_continents = sorted(df["continent"].dropna().unique())
sel_continents_raw = st.sidebar.pills(
    "Continent",
    all_continents,
    selection_mode="multi",
    key=f"f_continent_{_gen}",
    help="Click to filter by continent. Leave empty to show all.",
)
sel_continents = sel_continents_raw if sel_continents_raw else all_continents

# --- Type pills ---
all_types = sorted(df["type"].dropna().unique())
sel_types_raw = st.sidebar.pills(
    "Type",
    all_types,
    selection_mode="multi",
    key=f"f_type_{_gen}",
    help=(
        "Sovereign country = independent nation | "
        "Country = broadly recognised state | "
        "Dependency = governed by another country | "
        "Disputed = contested sovereignty | "
        "Indeterminate = status unclear"
    ),
)
sel_types = sel_types_raw if sel_types_raw else all_types

# --- Subregion multiselect (cascading from continent, empty = all) ---
available_subregions = sorted(
    df[df["continent"].isin(sel_continents)]["subregion"].dropna().unique()
)
sel_subregions_raw = st.sidebar.multiselect(
    "Subregion",
    available_subregions,
    key=f"f_subregion_{_gen}",
    placeholder="All subregions (click to narrow down)",
)
sel_subregions = sel_subregions_raw if sel_subregions_raw else available_subregions

# --- Numeric range sliders ---
# Population -- displayed in millions for readability
pop_range_m = (0.0, float(math.ceil(df["pop"].dropna().max() / 1_000_000)))
sel_pop_m = st.sidebar.slider(
    "Population (millions)",
    min_value=pop_range_m[0],
    max_value=pop_range_m[1],
    value=pop_range_m,
    step=1.0,
    key=f"f_pop_{_gen}",
    format="%.0f",
)
pop_is_default = sel_pop_m[0] == pop_range_m[0] and sel_pop_m[1] == pop_range_m[1]

# Life Expectancy
life_range = (
    float(math.floor(df["lifeExp"].dropna().min())),
    float(math.ceil(df["lifeExp"].dropna().max())),
)
sel_life = st.sidebar.slider(
    "Life Expectancy (years)",
    min_value=life_range[0],
    max_value=life_range[1],
    value=life_range,
    step=1.0,
    key=f"f_life_{_gen}",
    format="%.0f",
)
life_is_default = sel_life[0] == life_range[0] and sel_life[1] == life_range[1]

# GDP per Capita
gdp_range = (0.0, float(math.ceil(df["gdpPercap"].dropna().max() / 1000) * 1000))
sel_gdp = st.sidebar.slider(
    "GDP per Capita ($)",
    min_value=gdp_range[0],
    max_value=gdp_range[1],
    value=gdp_range,
    step=500.0,
    key=f"f_gdp_{_gen}",
    format="%.0f",
)
gdp_is_default = sel_gdp[0] == gdp_range[0] and sel_gdp[1] == gdp_range[1]

#### Apply all filters ####
# When a numeric slider is at its default (full range), all rows pass --
# including those with NaN. When the user actively adjusts a slider, only
# rows with a value inside the selected range pass; NaN rows are excluded
# because the user is explicitly filtering on that metric.

filtered = df[
    df["continent"].isin(sel_continents)
    & df["subregion"].isin(sel_subregions)
    & df["type"].isin(sel_types)
    & (pop_is_default | df["pop"].between(sel_pop_m[0] * 1_000_000, sel_pop_m[1] * 1_000_000))
    & (life_is_default | df["lifeExp"].between(sel_life[0], sel_life[1]))
    & (gdp_is_default | df["gdpPercap"].between(sel_gdp[0], sel_gdp[1]))
]

# --- Sidebar: active filter summary ---

active_filters: list[str] = []
if sel_continents_raw:
    active_filters.append(f"**Continent:** {', '.join(sel_continents_raw)}")
if sel_subregions_raw:
    active_filters.append(f"**Subregion:** {', '.join(sel_subregions_raw)}")
if sel_types_raw:
    active_filters.append(f"**Type:** {', '.join(sel_types_raw)}")
if not pop_is_default:
    active_filters.append(f"**Population:** {sel_pop_m[0]:.0f}M -- {sel_pop_m[1]:.0f}M")
if not life_is_default:
    active_filters.append(
        f"**Life Expectancy:** {sel_life[0]:.0f} -- {sel_life[1]:.0f} years"
    )
if not gdp_is_default:
    active_filters.append(f"**GDP per Capita:** \\${sel_gdp[0]:,.0f} -- \\${sel_gdp[1]:,.0f}")

c1, c2 = st.sidebar.columns(2)
c1.metric("Showing", len(filtered))
c2.metric("Total", len(df))

if active_filters:
    st.sidebar.markdown("**Active filters:**")
    for f in active_filters:
        st.sidebar.markdown(f"- {f}")


    # Reset button -- increments the generation counter so every widget
    # gets a fresh key on the next run, forcing default values.
    def _reset_filters() -> None:
        st.session_state.filter_gen += 1

    st.sidebar.button(
        "Reset all filters",
        on_click=_reset_filters,
        use_container_width=True,
    )
else:
    st.sidebar.caption("No filters active -- showing all data.")

#### Tabs ####

tab_stats, tab_viz = st.tabs(["Summary Statistics", "Visualisations"])

#### Tab 1: Summary Statistics ####

with tab_stats:
    if filtered.empty:
        st.warning("No countries match the current filters.")
    else:
        st.subheader("Summary Statistics")
        st.markdown(
            f"Descriptive statistics for the **{len(filtered)}** countries "
            "matching the current filters."
        )

        stat_cols = ["area_km2", "pop", "lifeExp", "gdpPercap"]
        stats = filtered[stat_cols].describe()
        stats.loc["total"] = filtered[stat_cols].sum()
        stats.index = [
            "Count", "Mean", "Std Dev", "Min",
            "25th Percentile", "Median", "75th Percentile", "Max", "Total",
        ]
        stats = stats.rename(columns={
            "area_km2": "Area (sq. km)",
            "pop": "Population",
            "lifeExp": "Life Expectancy (years)",
            "gdpPercap": "GDP per Capita ($)",
        })

        st.dataframe(
            stats.style.format("{:,.2f}", na_rep="--"),
            use_container_width=True,
        )

        st.subheader("Filtered Data")
        display_cols = [
            "iso_a2", "name_long", "continent", "region_un", "subregion",
            "type", "area_km2", "pop", "lifeExp", "gdpPercap",
        ]
        col_labels = {
            "iso_a2": "Code",
            "name_long": "Name",
            "continent": "Continent",
            "region_un": "Region",
            "subregion": "Subregion",
            "type": "Type",
            "area_km2": "Area (sq km)",
            "pop": "Population",
            "lifeExp": "Life Expectancy (years)",
            "gdpPercap": "GDP per Capita ($)",
        }
        st.dataframe(
            filtered[display_cols]
            .rename(columns=col_labels)
            .style.format(
                {
                    "Area (sq km)": "{:,.0f}",
                    "Population": "{:,.0f}",
                    "Life Expectancy (years)": "{:.2f}",
                    "GDP per Capita ($)": "{:,.2f}",
                },
                na_rep="--",
            ),
            use_container_width=True,
            height=400,
        )

#### Tab 2: Visualisations ####

with tab_viz:
    if filtered.empty:
        st.warning("No data to visualise -- broaden the filters above.")
    else:
        # Average population density by region (bar chart)
        st.subheader("Average Population Density by Region")
        st.markdown(
            "Population density = total population / total area (sq. km) for each region."
        )
        density_data = filtered.dropna(subset=["pop", "area_km2"]).copy()
        if not density_data.empty:
            region_agg = density_data.groupby("region_un").agg(
                total_pop=("pop", "sum"),
                total_area=("area_km2", "sum"),
            )
            region_agg["pop_density"] = region_agg["total_pop"] / region_agg["total_area"]
            density = (
                region_agg["pop_density"]
                .sort_values(ascending=False)
                .reset_index()
            )
            density.columns = ["Region", "Avg Pop. Density (per sq. km)"]

            fig_density = px.bar(
                density,
                x="Region",
                y="Avg Pop. Density (per sq. km)",
                title="Average Population Density by Region",
            )
            fig_density.update_layout(xaxis_tickangle=-45)
            fig_density.update_yaxes(tickformat=",.1f")
            fig_density.update_traces(hovertemplate="Region: %{x}<br>Avg Pop. Density: %{y:.1f} per sq. km<extra></extra>")
            st.plotly_chart(fig_density, use_container_width=True)

        # Life Expectancy vs GDP per Capita (Gapminder-style scatter)
        st.subheader("Life Expectancy vs GDP per Capita")
        st.markdown(
            "Bubble size represents population; colour represents continent."
        )
        scatter = filtered.dropna(subset=["lifeExp", "gdpPercap", "pop"])
        if not scatter.empty:
            fig_scatter = px.scatter(
                scatter,
                x="gdpPercap",
                y="lifeExp",
                size="pop",
                color="continent",
                custom_data=["name_long", "continent", "pop"],
                labels={
                    "gdpPercap": "GDP per Capita ($)",
                    "lifeExp": "Life Expectancy (years)",
                    "pop": "Population",
                },
                title="Life Expectancy vs GDP per Capita",
                size_max=55,
            )
            fig_scatter.update_xaxes(tickformat="$,.0f")
            fig_scatter.update_traces(
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Continent: %{customdata[1]}<br>"
                    "GDP per Capita: $%{x:,.0f}<br>"
                    "Life Expectancy: %{y:.2f} years<br>"
                    "Population: %{customdata[2]:,}"
                    "<extra></extra>"
                )
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Top 15 most populous countries
        st.subheader("Top 15 Most Populous Countries")
        pop_data = filtered.dropna(subset=["pop"]).nlargest(15, "pop")
        if not pop_data.empty:
            fig_pop = px.bar(
                pop_data.sort_values("pop", ascending=True),
                x="pop",
                y="name_long",
                orientation="h",
                color="continent",
                labels={"name_long": "Name", "continent": "Continent", "pop": "Population"},
                title="Top 15 Most Populous Countries (Filtered)",
            )
            fig_pop.update_yaxes(categoryorder="total ascending")
            fig_pop.update_xaxes(tickformat=",")
            fig_pop.update_traces(hovertemplate="%{y}<br>Population: %{x:,}<extra></extra>")
            st.plotly_chart(fig_pop, use_container_width=True)

        # GDP per Capita world map (choropleth)
        st.subheader("GDP per Capita -- World Map")
        st.markdown(
            "Choropleth map showing GDP per capita by country. "
            "Darker shading indicates higher GDP per capita."
        )
        map_data = filtered.dropna(subset=["gdpPercap", "name_long"])
        if not map_data.empty:
            fig_map = px.choropleth(
                map_data,
                locations="name_long",
                locationmode="country names",
                color="gdpPercap",
                hover_name="name_long",
                hover_data={
                    "name_long": False,
                    "gdpPercap": ":$,.0f",
                    "continent": True,
                    "lifeExp": ":.2f",
                    "pop": ":,",
                },
                labels={
                    "gdpPercap": "GDP per Capita",
                    "continent": "Continent",
                    "lifeExp": "Life Expectancy",
                    "pop": "Population",
                },
                color_continuous_scale="PuRd",
                title="GDP per Capita by Country",
            )
            fig_map.update_layout(
                geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
                coloraxis_colorbar=dict(title="GDP ($)", tickformat="$,.0f"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # GDP per Capita distribution by continent (box plot)
        st.subheader("GDP per Capita Distribution by Continent")
        gdp_data = filtered.dropna(subset=["gdpPercap"])
        if not gdp_data.empty:
            fig_box = px.box(
                gdp_data,
                x="continent",
                y="gdpPercap",
                color="continent",
                hover_data={"name_long": True, "continent": False, "gdpPercap": False},
                labels={
                    "gdpPercap": "GDP per Capita ($)",
                    "continent": "",
                    "name_long": "Country",
                },
                title="GDP per Capita Distribution by Continent",
            )
            fig_box.update_layout(showlegend=False)
            fig_box.update_yaxes(tickformat="$,.0f")
            st.plotly_chart(fig_box, use_container_width=True)
