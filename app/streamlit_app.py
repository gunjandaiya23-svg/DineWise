"""DineWise Streamlit interface."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import data_path, load_config  # noqa: E402
from dinewise.graph.workflow import build_graph, run_query  # noqa: E402
from dinewise.image.visual_index import build_visual_index  # noqa: E402
from dinewise.llm.generator import Generator  # noqa: E402
from dinewise.rag.bm25_index import load_bm25  # noqa: E402
from dinewise.rag.search import load_retriever  # noqa: E402

CITIES = ["Philadelphia"]
CUISINES = [
    "Italian", "Mexican", "Chinese", "Japanese", "Thai", "Indian",
    "Mediterranean", "American", "Seafood", "BBQ", "French", "Vietnamese",
]


@st.cache_resource(show_spinner="Loading retrieval stack ...")
def load_assets():
    cfg = load_config()
    name = cfg["chunking"]["active"]
    restaurants = pd.read_parquet(data_path(cfg, "processed_dir", "restaurants.parquet"))
    chunks = pd.read_parquet(data_path(cfg, "index_dir", "faiss", name, "chunks.parquet"))
    retriever = load_retriever(cfg, name)
    bm25 = load_bm25(cfg, name)
    return cfg, restaurants, chunks, retriever, bm25


@st.cache_resource(show_spinner="Building workflow ...")
def load_graph(use_llm: bool):
    cfg, restaurants, chunks, retriever, bm25 = load_assets()
    generator = Generator(cfg, use_llm=use_llm)
    return build_graph(cfg, retriever, bm25, chunks, restaurants, generator)


@st.cache_resource(show_spinner="Building visual index ...")
def load_visual():
    cfg, restaurants, _, _, _ = load_assets()
    return build_visual_index(restaurants, load_retriever(cfg).model)


def effective_query(query: str, profile: dict) -> str:
    parts = [query]
    lowered = query.lower()
    if profile.get("city") and profile["city"].lower() not in lowered:
        parts.append(f"in {profile['city']}")
    if profile.get("cuisines"):
        for cuisine in profile["cuisines"]:
            if cuisine.lower() not in lowered:
                parts.append(cuisine)
    if profile.get("max_price"):
        parts.append("under " + "$" * int(profile["max_price"]))
    if profile.get("ambience") and profile["ambience"].lower() not in lowered:
        parts.append(profile["ambience"])
    return " ".join(parts)


def render_result(state: dict) -> None:
    answer = state.get("answer", "")
    if state.get("passed"):
        st.success("Validated: all hard grounding checks passed.")
    elif state.get("out_of_scope") or state.get("fallback_used"):
        st.warning("This request is outside the dataset's scope.")
    st.markdown(answer)

    hits = state.get("hits", [])
    if hits:
        rows = [
            {
                "Restaurant": hit["chunk"].get("restaurant_name"),
                "ID": hit["chunk"].get("business_id"),
                "Cuisine": hit["chunk"].get("cuisine_primary"),
                "Price": "$" * int(hit["chunk"].get("price_range") or 0) or "unknown",
                "Rating": hit["chunk"].get("stars"),
                "Score": round(float(hit.get("score", 0.0)), 4),
            }
            for hit in hits
        ]
        st.subheader("Ranked candidates")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Debug: constraints, validation, trace"):
        constraints = state.get("constraints")
        if constraints is not None:
            st.json(
                {
                    "city": constraints.city,
                    "cuisines": constraints.cuisines,
                    "max_price_bucket": constraints.max_price_bucket,
                    "required_attributes": constraints.required_attributes,
                    "dietary": constraints.dietary,
                    "intent": constraints.intent,
                }
            )
        st.json(state.get("validation", {}))
        st.code("\n".join(state.get("trace", [])))


def main() -> None:
    st.set_page_config(page_title="DineWise", page_icon="🍽️", layout="wide")
    st.title("DineWise")
    st.caption(
        "Retrieval-augmented restaurant recommendations over a static dataset. "
        "Ratings and prices are dataset values, not live information."
    )

    with st.sidebar:
        st.header("Your profile")
        city = st.selectbox("City", CITIES)
        cuisines = st.multiselect("Preferred cuisines", CUISINES)
        max_price = st.slider("Maximum price bucket", 1, 4, 2)
        ambience = st.selectbox(
            "Ambience", ["", "romantic", "quiet", "casual", "classy", "family"]
        )
        dietary = st.multiselect(
            "Dietary requirements", ["vegetarian", "vegan", "gluten-free"]
        )
        use_llm = st.toggle("Use Qwen generation", value=False)
        st.divider()
        st.caption(
            "Use Qwen for natural-language answers, or leave it off for the "
            "fast deterministic template."
        )

    profile = {
        "city": city,
        "cuisines": cuisines,
        "max_price": max_price,
        "ambience": ambience,
        "dietary": dietary,
    }

    tab_recommend, tab_visual, tab_about = st.tabs(
        ["Recommend", "Visual search", "About"]
    )

    with tab_recommend:
        query = st.text_input(
            "What are you looking for?",
            value="Find Italian restaurants with vegetarian options in Philadelphia.",
        )
        if st.button("Recommend", type="primary") and query.strip():
            graph = load_graph(use_llm)
            with st.spinner("Searching the dataset ..."):
                state = run_query(graph, effective_query(query, profile), profile)
            render_result(state)

    with tab_visual:
        visual_query = st.text_input(
            "Describe the atmosphere or interior",
            value="romantic interior with warm lighting",
        )
        if st.button("Search visuals") and visual_query.strip():
            retriever = load_visual()
            hits = retriever.search(visual_query, k=5)
            rows = [
                {
                    "Restaurant": hit["row"].get("name"),
                    "Venue style": hit["row"].get("venue_style"),
                    "Colour tone": hit["row"].get("indoor_color_tone"),
                    "Special features": hit["row"].get("special_features"),
                    "Score": round(hit["score"], 4),
                }
                for hit in hits
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "Visual matching uses GPT-4o descriptions of photos, not the photos "
                "themselves. The dataset contains no image files."
            )

    with tab_about:
        st.markdown(
            """
DineWise is a laboratory prototype, not a booking service.

**What it can do**
- Recommend restaurants by cuisine, city, budget bucket, rating, and attributes.
- Quote review evidence and visual descriptions.
- Refuse live-availability, unknown-city, and unrecorded-amenity questions.

**What it cannot do**
- Provide live availability, reservations, current prices, or wait times.
- Guarantee dietary safety.
- Match an uploaded photo to a specific restaurant.
"""
        )


if __name__ == "__main__":
    main()
