from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


OWNER_NAME = "Hassan Khan"
OWNER_TAGLINE = "AI Engineer | RAG Engineer | Building RAG workflows that work in production"
OWNER_LOCATION = "Islamabad, Pakistan (Remote)"
OWNER_EMAIL = "hassanaiengineer@gmail.com"
OWNER_LINKEDIN = "https://www.linkedin.com/in/hassan-khan-4961b722b/"


@dataclass(frozen=True)
class Brand:
    name: str = OWNER_NAME
    tagline: str = OWNER_TAGLINE
    location: str = OWNER_LOCATION
    email: str = OWNER_EMAIL
    linkedin: str = OWNER_LINKEDIN


DEFAULT_BRAND = Brand()


def brand_markdown(brand: Brand = DEFAULT_BRAND) -> str:
    return (
        f"**Built by {brand.name}**  \n"
        f"{brand.tagline}  \n"
        f"{brand.location}  \n"
        f"Email: `{brand.email}`  \n"
        f"LinkedIn: {brand.linkedin}"
    )


def apply_streamlit_branding(
    st,
    *,
    title: str,
    page_icon: str = "🧠",
    layout: str = "wide",
    brand: Brand = DEFAULT_BRAND,
    sidebar: bool = True,
    caption: Optional[str] = None,
) -> None:
    st.set_page_config(page_title=title, page_icon=page_icon, layout=layout)
    st.title(title)
    if caption:
        st.caption(caption)
    if sidebar:
        with st.sidebar:
            st.markdown("---")
            st.markdown(brand_markdown(brand))

