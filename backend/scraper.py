from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from newspaper import Article, Config

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

def _safe_date_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _download_image_to_temp(image_url: str) -> str:
    if not image_url or not image_url.startswith("http"):
        return ""
    try:
        response = requests.get(
            image_url,
            timeout=15,
            stream=True,
            headers={"User-Agent": USER_AGENT},
            verify=False
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()

        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"
        elif "gif" in content_type:
            ext = ".gif"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            return temp_file.name
    except Exception:
        return ""


def _scrape_with_newspaper(url: str) -> Dict[str, Any]:
    config = Config()
    config.browser_user_agent = USER_AGENT
    config.request_timeout = 15

    article = Article(url, config=config)
    article.download()
    article.parse()

    return {
        "title": article.title or "",
        "text": article.text or "",
        "authors": article.authors if article.authors else [],
        "date": _safe_date_to_string(article.publish_date),
        "top_image_url": article.top_image or "",
    }


def _scrape_with_bs4(url: str) -> Dict[str, Any]:
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": USER_AGENT},
        verify=False
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script, style, nav, footer tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join([p for p in paragraphs if len(p.split()) > 3]).strip()

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Try to find top image
    top_image_url = ""
    og_img = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "twitter:image"})
    if og_img and og_img.get("content"):
        top_image_url = og_img["content"]
    else:
        img_tag = soup.find("img", src=True)
        if img_tag:
            top_image_url = img_tag["src"]

    return {
        "title": title,
        "text": text,
        "authors": [],
        "date": "",
        "top_image_url": top_image_url,
    }


def scrape_url(url: str) -> Dict[str, Any]:
    source_domain = urlparse(url).netloc
    result: Dict[str, Any] = {
        "title": "",
        "text": "",
        "image_path": "",
        "date": "",
        "authors": [],
        "source_domain": source_domain,
    }

    if not isinstance(url, str) or not url.strip():
        return {"error": "Could not scrape this URL"}

    top_image_url = ""
    try:
        newspaper_data = _scrape_with_newspaper(url)
        result["title"] = newspaper_data["title"]
        result["text"] = newspaper_data["text"]
        result["authors"] = newspaper_data["authors"]
        result["date"] = newspaper_data["date"]
        top_image_url = newspaper_data["top_image_url"]
    except Exception:
        pass

    if not result["text"] or len(result["text"]) < 50:
        try:
            bs4_data = _scrape_with_bs4(url)
            if not result["title"]:
                result["title"] = bs4_data["title"]
            if len(bs4_data["text"]) > len(result["text"]):
                result["text"] = bs4_data["text"]
            if not top_image_url:
                top_image_url = bs4_data["top_image_url"]
        except Exception:
            pass

    if not result["title"] and not result["text"]:
        return {"error": "Could not extract article content from this URL"}

    image_path = _download_image_to_temp(top_image_url)
    result["image_path"] = image_path

    if not isinstance(result["authors"], list):
        result["authors"] = [str(result["authors"])]
    else:
        result["authors"] = [str(author) for author in result["authors"] if str(author).strip()]

    return result
