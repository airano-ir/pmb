"""
Image attachment + optional CLIP embedding (Improvement J — image half).

Two modes:
  1. METADATA-ONLY (default, zero deps):
       record_image(path, description="screenshot of dashboard")
       → stored as event_type='image' with metadata.image_path
       → searchable via the TEXT description (BM25 + dense)
       → no image encoding; fast, no model download

  2. CLIP-EMBED (opt-in, requires open_clip or sentence-transformers CLIP):
       In addition to metadata-only, run CLIP image encoder → store
       image embedding in metadata. Cross-modal text→image search becomes
       possible via the same CLIP's text encoder at query time.

Image search at query time:
  - Text query → existing pipeline searches description text
  - Optional cross-modal: encode query with CLIP text encoder, cosine
    against stored image embeddings

CLIP model loading is LAZY — only happens on first encode call. If CLIP
not installed, we silently degrade to metadata-only.

For developers who keep screenshots / diagrams of work, this enables
"find the screenshot where I drew the auth flow" type queries.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ImageAttachment:
    path: str
    description: str = ""
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    clip_embedding: list[float] | None = None

    def to_metadata(self) -> dict:
        meta = {
            "image_path": self.path,
            "image_description": self.description,
        }
        if self.width:
            meta["image_width"] = self.width
        if self.height:
            meta["image_height"] = self.height
        if self.sha256:
            meta["image_sha256"] = self.sha256
        if self.clip_embedding is not None:
            # Store as base64 to keep JSON-friendly (will be small fp32 list)
            import json as _j
            meta["clip_embedding_json"] = _j.dumps(self.clip_embedding)
        return meta


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def attach_image(
    path: str,
    description: str = "",
    encode_clip: bool = False,
) -> ImageAttachment:
    """Build an ImageAttachment from a file path.

    Args:
      path:          absolute or relative path to image file
      description:   text description (becomes the event's content)
      encode_clip:   if True, attempt CLIP encoding (silently degrades if not installed)
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    att = ImageAttachment(path=str(p), description=description)
    try:
        att.sha256 = file_sha256(p)
    except Exception:
        pass
    # Try to read dimensions via PIL if available, else skip
    try:
        from PIL import Image  # noqa: F401
        with Image.open(p) as im:
            att.width, att.height = im.size
    except Exception:
        pass
    if encode_clip:
        try:
            emb = _clip_encode_image(p)
            if emb is not None:
                att.clip_embedding = list(emb)
        except Exception as e:
            log.debug("CLIP encode skipped: %s", e)
    return att


def _clip_encode_image(path: Path):
    """Lazy CLIP image encoder. Returns numpy array or None.

    Tries (in order):
      1. open_clip_torch (preferred for quality)
      2. sentence-transformers' clip-ViT-B-32
      3. None if neither is installed
    """
    try:
        import open_clip
        import torch
        from PIL import Image
        # Use small ViT-B-32 — ~150MB, balances speed/quality
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai",
        )
        model.eval()
        with torch.no_grad():
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            feat = model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()[0]
    except Exception:
        pass
    try:
        from PIL import Image
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("clip-ViT-B-32")
        emb = model.encode(Image.open(path), convert_to_numpy=True)
        # L2 normalize
        import numpy as np
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        return emb
    except Exception:
        return None


def clip_encode_text(query: str):
    """Encode a TEXT query via CLIP text encoder for cross-modal search.
    Returns numpy array or None if CLIP unavailable."""
    try:
        import open_clip
        import torch
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai",
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            toks = tokenizer([query])
            feat = model.encode_text(toks)
            feat /= feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()[0]
    except Exception:
        pass
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("clip-ViT-B-32")
        import numpy as np
        emb = model.encode(query, convert_to_numpy=True)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        return emb
    except Exception:
        return None
