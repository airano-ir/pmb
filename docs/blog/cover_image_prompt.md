# Cover image prompt — PMB HackerNoon launch post

HackerNoon требует cover-картинку **минимум 1200×630 px** (16:9 или 1.91:1).

---

## Главный промпт (универсальный — для DALL-E / Flux / Midjourney / GPT Image / Firefly)

```
A modern, minimalist tech illustration for an article about local-first AI memory.
Centerpiece: a stylized brain merged with a database cylinder — the brain half made
of glowing neural circuit lines, the database half made of stacked translucent disks
with small cyan data nodes. The composition is centered, with subtle floating
geometric shapes (cubes, dots, light rays) around it on a dark deep-blue background
gradient (#0d1117 to #1a1f3a). Tiny orbiting elements suggest data flowing in and out.
A hairline grid pattern in the background adds a technical feel. Color palette:
electric purple (#7c3aed), soft indigo (#5b4bff), cyan accents (#00d4ff), white
highlights. Style: clean vector illustration, flat with subtle gradients, futuristic
but approachable, similar to Stripe's marketing visuals or Linear's brand aesthetic.
NO TEXT, NO LETTERS, NO LOGOS, NO HUMAN FACES. 16:9 aspect ratio, 1920×1080.
```

---

## Версии под конкретные генераторы

### Midjourney v6

```
modern minimalist tech illustration, stylized brain merged with a database cylinder,
brain half made of glowing neural circuit lines, database half made of stacked
translucent disks with cyan data nodes, centered composition, floating geometric
shapes around it, dark deep-blue gradient background, hairline grid pattern,
electric purple #7c3aed and cyan #00d4ff accents, clean vector style like Stripe or
Linear, flat with subtle gradients, futuristic approachable feel, no text, no faces
--ar 16:9 --style raw --v 6
```

### DALL-E 3 / GPT Image / ChatGPT-4o

Вставь главный промпт выше. Добавь в конце:

```
Render in 1792×1024 landscape. No text or letters anywhere in the image.
Photorealistic-vector hybrid style.
```

### Flux.1 / Stable Diffusion XL

```
Prompt: minimalist vector illustration, brain-and-database hybrid icon centered,
neural circuit lines glowing, stacked translucent database disks with cyan nodes,
dark navy background with hairline grid, electric purple and cyan color palette,
clean modern tech aesthetic, Stripe-style marketing visual, flat design with subtle
gradients

Negative prompt: text, letters, words, watermark, signature, human face, hands,
cluttered, busy, photorealistic skin, lens flare, low quality, blurry

Settings: 1920×1080, CFG 6, 30 steps, DPM++ 2M Karras
```

### Adobe Firefly / Photoshop Generative Fill

```
Tech illustration: brain fused with database, glowing circuit lines, cyan data
nodes, dark blue gradient background, electric purple highlights, minimalist
vector style, no text, 16:9 landscape
```

---

## Альтернатива — "code aesthetic"

Если первый стиль покажется слишком "стоковым", второй вариант — упор на код:

```
A dark editor window in the center showing minimal Python code in light blue and
green syntax highlighting, set against a deep navy background. Floating around the
window: ghosted memory nodes connected by glowing thread-lines, suggesting recall.
A small brain icon glows softly in the corner, made of circuit traces. The
composition feels like a developer's desktop in a sci-fi film, but quiet and clean.
Color palette: VS Code dark theme (#1e1e1e, #569cd6, #4ec9b0) with subtle purple
accents. NO READABLE TEXT, just code-shaped silhouettes. 16:9 landscape, 1920×1080.
```

---

## Третий вариант — "graph network"

Самый абстрактный, хорошо работает если первый вариант выглядит слишком "иллюстративно":

```
An abstract knowledge graph network: dozens of small glowing nodes connected by
thin curved lines, forming a brain-like silhouette in the negative space. Some
nodes pulse brighter (the "remembered" facts), others are dim. The whole composition
sits on a dark gradient background going from #0a0e27 (top) to #1a1f3a (bottom).
Color accents: electric blue (#3b82f6), violet (#8b5cf6), mint (#10b981). Style:
data-viz meets cinematic — like a still frame from a Microsoft AI keynote video.
Clean, minimal, no clutter, 16:9 aspect ratio.
```

---

## Что сделать с промптом

1. **Сгенерируй 3-4 варианта** через любой инструмент (Midjourney / ChatGPT image gen / Flux)
2. Выбери тот где:
   - Цвета совпадают с твоим логотипом (фиолетовый + cyan)
   - Нет лишнего текста / лиц / артефактов
   - Хорошо смотрится при сжатии до 600px wide (HackerNoon делает миниатюру)
3. **Upscale до 1920×1080** если генератор выдал меньше
4. Сохрани как `docs/assets/cover_hackernoon.png` (или `.jpg` — для cover JPEG предпочтительнее, меньше вес)
5. В HackerNoon-редакторе на этапе "Cover image" загрузи этот файл

---

## Бонус: cover для dev.to и LinkedIn

dev.to использует **1000×420** (соотношение 2.38:1) — попроси генератор сделать ту же картинку в этом aspect ratio, либо обрежь существующую 1920×1080 по центру.

LinkedIn (если будешь шарить пост) предпочитает **1200×627** — то же самое 16:9, подойдёт та же картинка.

---

## Подсказка для финального ретуша

Если хочешь добавить **подпись бренда без вмешательства в иллюстрацию** — после генерации картинки открой её в Canva / Figma / Photoshop и наложи в нижний-правый угол:

- маленький логотип PMB (`docs/assets/logo.png`) — 80×80px
- белый текст "PMB · pmb-ai" — 16px, opacity 70%

Это не обязательно, но повышает recall (no pun intended) бренда у читателей.
