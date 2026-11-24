# Quick tutorial: edit and deploy the website

Every push to `master` rebuilds and publishes https://ykrmm.github.io automatically, in about 3–4 minutes.

## The 3-step loop

```bash
cd ~/Documents/Recherche/Thèse/site/ykrmm.github.io

# 1. edit files (see the table below)

# 2. (optional) preview locally at http://localhost:4000
docker run --rm -p 4000:4000 -v "$PWD":/srv/jekyll -w /srv/jekyll amirpourmand/al-folio:latest \
  bash -c 'bundle exec jekyll serve --host 0.0.0.0 --port 4000 --force_polling --destination /tmp/_site'
#    Ctrl+C to stop. The page refreshes when you save a file (except _config.yml: restart for that one).

# 3. publish
git add -A
git commit -m "Update news"
git push
```

Check the build at https://github.com/ykrmm/ykrmm.github.io/actions: green ✓ means it's live, red ✗ means it failed (click it to see why). If the live site looks unchanged, hard-refresh with Cmd+Shift+R.

**No terminal?** Open the file on github.com, click the ✏️ pencil, edit, then click "Commit changes". It deploys the same way. If you do this, run `git pull` before editing locally next time.

## Where things are

| I want to change… | File |
| --- | --- |
| Short bio (home page) | `_pages/about.md`, the text after the second `---` |
| Line under my name / text under the photo | `_pages/about.md`: `subtitle:` and `more_info:` |
| Profile photo | replace `assets/img/prof_pic.jpg` (keep the same name) |
| News | `_news/` (one file per item) |
| Publications | `_bibliography/papers.bib` |
| CV page (web version) | `_data/cv.yml` |
| CV PDF (download button) | replace `assets/pdf/CV_Karmim.pdf` (keep the same name) |
| Talks / Teaching | `_pages/talks.md` / `_pages/teaching.md` |
| Blog posts | `_posts/` |
| Social links (email, Scholar, GitHub…) | `_data/socials.yml` |
| Site title, favicon, settings | `_config.yml` |

## Home page

### Short bio

In `_pages/about.md`, everything below the second `---` line is the bio, in plain Markdown:

```markdown
I am a postdoctoral researcher at **INRIA**…      ← **bold**
See [my paper](https://arxiv.org/abs/...)          ← link
- a bullet point
```

At the top of the same file:
- `subtitle:` is the line under your name.
- `more_info:` is the text under the photo (one `<p>…</p>` per line).
- `announcements: limit: 8` sets how many news items show on the home page.
- `selected_papers: true` shows the papers marked `selected = {true}` in the `.bib`.

### Add a news item

Create `_news/2026-12-01-my-news.md`. The date in the filename is only for sorting; the `date:` line is what gets displayed:

```markdown
---
layout: post
date: 2026-12-01
inline: true
related_posts: false
---

Our paper [Title](https://arxiv.org/abs/xxxx) is accepted at **ICLR 2027**! :tada:
```

Remove a news item by deleting its file. Emojis work with `:name:` (`:tada:`, `:chile:`, `:mortar_board:`).

## CV

There are two independent things:

1. **PDF** (download button on the CV page and in the social icons): export your LaTeX CV and overwrite `assets/pdf/CV_Karmim.pdf`.
2. **Web CV** (the /cv/ page): edit `_data/cv.yml`.
   - `summary:` is the Professional Summary.
   - `Experience:` holds jobs and positions. Copy an existing entry block and keep the same indentation.
   - `Education:`, `Skills:` and `Languages:` work the same way.
   - `Teaching & service:` is a list of `label:` / `details:` pairs.
   - Dates are `YYYY-MM`. Use `end_date: present` for ongoing positions.
   - Section names must stay `Experience`, `Education`, `Skills`, `Languages`; other names only render `label/details` entries.

YAML is picky: use **spaces, not tabs**, keep the indentation aligned with the entry above, and put text containing `:` inside quotes.

## Publications

Add an entry at the top of `_bibliography/papers.bib` (copy an existing one). Useful fields:

| Field | Effect |
| --- | --- |
| `abbr = {ICLR}` | colored badge (colors in `_data/venues.yml`) |
| `selected = {true}` | also shown on the home page |
| `preview = {myfig.png}` | thumbnail; put the image in `assets/img/publication_preview/` |
| `arxiv = {2601.12345}`, `code = {https://github.com/...}`, `html = {...}`, `hal = {hal-...}`, `pdf = {...}` | buttons |
| `abstract = {...}` | "Abs" button |
| `annotation = {* Equal contribution}` | info icon next to the authors |

Your name is underlined automatically. In titles, escape `%` as `\%` and `&` as `\&`.

## Blog post

Create `_posts/2026-12-01-my-post.md`:

```markdown
---
layout: post
title: My post title
date: 2026-12-01
description: One-line summary shown in the blog list.
tags: llms reasoning
categories: llms
toc:
  sidebar: left        # optional table of contents on the left
---

Text in Markdown. Math with $x^2$ or $$ ... $$.
![caption](/assets/img/posts/my-post/figure.png)
```

- Hide a post without deleting it by adding `published: false` to its header.
- Make a blog entry that just links elsewhere with `redirect: https://...`.
- Put a post's images in `assets/img/posts/<post-name>/`.

## If the build fails (red ✗)

Click the failed run, then "Install and Build", and scroll to the error. Usual suspects:
- a YAML indentation error in `_data/*.yml` or a file header;
- a missing `}` or `,` in `papers.bib`;
- an image path that doesn't exist.

Fix it and push again. While a build is failing, the live site keeps showing the last successful version, so nothing breaks for visitors.
