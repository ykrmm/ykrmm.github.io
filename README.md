# ykrmm.github.io

Personal academic website of Yannis Karmim, built with [al-folio](https://github.com/alshedivat/al-folio) (v1.x).

## Where things live

| What | File |
| --- | --- |
| Bio, profile photo | `_pages/about.md`, `assets/img/prof_pic.png` |
| News | `_news/YYYY-MM-DD-*.md` (one file per item) |
| Publications | `_bibliography/papers.bib` (thumbnails in `assets/img/publication_preview/`, venue colors in `_data/venues.yml`) |
| Blog posts | `_posts/YYYY-MM-DD-title.md` |
| Talks, teaching | `_pages/talks.md`, `_pages/teaching.md` |
| CV page / PDF | `_data/cv.yml`, `assets/pdf/CV_Karmim.pdf` |
| Social links | `_data/socials.yml` |
| Site settings | `_config.yml` |

## Local preview (Docker)

```bash
docker run --rm -p 4000:4000 -v "$PWD":/srv/jekyll -w /srv/jekyll amirpourmand/al-folio:latest \
  bash -c 'bundle exec jekyll serve --host 0.0.0.0 --port 4000 --force_polling --destination /tmp/_site'
```

Then open http://localhost:4000. See [QUICK_TUTO.md](QUICK_TUTO.md) for how to edit content.

## Deployment

Pushing to `master` runs `.github/workflows/deploy.yml`, which builds the site and pushes it to the `gh-pages` branch.
GitHub Pages must be set to **Deploy from a branch → `gh-pages`** (Settings → Pages), and Actions need
**Read and write permissions** (Settings → Actions → General → Workflow permissions).
