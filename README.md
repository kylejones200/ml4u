# ML4U: Machine Learning for Power & Utilities

A Hugo-based site that provides hands-on chapters, code, and demos for applying Machine Learning to the power and utilities sector.

- Live site: https://kylejones200.github.io/ml4u/
- Generator: Hugo (extended)
- Theme: Blowfish (vendored in `themes/blowfish/`)

## Quick start (local)

Prerequisites:
- Hugo Extended (https://gohugo.io/installation/)

Run the dev server:
```bash
hugo server -D
```
Then open the URL shown in the terminal (typically http://localhost:1313/ml4u/).

Build the site:
```bash
hugo --minify
```
The static files are generated into `public/`.

## Content structure

- Homepage: `content/_index.md`
- Chapters: each chapter is a leaf bundle using `index.md` inside its folder for clean URLs
  - Example: `content/c1/index.md` → https://kylejones200.github.io/ml4u/c1/
  - Folders currently: `c1` … `c20`
- Additional docs: `content/docs/`

### Creating a new chapter
1. Create a folder `content/c21/` (or the next number).
2. Add `index.md` with front matter and content, e.g.:
   ```markdown
   ---
   title: "New Chapter Title"
   description: "Short summary"
   ---
   # Heading
   Content here…
   ```
3. Add any assets alongside `index.md` in the same folder if needed.

## Configuration

- `hugo.toml`
  - `baseURL = 'https://kylejones200.github.io/ml4u/'`
  - `theme = 'blowfish'`
  - Enable HTML in Markdown (Goldmark):
    ```toml
    [markup]
      [markup.goldmark]
        [markup.goldmark.renderer]
          unsafe = true
    ```
  - Disable RSS site-wide (we only publish HTML/JSON):
    ```toml
    disableKinds = ["RSS"]
    [outputs]
      home = ["HTML", "JSON"]
    ```

## Deployment (GitHub Pages)

- Workflow: `.github/workflows/hugo.yml`
  - Checks out repo (no submodules needed — theme is vendored)
  - Builds with `hugo --minify`
  - Uploads `public/` as Pages artifact
  - Deploys via `actions/deploy-pages`

### Triggering a deploy
- Push to `main` or run the workflow manually (Actions → Run workflow).

### Troubleshooting
- Seeing XML at the root?
  - Hard refresh `https://kylejones200.github.io/ml4u/?v=<random>`
  - Confirm latest Actions run is green.
  - Ensure `disableKinds = ["RSS"]` is in `hugo.toml` and that the workflow completed a fresh deploy.
- Theme not applied?
  - Ensure `theme = 'blowfish'` in `hugo.toml` and that the `themes/blowfish/` directory is present.

## Contributing

- Open a PR against `main`.
- Keep chapters self-contained. Prefer `index.md` per chapter for clean URLs.
- Use short, descriptive titles and add a `description` in front matter.

## License

Content copyright © its authors. Theme © Blowfish authors. See theme licensing in `themes/blowfish/`.
