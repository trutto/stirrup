# Stirrup — public writeup site

This folder is a self-contained static page (`index.html` + `assets/`).

## Current hosting
Published with **GitHub Pages**:

https://trutto.github.io/stirrup/

Repo: https://github.com/trutto/stirrup

A Tailscale Funnel mirror also exists at `https://omarchy.tail870208.ts.net/` while this laptop is online.

Funnel serves this directory directly; it stays up while the laptop is online and Tailscale is
connected. It is public (no tailnet login) and contains no secrets.

Useful commands:
```bash
tailscale funnel status          # show the public URL + served path
tailscale funnel reset           # stop publishing
```

## Permanent hosting (optional)
To make the link independent of this laptop:

1. Create a public GitHub repo, e.g. `stirrup`.
2. Upload everything in this folder (`index.html`, `assets/`) to the repo root.
3. Repo → Settings → Pages → Deploy from branch → `main` / root.
4. Link: `https://<username>.github.io/stirrup/`.

The page has OG/Twitter meta tags; update `og:image` to the final public URL if the host changes.
