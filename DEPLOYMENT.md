# Free deployment starter

This repository is configured for a static Vercel frontend and a Render FastAPI service. Import the GitHub repository into both services.

## Vercel

Use the repository root as the project root; `vercel.json` installs the frontend dependencies and builds `frontend/dist`. Add this environment variable before deploying:

| Name | Value |
| --- | --- |
| `VITE_API_URL` | The Render service URL, for example `https://akito-ai-api.onrender.com` |

## Render

Create a Blueprint from this repository and select the free web service defined in `render.yaml`. After Render assigns the service URL, set `VITE_API_URL` in Vercel to that URL and redeploy the frontend. The API's CORS setting is open by default for the static site; set `AKITO_CORS_ORIGINS` to a comma-separated list of exact site origins to restrict it.

## Runtime limitations

The API is configured to start, but its assistant currently calls a local Ollama instance and uses a local embedding model and Chroma database. Ollama is not bundled into Render, so chat, vision, and model-dependent research workflows need an accessible model runtime and deployment-specific configuration before they can work online. The Render free service sleeps after inactivity, and its filesystem is ephemeral: uploads, conversation files, the vector index, and other saved research data can disappear on restart, redeploy, or spin-down. Use persistent storage and a hosted database/model setup for a lasting public deployment.

See [Render's free service limits](https://render.com/docs/free) for current runtime and filesystem behavior. Vercel's Hobby plan is intended for personal projects; see [Vercel's plan details](https://vercel.com/docs/plans/hobby).
