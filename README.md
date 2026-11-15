# Free Model Aggregator

This Node 22 gateway reads provider, model rating, and workload details from a trial brief. It probes provider catalogs, selects a compatible free model for each chat request, translates between OpenAI, Anthropic, and Gemini request formats, and fails over across providers.

Run the server with node main.ts after setting MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE. Use node main.ts --list to print the discovered catalog.
