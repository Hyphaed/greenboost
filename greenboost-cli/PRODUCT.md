# Product

## Register

product

## Users

Solo developer (personal use). High tolerance for information density and power-user features. Context: switching between terminal sessions and the dashboard while actively coding — ambient glances more than deliberate navigation. The user knows what every panel does; the interface doesn't need to explain itself.

## Product Purpose

`greenboost-cli` (`gb`) is an AI coding assistant CLI with GPU memory integration. It combines a streaming agent loop powered exclusively by gb-synapse (GreenBoost's HuggingFace-pull + Ollama-index + cluster-distributed llama.cpp serving layer), RAG over local codebases, per-project brain and memory, token tracking, FLUX diffusion for UI asset generation, and an 8-page web dashboard — all auto-routing across T1 VRAM / T2 DDR / T3 NVMe / T4 tiers. Success is a tool that feels like an extension of the developer's mental workspace: always available, never in the way, visibly intelligent.

## Brand Personality

Technical, atmospheric, alive. The interface should feel like a machine that is actively working — data breathes, tiers glow, context surfaces. Not decorative atmosphere; earned atmosphere from real system state.

## Anti-references

- Generic SaaS cream/white: not the light-mode, rounded-corners, Inter-everything template
- VS Code clone: lives alongside VS Code; shouldn't feel like a cousin
- Neon cyberpunk: not the purple-on-black hacker aesthetic saturated in AI tooling since 2023
- Enterprise grey: not muted Grafana defaults or Jenkins-era dashboard monotony

## Design Principles

1. **Density over decoration** — every pixel earns its place; information should be immediately legible, not gestured at
2. **The machine speaks** — live GPU tier, token counts, RAG status, and backend state are first-class UI elements, not buried in settings
3. **Context collapses distance** — goals, history, and RAG surface before you ask; the tool anticipates the developer's next move
4. **Power without ceremony** — expert features are immediately accessible; no wizard gatekeeping for someone who already knows what they need
5. **Atmosphere earns trust** — ambient visual quality (glow, contrast, motion) should signal that the tool is good at its job, not compensate for a tool that isn't

## Accessibility & Inclusion

WCAG AA baseline. Standard contrast ratios and keyboard navigation. No specific reduced-motion or high-contrast requirements beyond baseline browser support.
