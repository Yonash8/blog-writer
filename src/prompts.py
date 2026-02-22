"""System and user prompts for the article-writing chatbot."""

DEEP_RESEARCH_PROMPT = """Research the topic "{topic}" exhaustively for a comprehensive SEO-optimized article.

Do:
- Search extensively for and analyze authoritative sources: engineering blogs, academic papers, \
official documentation, industry reports, case studies, and reputable publications.
- Include specific figures, statistics, trends, and measurable outcomes wherever available.
- Cover the topic from every relevant angle: definition, history/context, current state, \
key players, best practices, common pitfalls, and future outlook.
- Include inline citations and return all source metadata.
- Prioritize recent, up-to-date information (2025-2026).
- Be exhaustive: leave no relevant subtopic unexplored. Depth over breadth when both matter.

Output a comprehensive research report (2500-4000 words) structured as:
1. Executive Summary (2-3 paragraphs)
2. Key Facts and Statistics
3. Detailed Findings (organized by subtopic—each with multiple paragraphs)
4. Notable Sources and References
5. Suggested Article Angles

Be analytical, thorough, and data-driven. Avoid generalities. Favor concrete examples and quantifiable data."""

IMAGE_PLACEMENT_PROMPT = """Given this article in Markdown format, suggest 2-4 strategic places to insert relevant images.
Focus on: introduction section, key concept sections, and conclusion.
For each placement, provide:
- section_index: 0-based index of the section
- paragraph_after: A short unique snippet (first ~50 chars) of the paragraph after which to insert. Copy the exact text from the article, including markdown like [link](url), so the insertion point can be found accurately.
- image_prompt: A detailed prompt for DALL-E/Imagen to generate an appropriate illustration (no text in image)
- alt_text: Descriptive alt text for accessibility

Return valid JSON array only, no other text. Example format:
[{{"section_index": 0, "paragraph_after": "intro text...", "image_prompt": "Professional illustration of...", "alt_text": "..."}}]"""

HERO_IMAGE_PROMPT_TEMPLATE = (
    "Using the attached style references for the mascot and the monochromatic blue palette, "
    "generate a minimalist image with an animation outline aesthetic. "
    "Feature the PromptLayer Bot {description}"
)

INFOGRAPHIC_ANALYSIS_PROMPT = """You are an expert content strategist and data visualization specialist.

Analyze the following article and determine the BEST opportunity for a single infographic.

For the infographic, provide:
1. **position_after**: A short unique snippet (first ~60 chars) of the paragraph AFTER which the infographic should be placed. Pick a spot where visual data would enhance comprehension.
2. **infographic_type**: One of: "comparison_table", "flowchart", "bar_chart", "pie_chart", "timeline", "process_diagram", "statistics_highlight", "checklist"
3. **title**: A concise title for the infographic.
4. **description**: A detailed description of what the infographic should contain — specific data points, labels, comparisons, steps, etc. Be precise enough that a designer could create it.
5. **section_name**: The name/heading of the section where it will be placed.

Return ONLY valid JSON (no markdown code blocks, no explanation):
{{"position_after": "...", "infographic_type": "...", "title": "...", "description": "...", "section_name": "..."}}

Article:
---
{article}
---"""

INFOGRAPHIC_GENERATION_PROMPT_TEMPLATE = (
    "Refer to the attached style examples for the line weight, font style, and general color tone choice. "
    "Do not use the attachments for the layout or structure; follow the technical logic described below.\n\n"
    "Create a {infographic_type} infographic titled \"{title}\".\n\n"
    "{description}"
)

IMPROVE_ARTICLE_PROMPT = """You are an expert editor. Revise the following article based on the user's feedback.

CAPABILITIES:
- Edit text: rephrase, add, remove, or restructure any part of the article.
- Inject links: add Markdown links [anchor text](url) anywhere appropriate. Use links from the "Links to inject" section when provided.
- Preserve structure: keep headings, lists, and existing formatting unless the feedback asks to change them.

User feedback: {feedback}
{links_section}

Article to revise (Markdown):
---
{article}
---

RULES:
- Output ONLY the revised Markdown article. No explanations, no code blocks around it.
- For links: use strict Markdown syntax [anchor](url). Anchor text should be natural and descriptive.
- Place links inline where they add value (e.g. "see the [official docs](url) for details").
- Apply all feedback and link injections. When multiple links are provided, distribute them appropriately in the article."""
