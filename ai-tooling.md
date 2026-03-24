# AI Tooling Documentation

The following tools were used and for what purposes:

1. **Gemini CLI with Conductor** - this was the primary development tool used to build the project from the ground up, including creating the corpus documents, implementing the RAG pipeline, and iteratively modifying the prompt template in `rag_engine.py` to improve groundedness and citation accuracy scores during evaluation.

2. **Claude Code CLI** - used for debugging, deployment configuration, and documentation. Specifically used to set up the Render deployment (adding Gunicorn, creating `render.yaml`), and to write and update the `.md` documentation files.

3. **Gemini (browser chat)** - used as a sounding board throughout the project to chat through ideas, verify that requirements were being addressed, and clarify concepts.

---

## References

1. gemini-cli-extensions, Aitbayev, S., & Gana Obregón, M. (2025). "Conductor: A Gemini CLI extension for context-driven development." GitHub. https://github.com/gemini-cli-extensions/conductor

2. Google. "Gemini." https://gemini.google.com. Accessed: 2026-03-24.

3. Anthropic. "Claude Code." https://claude.ai/code. Accessed: 2026-03-24.

```bibtex
@MISC{gemini-cli-extensions2025conductor,
  author = {gemini-cli-extensions and Aitbayev, Sherzat and Gana Obreg{\'o}n, Mois{\'e}s},
  title = {Conductor: A Gemini CLI extension for context-driven development},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/gemini-cli-extensions/conductor}},
  note = {Accessed: 2026-03-24}
}

@MISC{google_gemini,
  author = {Google},
  title = {Gemini},
  howpublished = {\url{https://gemini.google.com}},
  note = {Accessed: 2026-03-24}
}

@MISC{anthropic_claude_code,
  author = {Anthropic},
  title = {Claude Code},
  howpublished = {\url{https://claude.ai/code}},
  note = {Accessed: 2026-03-24}
}
```
