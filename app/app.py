"""NexusRAG Gradio Interface - Premium Design."""

import html

import gradio as gr

from nexusrag.pipeline import NexusRAG, get_nexusrag

CUSTOM_CSS = """
/* ============================================
   NexusRAG Premium Design System
   ============================================ */

/* CSS Variables */
:root {
    --primary: #7C3AED;
    --primary-hover: #6D28D9;
    --primary-light: #EDE9FE;
    --primary-dark: #5B21B6;

    --bg-primary: #f8f7fc;
    --bg-secondary: #F3F4F6;
    --bg-card: #FFFFFF;

    --text-primary: #1F2937;
    --text-secondary: #6B7280;
    --text-muted: #9CA3AF;

    --border-light: #E5E7EB;
    --border-medium: #D1D5DB;

    --success: #10B981;
    --success-bg: #D1FAE5;
    --warning: #F59E0B;
    --warning-bg: #FEF3C7;
    --error: #EF4444;
    --error-bg: #FEE2E2;

    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.12);
    --shadow-hover: 0 8px 30px rgba(124, 58, 237, 0.15);

    --radius-sm: 6px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-full: 9999px;

    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #0F0F1A;
        --bg-secondary: #1A1A2E;
        --bg-card: #16162A;
        --text-primary: #F9FAFB;
        --text-secondary: #9CA3AF;
        --text-muted: #6B7280;
        --border-light: #2D2D44;
        --border-medium: #3D3D5C;
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
}

/* Global Reset */
.gradio-container {
    background: var(--bg-primary) !important;
    font-family: var(--font-sans) !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

.gradio-container * {
    font-family: var(--font-sans) !important;
}

/* Hide Gradio Footer */
footer {
    display: none !important;
}

/* ============================================
   Header Styles
   ============================================ */

.header-container {
    text-align: center;
    padding: 48px 0 32px;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 32px;
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-primary) 100%);
    margin: -24px -24px 32px -24px;
    padding: 48px 24px 32px;
}

.header-logo {
    font-size: 2.75rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 1.125rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin: 0;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 16px;
    padding: 6px 14px;
    background: var(--primary-light);
    color: var(--primary);
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: var(--radius-full);
}

/* ============================================
   Card Styles
   ============================================ */

.card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-md) !important;
    padding: 24px !important;
    transition: var(--transition) !important;
}

.card:hover {
    box-shadow: var(--shadow-hover) !important;
    border-color: var(--primary-light) !important;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 2px solid var(--primary-light);
}

.card-icon {
    width: 32px;
    height: 32px;
    background: var(--primary-light);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--primary);
    font-size: 1rem;
}

.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.01em;
}

/* ============================================
   Upload Area
   ============================================ */

.upload-container {
    border: 2px dashed var(--border-medium) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-secondary) !important;
    transition: var(--transition) !important;
    padding: 32px !important;
    text-align: center;
}

.upload-container:hover {
    border-color: var(--primary) !important;
    background: var(--primary-light) !important;
}

.upload-container.dragover {
    border-color: var(--primary) !important;
    background: var(--primary-light) !important;
    transform: scale(1.01);
}

/* ============================================
   Button Styles
   ============================================ */

.btn-primary {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3) !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
}

.btn-primary:active {
    transform: translateY(0) !important;
}

.btn-secondary {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 18px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
}

.btn-secondary:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    background: var(--primary-light) !important;
}

/* ============================================
   Chat Styles
   ============================================ */

.chat-container {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-secondary) !important;
    overflow: hidden;
}

.chatbot {
    background: transparent !important;
}

.chatbot .message {
    padding: 16px 20px !important;
    border-radius: var(--radius-md) !important;
    margin: 8px 12px !important;
    max-width: 85% !important;
    line-height: 1.6 !important;
    font-size: 0.95rem !important;
}

.chatbot .user {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    margin-left: auto !important;
    border-bottom-right-radius: 4px !important;
}

.chatbot .bot {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-light) !important;
    margin-right: auto !important;
    border-bottom-left-radius: 4px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ============================================
   Input Styles
   ============================================ */

.input-container textarea,
.input-container input {
    background: var(--bg-card) !important;
    border: 2px solid var(--border-light) !important;
    border-radius: var(--radius-sm) !important;
    padding: 14px 16px !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
    transition: var(--transition) !important;
}

.input-container textarea:focus,
.input-container input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-light) !important;
    outline: none !important;
}

.input-container textarea::placeholder,
.input-container input::placeholder {
    color: var(--text-muted) !important;
}

/* ============================================
   Sources Panel
   ============================================ */

.source-item {
    background: var(--bg-secondary);
    border-left: 3px solid var(--primary);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: var(--transition);
}

.source-item:hover {
    background: var(--primary-light);
    transform: translateX(4px);
}

.source-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--primary);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: var(--radius-full);
    margin-right: 10px;
}

.source-meta {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 8px;
}

.source-content {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.6;
    margin-top: 8px;
}

/* ============================================
   Confidence Badge
   ============================================ */

.confidence-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px;
    text-align: center;
}

.confidence-ring {
    width: 100px;
    height: 100px;
    border-radius: var(--radius-full);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
    position: relative;
}

.confidence-ring::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: var(--radius-full);
    padding: 4px;
    background: conic-gradient(var(--ring-color) calc(var(--confidence) * 360deg), var(--border-light) 0);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
}

.confidence-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
}

.confidence-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: var(--radius-full);
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 8px;
}

.confidence-high {
    background: var(--success-bg);
    color: var(--success);
}

.confidence-medium {
    background: var(--warning-bg);
    color: var(--warning);
}

.confidence-low {
    background: var(--error-bg);
    color: var(--error);
}

/* ============================================
   Stats Grid
   ============================================ */

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 16px;
}

.stat-item {
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    padding: 16px 12px;
    text-align: center;
    transition: var(--transition);
}

.stat-item:hover {
    background: var(--primary-light);
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1;
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* ============================================
   Status Indicators
   ============================================ */

.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 0;
    font-size: 0.85rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: var(--radius-full);
    flex-shrink: 0;
}

.status-dot.online {
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
}

.status-dot.offline {
    background: var(--error);
}

.status-label {
    color: var(--text-muted);
}

.status-value {
    color: var(--text-primary);
    font-weight: 500;
    margin-left: auto;
}

/* ============================================
   Document List
   ============================================ */

.doc-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    margin-bottom: 8px;
    transition: var(--transition);
}

.doc-item:hover {
    background: var(--primary-light);
}

.doc-icon {
    width: 36px;
    height: 36px;
    background: var(--primary-light);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--primary);
    font-size: 0.9rem;
}

.doc-info {
    flex: 1;
    min-width: 0;
}

.doc-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.doc-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* ============================================
   Empty States
   ============================================ */

.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: var(--text-muted);
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 16px;
    opacity: 0.5;
}

.empty-state-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.empty-state-desc {
    font-size: 0.875rem;
    color: var(--text-muted);
}

/* ============================================
   Animations
   ============================================ */

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

.animate-slide-in {
    animation: slideIn 0.3s ease forwards;
}

/* ============================================
   Responsive Design
   ============================================ */

@media (max-width: 768px) {
    .header-logo {
        font-size: 2rem;
    }

    .stats-grid {
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
    }

    .stat-value {
        font-size: 1.25rem;
    }
}

/* ============================================
   Gradio Component Overrides
   ============================================ */

.gradio-container .prose {
    max-width: none !important;
}

.gradio-container .gap-4 {
    gap: 24px !important;
}

.gradio-container label {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
}

.gradio-container .border-gray-200 {
    border-color: var(--border-light) !important;
}

/* Hide default gradio elements */
.gradio-container .gr-button-lg {
    display: none;
}

.contain {
    background: transparent !important;
}

#component-0 {
    background: transparent !important;
}
"""

# Global pipeline instance
rag: NexusRAG | None = None


def get_rag() -> NexusRAG:
    """Get or initialize RAG pipeline."""
    global rag
    if rag is None:
        rag = get_nexusrag()
    return rag


def format_sources_html(sources: list) -> str:
    """Format sources as styled HTML cards."""
    if not sources:
        return """
        <div class="empty-state">
            <div class="empty-state-icon">&#128269;</div>
            <div class="empty-state-title">No sources yet</div>
            <div class="empty-state-desc">Ask a question to see relevant sources</div>
        </div>
        """

    html_parts = ['<div class="animate-slide-in">']
    for source in sources:
        page_info = f" &middot; Page {source.page_number}" if source.page_number else ""
        section_info = f"<strong>{html.escape(source.section_title)}</strong>" if source.section_title else ""

        raw_preview = source.content[:250] + "..." if len(source.content) > 250 else source.content
        content_preview = html.escape(raw_preview)

        html_parts.append(f"""
        <div class="source-item">
            <div style="display: flex; align-items: flex-start; gap: 12px;">
                <span class="source-number">{source.index}</span>
                <div style="flex: 1;">
                    {f'<div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">{section_info}</div>' if section_info else ''}
                    <div class="source-content">{content_preview}</div>
                    <div class="source-meta">Source {source.index}{page_info}</div>
                </div>
            </div>
        </div>
        """)

    html_parts.append('</div>')
    return "".join(html_parts)


def format_confidence_html(confidence: float) -> str:
    """Format confidence as a visual gauge."""
    percentage = int(confidence * 100)

    if confidence >= 0.7:
        level, badge_class, ring_color = "High", "confidence-high", "var(--success)"
    elif confidence >= 0.4:
        level, badge_class, ring_color = "Medium", "confidence-medium", "var(--warning)"
    else:
        level, badge_class, ring_color = "Low", "confidence-low", "var(--error)"

    return f"""
    <div class="confidence-container animate-slide-in">
        <div class="confidence-ring" style="--confidence: {confidence}; --ring-color: {ring_color};">
            <span class="confidence-value">{percentage}%</span>
        </div>
        <div class="confidence-label">Confidence Score</div>
        <div class="confidence-badge {badge_class}">
            <span>&#9679;</span> {level} Confidence
        </div>
    </div>
    """


def format_empty_confidence() -> str:
    """Empty confidence state."""
    return """
    <div class="confidence-container">
        <div style="color: var(--text-muted); font-size: 0.9rem;">
            Confidence score will appear here after you ask a question
        </div>
    </div>
    """


def upload_files(files: list[str] | None) -> tuple[str, str]:
    """Process uploaded files."""
    if not files:
        return "No files selected.", get_stats_html()

    pipeline = get_rag()
    results = []
    success_count = 0

    for file_path in files:
        result = pipeline.ingest(file_path)
        if result.success:
            results.append(f"Processed {result.filename} ({result.chunk_count} chunks)")
            success_count += 1
        else:
            results.append(f"Failed: {result.filename} - {result.error}")

    status = f"Uploaded {success_count}/{len(files)} files\n" + "\n".join(results)
    return status, get_stats_html()


def get_stats_html() -> str:
    """Get system statistics as styled HTML."""
    try:
        pipeline = get_rag()
        stats = pipeline.get_stats()

        llm_status = "online" if stats.llm_available else "offline"

        return f"""
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{stats.total_documents}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats.total_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats.total_words:,}</div>
                <div class="stat-label">Words</div>
            </div>
        </div>

        <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border-light);">
            <div class="status-row">
                <span class="status-dot {llm_status}"></span>
                <span class="status-label">LLM</span>
                <span class="status-value">{stats.llm_model}</span>
            </div>
            <div class="status-row">
                <span class="status-dot online"></span>
                <span class="status-label">Embeddings</span>
                <span class="status-value">{stats.embedding_model}</span>
            </div>
        </div>
        """
    except Exception as e:
        return f"""
        <div class="empty-state">
            <div class="empty-state-desc" style="color: var(--error);">Error loading stats: {e}</div>
        </div>
        """


def process_query(
    message: str,
    history: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]], str, str]:
    """Process user query and return response with sources."""
    if not message.strip():
        return "", history, format_sources_html([]), format_empty_confidence()

    pipeline = get_rag()

    try:
        response = pipeline.query(message)
        history.append((message, response.answer))
        sources_html = format_sources_html(response.sources)
        confidence_html = format_confidence_html(response.confidence)
        return "", history, sources_html, confidence_html

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        history.append((message, error_msg))
        return "", history, format_sources_html([]), format_empty_confidence()


def clear_all() -> tuple[str, list, str, str, str]:
    """Clear all data and reset UI."""
    try:
        pipeline = get_rag()
        pipeline.clear_all()
        return (
            "All documents cleared successfully.",
            [],
            format_sources_html([]),
            format_empty_confidence(),
            get_stats_html(),
        )
    except Exception as e:
        return (
            f"Error clearing data: {e}",
            [],
            format_sources_html([]),
            format_empty_confidence(),
            get_stats_html(),
        )


def list_documents() -> str:
    """List all ingested documents."""
    try:
        pipeline = get_rag()
        docs = pipeline.list_documents()

        if not docs:
            return "No documents uploaded yet."

        lines = []
        for doc in docs:
            name = doc.get('filename', 'Unknown')
            words = doc.get('word_count', 0)
            lines.append(f"{name} ({words:,} words)")

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing documents: {e}"


# Build the interface
with gr.Blocks(title="NexusRAG") as app:

    # Header
    gr.HTML("""
    <div class="header-container">
        <h1 class="header-logo">NexusRAG</h1>
        <p class="header-subtitle">Self-correcting retrieval for scientific literature synthesis</p>
        <div class="header-badge">
            <span>&#9889;</span> Powered by Local LLM
        </div>
    </div>
    """)

    with gr.Row():
        # Left Column - Documents
        with gr.Column(scale=1):

            # Upload Card
            with gr.Group(elem_classes="card"):
                gr.HTML("""
                <div class="card-header">
                    <div class="card-icon">&#128196;</div>
                    <h3 class="card-title">Documents</h3>
                </div>
                """)

                file_upload = gr.File(
                    label="",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                    elem_classes="upload-container",
                )

                gr.HTML('<p style="text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 12px;">Drop PDF, DOCX, TXT, or MD files here</p>')

                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                    max_lines=5,
                    elem_classes="input-container",
                )

                with gr.Row():
                    list_btn = gr.Button("List All", variant="secondary", elem_classes="btn-secondary")
                    clear_btn = gr.Button("Clear All", variant="secondary", elem_classes="btn-secondary")

            # Stats Card
            with gr.Group(elem_classes="card"):
                gr.HTML("""
                <div class="card-header">
                    <div class="card-icon">&#128200;</div>
                    <h3 class="card-title">System Status</h3>
                </div>
                """)
                stats_html = gr.HTML(value=get_stats_html)

        # Right Column - Chat & Results
        with gr.Column(scale=2):

            # Chat Card
            with gr.Group(elem_classes="card"):
                gr.HTML("""
                <div class="card-header">
                    <div class="card-icon">&#128172;</div>
                    <h3 class="card-title">Research Assistant</h3>
                </div>
                """)

                chatbot = gr.Chatbot(
                    label="",
                    height=380,
                    elem_classes="chat-container",
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        query_input = gr.Textbox(
                            label="",
                            placeholder="Ask a question about your documents...",
                            lines=2,
                            elem_classes="input-container",
                        )
                    with gr.Column(scale=1, min_width=100):
                        submit_btn = gr.Button("Ask", variant="primary", elem_classes="btn-primary")

            # Results Row
            with gr.Row():
                # Confidence Card
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        gr.HTML("""
                        <div class="card-header">
                            <div class="card-icon">&#127919;</div>
                            <h3 class="card-title">Confidence</h3>
                        </div>
                        """)
                        confidence_html = gr.HTML(value=format_empty_confidence)

                # Sources Card
                with gr.Column(scale=2):
                    with gr.Group(elem_classes="card"):
                        gr.HTML("""
                        <div class="card-header">
                            <div class="card-icon">&#128209;</div>
                            <h3 class="card-title">Sources</h3>
                        </div>
                        """)
                        sources_html = gr.HTML(value=format_sources_html([]))

    # Event handlers
    file_upload.change(
        fn=upload_files,
        inputs=[file_upload],
        outputs=[upload_status, stats_html],
    )

    submit_btn.click(
        fn=process_query,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot, sources_html, confidence_html],
    )

    query_input.submit(
        fn=process_query,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot, sources_html, confidence_html],
    )

    list_btn.click(
        fn=list_documents,
        outputs=[upload_status],
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[upload_status, chatbot, sources_html, confidence_html, stats_html],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )
