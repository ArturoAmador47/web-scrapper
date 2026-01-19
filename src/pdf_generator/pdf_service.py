"""PDF generation module using WeasyPrint."""

import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import markdown

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFGenerator:
    """Generate PDF reports from news articles."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown(
        self,
        articles: List[Dict[str, Any]],
        title: str = None
    ) -> str:
        """Generate markdown content from articles (flat list, no grouping)."""
        title = title or settings.pdf_title

        md_content = f"# {title}\n\n"
        md_content += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        md_content += f"---\n\n"
        md_content += f"**Total Articles: {len(articles)}**\n\n"
        md_content += "---\n\n"

        for idx, article in enumerate(articles, 1):
            md_content += f"## {idx}. {article.get('title', 'No Title')}\n\n"

            # Metadata
            source = article.get('source', 'Unknown')
            author = article.get('author', 'Unknown')
            date = article.get('published_date', 'N/A')
            url = article.get('url', '#')

            md_content += f"**Source:** {source}  \n"
            md_content += f"**Author:** {author}  \n"
            md_content += f"**Published:** {date}  \n"
            md_content += f"**URL:** [{url}]({url})  \n\n"

            # Content
            content = article.get('content', 'No content available.')
            md_content += f"{content}\n\n"
            md_content += "---\n\n"

        return md_content

    def generate_markdown_grouped(
        self,
        grouped_articles: List[Dict[str, Any]],
        title: str = None
    ) -> str:
        """Generate markdown content from articles grouped by topic (simple version).

        Args:
            grouped_articles: List of dicts with 'topic_name' and 'articles' keys
            title: Optional title for the document
        """
        title = title or settings.pdf_title

        # Count total articles
        total_articles = sum(len(g["articles"]) for g in grouped_articles)

        md_content = f"# {title}\n\n"
        md_content += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        md_content += f"**Total Articles: {total_articles}** | **Topics: {len(grouped_articles)}**\n\n"
        md_content += "---\n\n"

        # Generate Table of Contents
        md_content += "## Table of Contents\n\n"
        for idx, group in enumerate(grouped_articles, 1):
            topic = group["topic_name"]
            count = len(group["articles"])
            # Create anchor-friendly ID
            anchor = topic.lower().replace(" ", "-").replace("&", "and")
            md_content += f"{idx}. [{topic}](#{anchor}) ({count} articles)\n"
        md_content += "\n---\n\n"

        # Generate content for each topic
        article_counter = 1
        for group in grouped_articles:
            topic = group["topic_name"]
            articles = group["articles"]
            anchor = topic.lower().replace(" ", "-").replace("&", "and")

            md_content += f"## {topic} {{#{anchor}}}\n\n"
            md_content += f"*{len(articles)} articles in this topic*\n\n"

            for article in articles:
                md_content += f"### {article_counter}. {article.get('title', 'No Title')}\n\n"

                # Metadata
                source = article.get('source', 'Unknown')
                author = article.get('author', 'Unknown')
                date = article.get('published_date', 'N/A')
                url = article.get('url', '#')

                md_content += f"**Source:** {source}  \n"
                md_content += f"**Author:** {author}  \n"
                md_content += f"**Published:** {date}  \n"
                md_content += f"**URL:** [{url}]({url})  \n\n"

                # Content
                content = article.get('content', 'No content available.')
                md_content += f"{content}\n\n"
                md_content += "---\n\n"

                article_counter += 1

        return md_content

    def generate_markdown_enriched(
        self,
        enriched_data: Dict[str, Any],
        title: str = None
    ) -> str:
        """Generate markdown with executive summary, top picks, narratives, and sources.

        Optimized for NotebookLM podcast generation with narrative-first structure.

        Args:
            enriched_data: Dict with 'executive_summary', 'top_articles', and 'topics'
            title: Optional title for the document
        """
        title = title or settings.pdf_title

        executive_summary = enriched_data.get("executive_summary", "")
        top_articles = enriched_data.get("top_articles", [])
        topics = enriched_data.get("topics", [])

        # Count total articles
        total_articles = sum(len(t["articles"]) for t in topics)

        md_content = f"# {title}\n\n"
        md_content += f"*Your weekly tech intelligence briefing for consultants*\n\n"
        md_content += f"*Generated on {datetime.now().strftime('%B %d, %Y')}*\n\n"
        md_content += "---\n\n"

        # Executive Summary
        md_content += "## Executive Summary\n\n"
        md_content += f"{executive_summary}\n\n"
        md_content += "---\n\n"

        # Top 3 Must-Read Articles (narrative style)
        if top_articles:
            md_content += "## This Week's Must-Know Stories\n\n"
            md_content += "Before diving into the details, here are the three stories that deserve your immediate attention this week.\n\n"
            for idx, article in enumerate(top_articles, 1):
                title_text = article.get('title', 'No Title')
                reason = article.get('relevance_reason', '')
                source = article.get('source', 'Unknown')
                content = article.get('content', '')[:300]

                md_content += f"**{idx}. {title_text}**\n\n"
                if reason:
                    md_content += f"{reason} "
                if content:
                    md_content += f"{content}\n\n"
                md_content += f"*Source: {source}*\n\n"
            md_content += "---\n\n"

        # Topics Overview
        md_content += "## What We're Covering\n\n"
        md_content += f"This briefing covers {total_articles} developments across {len(topics)} key areas:\n\n"
        for idx, topic in enumerate(topics, 1):
            topic_name = topic["topic_name"]
            count = len(topic["articles"])
            md_content += f"- **{topic_name}** ({count} stories)\n"
        md_content += "\n---\n\n"

        # Generate narrative content for each topic
        for topic in topics:
            topic_name = topic["topic_name"]
            narrative = topic.get("narrative", "")
            brief = topic.get("brief", "")
            articles = topic["articles"]

            md_content += f"## {topic_name}\n\n"

            # Main narrative (this is the key content for NotebookLM)
            if narrative:
                md_content += f"{narrative}\n\n"
            elif brief:
                md_content += f"{brief}\n\n"

            # Sources section (compact, for reference)
            md_content += "**Sources for this section:**\n\n"
            for article in articles:
                title_text = article.get('title', 'No Title')
                source = article.get('source', 'Unknown')
                url = article.get('url', '#')
                md_content += f"- [{title_text}]({url}) — *{source}*\n"

            md_content += "\n---\n\n"

        # Closing
        md_content += "## Wrapping Up\n\n"
        md_content += f"That's your tech briefing for {datetime.now().strftime('%B %d, %Y')}. "
        md_content += f"We covered {total_articles} stories across {len(topics)} topics. "
        md_content += "Stay informed, stay ahead, and use these insights to deliver more value to your clients.\n\n"
        md_content += "*This briefing was automatically generated and curated for tech consultants.*\n"

        return md_content
    
    def generate_html(self, markdown_content: str) -> str:
        """Convert markdown to HTML with styling."""
        # Convert markdown to HTML
        html_body = markdown.markdown(
            markdown_content,
            extensions=['extra', 'codehilite', 'tables']
        )

        # Create full HTML document with CSS
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{settings.pdf_title}</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}

        body {{
            font-family: 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}

        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }}

        a {{
            color: #3498db;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        hr {{
            border: none;
            border-top: 1px solid #ecf0f1;
            margin: 30px 0;
        }}

        strong {{
            color: #2c3e50;
        }}

        em {{
            color: #7f8c8d;
        }}

        code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}

        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}

        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>
"""
        return html

    def generate_html_enriched(self, enriched_data: Dict[str, Any], title: str = None) -> str:
        """Generate visually rich HTML with badges, cards, and better hierarchy.

        Optimized for PDF generation with visual aids for easier reading.
        """
        title = title or settings.pdf_title
        executive_summary = enriched_data.get("executive_summary", "")
        top_articles = enriched_data.get("top_articles", [])
        topics = enriched_data.get("topics", [])
        total_articles = sum(len(t["articles"]) for t in topics)

        # Topic colors for visual distinction
        topic_colors = [
            "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
            "#f39c12", "#1abc9c", "#e67e22", "#34495e"
        ]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @page {{
            size: A4;
            margin: 1.5cm;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.7;
            color: #2d3748;
            max-width: 100%;
            margin: 0;
            padding: 0;
            font-size: 11pt;
        }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%);
            color: white;
            padding: 30px;
            margin: -1.5cm -1.5cm 30px -1.5cm;
            text-align: center;
        }}

        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 28pt;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}

        .header .subtitle {{
            font-size: 12pt;
            opacity: 0.9;
            margin-bottom: 15px;
        }}

        .header .meta {{
            display: inline-block;
            background: rgba(255,255,255,0.15);
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 10pt;
        }}

        /* Stats badges */
        .stats {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 25px 0;
        }}

        .stat-badge {{
            background: #f7fafc;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px 25px;
            text-align: center;
        }}

        .stat-badge .number {{
            font-size: 24pt;
            font-weight: 700;
            color: #2d3748;
            display: block;
        }}

        .stat-badge .label {{
            font-size: 9pt;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Section titles */
        .section-title {{
            font-size: 16pt;
            color: #1a365d;
            border-bottom: 3px solid #3182ce;
            padding-bottom: 10px;
            margin: 35px 0 20px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section-icon {{
            font-size: 20pt;
        }}

        /* Executive summary card */
        .executive-summary {{
            background: linear-gradient(135deg, #ebf8ff 0%, #f0fff4 100%);
            border-left: 5px solid #3182ce;
            border-radius: 0 12px 12px 0;
            padding: 25px;
            margin: 20px 0;
        }}

        .executive-summary h3 {{
            margin: 0 0 15px 0;
            color: #2c5282;
            font-size: 13pt;
        }}

        .executive-summary p {{
            margin: 0;
            color: #2d3748;
        }}

        /* Must-read cards */
        .must-read-card {{
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            page-break-inside: avoid;
        }}

        .must-read-card.priority-1 {{
            border-left: 5px solid #e53e3e;
        }}

        .must-read-card.priority-2 {{
            border-left: 5px solid #dd6b20;
        }}

        .must-read-card.priority-3 {{
            border-left: 5px solid #d69e2e;
        }}

        .priority-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 8pt;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        .priority-badge.p1 {{
            background: #fed7d7;
            color: #c53030;
        }}

        .priority-badge.p2 {{
            background: #feebc8;
            color: #c05621;
        }}

        .priority-badge.p3 {{
            background: #fefcbf;
            color: #b7791f;
        }}

        .must-read-card h4 {{
            margin: 10px 0;
            font-size: 12pt;
            color: #1a365d;
        }}

        .why-matters {{
            background: #f7fafc;
            padding: 12px 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 10pt;
        }}

        .why-matters strong {{
            color: #2c5282;
        }}

        /* Topic overview */
        .topics-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }}

        .topic-pill {{
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 9pt;
            font-weight: 500;
            color: white;
        }}

        /* Topic sections */
        .topic-section {{
            margin: 30px 0;
            page-break-inside: avoid;
        }}

        .topic-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }}

        .topic-badge {{
            padding: 10px 20px;
            border-radius: 25px;
            color: white;
            font-weight: 600;
            font-size: 12pt;
        }}

        .topic-count {{
            background: #edf2f7;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 9pt;
            color: #4a5568;
        }}

        .narrative-box {{
            background: #f8fafc;
            border-radius: 12px;
            padding: 25px;
            margin: 15px 0;
            border: 1px solid #e2e8f0;
        }}

        .narrative-box p {{
            margin: 0 0 15px 0;
            text-align: justify;
        }}

        .narrative-box p:last-child {{
            margin-bottom: 0;
        }}

        /* Sources list */
        .sources-section {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px dashed #cbd5e0;
        }}

        .sources-title {{
            font-size: 9pt;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        .source-item {{
            display: flex;
            align-items: baseline;
            gap: 8px;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
            font-size: 9pt;
        }}

        .source-item:last-child {{
            border-bottom: none;
        }}

        .source-bullet {{
            color: #a0aec0;
        }}

        .source-item a {{
            color: #3182ce;
            text-decoration: none;
            flex: 1;
        }}

        .source-tag {{
            background: #edf2f7;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 8pt;
            color: #718096;
        }}

        /* Footer */
        .footer {{
            background: #f7fafc;
            border-radius: 12px;
            padding: 25px;
            margin-top: 40px;
            text-align: center;
        }}

        .footer h3 {{
            margin: 0 0 10px 0;
            color: #2d3748;
        }}

        .footer p {{
            margin: 0;
            color: #718096;
            font-size: 10pt;
        }}

        /* Utilities */
        .divider {{
            height: 2px;
            background: linear-gradient(90deg, #e2e8f0 0%, transparent 100%);
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">Your Weekly Tech Intelligence Briefing for Consultants</div>
        <div class="meta">Generated on {datetime.now().strftime('%B %d, %Y')}</div>
    </div>

    <div class="stats">
        <div class="stat-badge">
            <span class="number">{total_articles}</span>
            <span class="label">Articles</span>
        </div>
        <div class="stat-badge">
            <span class="number">{len(topics)}</span>
            <span class="label">Topics</span>
        </div>
        <div class="stat-badge">
            <span class="number">{len(top_articles)}</span>
            <span class="label">Must-Reads</span>
        </div>
    </div>

    <div class="section-title">
        <span class="section-icon">📋</span>
        Executive Summary
    </div>

    <div class="executive-summary">
        <p>{executive_summary.replace(chr(10), '</p><p>')}</p>
    </div>
"""

        # Top articles section
        if top_articles:
            html += """
    <div class="section-title">
        <span class="section-icon">⭐</span>
        This Week's Must-Read Stories
    </div>
"""
            priority_classes = ["p1", "p2", "p3"]
            priority_labels = ["Top Priority", "High Priority", "Notable"]

            for idx, article in enumerate(top_articles[:3]):
                p_class = priority_classes[idx] if idx < 3 else "p3"
                p_label = priority_labels[idx] if idx < 3 else "Notable"
                card_class = f"priority-{idx+1}" if idx < 3 else "priority-3"

                title_text = article.get('title', 'No Title')
                reason = article.get('relevance_reason', '')
                source = article.get('source', 'Unknown')
                content = article.get('content', '')[:250]

                html += f"""
    <div class="must-read-card {card_class}">
        <span class="priority-badge {p_class}">{p_label}</span>
        <h4>{title_text}</h4>
        <div class="why-matters">
            <strong>Why it matters:</strong> {reason}
        </div>
        <p style="font-size: 10pt; color: #4a5568; margin: 10px 0;">{content}...</p>
        <div style="font-size: 9pt; color: #718096;">Source: {source}</div>
    </div>
"""

        # Topics overview
        html += """
    <div class="section-title">
        <span class="section-icon">📚</span>
        Topics Covered
    </div>
    <div class="topics-grid">
"""
        for idx, topic in enumerate(topics):
            color = topic_colors[idx % len(topic_colors)]
            topic_name = topic["topic_name"]
            count = len(topic["articles"])
            html += f'        <span class="topic-pill" style="background: {color};">{topic_name} ({count})</span>\n'

        html += "    </div>\n\n    <div class=\"divider\"></div>\n"

        # Topic sections with narratives
        for idx, topic in enumerate(topics):
            color = topic_colors[idx % len(topic_colors)]
            topic_name = topic["topic_name"]
            narrative = topic.get("narrative", "")
            articles = topic["articles"]

            html += f"""
    <div class="topic-section">
        <div class="topic-header">
            <span class="topic-badge" style="background: {color};">{topic_name}</span>
            <span class="topic-count">{len(articles)} articles</span>
        </div>

        <div class="narrative-box">
            <p>{narrative.replace(chr(10), '</p><p>')}</p>
        </div>

        <div class="sources-section">
            <div class="sources-title">📎 Sources</div>
"""
            for article in articles:
                a_title = article.get('title', 'No Title')
                a_source = article.get('source', 'Unknown')
                a_url = article.get('url', '#')
                html += f"""            <div class="source-item">
                <span class="source-bullet">•</span>
                <a href="{a_url}">{a_title}</a>
                <span class="source-tag">{a_source}</span>
            </div>
"""
            html += "        </div>\n    </div>\n"

        # Footer
        html += f"""
    <div class="footer">
        <h3>That's Your Briefing</h3>
        <p>Stay informed, stay ahead. Use these insights to deliver more value to your clients.</p>
        <p style="margin-top: 10px; font-size: 9pt;">Auto-generated tech intelligence • {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
</body>
</html>
"""
        return html
    
    def generate_pdf(
        self,
        articles: List[Dict[str, Any]],
        filename: str = None,
        title: str = None
    ) -> Path:
        """Generate PDF from articles (flat list)."""
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("WeasyPrint is not installed. Install it with: pip install weasyprint")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tech_news_digest_{timestamp}.pdf"

        output_path = self.output_dir / filename

        try:
            # Generate markdown
            markdown_content = self.generate_markdown(articles, title)

            # Convert to HTML
            html_content = self.generate_html(markdown_content)

            # Generate PDF
            HTML(string=html_content).write_pdf(output_path)

            logger.info(f"PDF generated successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise

    def generate_pdf_grouped(
        self,
        grouped_articles: List[Dict[str, Any]],
        filename: str = None,
        title: str = None
    ) -> Path:
        """Generate PDF from articles grouped by topic.

        Args:
            grouped_articles: List of dicts with 'topic_name' and 'articles' keys
            filename: Optional output filename
            title: Optional document title
        """
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("WeasyPrint is not installed. Install it with: pip install weasyprint")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tech_news_digest_{timestamp}.pdf"

        output_path = self.output_dir / filename

        try:
            # Generate grouped markdown
            markdown_content = self.generate_markdown_grouped(grouped_articles, title)

            # Convert to HTML
            html_content = self.generate_html(markdown_content)

            # Generate PDF
            HTML(string=html_content).write_pdf(output_path)

            logger.info(f"Grouped PDF generated successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating grouped PDF: {e}")
            raise

    def generate_pdf_enriched(
        self,
        enriched_data: Dict[str, Any],
        filename: str = None,
        title: str = None
    ) -> Path:
        """Generate visually rich PDF with badges, cards, and narratives.

        Args:
            enriched_data: Dict with 'executive_summary', 'top_articles', and 'topics'
            filename: Optional output filename
            title: Optional document title
        """
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("WeasyPrint is not installed. Install it with: pip install weasyprint")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tech_news_digest_{timestamp}.pdf"

        output_path = self.output_dir / filename

        try:
            # Generate visually rich HTML directly
            html_content = self.generate_html_enriched(enriched_data, title)

            # Generate PDF
            HTML(string=html_content).write_pdf(output_path)

            logger.info(f"Enriched PDF generated successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating enriched PDF: {e}")
            raise
    
    def generate_pdf_from_markdown(
        self,
        markdown_content: str,
        filename: str = None
    ) -> Path:
        """Generate PDF directly from markdown content."""
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("WeasyPrint is not installed. Install it with: pip install weasyprint")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.pdf"
        
        output_path = self.output_dir / filename
        
        try:
            html_content = self.generate_html(markdown_content)
            HTML(string=html_content).write_pdf(output_path)
            
            logger.info(f"PDF generated from markdown: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF from markdown: {e}")
            raise


def main():
    """Test PDF generation."""
    generator = PDFGenerator()
    
    # Sample articles
    sample_articles = [
        {
            "title": "AI Breakthrough: New Model Achieves Human-Level Performance",
            "source": "TechCrunch",
            "author": "John Doe",
            "published_date": "2024-01-18",
            "url": "https://example.com/article1",
            "content": "A new artificial intelligence model has achieved human-level performance on a wide range of tasks. The model, developed by researchers at a leading tech company, demonstrates unprecedented capabilities in natural language understanding and generation."
        },
        {
            "title": "Quantum Computing Makes Major Leap Forward",
            "source": "The Verge",
            "author": "Jane Smith",
            "published_date": "2024-01-17",
            "url": "https://example.com/article2",
            "content": "Scientists have announced a major breakthrough in quantum computing, successfully running complex algorithms on a 1000-qubit processor. This achievement brings us closer to practical quantum computers that can solve real-world problems."
        }
    ]
    
    output_path = generator.generate_pdf(sample_articles, "test_report.pdf")
    print(f"Test PDF generated at: {output_path}")


if __name__ == "__main__":
    main()
