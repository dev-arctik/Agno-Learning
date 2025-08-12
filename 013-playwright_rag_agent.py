#!/usr/bin/env python3
"""
Playwright-powered RAG Agent using Agno

This agent creates a comprehensive RAG system that:
1. Uses Playwright to discover URLs by navigating and clicking through a website
2. Extracts content from all discovered URLs using BeautifulSoup
3. Uses Agentic Chunking for intelligent document processing
4. Stores data in LanceDB with OpenAI embeddings
5. Provides intelligent conversational interface using GPT-4o-mini

Prerequisites:
- pip install agno playwright beautifulsoup4 lxml requests
- playwright install chromium
- export OPENAI_API_KEY=your_openai_api_key

Usage:
python 013-playwright_rag_agent.py
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from urllib.parse import urljoin, urlparse
import typer
from rich.prompt import Prompt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.agentic import AgenticChunking
from agno.knowledge.agent import AgentKnowledge
from agno.document.base import Document
from agno.tools.reasoning import ReasoningTools

# Try to import from config.secrets, fallback to environment variables
try:
    from config.secrets import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Rich console for beautiful output
console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

class PlaywrightRAGConfig:
    """Configuration class for the Playwright RAG system"""
    
    def __init__(
        self,
        base_url: str,
        company_name: str = "WebSite",
        ai_name: str = "PlaywrightBot",
        ai_instructions: List[str] = None,
        lancedb_path: str = "./temp/playwright_lancedb",
        table_name: str = "",
        max_urls: int = 50,
        max_depth: int = 3,
        wait_time: int = 2000,  # milliseconds
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None
    ):
        self.base_url = base_url
        self.company_name = company_name
        self.ai_name = ai_name
        self.ai_instructions = ai_instructions or self._default_instructions()
        self.lancedb_path = lancedb_path
        self.table_name = table_name or f"{company_name.lower().replace(' ', '_')}_knowledge"
        self.max_urls = max_urls
        self.max_depth = max_depth
        self.wait_time = wait_time
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or [
            'logout', 'login', 'signup', 'register', 'admin', 
            '.pdf', '.jpg', '.png', '.gif', '.zip', '.exe',
            'mailto:', 'tel:', 'javascript:'
        ]
    
    def _default_instructions(self) -> List[str]:
        return [
            f"You are {self.ai_name}, an AI assistant with comprehensive knowledge about {self.company_name}.",
            "You have access to detailed information extracted from the company's website.",
            "Always search your knowledge base before answering questions.",
            "Provide accurate, helpful, and contextual responses based on the website content.",
            "If you cannot find relevant information in your knowledge base, clearly state this.",
            "Include sources and references when available.",
            "Be professional, friendly, and informative in your responses."
        ]

# ============================================================================
# PLAYWRIGHT URL DISCOVERY
# ============================================================================

class PlaywrightURLDiscovery:
    """Class to discover URLs using Playwright by navigating and clicking through website"""
    
    def __init__(self, config: PlaywrightRAGConfig):
        self.config = config
        self.discovered_urls: Set[str] = set()
        self.visited_urls: Set[str] = set()
        self.browser: Optional[Browser] = None
        self.base_domain = urlparse(config.base_url).netloc
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be included based on patterns and domain"""
        if not url or url in self.visited_urls:
            return False
        
        # Check if it's from the same domain
        parsed_url = urlparse(url)
        if parsed_url.netloc and parsed_url.netloc != self.base_domain:
            return False
        
        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if pattern.lower() in url.lower():
                return False
        
        # Check include patterns (if any specified)
        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if pattern.lower() in url.lower():
                    return True
            return False
        
        return True
    
    async def _extract_links_from_page(self, page: Page) -> List[str]:
        """Extract all links from current page"""
        try:
            # Wait for page to load
            await page.wait_for_load_state("networkidle", timeout=10000)
            
            # Get all links
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => link.href).filter(href => href && href.trim() !== '');
                }
            """)
            
            # Filter and normalize URLs
            valid_links = []
            for link in links:
                if self._is_valid_url(link):
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(self.config.base_url, link)
                    if self._is_valid_url(absolute_url):
                        valid_links.append(absolute_url)
            
            return valid_links
            
        except Exception as e:
            console.print(f"[yellow]Warning: Error extracting links: {str(e)}[/yellow]")
            return []
    
    async def _navigate_and_discover(self, page: Page, url: str, depth: int = 0) -> None:
        """Navigate to URL and discover more URLs by clicking through the site"""
        if depth >= self.config.max_depth or len(self.discovered_urls) >= self.config.max_urls:
            return
        
        if url in self.visited_urls:
            return
        
        try:
            console.print(f"[cyan]Navigating to: {url} (depth: {depth})[/cyan]")
            
            # Navigate to the page
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self.visited_urls.add(url)
            self.discovered_urls.add(url)
            
            # Wait for dynamic content
            await page.wait_for_timeout(self.config.wait_time)
            
            # Extract links from current page
            links = await self._extract_links_from_page(page)
            
            console.print(f"[dim]Found {len(links)} links on this page[/dim]")
            
            # Add links to discovered URLs
            for link in links:
                if len(self.discovered_urls) < self.config.max_urls:
                    self.discovered_urls.add(link)
            
            # Navigate to some of the discovered links (limited to prevent infinite recursion)
            if depth < self.config.max_depth - 1:
                # Take first few links for deeper navigation
                navigation_links = [link for link in links if link not in self.visited_urls][:3]
                
                for nav_link in navigation_links:
                    if len(self.discovered_urls) >= self.config.max_urls:
                        break
                    await self._navigate_and_discover(page, nav_link, depth + 1)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Error navigating to {url}: {str(e)}[/yellow]")
    
    async def discover_urls(self) -> List[str]:
        """Main method to discover URLs using Playwright"""
        console.print(f"[blue]Starting URL discovery for: {self.config.base_url}[/blue]")
        console.print(f"[dim]Max URLs: {self.config.max_urls}, Max Depth: {self.config.max_depth}[/dim]")
        
        async with async_playwright() as p:
            # Launch browser
            self.browser = await p.chromium.launch(headless=True)
            context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            try:
                # Start discovery from base URL
                await self._navigate_and_discover(page, self.config.base_url)
                
                console.print(f"[green]✓ Discovery completed! Found {len(self.discovered_urls)} URLs[/green]")
                
                return list(self.discovered_urls)
                
            finally:
                await context.close()
                await self.browser.close()

# ============================================================================
# CONTENT EXTRACTOR
# ============================================================================

class WebContentExtractor:
    """Extract content from URLs using requests and BeautifulSoup"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def extract_content(self, url: str) -> Optional[str]:
        """Extract main content from a URL"""
        try:
            console.print(f"[dim]Extracting content from: {url}[/dim]")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.main-content', '#main-content', '.post-content',
                '.entry-content', 'body'
            ]
            
            content_text = ""
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text()
                    break
            
            # Fallback to body if no specific content area found
            if not content_text:
                content_text = soup.get_text()
            
            # Clean the text
            cleaned_content = self._clean_text(content_text)
            
            if len(cleaned_content) > 100:  # Only return if substantial content
                return cleaned_content
            
            return None
            
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to extract content from {url}: {str(e)}[/yellow]")
            return None

# ============================================================================
# DOCUMENT LOADER
# ============================================================================

class PlaywrightDocumentLoader:
    """Load documents using Playwright URL discovery and content extraction"""
    
    def __init__(self, config: PlaywrightRAGConfig, chunking_strategy=None):
        self.config = config
        self.url_discovery = PlaywrightURLDiscovery(config)
        self.content_extractor = WebContentExtractor()
        self.chunking_strategy = chunking_strategy or AgenticChunking()
    
    async def load_documents(self) -> List[Document]:
        """Load documents by discovering URLs and extracting content"""
        
        # Discover URLs using Playwright
        urls = await self.url_discovery.discover_urls()
        
        if not urls:
            console.print("[red]No URLs were discovered![/red]")
            return []
        
        console.print(f"[blue]Extracting content from {len(urls)} discovered URLs...[/blue]")
        
        all_documents = []
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Extracting content...", total=len(urls))
            
            for i, url in enumerate(urls, 1):
                progress.update(task, advance=1)
                
                content = self.content_extractor.extract_content(url)
                
                if content:
                    doc = Document(
                        name=f"page_{i}_{urlparse(url).path.replace('/', '_')}",
                        content=content,
                        meta_data={
                            "source_url": url,
                            "extracted_at": str(int(time.time() * 1000)),
                            "document_type": "web_page",
                            "extraction_method": "playwright_discovery"
                        }
                    )
                    all_documents.append(doc)
                    console.print(f"[green]✓ Extracted content from {url}[/green]")
                else:
                    console.print(f"[yellow]⚠ No content extracted from {url}[/yellow]")
        
        if not all_documents:
            console.print("[red]No content was extracted from any URLs![/red]")
            return []
        
        # Apply chunking
        console.print(f"[blue]Processing {len(all_documents)} documents with agentic chunking...[/blue]")
        
        chunked_documents = []
        for doc in all_documents:
            try:
                chunks = self.chunking_strategy.chunk(doc)
                chunked_documents.extend(chunks)
            except Exception as e:
                console.print(f"[yellow]Warning: Error chunking document {doc.name}: {str(e)}[/yellow]")
                chunked_documents.append(doc)
        
        console.print(f"[blue]Created {len(chunked_documents)} chunks from {len(all_documents)} documents[/blue]")
        return chunked_documents

# ============================================================================
# MAIN RAG AGENT
# ============================================================================

class PlaywrightRAGAgent:
    """Main RAG Agent class using Playwright for URL discovery"""
    
    def __init__(self, config: PlaywrightRAGConfig, openai_api_key: Optional[str] = None):
        self.config = config
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        self.agent = None
        self.knowledge_base = None
        self.document_loader = None
        self.chunking_strategy = AgenticChunking()
        
        self._setup_agent()
    
    def _setup_agent(self):
        """Initialize the agent with all components"""
        console.print(f"[blue]Initializing {self.config.ai_name} for {self.config.company_name}...[/blue]")
        
        # Create embedder
        embedder = OpenAIEmbedder(
            id="text-embedding-3-small",
            dimensions=1536,
            api_key=self.openai_api_key
        )
        
        # Create vector database
        vector_db = LanceDb(
            table_name=self.config.table_name,
            uri=self.config.lancedb_path,
            search_type=SearchType.vector,
            embedder=embedder
        )
        
        # Create knowledge base
        self.knowledge_base = AgentKnowledge(
            vector_db=vector_db,
            chunking_strategy=self.chunking_strategy
        )
        
        # Create document loader
        self.document_loader = PlaywrightDocumentLoader(
            config=self.config,
            chunking_strategy=self.chunking_strategy
        )
        
        # Create agent
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key=self.openai_api_key),
            knowledge=self.knowledge_base,
            tools=[
                ReasoningTools(
                    add_instructions=True,
                    instructions="Always search the knowledge base thoroughly before answering questions.",
                    think=True
                )
            ],
            instructions=self.config.ai_instructions,
            add_datetime_to_instructions=True,
            add_history_to_messages=True,
            num_history_runs=5,
            search_knowledge=True,
            show_tool_calls=True,
            markdown=True,
            debug_mode=False
        )
        
        console.print(f"[green]✓ {self.config.ai_name} initialized successfully![/green]")
    
    async def load_knowledge(self, recreate: bool = False):
        """Load the knowledge base using Playwright discovery"""
        console.print(f"[blue]Loading knowledge base for {self.config.base_url}...[/blue]")
        
        if recreate:
            console.print(f"[yellow]Recreating knowledge base...[/yellow]")
            self.knowledge_base.vector_db.clear()
        
        # Check if knowledge base already exists
        if not recreate:
            try:
                existing_docs = self.knowledge_base.vector_db.search("test", limit=1)
                if existing_docs:
                    console.print(f"[green]Knowledge base already loaded with content. Skipping load.[/green]")
                    return
            except:
                pass
        
        # Load documents using Playwright
        documents = await self.document_loader.load_documents()
        
        if documents:
            console.print(f"[blue]Storing {len(documents)} chunks in vector database...[/blue]")
            self.knowledge_base.vector_db.insert(documents)
            console.print(f"[green]✓ Knowledge base loaded successfully with {len(documents)} chunks![/green]")
        else:
            console.print("[red]No documents were loaded![/red]")
    
    def chat(self):
        """Start interactive chat session"""
        console.print(f"\n[bold green]🤖 {self.config.ai_name} is ready to chat![/bold green]")
        console.print(f"[dim]Knowledge about {self.config.company_name} has been loaded from website discovery[/dim]")
        console.print(f"[dim]Base URL: {self.config.base_url}[/dim]")
        console.print(f"[dim]Type 'exit', 'quit', or 'bye' to end the conversation[/dim]\n")
        
        while True:
            try:
                user_input = Prompt.ask(f"[bold blue]You[/bold blue]")
                
                if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                    console.print(f"[yellow]👋 Goodbye! Thanks for chatting with {self.config.ai_name}![/yellow]")
                    break
                
                console.print(f"\n[bold green]{self.config.ai_name}[/bold green]:")
                self.agent.print_response(user_input, stream=True)
                console.print("\n" + "─" * 60 + "\n")
                
            except KeyboardInterrupt:
                console.print(f"\n[yellow]👋 Goodbye! Thanks for chatting with {self.config.ai_name}![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    
    def print_config(self):
        """Print current configuration"""
        table = Table(title=f"{self.config.company_name} Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Company Name", self.config.company_name)
        table.add_row("AI Name", self.config.ai_name)
        table.add_row("Base URL", self.config.base_url)
        table.add_row("Max URLs", str(self.config.max_urls))
        table.add_row("Max Depth", str(self.config.max_depth))
        table.add_row("Wait Time", f"{self.config.wait_time}ms")
        table.add_row("LanceDB Path", self.config.lancedb_path)
        table.add_row("Table Name", self.config.table_name)
        
        console.print(table)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def main_async(
    base_url: str = "https://docs.agno.com",
    company_name: str = "Agno Docs",
    ai_name: str = "AgnoDocsBot",
    max_urls: int = 20,
    max_depth: int = 5,
    recreate: bool = False,
    config_only: bool = False
):
    """
    Playwright-powered RAG Agent using Agno Framework
    
    This tool creates an intelligent RAG agent that:
    1. Uses Playwright to discover URLs by navigating through a website
    2. Extracts content from all discovered URLs
    3. Creates embeddings and stores in LanceDB with agentic chunking
    4. Provides conversational AI interface
    """
    
    # Validate OpenAI API key
    if not OPENAI_API_KEY:
        console.print("[red]Error: OPENAI_API_KEY environment variable is not set![/red]")
        return
    
    # Create configuration
    config = PlaywrightRAGConfig(
        base_url=base_url,
        company_name=company_name,
        ai_name=ai_name,
        max_urls=max_urls,
        max_depth=max_depth
    )
    
    # Create agent system
    agent_system = PlaywrightRAGAgent(config, openai_api_key=OPENAI_API_KEY)
    agent_system.print_config()
    
    if config_only:
        return
    
    try:
        # Load knowledge base
        await agent_system.load_knowledge(recreate=recreate)
        
        # Start chat
        agent_system.chat()
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise

def main(
    base_url: str = typer.Option("https://docs.agno.com", help="Base URL to start discovery from"),
    company_name: str = typer.Option("Agno Docs", help="Name of the company/website"),
    ai_name: str = typer.Option("AgnoDocsBot", help="Name of your AI assistant"),
    max_urls: int = typer.Option(100, help="Maximum number of URLs to discover"),
    max_depth: int = typer.Option(5, help="Maximum depth for URL discovery"),
    recreate: bool = typer.Option(False, help="Recreate the knowledge base"),
    config_only: bool = typer.Option(False, help="Only show configuration and exit")
):
    asyncio.run(main_async(
        base_url=base_url,
        company_name=company_name,
        ai_name=ai_name,
        max_urls=max_urls,
        max_depth=max_depth,
        recreate=recreate,
        config_only=config_only
    ))


if __name__ == "__main__":
    typer.run(main)

# TO RUN THE AGENT
# python 013-playwright_rag_agent.py --base-url "https://docs.agno.com" --max-urls 20 --max-depth 2 --company-name "Agno Docs" --ai-name "AgnoDocsBot"