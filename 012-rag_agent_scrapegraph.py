#!/usr/bin/env python3
"""
Web Content Extractor RAG Agent using Agno with ScrapeGraph

This agent creates a comprehensive RAG system that:
1. Extracts content from websites using ScrapeGraph's markdownify API
2. Uses Agentic Chunking for intelligent document processing
3. Stores data in LanceDB with OpenAI embeddings
4. Provides intelligent conversational interface using GPT-4o-mini

Prerequisites:
- pip install agno scrapegraph-py lancedb openai
- export SGAI_API_KEY=your_scrapegraph_api_key
- export OPENAI_API_KEY=your_openai_api_key

Usage:
python scrapegraph_rag_agent.py
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import typer
from rich.prompt import Prompt
from rich.console import Console
from rich.table import Table

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.scrapegraph import ScrapeGraphTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.agentic import AgenticChunking
from agno.knowledge.agent import AgentKnowledge
from agno.document.base import Document
from agno.tools.reasoning import ReasoningTools

# Try to import from config.secrets, fallback to environment variables
try:
    from config.secrets import OPENAI_API_KEY, SGAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SGAI_API_KEY = os.getenv("SGAI_API_KEY")

# Initialize Rich console for beautiful output
console = Console()

# ============================================================================
# COMPANY CONFIGURATION
# ============================================================================

class CompanyConfig:
    """Configuration class for the company and AI agent"""
    
    def __init__(
        self,
        company_name: str,
        ai_name: str,
        ai_instructions: List[str] = None,
        urls_to_scrape: List[str] = None,
        lancedb_path: str = "./temp/lancedb",
        table_name: str = ""
    ):
        self.company_name = company_name
        self.ai_name = ai_name
        self.ai_instructions = ai_instructions or self._default_instructions()
        self.urls_to_scrape = urls_to_scrape or self._default_urls()
        self.lancedb_path = lancedb_path
        if table_name == "":
            self.table_name = self.company_name.lower().replace(" ", "_") + "_knowledge"
        else:
            self.table_name = table_name
    
    def _default_instructions(self) -> List[str]:
        return [
            f"You are {self.ai_name}, an AI assistant for {self.company_name}.",
            "You have access to comprehensive knowledge about the company from scraped web content.",
            "Always search your knowledge base before answering questions.",
            "Provide accurate, helpful, and contextual responses based on the company's information.",
            "If you cannot find relevant information in your knowledge base, clearly state this.",
            "Include sources and references when available.",
            "Be professional, friendly, and informative in your responses."
        ]

# ============================================================================
# CUSTOM SCRAPEGRAPH CONTENT EXTRACTOR
# ============================================================================

class ScrapeGraphDocumentLoader:
    """Helper class to load documents using ScrapeGraph markdownify and process them"""
    
    def __init__(self, sgai_api_key: Optional[str] = None, chunking_strategy = AgenticChunking()):
        self.sgai_api_key = sgai_api_key or SGAI_API_KEY
        self.chunking_strategy = chunking_strategy

        if not self.sgai_api_key:
            raise ValueError("ScrapeGraph API key is required. Set SGAI_API_KEY environment variable.")
    
    def _extract_content_with_fallbacks(self, client, url: str, url_index: int) -> str:
        """Extract content using multiple fallback strategies"""
        
        # Strategy 1: Try markdownify first
        try:
            print(f"[blue]STRATEGY 1: Using markdownify for URL {url_index}: {url}[/blue]")
            result = client.markdownify(website_url=url)
            if result and 'result' in result and result['result'].strip():
                content = result['result'].strip()
                if len(content) > 10:  # More than just whitespace
                    console.print(f"[dim]✓ Markdownify successful[/dim]")
                    return content
                else:
                    console.print(f"[yellow]⚠ Markdownify returned minimal content, trying alternatives...[/yellow]")
        except Exception as e:
            console.print(f"[dim]⚠ Markdownify failed: {str(e)}, trying alternatives...[/dim]")
        
        # Strategy 2: Try smartscraper with markdown prompt
        try:
            print(f"[blue]STRATEGY 2: Using smartscraper markdown prompt for URL {url_index}: {url}[/blue]")
            markdown_prompt = """
            Extract all content from this webpage and format it as clean markdown.
            Include:
            - All headings (# ## ### format)
            - All text content and paragraphs  
            - Lists and bullet points
            - Links in [text](url) format
            - Any important information
            
            Return only the markdown content, no explanations.
            """
            
            result = client.smartscraper(website_url=url, user_prompt=markdown_prompt)
            if result and 'result' in result and result['result']:
                content = str(result['result']).strip()
                if len(content) > 10:
                    console.print(f"[dim]✓ Smartscraper with markdown prompt successful[/dim]")
                    return content
        except Exception as e:
            console.print(f"[dim]⚠ Smartscraper markdown failed: {str(e)}, trying simple extraction...[/dim]")
        
        # Strategy 3: Try smartscraper with simple prompt
        try:
            print(f"[blue]STRATEGY 3: Using smartscraper simple prompt for URL {url_index}: {url}[/blue]")
            simple_prompt = "Extract all the text content, headings, and main information from this webpage"
            
            result = client.smartscraper(website_url=url, user_prompt=simple_prompt)
            if result and 'result' in result and result['result']:
                content = str(result['result']).strip()
                if len(content) > 10:
                    console.print(f"[dim]✓ Smartscraper with simple prompt successful[/dim]")
                    return content
        except Exception as e:
            console.print(f"[dim]⚠ Simple extraction failed: {str(e)}[/dim]")
        
        return None
    
    def load_documents(self, urls: List[str], chunking_strategy=None) -> List[Document]:
        """Load documents from URLs using ScrapeGraph with fallback strategies"""
        
        console.print(f"[blue]Starting to extract content from {len(urls)} URLs with ScrapeGraph...[/blue]")
        
        # Import ScrapeGraph Client
        try:
            from scrapegraph_py import Client
        except ImportError:
            console.print("[red]Error: scrapegraph-py package not found. Please install with: pip install scrapegraph-py[/red]")
            return []
        
        # Initialize ScrapeGraph client
        client = Client(api_key=self.sgai_api_key)
        
        all_documents = []
        
        # Process URLs individually with fallback strategies
        for i, url in enumerate(urls, 1):
            console.print(f"[cyan]Extracting {i}/{len(urls)}: {url}[/cyan]")
            
            content = self._extract_content_with_fallbacks(client, url, i)
            
            if content:
                doc = Document(
                    name=f"extracted_content_{i}",
                    content=content,
                    meta_data={
                        "source_url": url,
                        "extracted_at": str(int(__import__('time').time() * 1000)),
                        "document_type": "web_page_content",
                        "extraction_method": "scrapegraph_multi_strategy"
                    }
                )
                all_documents.append(doc)
                console.print(f"[green]✓ Successfully extracted content from {url}[/green]")
                console.print(f"[dim]Content length: {len(content)} characters[/dim]")
            else:
                console.print(f"[red]✗ Failed to extract meaningful content from {url}[/red]")
                console.print(f"[dim]This URL may have anti-bot protection or require JavaScript rendering[/dim]")
        
        if not all_documents:
            console.print("[red]No documents were successfully extracted![/red]")
            return []
        
        # Apply chunking if strategy provided
        if self.chunking_strategy:
            console.print(f"[blue]Processing {len(all_documents)} documents with chunking strategy...[/blue]")
            
            chunked_documents = []
            for doc in all_documents:
                try:
                    chunks = self.chunking_strategy.chunk(doc)
                    chunked_documents.extend(chunks)
                except Exception as e:
                    console.print(f"[yellow]Warning: Error chunking document {doc.name}: {str(e)}[/yellow]")
                    # Add original document if chunking fails
                    chunked_documents.append(doc)
            
            console.print(f"[blue]Created {len(chunked_documents)} chunks from {len(all_documents)} documents[/blue]")
            return chunked_documents
        
        return all_documents

# ============================================================================
# WEB CONTENT EXTRACTOR RAG AGENT
# ============================================================================

class WebScraperRAGAgent:
    """Main RAG Agent class that combines all components using ScrapeGraph"""
    
    def __init__(self, config: CompanyConfig, openai_api_key: Optional[str] = None, sgai_api_key: Optional[str] = None):
        """
        Initialize the WebScraperRAGAgent with configuration and API keys.
        """
        self.config = config
        self.agent = None
        self.knowledge_base = None
        self.document_loader = None
        
        # Set API keys and other attributes BEFORE calling _setup_agent()
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.sgai_api_key = sgai_api_key or SGAI_API_KEY
        self.chunking_strategy = AgenticChunking()

        # Validate API keys
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the agent.")
        
        if not self.sgai_api_key:
            raise ValueError("SGAI_API_KEY environment variable is not set. Please set it before running the agent.")
        
        # Now setup the agent with all attributes available
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
        
        # Create knowledge base using AgentKnowledge with custom document loading
        self.knowledge_base = AgentKnowledge(
            vector_db=vector_db,
            chunking_strategy=self.chunking_strategy
        )
        
        # Create document loader
        self.document_loader = ScrapeGraphDocumentLoader(
            sgai_api_key=self.sgai_api_key,
            chunking_strategy=self.chunking_strategy
        )
        
        # Create agent with ScrapeGraph tools
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key=self.openai_api_key),
            knowledge=self.knowledge_base,
            tools=[
                ReasoningTools(
                    add_instructions=True,
                    instructions="Always make sure you look at the knowledge base before answering questions.",
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
    
    def load_knowledge(self, recreate: bool = False):
        """Load the knowledge base using ScrapeGraph markdownify"""
        console.print(f"[blue]Loading knowledge base...[/blue]")
        
        if recreate:
            console.print(f"[yellow]Recreating knowledge base...[/yellow]")
            self.knowledge_base.vector_db.clear()
        
        # Check if knowledge base already exists and has content
        if not recreate:
            try:
                existing_docs = self.knowledge_base.vector_db.search("test", limit=1)
                if existing_docs:
                    console.print(f"[green]Knowledge base already loaded with content. Skipping load.[/green]")
                    return
            except:
                pass  # Continue with loading if we can't check existing content
        
        # Load documents using ScrapeGraph markdownify
        documents = self.document_loader.load_documents(
            urls=self.config.urls_to_scrape,
            chunking_strategy=self.chunking_strategy
        )
        
        if documents:
            console.print(f"[blue]Storing {len(documents)} chunks in vector database...[/blue]")
            
            # Store in vector database
            self.knowledge_base.vector_db.insert(documents)
            
            console.print(f"[green]✓ Knowledge base loaded successfully with {len(documents)} chunks![/green]")
        else:
            console.print("[red]No documents were loaded![/red]")
    
    def chat(self):
        """Start interactive chat session"""
        console.print(f"\n[bold green]🤖 {self.config.ai_name} is ready to chat![/bold green]")
        console.print(f"[dim]Knowledge about {self.config.company_name} has been loaded from {len(self.config.urls_to_scrape)} URLs[/dim]")
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
    
    def query(self, question: str) -> str:
        """Single query method"""
        return self.agent.run(question)
    
    def print_config(self):
        """Print current configuration"""
        table = Table(title=f"{self.config.company_name} Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Company Name", self.config.company_name)
        table.add_row("AI Name", self.config.ai_name)
        table.add_row("URLs to Extract", str(len(self.config.urls_to_scrape)))
        table.add_row("LanceDB Path", self.config.lancedb_path)
        table.add_row("Table Name", self.config.table_name)
        
        console.print(table)
        
        console.print(f"\n[bold]URLs to extract from:[/bold]")
        for i, url in enumerate(self.config.urls_to_scrape, 1):
            console.print(f"  {i}. {url}")

    def test_connection(self):
        """Test API connections and basic functionality"""
        console.print("[blue]Testing API connections...[/blue]")
        
        try:
            # Test OpenAI connection
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            console.print("[green]✓ OpenAI API key configured[/green]")
        except Exception as e:
            console.print(f"[red]✗ OpenAI API test failed: {str(e)}[/red]")
        
        try:
            # Test ScrapeGraph connection
            from scrapegraph_py import Client
            client = Client(api_key=self.sgai_api_key)
            console.print("[green]✓ ScrapeGraph API key configured[/green]")
        except Exception as e:
            console.print(f"[red]✗ ScrapeGraph API test failed: {str(e)}[/red]")
        
        console.print("[blue]✓ Connection tests completed[/blue]")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def create_sample_config() -> CompanyConfig:
    """Create a sample configuration for demonstration"""
    return CompanyConfig(
        company_name="Example Company",
        ai_name="ExampleBot",
        ai_instructions=[
            "You are ExampleBot, an AI assistant with knowledge from various web sources.",
            "You have comprehensive knowledge extracted from multiple websites.",
            "Always search your knowledge base before answering questions.",
            "Provide accurate, helpful responses based on the extracted content.",
            "If you cannot find relevant information, clearly state this.",
            "Be professional, friendly, and informative in your responses."
        ],
        urls_to_scrape=[
            "https://example.com",              # Simple example site
            "https://httpbin.org/html",         # Test HTML page
            "https://www.python.org",           # Python documentation  
            "https://docs.github.com/en/get-started",  # GitHub docs
            "https://www.wikipedia.org/wiki/Artificial_intelligence"  # Wikipedia article
        ]
    )

def main(
    company_name: str = typer.Option("Agno AI", help="Name of your company"),
    ai_name: str = typer.Option("AgnoBot", help="Name of your AI assistant"),
    urls: str = typer.Option("", help="Comma-separated list of URLs to extract from"),
    recreate: bool = typer.Option(False, help="Recreate the knowledge base"),
    config_only: bool = typer.Option(False, help="Only show configuration and exit")
):
    """
    Web Content Extractor RAG Agent using Agno Framework with ScrapeGraph
    
    This tool creates an intelligent RAG agent that extracts content from websites
    using ScrapeGraph's markdownify API, processes content with agentic chunking, 
    and provides conversational AI.
    """
    
    # Create configuration
    if urls:
        url_list = [url.strip() for url in urls.split(",") if url.strip()]
        config = CompanyConfig(
            company_name=company_name,
            ai_name=ai_name,
            urls_to_scrape=url_list
        )
    else:
        # Use sample configuration
        config = create_sample_config()
    
    # Show configuration
    agent_system = WebScraperRAGAgent(config, openai_api_key=OPENAI_API_KEY, sgai_api_key=SGAI_API_KEY)
    agent_system.print_config()
    
    if config_only:
        return
    
    try:
        # Test connections
        agent_system.test_connection()
        
        # Load knowledge base
        agent_system.load_knowledge(recreate=recreate)
        
        # Start chat
        agent_system.chat()
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise

if __name__ == "__main__":
    typer.run(main)