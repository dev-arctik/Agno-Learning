#!/usr/bin/env python3
"""
Web Content Extractor RAG Agent using Agno

This agent creates a comprehensive RAG system that:
1. Extracts content from websites using Firecrawl's Extract API (avoids rate limits)
2. Uses Agentic Chunking for intelligent document processing
3. Stores data in LanceDB with OpenAI embeddings
4. Provides intelligent conversational interface using GPT-4o-mini

Prerequisites:
- pip install agno firecrawl-py lancedb openai
- export FIRECRAWL_API_KEY=your_firecrawl_api_key
- export OPENAI_API_KEY=your_openai_api_key

Usage:
python 011-rag_agent.py
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
from agno.tools.firecrawl import FirecrawlTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.agentic import AgenticChunking
from agno.knowledge.agent import AgentKnowledge
from agno.document.base import Document
from agno.tools.reasoning import ReasoningTools

# Try to import from config.secrets, fallback to environment variables
try:
    from config.secrets import OPENAI_API_KEY, FIRECRAWL_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Initialize Rich console for beautiful output
console = Console()

# ============================================================================
# COMPANY CONFIGURATION
# ============================================================================

class CompanyConfig:
    """Configuration class for the company and AI agent"""
    
    def __init__(
        self,
        company_name: str = "TechCorp",
        ai_name: str = "WebScout",
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
    
    def _default_urls(self) -> List[str]:
        return [
            "https://docs.agno.com/introduction",
            "https://docs.agno.com/agents/introduction",
            "https://docs.agno.com/agents/tools",
            "https://docs.agno.com/agents/knowledge",
            "https://docs.agno.com/examples"
        ]

# ============================================================================
# CUSTOM FIRECRAWL CONTENT EXTRACTOR
# ============================================================================

class FirecrawlDocumentLoader:
    """Helper class to load documents using Firecrawl Extract and process them"""
    
    def __init__(self, firecrawl_api_key: Optional[str] = None, chunking_strategy = AgenticChunking()):
        self.firecrawl_api_key = firecrawl_api_key or FIRECRAWL_API_KEY
        self.chunking_strategy = chunking_strategy

        if not self.firecrawl_api_key:
            raise ValueError("Firecrawl API key is required. Set FIRECRAWL_API_KEY environment variable.")
    
    def load_documents(self, urls: List[str], chunking_strategy=None) -> List[Document]:
        """Load documents from URLs using Firecrawl Extract (avoids rate limits)"""
        
        console.print(f"[blue]Starting to extract content from {len(urls)} URLs with Firecrawl Extract...[/blue]")
        
        # Import FirecrawlApp for direct API usage
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            console.print("[red]Error: firecrawl-py package not found. Please install with: pip install firecrawl-py[/red]")
            return []
        
        # Initialize Firecrawl app
        app = FirecrawlApp(api_key=self.firecrawl_api_key)
        
        all_documents = []
        
        # Use extract feature to get structured content
        # Process URLs individually for better success rate and debugging
        extraction_prompt = """
        Extract all the meaningful content from this webpage including:
        - Main content and text
        - Key information, features, and descriptions
        - Any important details about products, services, or topics
        - Navigation and menu content that provides context
        - FAQ content, documentation, or help information
        
        Structure the content as readable text that preserves the original meaning and context.
        """
        
        # Process URLs individually
        for i, url in enumerate(urls, 1):
            console.print(f"[cyan]Extracting {i}/{len(urls)}: {url}[/cyan]")
            
            try:
                result = app.extract(
                    urls=[url],
                    prompt=extraction_prompt
                )
                
                if result.success and result.data:
                    content_data = result.data
                    
                    # Convert to text
                    if isinstance(content_data, dict):
                        content_text = ""
                        for key, value in content_data.items():
                            if isinstance(value, (str, int, float)):
                                content_text += f"{key}: {value}\n\n"
                            elif isinstance(value, list):
                                content_text += f"{key}:\n"
                                for item in value:
                                    content_text += f"- {item}\n"
                                content_text += "\n"
                    else:
                        content_text = str(content_data)
                    
                    if content_text.strip():
                        doc = Document(
                            name=f"extracted_content_{i}",
                            content=content_text,
                            meta_data={
                                "source_url": url,
                                "extracted_at": str(int(asyncio.get_event_loop().time() * 1000)) if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop() else str(int(__import__('time').time() * 1000)),
                                "document_type": "web_page_extract",
                                "extraction_method": "firecrawl_extract"
                            }
                        )
                        all_documents.append(doc)
                        console.print(f"[green]✓ Successfully extracted content from {url}[/green]")
                    else:
                        console.print(f"[yellow]⚠ No meaningful content extracted from {url}[/yellow]")
                else:
                    error_msg = getattr(result, 'error', 'Unknown error')
                    console.print(f"[red]✗ Failed to extract from {url}: {error_msg}[/red]")
                    
            except Exception as e:
                console.print(f"[red]✗ Error extracting {url}: {str(e)}[/red]")
                continue
        
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
    """Main RAG Agent class that combines all components using Firecrawl Extract"""
    
    def __init__(self, config: CompanyConfig, openai_api_key: Optional[str] = None, firecrawl_api_key: Optional[str] = None):
        """
        Initialize the WebScraperRAGAgent with configuration and API keys.
        """
        self.config = config
        self.agent = None
        self.knowledge_base = None
        self.document_loader = None
        
        # Set API keys and other attributes BEFORE calling _setup_agent()
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.firecrawl_api_key = firecrawl_api_key or FIRECRAWL_API_KEY
        self.chunking_strategy = AgenticChunking()

        # Validate API keys
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the agent.")
        
        if not self.firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable is not set. Please set it before running the agent.")
        
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
        self.document_loader = FirecrawlDocumentLoader(
            firecrawl_api_key=self.firecrawl_api_key,
            chunking_strategy=self.chunking_strategy
        )
        
        # Create agent
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key=self.openai_api_key),
            knowledge=self.knowledge_base,
            tools=[
                ReasoningTools(add_instructions=True,instructions="Always make sure you look at the knowledge base before answering questions.",think=True),
                FirecrawlTools(
                    api_key=self.firecrawl_api_key,
                    scrape=True,
                    crawl=True,
                    mapping=True,
                    search=True,
                    limit=10
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
        """Load the knowledge base using Firecrawl Extract"""
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
        
        # Load documents using Firecrawl Extract
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
            # This doesn't make an actual API call, just tests client creation
            console.print("[green]✓ OpenAI API key configured[/green]")
        except Exception as e:
            console.print(f"[red]✗ OpenAI API test failed: {str(e)}[/red]")
        
        try:
            # Test Firecrawl connection by creating app instance (doesn't make API call)
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=self.firecrawl_api_key)
            console.print("[green]✓ Firecrawl API key configured[/green]")
        except Exception as e:
            console.print(f"[red]✗ Firecrawl API test failed: {str(e)}[/red]")
        
        console.print("[blue]✓ Connection tests completed[/blue]")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def create_sample_config() -> CompanyConfig:
    """Create a sample configuration for demonstration with example.com data"""
    return CompanyConfig(
        company_name="Example Inc.",
        ai_name="ExampleBot",
        ai_instructions=[
            "You are ExampleBot, an AI assistant for Example Inc.",
            "You have comprehensive knowledge about Example Inc. and its website.",
            "Always search your knowledge base before answering questions about Example Inc.",
            "Provide accurate, helpful responses based on the extracted website content.",
            "If you cannot find relevant information, clearly state this.",
            "Be professional, friendly, and informative in your responses."
        ],
        urls_to_scrape=[
            "https://example.com",
            "https://example.com/about",
            "https://example.com/contact",
            "https://example.com/privacy",
            "https://example.com/terms"
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
    Web Content Extractor RAG Agent using Agno Framework
    
    This tool creates an intelligent RAG agent that extracts content from websites
    using Firecrawl's Extract API, processes content with agentic chunking, 
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
    agent_system = WebScraperRAGAgent(config, openai_api_key=OPENAI_API_KEY, firecrawl_api_key=FIRECRAWL_API_KEY)
    agent_system.print_config()
    
    if config_only:
        return
    
    try:
        # Load knowledge base
        agent_system.load_knowledge(recreate=recreate)
        
        # Start chat
        agent_system.chat()
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise

if __name__ == "__main__":
    typer.run(main)