"""Configuration management for the book visualization tool."""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration settings."""
    anthropic_api_key: str
    output_dir: Path = Path("visualizations")
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set") 