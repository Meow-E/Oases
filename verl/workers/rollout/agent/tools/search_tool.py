import os
import json
import requests
import random
from typing import Optional, Dict, List, Any

from verl.workers.rollout.agent.tools.base_tool import BaseTool


class SearchTool(BaseTool):
    """
    Search tool that provides web search capability with caching.
    """

    def __init__(
        self,
        host: str,
        ports: List[str],
        max_results: int = 10,
        result_length: int = 1000,
        max_attempts: int = 10
    ):
        """
        Initialize the Google search tool.
        
        Args:
            host: 
            ports: 
            max_results: Maximum number of search results to return
            result_length: Maximum length of each result snippet
        """
        # API configuration
        self.headers = {'Content-Type': 'application/json'}
        self.api_urls = [f"http://{host}:{port}/search" for port in ports]

        self.max_results = max_results
        self.result_length = result_length
        self.max_attempts = max_attempts
        

    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "search"

    @property
    def trigger_tag(self) -> str:
        """Tag used to trigger this tool."""
        return "search"

    
    def execute(self, query: str, timeout: int = 60) -> str:
        """
        Execute Google search query.

        Args:
            query: Search query string
            timeout: request timeout in seconds

        Returns:
            Formatted search results as string
        """
        query = query.replace('"', '')
        
        attempts = 0
        tried_urls = set()

        payload = {'questions': [query],'N': self.max_results}
        
        while attempts < self.max_attempts:
            available_urls = [url for url in self.api_urls if url not in tried_urls]
            if not available_urls:
                available_urls = self.api_urls

            api_url = random.choice(available_urls)
            tried_urls.add(api_url)

            try:
                response = requests.post(api_url, headers=self.headers, data=json.dumps(payload), timeout=timeout)
                response.raise_for_status()

                data = response.json()
                if not data or "top_k_docs" not in data[0]:
                    raise ValueError("Unexpected API response format")

                return data[0]["top_k_docs"]

            except (requests.exceptions.RequestException, ValueError) as e:
                attempts += 1
                print(f"⚠️ 第 {attempts} 次尝试失败（URL: {api_url}）：{e}")

        raise RuntimeError(f"❌ 请求失败次数超过 {self.max_attempts} 次，无法获取文档")
