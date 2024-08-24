import os, re
from diskcache import Cache
from langchain_community.utilities import BingSearchAPIWrapper

class BingSearch():
    def __init__(self) -> None:
        bing_subscription_key = os.environ.get('BING_SUBSCRIPTION_KEY')
        bing_search_url = 'https://api.bing.microsoft.com/v7.0/search'
        self.search = BingSearchAPIWrapper(bing_subscription_key=bing_subscription_key,
                                           bing_search_url=bing_search_url)
        self.cache = Cache('.bing_search_cache')

    def __call__(self, query, num_results=10):
        # Check if the query is already in the cache
        if query in self.cache:
            return self.cache[query]
        
        # If the query is not in the cache, perform a search
        results = self.search.results(query, num_results)
        results_string = self._process_results(results)
        self.cache[query] = results_string
        
        return results_string

    def _process_results(self, results):
        return "\n".join([self._process_result(result) for result in results])
    
    def _process_result(self, result: dict):
        # 1. Extract the title and remove all HTML tags
        title = result.get('title', '')
        title_cleaned = re.sub(r'<[^>]+>', '', title)  # Remove all HTML tags
        title_cleaned = re.sub(r'</b>', '', title_cleaned)  # Remove </b> tags
        title_cleaned = re.sub(r'<b>', '', title_cleaned)  # Remove <b> tags
        title_cleaned = title_cleaned.strip()

        # 2. Extract the snippet and remove HTML tags and </b> tags
        snippet = result.get('snippet', '')
        snippet_cleaned = re.sub(r'<[^>]+>', '', snippet)  # Remove all HTML tags
        snippet_cleaned = re.sub(r'</b>', '', snippet_cleaned)  # Remove </b> tags
        snippet_cleaned = re.sub(r'<b>', '', snippet_cleaned)  # Remove <b> tags
        snippet_cleaned = snippet_cleaned.strip()

        # 3. Concatenate processed search results into a string, each result consisting of a title and a snippet
        result_string = f'Title: {title_cleaned}\nSnippet: {snippet_cleaned}\n'
        return result_string
    
if __name__ == '__main__':
    bing_search = BingSearch()
    results = bing_search('How to make a sandwich?')
    print(results)
