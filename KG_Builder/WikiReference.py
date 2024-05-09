import time 
import wikipedia

class WikiAPI:
    def call_wiki_api(self, item, max_retries=2):
        retry_count = 0
        while retry_count < max_retries:
            try:
                data = wikipedia.search(item, results=1)
                if data:
                    return data
                else:
                    time.sleep(2 ** retry_count)
                    retry_count += 1
            except Exception as e:
                time.sleep(2 ** retry_count)
                retry_count += 1
        return None
