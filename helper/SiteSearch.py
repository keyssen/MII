from helper.BloomFilter import BloomFilter


class SiteSearch:
    def __init__(self):
        self.filter: BloomFilter = BloomFilter(100000, 5)
        self.keyword_urls: dict[str, list[str]] = {}

    def add(self, url: str, keywords: list[str]) -> None:
        for keyword in keywords:
            lowercase_string = keyword.lower()
            self.filter.add(lowercase_string)
            if lowercase_string not in self.keyword_urls:
                self.keyword_urls[lowercase_string] = []
            self.keyword_urls[lowercase_string].append(url)

    def find_url(self, keyword: str) -> list[str]:
        lowercase_string = keyword.lower()
        if self.filter.contains(lowercase_string):
            return self.keyword_urls.get(lowercase_string)
        else:
            return []

    def contains(self, keyword: str) -> list[str]:
        lowercase_string = keyword.lower()
        if self.filter.contains(lowercase_string):
            return True
        else:
            return False