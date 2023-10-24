import hashlib
import mmh3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, item):
        for seed in range(self.hash_count):
            index = mmh3.hash(item, seed) % self.size
            self.bit_array[index] = 1

    def contains(self, item):
        for seed in range(self.hash_count):
            index = mmh3.hash(item, seed) % self.size
            if self.bit_array[index] == 0:
                return False
        return True


# search_engine = SiteSearch()
#
# # Company Valuation Country State City Industries Founded Year Name of Founders Total Funding Number of Employees
# # https://www.kaggle.com/datasets/ankanhore545/100-highest-valued-unicorns
# # search_engine.add_site("www.python.org", ["python", "code", "programming"])
# # search_engine.add_site("www.chatbot.com", ["chatbot", "AI", "messaging"])
# # search_engine.add_site("www.ai.com", ["AI", "machine learning", "data"])
# search_engine.add("https://www.kaggle.com/datasets/ankanhore545/100-highest-valued-unicorns", ["Company", "Valuation", "Country", "State", "City", "Industries", "Founded Year", "Name of Founders", "Total Funding", "Number of Employees"])
# search_engine.add("https://www.kaggle.com/datasets/ilyaryabov/tesla-insider-trading", ["Insider Trading", "Relationship", "Date", "Transaction", "Cost", "Shares", "Value", "Shares Total", "SEC Form 4"])
# search_engine.add("https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects", ["NASA", "est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "orbiting_body", "sentry_object", "absolute_magnitude", "hazardous"])
#
# keywords = search_engine.find_url("Country")
# print(keywords)  # ['python', 'code', 'programming']
#
# keywords = search_engine.find_url("Insider Trading")
# print(keywords)  # ['AI', 'machine learning', 'data']
#
# keywords = search_engine.find_url("NASA")
# print(keywords)  # []