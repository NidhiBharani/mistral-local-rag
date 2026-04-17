# import hashlib

# from gptcache import Cache
# from gptcache.adapter.api import init_similar_cache
# from langchain.cache import GPTCache

# from langchain import globals
# # Set llm_cache
# globals.set_llm_cache(True)

# def get_hashed_name(name):
#     return hashlib.sha256(name.encode()).hexdigest()


# def init_gptcache(cache_obj: Cache, llm: str):
#     hashed_llm = get_hashed_name(llm)
#     init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}")


# set_llm_cache(GPTCache(init_gptcache))


# from langchain.cache import InMemoryCache

# set_llm_cache(InMemoryCache())

import langchain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()

from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()
