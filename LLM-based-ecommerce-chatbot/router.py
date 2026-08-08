from semantic_router import Route, SemanticRouter
from semantic_router.encoders import FastEmbedEncoder

# 1. Define routes
politics = Route(
    name="politics",
    utterances=[
        "who's the president of the country",
        "what's the latest news on the election",
        "what's the government's policy on healthcare",
        "tell me about the senate vote",
    ],
)

chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today",
        "hi, how are you doing",
        "tell me a joke",
        "what's your favorite movie",
    ],
)

code_help = Route(
    name="code_help",
    utterances=[
        "why is my python code throwing an error",
        "help me debug this function",
        "how do I fix this traceback",
        "explain this stack overflow error",
    ],
)

routes = [politics, chitchat, code_help]

# 2. Local, free encoder
encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")

# 3. Build the router — auto_sync forces the index to build before use
router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

# 4. Route a query
result = router("what do you think about the new tax policy?")
print(result.name)

result = router("what's up, how's it going?")
print(result.name)

result = router("what is python")
print(result.name)

