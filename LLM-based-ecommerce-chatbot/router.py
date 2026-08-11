from semantic_router import Route, SemanticRouter
from semantic_router.encoders import FastEmbedEncoder

# ---- 1. Define routes ----
faq_route = Route(
    name="faq",
    utterances=[
        "how do i track my order?",
        "what is the refund policy",
        "what if i dont like my order",
    ],
)

sql = Route(
    name="sql",
    utterances=[
        "I want to buy yoga mat with 50% discount",
        "what is the price of workout shoes ",
        "Are there any mens shoes available on sale?",
    ],
)

code_help_route = Route(
    name="code_help",
    utterances=[
        "why is my python code throwing an error",
        "help me debug this function",
        "how do I fix this traceback",
        "explain this stack overflow error",
    ],
)

routes = [faq_route, sql, code_help_route]

# ---- 2. Local, free encoder ----
encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")

# ---- 3. Build the router ----
router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

# ---- 4. get_route — single entry point main.py calls ----
def get_route(query: str) -> str | None:
    result = router(query)
    return result.name
