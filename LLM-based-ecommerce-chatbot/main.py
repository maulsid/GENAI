from router import get_route
from faq import faq_chain
from sql import sql_chain

def handle_query(query: str) -> str:
    route_name = get_route(query)

    if route_name == "faq":
        return faq_chain(query)

    elif route_name == "sql":
        return sql_chain(query)
    elif route_name == "code_help":
        return "I can't help with coding questions — I'm an ecommerce support assistant."

    else:
        return "Sorry, I didn't understand that. Could you rephrase your question?"

if __name__ == "__main__":
    query = input("Enter your question: ")
    answer = handle_query(query)
    print(answer)
