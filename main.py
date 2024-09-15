from enum import Enum

from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "hello "}

@app.post("/")
async def create_post():
    return {"message": "hello post "}

@app.put("/")
async def update_put():
    return {"message": "hello put "}

@app.get("/users")
async def get_all_users():
    return {"message": "list users route"}

@app.get("/users/me")
async def get_logged_in_user():
    return {"Message": "this is the current user"}

@app.get("/users/{user_id}")
async def get_specific_user(user_id: str):
    return {"user_id": user_id}

class CategoryEnum(str, Enum):
    fruits = "fruits"
    vegetables = "vegetables"
    dairy = "dairy"

@app.get("/categories/{category_name}")
async def get_category(category_name: CategoryEnum):
    if category_name == CategoryEnum.vegetables:
        return {"category_name": category_name, "message": "you are healthy"}

    if category_name.value == "fruits":
        return {
            "category_name": category_name,
            "message": "you are still healthy",
        }
    return {"category_name": category_name, "message": "i like chocolate milk"}

dummy_database = [{"item_title": "Foo"}, {"item_title": "Bar"}, {"item_title": "Baz"}]

@app.get("/old_data")
async def fetch_old_data(skip: int = 0, limit: int = 10):
    return dummy_database[skip : skip + limit]

@app.get("/items/{item_id}")
async def fetch_item(
    item_id: str, query_param: str, q: str | None = None, is_short: bool = False
):
    item = {"item_id": item_id, "query_param": query_param}
    if q:
        item.update({"q": q})
    if not is_short:
        item.update(
            {
                "description": " update description of an item"
            }
        )
    return item

@app.get("/users/{user_id}/items/{item_id}")
async def fetch_user_item(
    user_id: int, item_id: str, q: str | None = None, is_short: bool = False
):
    user_item = {"item_id": item_id, "owner_id": user_id}
    if q:
        user_item.update({"q": q})
    if not is_short:
        user_item.update(
            {
                "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut consectetur."
            }
        )
        return user_item

class Product(BaseModel):
    title: str
    description: str | None = None
    cost: float
    tax_amount: float | None = None

@app.post("/items")
async def add_item(product: Product):
    product_dict = product.dict()
    if product.tax_amount:
        total_price = product.cost + product.tax_amount
        product_dict.update({"total_price": total_price})
    return product_dict

@app.put("/items/{item_id}")
async def update_item_with_put(item_id: int, product: Product, q: str | None = None):
    response = {"item_id": item_id, **product.dict()}
    if q:
        response.update({"q": q})
    return response

@app.get("/search_items")
async def search_items(
    query: str
    | None = Query(
        None,
        min_length=3,
        max_length=10,
        title="Query string",
        description="A sample query string.",
        alias="search-query",
    )
):
    result_set = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if query:
        result_set.update({"query": query})
    return result_set


