from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional
from uuid import UUID

from fastapi import Body, Cookie, FastAPI, Header, Query, Path
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI()

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


@app.get("/items_hidden")
async def hidden_query_route(
    hidden_query: str | None = Query(None, include_in_schema=False)
):
    if hidden_query:
        return {"hidden_query": hidden_query}
    return {"hidden_query": "Not found"}

@app.get("/items_validation/{item_id}")
async def read_items_validation(
    *,
    #item_id must be path parameter
    item_id: int = Path(..., title="The ID of the item to get", gt=10, le=100),
    q: str = "hello",
    #size must be query parameter
    size: float = Query(..., gt=0, lt=7.75)
):
    
    results = {"item_id": item_id, "size": size}
    if q:
        results.update({"q": q})
    return results


class Item(BaseModel):
    name: str
    description: str | None = Field(
        None, title="The description of the item", max_length=300
    )
    price: float = Field(..., gt=0, description="The price must be greater than zero.")
    tax: float | None = None

#passed request body in query parameter
@app.put("/items/{item_id}")
async def update_item1(item_id: int, item: Item = Body(..., embed=True)):
    results = {"item_id": item_id, "item": item}
    return results

class Image(BaseModel):
    url: HttpUrl
    name: str


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: set[str] = []
    image: list[Image] | None = None


class Offer(BaseModel):
    name: str
    description: str | None = None
    price: float
    items: list[Item]


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results

@app.post("/offers")
async def create_offer(offer: Offer = Body(..., embed=True)):
    return offer


@app.post("/images/multiple")
async def create_multiple_images(images: list[Image]):
    return images


@app.post("/blah")
async def create_some_blahs(blahs: dict[int, float]):
    return blahs

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None



@app.put("/items/{item_id}")
async def update_items2(
    item_id: int,
    item: Item = Body(
        ...,
        examples={
            "normal": {
                "summary": "A normal example",
                "description": "A __normal__ item works _correctly_",
                "value": {
                    "name": "Foo",
                    "description": "A very nice Item",
                    "price": 16.25,
                    "tax": 1.67,
                },
            },          
            "converted": {
                "summary": "An example with converted data",
                "description": "FastAPI can convert price `strings` to actual `numbers` automatically",
                "value": {"name": "Bar", "price": "16.25"},
            },
            "invalid": {
                "summary": "Invalid data is rejected with an error",
                "description": "Hello youtubers",
                "value": {"name": "Baz", "price": "sixteen point two five"},
            },
        },
    ),
):
    results = {"item_id": item_id, "item": item}
    return results


@app.put("/items/{item_id}")
async def read_items(
    item_id: UUID,
    start_date: Optional[datetime] = Body(None),
    end_date: Optional[datetime] = Body(None),
    repeat_at: Optional[time] = Body(None),
    process_after: Optional[timedelta] = Body(None),
):
    if start_date and process_after :
        start_process = start_date + process_after 
    else:
        None
    if start_process and end_date:
        duration = end_date - start_process 
    else:
        None
    return {
        "item_id": item_id,
        "start_date": start_date,
        "end_date": end_date,
        "repeat_at": repeat_at,
        "process_after": process_after,
        "start_process": start_process,
        "duration": duration,
    }

@app.get("/items")
async def read_items(
    cookie_id: str | None = Cookie(None),
    accept_encoding: str | None = Header(None),
    sec_ch_ua: str | None = Header(None),
    user_agent: str | None = Header(None),
    x_token: list[str] | None = Header(None),
):
    return {
        "cookie_id": cookie_id,
        "Accept-Encoding": accept_encoding,
        "sec-ch-ua": sec_ch_ua,
        "User-Agent": user_agent,
        "X-Token values": x_token,
    }